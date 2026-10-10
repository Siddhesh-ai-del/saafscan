"""ML runtime: ONNX classifier as primary, rule-table as total fallback.

Decision order for :func:`classify`:

1. ``SAAFSCAN_CLASSIFIER=rules`` → rule-table, always.
2. Control-zone drift beyond tolerance → ``inconclusive`` (strip/image bad;
   ML is never consulted).
3. Load the active model spec + ONNX session. Missing / unloadable → rule-table.
4. Run the model, temperature-calibrate, and abstain to ``inconclusive`` when
   the calibrated confidence is below the threshold.

Any failure in the ML path degrades to the rule-table verdict rather than
crashing the demo. The rule-table is never rewritten — only wrapped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from saafscan.classifier import _ADVICE, CONTROL_TOLERANCE, classify_zones, zone_features
from saafscan.ml.calibration import apply_temperature
from saafscan.ml.features import extract_features
from saafscan.roi import ZONE_NOMINAL
from saafscan.verdict import Verdict

# Order the model was trained with; index positions are the argmax → class map.
CLASSES: tuple[str, ...] = ("none", "starch", "detergent", "both", "inconclusive")

_SEVERITY_NONE = "none"
_SEVERITY_LOW = "low"
_SEVERITY_MODERATE = "moderate"
_SEVERITY_HIGH = "high"

# Calibrated-confidence → severity bands (severity only applies to a real verdict).
_SEVERITY_BANDS = ((0.90, _SEVERITY_HIGH), (0.70, _SEVERITY_MODERATE), (0.0, _SEVERITY_LOW))


def _default_models_dir() -> Path:
    env = os.environ.get("SAAFSCAN_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "models"


@dataclass(frozen=True)
class ModelSpec:
    """A trained model's metadata, loaded from ``<name>.json``."""

    name: str
    kind: str  # "features" | "cnn"
    classes: tuple[str, ...]
    temperature: float
    abstain_threshold: float
    onnx_path: str

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any], models_dir: Path) -> ModelSpec:
        return cls(
            name=data.get("name", name),
            kind=data["kind"],
            classes=tuple(data["classes"]),
            temperature=float(data.get("temperature", 1.0)),
            abstain_threshold=float(data.get("abstain_threshold", 0.5)),
            onnx_path=str(models_dir / f"{name}.onnx"),
        )


_SESSIONS: dict[str, Any] = {}
_SPEC_CACHE: dict[str, ModelSpec | None] = {}


def reset_cache() -> None:
    """Drop cached sessions and specs (called between tests / after retraining)."""
    _SESSIONS.clear()
    _SPEC_CACHE.clear()


def _resolve_spec_name() -> str | None:
    """Which model is active: env override, else ``active.json``, else None."""
    env = os.environ.get("SAAFSCAN_ML_MODEL")
    if env:
        return env
    pointer = _default_models_dir() / "active.json"
    if not pointer.is_file():
        return None
    try:
        return json.loads(pointer.read_text()).get("model")
    except (json.JSONDecodeError, OSError):
        return None


def load_active_spec() -> ModelSpec | None:
    """Load the active model spec, or ``None`` if absent/unreadable."""
    name = _resolve_spec_name()
    if not name:
        return None
    if name in _SPEC_CACHE:
        return _SPEC_CACHE[name]
    models_dir = _default_models_dir()
    spec_file = models_dir / f"{name}.json"
    spec: ModelSpec | None = None
    if spec_file.is_file():
        try:
            spec = ModelSpec.from_json(name, json.loads(spec_file.read_text()), models_dir)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            spec = None
    _SPEC_CACHE[name] = spec
    return spec


def _open_session(spec: ModelSpec):
    """Open (and cache) a CPU-only ONNX session for ``spec``.

    Raises:
        FileNotFoundError: If the ``.onnx`` file does not exist.
        RuntimeError: If onnxruntime cannot load the graph.
    """
    path = Path(spec.onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"model artifact not found: {path}")
    cached = _SESSIONS.get(spec.name)
    if cached is not None:
        return cached
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("onnxruntime is not installed") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _SESSIONS[spec.name] = session
    return session


def _severity_for(confidence: float, adulterant: str) -> str:
    if adulterant == "none":
        return _SEVERITY_NONE
    for floor, label in _SEVERITY_BANDS:
        if confidence >= floor:
            return label
    return _SEVERITY_LOW


def _make_evidence(
    zones: dict[str, tuple[float, float, float]], probs: np.ndarray | None, ml_used: bool
) -> dict[str, float]:
    """ΔE evidence always; per-class probabilities when ML decided."""
    evidence = {
        f"{k}_deltaE": round(float(v), 2) for k, v in zone_features(zones, ZONE_NOMINAL).items()
    }
    evidence["ml_used"] = 1.0 if ml_used else 0.0
    if probs is not None:
        for cls, p in zip(CLASSES, probs, strict=True):
            evidence[f"prob_{cls}"] = round(float(p), 4)
    return evidence


def _rules_with_evidence(
    zones: dict[str, tuple[float, float, float]], evidence: dict[str, float]
) -> Verdict:
    """Rule-table verdict, overlaid with the wrapper's evidence dict."""
    verdict = classify_zones(zones)
    return Verdict(
        adulterant=verdict.adulterant,
        confidence=verdict.confidence,
        severity=verdict.severity,
        advice=verdict.advice,
        evidence=evidence,
    )


def verdict_from_probs(probs: np.ndarray, spec: ModelSpec) -> Verdict:
    """Map a calibrated probability vector to a Verdict, abstaining when unsure.

    Args:
        probs: Length-``len(spec.classes)`` calibrated probabilities.
        spec: The model spec providing classes and the abstention threshold.

    Returns:
        A Verdict; ``inconclusive`` when the top class is inconclusive or its
        confidence falls below ``spec.abstain_threshold``.

    Raises:
        ValueError: If ``probs`` length does not match ``spec.classes``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[-1] != len(spec.classes):
        raise ValueError(f"expected {len(spec.classes)} probabilities, got {probs.shape[-1]}")

    top_idx = int(np.argmax(probs))
    top_class = spec.classes[top_idx]
    confidence = float(probs[top_idx])
    evidence: dict[str, float] = {"ml_used": 1.0}
    for cls, p in zip(spec.classes, probs, strict=True):
        evidence[f"prob_{cls}"] = round(float(p), 4)

    if top_class == "inconclusive" or confidence < spec.abstain_threshold:
        return Verdict(
            adulterant="inconclusive",
            confidence=round(confidence, 4),
            severity=_SEVERITY_LOW,
            advice=_ADVICE["inconclusive"],
            evidence=evidence,
        )

    return Verdict(
        adulterant=top_class,
        confidence=round(confidence, 4),
        severity=_severity_for(confidence, top_class),
        advice=_ADVICE.get(top_class, _ADVICE["inconclusive"]),
        evidence=evidence,
    )


def select_probability_output(
    outputs: list[Any], n_classes: int, batch: int | None = None
) -> np.ndarray:
    """Pick the float probability matrix out of a session's raw outputs.

    Classifier exports emit an int64 ``label`` output *first*; reading it as a
    confidence would silently fabricate a verdict from a class index. This
    selects by dtype and width instead of by position, so the rule holds for
    any export order or graph that hides extra outputs.

    Args:
        outputs: Raw list returned by ``session.run(None, ...)``.
        n_classes: Expected class dimension.
        batch: Expected batch size, or None to accept any.

    Returns:
        float64 probability array of shape (batch, n_classes).

    Raises:
        ValueError: If no output matches — callers fall back to the rule-table
            rather than guessing which tensor is the confidence.
    """
    for raw in outputs:
        arr = np.asarray(raw)
        if arr.dtype.kind != "f":
            continue
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            if batch is None or arr.shape[0] == batch:
                return arr.astype(np.float64)
        elif arr.ndim == 1 and arr.shape[0] == n_classes and batch in (None, 1):
            return arr.reshape(1, -1).astype(np.float64)
    raise ValueError(
        f"no {n_classes}-class float probability output among {len(outputs)} session outputs; "
        "refusing to score it"
    )


def _predict_probs(
    session: Any, spec: ModelSpec, zones: dict[str, tuple[float, float, float]]
) -> np.ndarray:
    """Run the session on the engineered features and temperature-calibrate.

    The probability row is selected by dtype/width (:func:`select_probability_output`)
    rather than by index, and a graph without one raises so ``classify`` falls
    back to the rule-table instead of trusting a class index as confidence.
    """
    features = extract_features(zones).reshape(1, -1).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features})
    probs = select_probability_output(outputs, len(spec.classes), batch=1).reshape(-1)
    return apply_temperature(probs, spec.temperature)


def classify(
    zones: dict[str, tuple[float, float, float]], strip: np.ndarray | None = None
) -> Verdict:
    """Classify reaction zones: ML-primary, rule-table fallback.

    Args:
        zones: Measured zone colours (control/starch/detergent → RGB).
        strip: Unused for the features model; accepted so the CNN kind can share
            the call signature. Safe to pass ``None``.

    Returns:
        A Verdict. Falls back to the rule-table whenever the ML path is
        unavailable or the input is out of distribution.
    """
    del strip  # features model does not consume the strip image
    if os.environ.get("SAAFSCAN_CLASSIFIER", "ml") == "rules":
        return _rules_with_evidence(zones, _make_evidence(zones, None, ml_used=False))

    if "control" in zones and zone_features(zones, ZONE_NOMINAL)["control"] > CONTROL_TOLERANCE:
        # Expired strip / bad imaging: the control zone reacted. Never diagnose.
        return Verdict(
            adulterant="inconclusive",
            confidence=0.3,
            severity=_SEVERITY_LOW,
            advice=_ADVICE["inconclusive"],
            evidence=_make_evidence(zones, None, ml_used=False),
        )

    spec = load_active_spec()
    if spec is None:
        return _rules_with_evidence(zones, _make_evidence(zones, None, ml_used=False))

    try:
        session = _open_session(spec)
        probs = _predict_probs(session, spec, zones)
    except Exception:
        # Total fallback: a broken/absent model must not break the demo.
        return _rules_with_evidence(zones, _make_evidence(zones, None, ml_used=False))

    verdict = verdict_from_probs(probs, spec)
    # Preserve the always-on ΔE evidence alongside the ML probabilities.
    merged = _make_evidence(zones, probs, ml_used=True)
    return Verdict(
        adulterant=verdict.adulterant,
        confidence=verdict.confidence,
        severity=verdict.severity,
        advice=verdict.advice,
        evidence=merged,
    )


def active_model_name() -> str:
    """Name of the active model, or ``"rules"`` when no artifact is active."""
    if os.environ.get("SAAFSCAN_CLASSIFIER", "ml") == "rules":
        return "rules"
    spec = load_active_spec()
    return spec.name if spec else "rules"
