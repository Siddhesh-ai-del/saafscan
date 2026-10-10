"""Held-out-lighting evaluation: ML (primary) vs rule-table baseline.

The honest delta report for the competition deck. Everything is measured on
the **test split — lighting conditions the model never saw during training or
calibration** (split decided by ``condition.seed % 10``, never by random
images). Reproduce with::

    .venv/bin/python -m saafscan.ml.eval --dataset data/trainset \
        --models models --out models/eval_report.json

Semantics (kept identical to serving so the numbers describe the demo):

- ML verdicts follow ``runtime.verdict_from_probs``: argmax, but abstain to
  ``inconclusive`` when the calibrated confidence is below the threshold or
  the inconclusive class wins.
- ``ml.accuracy`` counts an abstention as a prediction of ``inconclusive`` —
  right only when the truth really is inconclusive; a fabricated verdict on a
  detected sample is always an error.
- ``ml.coverage`` is the fraction of samples that received a real food verdict
  (exactly what the demo would display), and the coverage curve sweeps the
  abstention threshold with selective accuracy (accuracy on decided samples).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from saafscan.classifier import classify_zones
from saafscan.ml.calibration import (
    apply_temperature,
    confusion_matrix,
    predict_with_abstention,
)
from saafscan.ml.features import FEATURE_NAMES

# Column positions of the raw zone RGB inside the feature vector, so the
# rule-table baseline can be re-run from exported rows without the strip image.
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_ZONE_RGB = ("control", "starch", "detergent")

# Threshold sweep for the accuracy-vs-coverage curve; the operating threshold
# itself is inserted so the curve always contains the shipped point.
_SWEEP = tuple(float(t) for t in np.linspace(0.0, 1.0, 11))


def _zones_from_features(row: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Reconstruct zone colours from a feature row (RGB columns are verbatim)."""
    return {
        zone: (
            float(row[_FEATURE_INDEX[f"{zone}_r"]]),
            float(row[_FEATURE_INDEX[f"{zone}_g"]]),
            float(row[_FEATURE_INDEX[f"{zone}_b"]]),
        )
        for zone in _ZONE_RGB
    }


def _inconclusive_index(classes: tuple[str, ...]) -> int:
    return classes.index("inconclusive") if "inconclusive" in classes else -1


def _accepted_mask(preds: np.ndarray, inconclusive_idx: int) -> np.ndarray:
    """Samples that received a real (non-inconclusive) verdict."""
    return preds >= 0 if inconclusive_idx < 0 else preds != inconclusive_idx


def _coverage_curve(
    ml_probs: np.ndarray,
    y_true: np.ndarray,
    classes: tuple[str, ...],
    abstain_threshold: float,
) -> list[dict[str, float]]:
    """Selective accuracy vs coverage across ascending abstention thresholds."""
    inconclusive_idx = _inconclusive_index(classes)
    thresholds = sorted(set(_SWEEP) | {float(abstain_threshold)})
    points: list[dict[str, float]] = []
    for tau in thresholds:
        preds = predict_with_abstention(ml_probs, tau, inconclusive_idx)
        accepted = _accepted_mask(preds, inconclusive_idx)
        coverage = float(accepted.mean()) if len(y_true) else 0.0
        n_acc = int(accepted.sum())
        accuracy = float((preds[accepted] == y_true[accepted]).mean()) if n_acc else 0.0
        points.append({"threshold": float(tau), "coverage": coverage, "accuracy": accuracy})
    return points


def build_eval_report(
    y_true: np.ndarray,
    ml_probs: np.ndarray,
    rule_preds: np.ndarray,
    classes: tuple[str, ...],
    abstain_threshold: float,
) -> dict[str, Any]:
    """Compare ML (with abstention) against the rule-table on one split.

    Args:
        y_true: True class indices, shape (n,).
        ml_probs: Temperature-calibrated ML probabilities, shape (n, n_classes).
        rule_preds: Rule-table class indices, shape (n,).
        classes: Class names in argmax order.
        abstain_threshold: The shipped abstention threshold (also swept).

    Returns:
        JSON-serialisable report with ``classes``, ``ml`` (accuracy, selective
        accuracy, coverage, abstained count, confusion, coverage curve),
        ``rules`` (accuracy, coverage, confusion) and ``delta_accuracy``
        (ml accuracy − rules accuracy, both overall on the same samples).

    Raises:
        ValueError: If array shapes disagree with ``classes``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    ml_probs = np.asarray(ml_probs, dtype=np.float64)
    rule_preds = np.asarray(rule_preds, dtype=np.int64)
    classes = tuple(classes)
    n_classes = len(classes)
    n = len(y_true)
    if ml_probs.shape != (n, n_classes):
        raise ValueError(f"ml_probs must be ({n}, {n_classes}), got {ml_probs.shape}")
    if rule_preds.shape != (n,):
        raise ValueError(f"rule_preds must be ({n},), got {rule_preds.shape}")

    inconclusive_idx = _inconclusive_index(classes)
    ml_preds = predict_with_abstention(ml_probs, abstain_threshold, inconclusive_idx)
    accepted = _accepted_mask(ml_preds, inconclusive_idx)
    n_acc = int(accepted.sum())

    ml_accuracy = float((ml_preds == y_true).mean()) if n else 0.0
    ml_coverage = float(n_acc / n) if n else 0.0
    ml_selective = float((ml_preds[accepted] == y_true[accepted]).mean()) if n_acc else 0.0

    rules_accuracy = float((rule_preds == y_true).mean()) if n else 0.0
    rules_coverage = float(_accepted_mask(rule_preds, inconclusive_idx).mean()) if n else 0.0

    return {
        "classes": list(classes),
        "abstain_threshold": float(abstain_threshold),
        "ml": {
            "accuracy": ml_accuracy,
            "selective_accuracy": ml_selective,
            "coverage": ml_coverage,
            "abstained": int(n - n_acc),
            "confusion": confusion_matrix(y_true, ml_preds, n_classes) if n else [],
            "coverage_curve": _coverage_curve(ml_probs, y_true, classes, abstain_threshold),
        },
        "rules": {
            "accuracy": rules_accuracy,
            "coverage": rules_coverage,
            "confusion": confusion_matrix(y_true, rule_preds, n_classes) if n else [],
        },
        "delta_accuracy": float(ml_accuracy - rules_accuracy),
    }


def _rule_predictions(x: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    """Run the rule-table baseline from feature rows (zones reconstructed)."""
    preds: list[int] = []
    for row in x:
        verdict = classify_zones(_zones_from_features(row))
        if verdict.adulterant not in classes:
            raise ValueError(f"rule-table returned unknown class {verdict.adulterant!r}")
        preds.append(classes.index(verdict.adulterant))
    return np.asarray(preds, dtype=np.int64)


def _load_spec(models_dir: Path, name: str) -> dict[str, Any]:
    spec_file = models_dir / f"{name}.json"
    if not spec_file.is_file():
        raise FileNotFoundError(f"model spec not found: {spec_file}")
    return json.loads(spec_file.read_text())


def _feature_probs(models_dir: Path, name: str, spec: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Batch inference for the features model, temperature-calibrated."""
    import onnxruntime as ort

    from saafscan.ml.runtime import select_probability_output

    onnx_file = models_dir / f"{name}.onnx"
    if not onnx_file.is_file():
        raise FileNotFoundError(f"model artifact not found: {onnx_file}")
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: x.astype(np.float32)})
    probs = select_probability_output(outputs, len(spec["classes"]), batch=len(x))
    return apply_temperature(probs, float(spec.get("temperature", 1.0)))


def _cnn_probs(
    dataset_dir: Path,
    manifest: dict[str, Any],
    models_dir: Path,
    name: str,
    spec: dict[str, Any],
    classes: tuple[str, ...],
    y_test: np.ndarray,
) -> np.ndarray:
    """CNN inference on the test-split strip crops, temperature-calibrated.

    Uses the exact preprocessing the CNN was trained with (``strip_to_chw``)
    so train and eval see identical tensors.
    """
    import onnxruntime as ort

    from saafscan.ml.preprocess import strip_to_chw
    from saafscan.ml.runtime import select_probability_output

    onnx_file = models_dir / f"{name}.onnx"
    if not onnx_file.is_file():
        raise FileNotFoundError(f"model artifact not found: {onnx_file}")

    test_samples = [s for s in manifest.get("samples", []) if s["split"] == "test"]
    if len(test_samples) != len(y_test):
        raise ValueError(
            f"manifest has {len(test_samples)} test samples but exported split has "
            f"{len(y_test)} — dataset and export are out of sync"
        )
    crops = []
    for sample in test_samples:
        if classes.index(sample["label"]) != int(y_test[len(crops)]):
            raise ValueError("test-split label order diverged between manifest and export")
        crop = np.load(dataset_dir / sample["path"])
        crops.append(strip_to_chw(crop))
    x = np.stack(crops).astype(np.float32)

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: x})
    probs = select_probability_output(outputs, len(classes), batch=len(x))
    return apply_temperature(probs, float(spec.get("temperature", 1.0)))


def run_eval(
    *,
    exported: dict[str, tuple[np.ndarray, np.ndarray]],
    models_dir: Path | str,
    active_name: str,
    classes: tuple[str, ...],
    dataset_manifest: dict[str, Any],
    dataset_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate one model against the rule-table on the held-out test split.

    Args:
        exported: Output of ``train.export_dataset``.
        models_dir: Directory holding ``<name>.json`` / ``<name>.onnx``.
        active_name: Which model to evaluate.
        classes: Class names in argmax order.
        dataset_manifest: The dataset manifest (for CNN crops + provenance).
        dataset_dir: Dataset root; required for ``kind == "cnn"`` models,
            which score strip crops rather than feature rows.

    Returns:
        Report from :func:`build_eval_report` plus ``active_model`` and a
        ``dataset`` provenance block.

    Raises:
        ValueError: CNN requested without ``dataset_dir``, or the test split
            is missing/out of sync.
        FileNotFoundError: Spec or ONNX artifact missing.
    """
    models_dir = Path(models_dir)
    classes = tuple(classes)
    spec = _load_spec(models_dir, active_name)
    if "test" not in exported:
        raise ValueError("exported dataset has no test split — nothing held out to evaluate")
    x_test, y_test = exported["test"]

    if spec.get("kind") == "cnn":
        if dataset_dir is None:
            raise ValueError(
                "CNN evaluation requires dataset_dir=... (its inputs are strip crops "
                "from the dataset, not exported feature rows)"
            )
        ml_probs = _cnn_probs(
            Path(dataset_dir), dataset_manifest, models_dir, active_name, spec, classes, y_test
        )
    else:
        ml_probs = _feature_probs(models_dir, active_name, spec, x_test)

    rule_preds = _rule_predictions(x_test, classes)
    report = build_eval_report(
        y_true=y_test,
        ml_probs=ml_probs,
        rule_preds=rule_preds,
        classes=classes,
        abstain_threshold=float(spec.get("abstain_threshold", 0.5)),
    )
    report["active_model"] = active_name
    test_conditions = {
        s["condition_key"] for s in dataset_manifest.get("samples", []) if s["split"] == "test"
    }
    report["dataset"] = {
        "split_policy": "by lighting condition (condition.seed % 10: "
        "0-6 train, 7-8 val, 9 test) — never random images",
        "n_test_samples": int(len(y_test)),
        "n_test_conditions": len(test_conditions),
        "skipped_undetectable": int(dataset_manifest.get("skipped_undetectable", 0)),
        "classes": list(classes),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate every model spec in a models dir against the baseline."""
    parser = argparse.ArgumentParser(
        prog="python -m saafscan.ml.eval",
        description="Held-out-lighting eval: ML vs rule-table baseline.",
    )
    parser.add_argument("--dataset", required=True, help="dataset dir with manifest.json")
    parser.add_argument("--models", default="models", help="dir holding model specs")
    parser.add_argument("--name", default=None, help="evaluate only this model")
    parser.add_argument("--out", default=None, help="write the full JSON report here")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    models_dir = Path(args.models)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    classes = tuple(manifest["classes"])

    if args.name:
        names = [args.name]
    else:
        names = sorted(p.stem for p in models_dir.glob("*.json") if p.stem != "active")
    if not names:
        print(f"no model specs found in {models_dir}")
        return 1

    from saafscan.ml.train import export_dataset

    exported = export_dataset(dataset_dir)
    active_pointer = models_dir / "active.json"
    active_model = None
    if active_pointer.is_file():
        active_model = json.loads(active_pointer.read_text()).get("model")

    reports: dict[str, Any] = {}
    for name in names:
        reports[name] = run_eval(
            exported=exported,
            models_dir=models_dir,
            active_name=name,
            classes=classes,
            dataset_manifest=manifest,
            dataset_dir=dataset_dir,
        )
        r = reports[name]
        print(
            f"{name:12s} ml_acc={r['ml']['accuracy']:.3f} "
            f"coverage={r['ml']['coverage']:.3f} "
            f"rules_acc={r['rules']['accuracy']:.3f} "
            f"delta={r['delta_accuracy']:+.3f}"
        )

    result = {"active_model": active_model, "reports": reports}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
