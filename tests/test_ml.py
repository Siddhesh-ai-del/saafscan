"""RED tests: ML classifier runtime — primary decision-maker with rules fallback.

Contracts under test:
- Without a trained artifact (or with SAAFSCAN_CLASSIFIER=rules) the wrapper
  returns exactly the rule-table verdict — the fallback is total, never fabricates.
- With a shipped artifact the pipeline is ML-primary and marks its evidence.
- Temperature scaling + abstention: low calibrated confidence → inconclusive.
- Control-drift (expired strip / bad imaging) is rejected before ML ever sees it.
- Determinism and live-demo latency budgets hold on the ML path too.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from saafscan.classifier import CONTROL_TOLERANCE, classify_zones, zone_features
from saafscan.roi import ZONE_NOMINAL
from saafscan.synthetic import render_sample
from saafscan.verdict import Verdict

REACTED = {
    "none": {
        "control": ZONE_NOMINAL["control"],
        "starch": ZONE_NOMINAL["starch"],
        "detergent": ZONE_NOMINAL["detergent"],
    },
    "starch": {
        "control": ZONE_NOMINAL["control"],
        "starch": (45, 40, 95),
        "detergent": ZONE_NOMINAL["detergent"],
    },
    "detergent": {
        "control": ZONE_NOMINAL["control"],
        "starch": ZONE_NOMINAL["starch"],
        "detergent": (214, 216, 220),
    },
    "both": {
        "control": ZONE_NOMINAL["control"],
        "starch": (45, 40, 95),
        "detergent": (214, 216, 220),
    },
}


def zones_for(label: str) -> dict[str, tuple[float, float, float]]:
    return {k: tuple(float(c) for c in v) for k, v in REACTED[label].items()}


@pytest.fixture(autouse=True)
def _fresh_runtime_cache(monkeypatch):
    """Every test starts without cached ONNX sessions or spec lookups."""
    from saafscan.ml import runtime

    runtime.reset_cache()
    yield
    runtime.reset_cache()


# --- calibration primitives (pure) --------------------------------------------


def test_apply_temperature_preserves_argmax_and_normalises() -> None:
    from saafscan.ml.calibration import apply_temperature

    probs = np.array([0.70, 0.20, 0.05, 0.04, 0.01])
    cooled = apply_temperature(probs, 2.0)
    assert cooled.sum() == pytest.approx(1.0)
    assert int(np.argmax(cooled)) == 0
    assert cooled[0] < probs[0], "T>1 must soften an overconfident distribution"


def test_apply_temperature_identity_at_one() -> None:
    from saafscan.ml.calibration import apply_temperature

    probs = np.array([0.5, 0.3, 0.2])
    assert np.allclose(apply_temperature(probs, 1.0), probs / probs.sum())


def test_fit_temperature_reduces_nll_and_is_finite() -> None:
    from saafscan.ml.calibration import apply_temperature, fit_temperature

    rng = np.random.default_rng(0)
    # Overconfident teacher: sharpen correct-class probs, then add label noise.
    probs = rng.dirichlet(np.full(5, 0.4), size=400)
    labels = np.array([rng.integers(0, 5) for _ in range(400)])
    labels[::7] = (labels[::7] + 1) % 5  # noise → model is overconfident
    temp = fit_temperature(probs, labels, n_classes=5)
    assert 0.1 <= temp <= 10.0 and math.isfinite(temp)

    def nll(t: float) -> float:
        scaled = apply_temperature(probs, t)
        return float(-np.mean(np.log(scaled[np.arange(len(labels)), labels] + 1e-12)))

    assert nll(temp) <= nll(1.0) + 1e-9, "fitted temperature must not be worse than T=1"


def test_select_abstain_threshold_meets_precision_target() -> None:
    from saafscan.ml.calibration import select_abstain_threshold

    # 20 clean high-confidence correct samples + 5 ambiguous low-confidence ones.
    probs = np.zeros((25, 5))
    labels = np.zeros(25, dtype=int)
    for i in range(20):
        probs[i] = [0.95, 0.02, 0.01, 0.1, 0.02] if i % 4 else [0.9, 0.05, 0.02, 0.01, 0.02]
        labels[i] = 0
    for i in range(20, 25):
        probs[i] = [0.3, 0.25, 0.2, 0.15, 0.1]
        labels[i] = i % 5  # mostly wrong / ambiguous
    tau = select_abstain_threshold(probs, labels, n_classes=5, min_precision=1.0)
    assert 0.0 < tau <= 1.0
    accepted = probs.max(axis=1) >= tau
    # At the precision target, every accepted prediction must be correct.
    assert all(probs[i].argmax() == labels[i] for i in np.where(accepted)[0])


def test_fit_abstention_policy_is_shared_by_both_trainers() -> None:
    """Model A and Model B must calibrate identically or the delta is a lie."""
    from saafscan.ml.calibration import fit_abstention_policy

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 60)
    probs = np.eye(5)[y] * 0.9 + 0.025  # row sums: 0.9 + 4*0.025 = 1.0
    temperature, tau = fit_abstention_policy(probs, y, 5)
    assert 0.1 <= temperature <= 10.0
    assert 0.55 <= tau <= 0.999

    # No calibration data → identity temperature, policy floor (nothing fitted).
    t0, tau0 = fit_abstention_policy(None, np.empty(0, np.int64), 5)
    assert t0 == 1.0 and tau0 == 0.55


def test_abstention_metrics_counts_abstentions_as_inconclusive() -> None:
    """Held-out metrics: abstaining is only correct when the truth is inconclusive."""
    from saafscan.ml.calibration import abstention_metrics

    probs = np.array(
        [
            [0.9, 0.025, 0.025, 0.025, 0.025],  # accepted, correct
            [0.3, 0.4, 0.1, 0.1, 0.1],  # conf 0.4 < 0.6 → abstain, truth 1 → wrong
            [0.05, 0.05, 0.05, 0.05, 0.8],  # top = inconclusive → abstain, truth 4 → right
        ]
    )
    y = np.array([0, 1, 4])
    m = abstention_metrics(probs, y, inconclusive_index=4, temperature=1.0, threshold=0.6)
    assert m["n"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["coverage"] == pytest.approx(1 / 3)
    assert m["abstained"] == 2
    assert m["confusion"][1][4] == 1  # truth 1 recorded as inconclusive
    assert m["confusion"][4][4] == 1

    # Empty split → zeros, never NaN (JSON reports must serialise).
    e = abstention_metrics(
        np.empty((0, 5)),
        np.empty(0, np.int64),
        inconclusive_index=4,
        temperature=1.0,
        threshold=0.6,
    )
    assert e["accuracy"] == 0.0 and e["coverage"] == 0.0 and e["confusion"] == []


def test_confusion_matrix_is_counts_over_classes() -> None:
    from saafscan.ml.calibration import confusion_matrix

    y_true = np.array([0, 0, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0, 2])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    assert cm == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]
    assert sum(sum(row) for row in cm) == len(y_true)


def test_accuracy_at_threshold_reports_coverage() -> None:
    from saafscan.ml.calibration import accuracy_at_threshold

    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    labels = np.array([0, 0, 0, 1])
    full = accuracy_at_threshold(probs, labels, n_classes=2, threshold=0.0)
    assert full["coverage"] == pytest.approx(1.0)
    assert full["accuracy"] == pytest.approx(0.75)  # 3/4 accepted correctly
    strict = accuracy_at_threshold(probs, labels, n_classes=2, threshold=0.8)
    assert strict["coverage"] == pytest.approx(0.5)
    assert strict["accuracy"] == pytest.approx(1.0)  # both accepted samples correct


# --- verdict construction from probabilities ----------------------------------


def _spec(**overrides):
    from saafscan.ml.runtime import ModelSpec

    base = dict(
        name="unit",
        kind="features",
        classes=("none", "starch", "detergent", "both", "inconclusive"),
        temperature=1.0,
        abstain_threshold=0.6,
        onnx_path="/nonexistent/model.onnx",
    )
    base.update(overrides)
    return ModelSpec(**base)


def test_predict_with_abstention_mirrors_serving_rule() -> None:
    """The eval harness must abstain exactly like verdict_from_probs does.

    Serving rule: top class is inconclusive OR confidence < τ → abstain.
    If eval and serving disagreed on abstention, the reported coverage would
    be a lie about what the demo actually does.
    """
    from saafscan.ml.calibration import predict_with_abstention

    probs = np.array(
        [
            [0.9, 0.025, 0.025, 0.025, 0.025],  # confident class 0 → verdict
            [0.05, 0.05, 0.05, 0.05, 0.8],  # top = inconclusive → abstain
            [0.3, 0.4, 0.1, 0.1, 0.1],  # conf 0.4 < τ=0.6 → abstain
            [0.1, 0.7, 0.1, 0.05, 0.05],  # confident class 1 → verdict
        ]
    )
    pred = predict_with_abstention(probs, threshold=0.6, inconclusive_index=4)
    assert pred.tolist() == [0, 4, 4, 1]
    assert pred.dtype == np.int64

    # No inconclusive class in the label set → threshold-only abstention,
    # with -1 as the "no verdict issued" sentinel.
    no_ic = predict_with_abstention(probs, threshold=0.6, inconclusive_index=-1)
    assert no_ic.tolist() == [0, 4, -1, 1]

    # Empty input stays empty (no crash, no fabricated rows).
    empty = predict_with_abstention(np.zeros((0, 5)), threshold=0.6, inconclusive_index=4)
    assert empty.shape == (0,)

    # Wrong class width is caught at the boundary.
    with pytest.raises(ValueError):
        predict_with_abstention(np.zeros((3, 4)), threshold=0.6, inconclusive_index=4)


def test_verdict_from_probs_maps_class_and_severity() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    probs = np.array([0.02, 0.95, 0.01, 0.01, 0.01])
    verdict = verdict_from_probs(probs, _spec())
    assert verdict.adulterant == "starch"
    assert verdict.confidence == pytest.approx(0.95)
    assert verdict.severity == "high"
    assert verdict.evidence["ml_used"] == 1.0


def test_verdict_from_probs_abstains_below_threshold() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    probs = np.array([0.34, 0.30, 0.20, 0.10, 0.06])
    verdict = verdict_from_probs(probs, _spec())
    assert verdict.adulterant == "inconclusive"
    assert verdict.confidence <= 0.6
    assert verdict.severity == "low"


def test_verdict_from_probs_never_accepts_out_of_domain_argmax() -> None:
    """A prob vector whose argmax is not a known class must abstain."""
    from saafscan.ml.runtime import verdict_from_probs

    probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # 6 entries vs 5 classes
    with pytest.raises(ValueError):
        verdict_from_probs(probs, _spec())


def test_verdict_from_probs_none_is_not_high_severity() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    probs = np.array([0.97, 0.01, 0.01, 0.0, 0.01])
    verdict = verdict_from_probs(probs, _spec())
    assert verdict.adulterant == "none"
    assert verdict.severity == "none"


# --- feature extraction --------------------------------------------------------


def test_extract_features_is_deterministic_and_fixed_length() -> None:
    from saafscan.ml.features import FEATURE_NAMES, extract_features

    a = extract_features(zones_for("starch"))
    b = extract_features(zones_for("starch"))
    assert a.shape == (len(FEATURE_NAMES),)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_extract_features_separates_classes_by_zone_delta() -> None:
    from saafscan.ml.features import FEATURE_NAMES, extract_features

    idx = FEATURE_NAMES.index("starch_deltaE")
    none_v = extract_features(zones_for("none"))
    starch_v = extract_features(zones_for("starch"))
    assert starch_v[idx] > none_v[idx] + 20, "starch reaction must raise its ΔE feature"


def test_extract_features_rejects_incomplete_zones() -> None:
    from saafscan.ml.features import extract_features

    with pytest.raises(ValueError):
        extract_features({"control": (1, 2, 3)})


# --- wrapper fallback -----------------------------------------------------------


def test_fallback_identical_to_rules_without_model(tmp_path, monkeypatch) -> None:
    from saafscan.ml import runtime

    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))  # empty → no artifact
    for label in ("none", "starch", "detergent", "both"):
        zones = zones_for(label)
        ml_v = runtime.classify(zones)
        rule_v = classify_zones(zones)
        assert ml_v.adulterant == rule_v.adulterant
        assert ml_v.confidence == pytest.approx(rule_v.confidence)
        assert ml_v.evidence["ml_used"] == 0.0


def test_env_flag_forces_rules_even_with_models(tmp_path, monkeypatch) -> None:
    from saafscan.ml import runtime

    monkeypatch.setenv("SAAFSCAN_CLASSIFIER", "rules")
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    verdict = runtime.classify(zones_for("starch"))
    assert verdict.evidence["ml_used"] == 0.0
    assert verdict.adulterant == "starch"


def test_corrupt_artifact_falls_back_to_rules(tmp_path, monkeypatch) -> None:
    """A broken model file must degrade to rules, never crash the demo."""
    import json

    from saafscan.ml import runtime

    spec = {
        "name": "broken",
        "kind": "features",
        "classes": ["none", "starch", "detergent", "both", "inconclusive"],
        "temperature": 1.0,
        "abstain_threshold": 0.6,
    }
    (tmp_path / "broken.json").write_text(json.dumps(spec))
    (tmp_path / "broken.onnx").write_bytes(b"this is not an onnx graph")
    (tmp_path / "active.json").write_text(json.dumps({"model": "broken"}))
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    verdict = runtime.classify(zones_for("detergent"))
    assert verdict.adulterant == "detergent"
    assert verdict.evidence["ml_used"] == 0.0


def test_missing_active_pointer_falls_back_to_rules(tmp_path, monkeypatch) -> None:
    from saafscan.ml import runtime

    (tmp_path / "some_model.json").write_text("{}")
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    verdict = runtime.classify(zones_for("both"))
    assert verdict.evidence["ml_used"] == 0.0


# --- shipped artifact behaviour --------------------------------------------------


def _active_spec():
    from saafscan.ml.runtime import load_active_spec

    return load_active_spec()


def test_shipped_model_metadata_is_valid() -> None:
    spec = _active_spec()
    assert spec is not None, "Phase 7 must ship a trained active model"
    assert spec.kind in {"features", "cnn"}
    assert spec.classes == ("none", "starch", "detergent", "both", "inconclusive")
    assert 0.1 <= spec.temperature <= 10.0
    assert 0.5 < spec.abstain_threshold <= 1.0
    from pathlib import Path

    assert Path(spec.onnx_path).is_file(), "active spec must point at a real .onnx"


def test_shipped_model_used_by_default_and_marks_evidence() -> None:
    import os

    from saafscan.ml import runtime

    assert os.environ.get("SAAFSCAN_CLASSIFIER", "ml") == "ml"
    verdict = runtime.classify(zones_for("starch"), strip=np.zeros((90, 360, 3), np.uint8))
    assert verdict.evidence["ml_used"] == 1.0
    assert "prob_starch" in verdict.evidence
    assert verdict.adulterant == "starch"


@pytest.mark.parametrize("label", ["none", "starch", "detergent", "both"])
def test_shipped_model_agrees_with_rules_on_golden_scenes(label: str) -> None:
    """The demo gauntlet scenes must survive the ML swap unchanged."""
    from saafscan.pipeline import analyze

    verdict = analyze(render_sample(label, seed=7)).verdict
    assert verdict.adulterant == label


def test_ml_path_is_deterministic() -> None:
    from saafscan.ml import runtime

    a = runtime.classify(zones_for("both"), strip=np.zeros((90, 360, 3), np.uint8))
    b = runtime.classify(zones_for("both"), strip=np.zeros((90, 360, 3), np.uint8))
    assert a.adulterant == b.adulterant
    assert a.confidence == pytest.approx(b.confidence)


def test_ml_inference_is_fast_enough_for_live_demo() -> None:
    from saafscan.ml import runtime

    strip = np.zeros((90, 360, 3), np.uint8)
    runtime.classify(zones_for("starch"), strip=strip)  # warm-up (session load)
    start = time.perf_counter()
    runtime.classify(zones_for("starch"), strip=strip)
    assert (time.perf_counter() - start) * 1000 < 500


# --- abstention and safety --------------------------------------------------------


def test_control_drift_is_rejected_before_ml() -> None:
    """Expired strip / bad imaging: control moved → inconclusive, ML not consulted."""
    from saafscan.ml import runtime

    drifted = dict(zones_for("starch"))
    drifted["control"] = (250.0, 240.0, 240.0)  # far beyond CONTROL_TOLERANCE
    assert zone_features(drifted, ZONE_NOMINAL)["control"] > CONTROL_TOLERANCE
    verdict = runtime.classify(drifted, strip=np.zeros((90, 360, 3), np.uint8))
    assert verdict.adulterant == "inconclusive"
    assert verdict.confidence <= 0.5


def test_flat_distribution_always_abstains() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    flat = np.full(5, 0.2)
    verdict = verdict_from_probs(flat, _spec(abstain_threshold=0.6))
    assert verdict.adulterant == "inconclusive"
    assert verdict.evidence["ml_used"] == 1.0


def test_inconclusive_class_argmax_stays_inconclusive() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    probs = np.array([0.05, 0.05, 0.05, 0.05, 0.8])
    verdict = verdict_from_probs(probs, _spec())
    assert verdict.adulterant == "inconclusive"
    assert verdict.severity == "low"


def test_ml_verdicts_are_valid_verdict_objects() -> None:
    from saafscan.ml.runtime import verdict_from_probs

    for probs in (np.array([0.9, 0.03, 0.02, 0.03, 0.02]), np.array([0.2] * 5)):
        verdict = verdict_from_probs(probs, _spec())
        assert isinstance(verdict, Verdict)  # validates ranges in __post_init__


# --- explicit model selection (loser behind a flag) -------------------------------


def _write_tiny_model(models_dir, name: str, kind: str) -> None:
    """Write a valid 2-node ONNX graph + spec JSON so the loader accepts it."""
    import json

    from onnx import TensorProto, helper

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.onnx").write_bytes(model.SerializeToString())
    (models_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "kind": kind,
                "classes": ["none", "starch", "detergent", "both", "inconclusive"],
                "temperature": 1.0,
                "abstain_threshold": 0.6,
            }
        )
    )


def test_ml_model_env_overrides_active_pointer(tmp_path, monkeypatch) -> None:
    """The loser model stays servable behind SAAFSCAN_ML_MODEL."""
    import json

    from saafscan.ml import runtime

    _write_tiny_model(tmp_path, "cnn", "cnn")
    _write_tiny_model(tmp_path, "features", "features")
    (tmp_path / "active.json").write_text(json.dumps({"model": "cnn"}))
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))

    assert runtime.load_active_spec().name == "cnn"
    monkeypatch.setenv("SAAFSCAN_ML_MODEL", "features")
    chosen = runtime.load_active_spec()
    assert chosen.name == "features" and chosen.kind == "features"
    monkeypatch.delenv("SAAFSCAN_ML_MODEL")
    assert runtime.load_active_spec().name == "cnn"


def test_active_pointer_reports_winner() -> None:
    from saafscan.ml.runtime import active_model_name

    name = active_model_name()
    assert name in {"cnn", "features", "rules"}


# --- CNN serving path ---------------------------------------------------------------


def _write_cnn_stub(models_dir, name: str = "cnn") -> None:
    """Minimal image-input graph: global-average-pool → pad to 5 → softmax.

    Accepts exactly the tensor the CNN kind is served (1, 3, 90, 360) and
    emits a single float probability row.
    """
    import json

    from onnx import TensorProto, helper

    pad = helper.make_tensor("pad", TensorProto.FLOAT, [1, 2], [0.0, 0.0])
    nodes = [
        helper.make_node("GlobalAveragePool", ["x"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("Constant", [], ["z"], value=pad),
        helper.make_node("Concat", ["flat", "z"], ["cat"], axis=1),
        helper.make_node("Softmax", ["cat"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "cnn_stub",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 90, 360])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 13  # onnxruntime 1.31 supports IR ≤ 13
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.onnx").write_bytes(model.SerializeToString())
    (models_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "name": name,
                "kind": "cnn",
                "classes": ["none", "starch", "detergent", "both", "inconclusive"],
                "temperature": 1.0,
                "abstain_threshold": 0.6,
            }
        )
    )


def test_cnn_kind_consumes_strip_image(tmp_path, monkeypatch) -> None:
    """kind="cnn" scores the strip crop, not the engineered feature row."""
    import json

    from saafscan.ml import runtime

    _write_cnn_stub(tmp_path, "cnn")
    (tmp_path / "active.json").write_text(json.dumps({"model": "cnn"}))
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SAAFSCAN_ML_MODEL", raising=False)

    strip = np.full((90, 360, 3), 250, np.uint8)
    verdict = runtime.classify(zones_for("starch"), strip=strip)
    assert verdict.evidence["ml_used"] == 1.0
    assert {k for k in verdict.evidence if k.startswith("prob_")} == {
        "prob_none",
        "prob_starch",
        "prob_detergent",
        "prob_both",
        "prob_inconclusive",
    }


def test_cnn_kind_without_strip_falls_back_to_rules(tmp_path, monkeypatch) -> None:
    """No strip image → the CNN cannot be consulted; rules answer instead.

    Feeding feature rows into an image-shaped graph would crash — and
    guessing a score instead would fabricate a verdict.
    """
    import json

    from saafscan.classifier import classify_zones
    from saafscan.ml import runtime

    _write_cnn_stub(tmp_path, "cnn")
    (tmp_path / "active.json").write_text(json.dumps({"model": "cnn"}))
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SAAFSCAN_ML_MODEL", raising=False)

    verdict = runtime.classify(zones_for("starch"))
    assert verdict.evidence["ml_used"] == 0.0
    assert not any(k.startswith("prob_") for k in verdict.evidence)
    rules = classify_zones(zones_for("starch"))
    assert verdict.adulterant == rules.adulterant
    assert verdict.confidence == rules.confidence


def test_cnn_kind_with_bad_strip_shape_falls_back_to_rules(tmp_path, monkeypatch) -> None:
    """A crop the preprocessing rejects must reach the rules, not crash."""
    import json

    from saafscan.ml import runtime

    _write_cnn_stub(tmp_path, "cnn")
    (tmp_path / "active.json").write_text(json.dumps({"model": "cnn"}))
    monkeypatch.setenv("SAAFSCAN_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("SAAFSCAN_ML_MODEL", raising=False)

    verdict = runtime.classify(zones_for("starch"), strip=np.zeros((10, 10, 3), np.uint8))
    assert verdict.evidence["ml_used"] == 0.0
    assert verdict.adulterant != ""


def test_pipeline_supplies_strip_crop_to_classifier(monkeypatch) -> None:
    """The pipeline hands classify() the canonical crop the CNN was trained on.

    The crop must be the white-balanced warp whose zone means are exactly the
    zones the same call receives — otherwise model and rules would reason
    about different pixels.
    """
    import saafscan.pipeline as pipeline_mod
    from saafscan.roi import zones_from_strip
    from saafscan.synthetic import render_sample

    seen: dict = {}
    real = pipeline_mod.ml_classify

    def spy(zones, strip=None):
        seen["zones"], seen["strip"] = zones, strip
        return real(zones, strip=strip)

    monkeypatch.setattr(pipeline_mod, "ml_classify", spy)
    analysis = pipeline_mod.analyze(render_sample("starch", seed=7))

    strip = seen["strip"]
    assert strip is not None and strip.shape == (90, 360, 3)
    for zone, rgb in zones_from_strip(strip).items():
        assert np.allclose(rgb, seen["zones"][zone])
    assert analysis.verdict.adulterant == "starch"


# --- probability-output selection -------------------------------------------------


def test_predict_probs_skips_int64_label_output(tmp_path) -> None:
    """Classifier exports emit [label(int64), probabilities(float)] in that order.

    The runtime must temperature-scale the probability row, never the class
    index — scaling a label would fabricate a confident verdict from nothing.
    """
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from saafscan.ml import runtime as rt

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [5])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    label_value = helper.make_tensor("label_value", TensorProto.INT64, [1], [3])
    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["probabilities"]),
        helper.make_node("Constant", [], ["label"], value=label_value),
    ]
    graph = helper.make_graph(
        nodes,
        "label_first",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 12])],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 5]),
        ],
        initializer=[starts, ends, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 13  # onnxruntime 1.31 supports IR ≤ 13
    path = tmp_path / "label_first.onnx"
    path.write_bytes(model.SerializeToString())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    spec = _spec(name="label_first", kind="features", onnx_path=str(path), temperature=1.0)
    probs = rt._predict_probs(session, spec, zones_for("starch"))
    assert probs.shape == (5,), "must select the 5-class probability row, not the label"
    assert probs.sum() == pytest.approx(1.0)
    assert float(probs.min()) > 0.0


def test_predict_probs_raises_when_no_probability_output(tmp_path) -> None:
    """No float n_classes-wide output → raise, so classify() falls back to rules."""
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from saafscan.ml import runtime as rt

    label_value = helper.make_tensor("label_value", TensorProto.INT64, [1], [3])
    nodes = [helper.make_node("Constant", [], ["label"], value=label_value)]
    graph = helper.make_graph(
        nodes,
        "label_only",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 12])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 13  # onnxruntime 1.31 supports IR ≤ 13
    path = tmp_path / "label_only.onnx"
    path.write_bytes(model.SerializeToString())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    spec = _spec(name="label_only", kind="features", onnx_path=str(path))
    with pytest.raises(ValueError):
        rt._predict_probs(session, spec, zones_for("starch"))
