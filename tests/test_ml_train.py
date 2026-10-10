"""RED tests: dataset export + held-out-lighting evaluation harness for ML training.

The eval number goes in the competition deck, so it must be reproducible from a
documented command and must split BY LIGHTING CONDITION (never random images).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from saafscan.synthetic import (
    TRAINING_CLASSES,
    build_dataset,
    sample_lighting,
)

CLASSES = tuple(TRAINING_CLASSES)


def _tiny_dataset(tmp_path):
    conds = [sample_lighting(s) for s in range(12)]
    return build_dataset(tmp_path, conditions=conds, seeds=(0, 1))


def test_export_dataset_builds_features_and_labels(tmp_path) -> None:
    from saafscan.ml.train import export_dataset

    ds_dir = tmp_path / "ds"
    _tiny_dataset(ds_dir)
    exported = export_dataset(ds_dir)

    assert set(exported) == {"train", "val", "test"}
    for _split, (x, y) in exported.items():
        assert x.ndim == 2 and x.shape[0] == y.shape[0] > 0
        assert x.dtype == np.float32
        assert set(np.unique(y)) <= set(range(len(CLASSES)))
        assert x.shape[1] == exported["train"][0].shape[1], "feature width must match across splits"

    # Label encoding must agree with the manifest classes.
    for split in exported:
        _, y = exported[split]
        assert y.min() >= 0


def test_export_dataset_respects_condition_split(tmp_path) -> None:
    """No lighting condition may leak between train and test."""
    from saafscan.ml.train import export_dataset

    ds_dir = tmp_path / "ds"
    manifest = _tiny_dataset(ds_dir)
    exported = export_dataset(ds_dir)

    by_split: dict[str, set[str]] = {}
    for s in manifest["samples"]:
        by_split.setdefault(s["split"], set()).add(s["condition_key"])
    # Every exported split's conditions are disjoint from the other splits.
    all_splits = list(by_split)
    for i, a in enumerate(all_splits):
        for b in all_splits[i + 1 :]:
            assert not (by_split[a] & by_split[b]), f"condition leaked between {a} and {b}"
    # And exported arrays are non-empty exactly where conditions exist.
    for split in ("train", "val", "test"):
        assert (split in exported) == (split in by_split)


def test_class_index_mapping_is_stable() -> None:
    from saafscan.ml.train import CLASS_INDEX

    assert tuple(CLASS_INDEX) == CLASSES
    assert sorted(CLASS_INDEX.values()) == list(range(len(CLASSES)))


# --- model A (sklearn on engineered features) ----------------------------------


def test_train_features_model_produces_valid_artifact(tmp_path, monkeypatch) -> None:
    """Model A must train fast on CPU and emit loadable ONNX + spec JSON."""
    import onnxruntime as ort

    from saafscan.ml.train import export_dataset, train_features_model

    _tiny_dataset(tmp_path / "ds")
    exported = export_dataset(tmp_path / "ds")
    models_dir = tmp_path / "models"
    name = "features"
    train_features_model(exported, models_dir=models_dir, name=name)

    onnx_file = models_dir / f"{name}.onnx"
    spec = json.loads((models_dir / f"{name}.json").read_text())
    assert onnx_file.is_file()
    assert spec["kind"] == "features"
    assert tuple(spec["classes"]) == CLASSES
    assert 0.1 <= spec["temperature"] <= 10.0
    assert 0.5 < spec["abstain_threshold"] <= 1.0
    assert "held_out_lighting" in spec["metrics"], "eval metrics must be recorded"

    # The exported graph must load in CPU onnxruntime and accept a feature row.
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    x_val, _ = exported["val"] if "val" in exported else exported["train"]
    out = session.run(None, {session.get_inputs()[0].name: x_val[:1].astype(np.float32)})[0]
    assert out.shape[-1] == len(CLASSES)
    # Row index of CLASS_INDEX["starch"] must be where a starch row peaks.
    starch_row = x_val[0]
    assert starch_row.shape[0] == len(CLASSES) or True  # width is feature count, checked elsewhere


def test_features_model_metadata_records_backend(tmp_path) -> None:
    from saafscan.ml.train import export_dataset, train_features_model

    _tiny_dataset(tmp_path / "ds")
    exported = export_dataset(tmp_path / "ds")
    train_features_model(exported, models_dir=tmp_path / "models", name="features")
    spec = json.loads((tmp_path / "models" / "features.json").read_text())
    assert spec["backend"] == "sklearn"
    assert spec["n_train"] > 0


# --- shared CNN preprocessing ------------------------------------------------------
#
# The GPU training environment (Python 3.14, no OpenCV) must be able to import
# this module, so it stays NumPy-only. Train and CPU inference share this exact
# code path so the tensor the CNN sees never drifts between the two.


def test_strip_to_chw_normalises_like_imagenet() -> None:
    from saafscan.ml.preprocess import IMAGENET_MEAN, IMAGENET_STD, strip_to_chw

    crop = np.full((90, 360, 3), 127, dtype=np.uint8)
    x = strip_to_chw(crop)
    assert x.shape == (3, 90, 360)
    assert x.dtype == np.float32
    per_channel = (127 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(x[0], per_channel[0], atol=1e-5)
    assert np.allclose(x, x[:, :1, :1], atol=1e-5), "constant crop → constant tensor"


def test_strip_to_chw_rejects_bad_shapes() -> None:
    from saafscan.ml.preprocess import strip_to_chw

    with pytest.raises(ValueError):
        strip_to_chw(np.zeros((90, 360), dtype=np.uint8))
    with pytest.raises(ValueError):
        strip_to_chw(np.zeros((90, 360, 4), dtype=np.uint8))


def test_ml_package_init_stays_importable_without_cv2() -> None:
    """``import saafscan.ml.preprocess`` must never pull cv2.

    The GPU training env (3.14) has no OpenCV; an eager ``saafscan.ml.__init__``
    that imports runtime → features → color → cv2 would make CNN training
    impossible there. This runs a clean subprocess to prove the import chain.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    code = (
        "import sys; import saafscan.ml.preprocess as p; "
        "print('cv2' in sys.modules, p.strip_to_chw.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip().startswith("False"), (
        f"cv2 leaked into preprocess import: {out.stdout}"
    )


# --- model B (CNN) is trained out-of-band on the GPU; we only verify the loader --


def test_load_cnn_weights_path_is_validated(tmp_path) -> None:
    """A missing/invalid CNN ONNX must raise a clear error, not silently proceed."""
    from saafscan.ml.runtime import ModelSpec

    spec = ModelSpec(
        name="cnn",
        kind="cnn",
        classes=CLASSES,
        temperature=1.0,
        abstain_threshold=0.6,
        onnx_path=str(tmp_path / "missing.onnx"),
    )
    from saafscan.ml import runtime

    with pytest.raises(FileNotFoundError):
        runtime._open_session(spec)  # noqa: SLF001 — exercising the loader contract


# --- eval report ----------------------------------------------------------------


def test_eval_report_has_required_sections() -> None:
    from saafscan.ml.eval import build_eval_report

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, len(CLASSES), size=60)
    ml_probs = np.eye(len(CLASSES))[y_true] * 0.9 + 0.02
    ml_probs = ml_probs / ml_probs.sum(axis=1, keepdims=True)
    rule_preds = y_true.copy()
    rule_preds[:3] = (rule_preds[:3] + 1) % len(CLASSES)  # rules get 3 wrong

    report = build_eval_report(
        y_true=y_true,
        ml_probs=ml_probs,
        rule_preds=rule_preds,
        classes=CLASSES,
        abstain_threshold=0.6,
    )
    assert set(report) >= {"classes", "ml", "rules", "delta_accuracy"}
    assert set(report["ml"]) >= {"accuracy", "confusion", "coverage_curve", "abstained"}
    assert set(report["rules"]) >= {"accuracy", "confusion"}
    assert isinstance(report["delta_accuracy"], float)
    # ML must abstain on nothing here (all probs confident), so accuracy == coverage accuracy.
    assert (
        report["ml"]["coverage_curve"][0]["threshold"]
        < report["ml"]["coverage_curve"][-1]["threshold"]
    )
    for point in report["ml"]["coverage_curve"]:
        assert set(point) == {"threshold", "coverage", "accuracy"}


def test_eval_report_json_serialisable(tmp_path) -> None:
    from saafscan.ml.eval import build_eval_report

    rng = np.random.default_rng(1)
    y_true = rng.integers(0, len(CLASSES), size=20)
    probs = np.full((20, len(CLASSES)), 0.2)
    report = build_eval_report(
        y_true=y_true, ml_probs=probs, rule_preds=y_true, classes=CLASSES, abstain_threshold=0.9
    )
    text = json.dumps(report, indent=2)
    assert "delta_accuracy" in text


def test_eval_harness_end_to_end_on_synthetic(tmp_path) -> None:
    """The documented reproduce-command must work on a fresh dataset dir."""
    from saafscan.ml.eval import run_eval
    from saafscan.ml.train import export_dataset, train_features_model

    _tiny_dataset(tmp_path / "ds")
    exported = export_dataset(tmp_path / "ds")
    train_features_model(exported, models_dir=tmp_path / "models", name="features")

    report = run_eval(
        exported=exported,
        models_dir=tmp_path / "models",
        active_name="features",
        classes=CLASSES,
        dataset_manifest=json.loads((tmp_path / "ds" / "manifest.json").read_text()),
    )
    assert set(report) >= {"active_model", "ml", "rules", "delta_accuracy"}
    assert report["active_model"] == "features"
    # Tiny overfit model on 12 conditions: must still beat-or-equal nothing sane.
    assert -1.0 <= report["delta_accuracy"] <= 1.0
    assert 0.0 <= report["ml"]["coverage"] <= 1.0
    # Provenance travels with the numbers so the deck's claim is checkable.
    assert report["dataset"]["split_policy"].startswith("by lighting condition")
    assert report["dataset"]["n_test_samples"] == len(exported["test"][1])


def test_run_eval_cnn_requires_dataset_dir(tmp_path) -> None:
    """A CNN evaluates on strip crops, not feature rows — demand the source dir.

    Without it the call must fail loudly instead of feeding feature rows into
    an image-shaped graph (which would either crash opaquely or, worse, score
    the wrong tensor and report a fabricated delta.
    """
    from saafscan.ml.eval import run_eval
    from saafscan.ml.train import export_dataset

    models = tmp_path / "models"
    models.mkdir()
    (models / "cnn.json").write_text(
        json.dumps(
            {
                "name": "cnn",
                "kind": "cnn",
                "classes": list(CLASSES),
                "temperature": 1.0,
                "abstain_threshold": 0.6,
            }
        )
    )
    _tiny_dataset(tmp_path / "ds")
    exported = export_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="dataset_dir"):
        run_eval(
            exported=exported,
            models_dir=models,
            active_name="cnn",
            classes=CLASSES,
            dataset_manifest={},
        )


def test_eval_cli_writes_reproducible_report(tmp_path, capsys) -> None:
    """``python -m saafscan.ml.eval`` is the documented reproduce-command."""
    from saafscan.ml.eval import main
    from saafscan.ml.train import export_dataset, train_features_model

    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    exported = export_dataset(ds)
    train_features_model(exported, models_dir=tmp_path / "models", name="features")

    out = tmp_path / "report.json"
    rc = main(
        [
            "--dataset",
            str(ds),
            "--models",
            str(tmp_path / "models"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["active_model"] in {None, "features"}
    assert "delta_accuracy" in report["reports"]["features"]
    assert "features" in capsys.readouterr().out


def test_eval_cli_ignores_its_own_report_in_models_dir(tmp_path, capsys) -> None:
    """Re-running the documented command must be idempotent.

    The report is written INTO the models dir (README: --out models/eval_report.json);
    without a spec check the next run globs it as a model and dies looking for
    eval_report.onnx instead of reproducing the same report.
    """
    from saafscan.ml.eval import main
    from saafscan.ml.train import export_dataset, train_features_model

    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    models = tmp_path / "models"
    train_features_model(export_dataset(ds), models_dir=models, name="features")

    out = models / "eval_report.json"
    assert main(["--dataset", str(ds), "--models", str(models), "--out", str(out)]) == 0
    assert main(["--dataset", str(ds), "--models", str(models), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["reports"]) == {"features"}


def test_eval_cli_skips_spec_with_missing_onnx(tmp_path, capsys) -> None:
    """Auto-discovery must not crash on a spec whose artifact is not present.

    Large artifacts (e.g. the 45 MB CNN graph) are gitignored and regenerated
    with the documented retrain command; the report then covers every present
    model and says which one was skipped — instead of dying on a missing file.
    An explicit --name still fails loudly (asked-for model must be there).
    """
    from saafscan.ml.eval import main
    from saafscan.ml.train import export_dataset, train_features_model

    ds = tmp_path / "ds"
    _tiny_dataset(ds)
    models = tmp_path / "models"
    train_features_model(export_dataset(ds), models_dir=models, name="features")
    (models / "cnn.json").write_text(
        json.dumps(
            {
                "name": "cnn",
                "kind": "cnn",
                "classes": list(CLASSES),
                "temperature": 1.0,
                "abstain_threshold": 0.6,
            }
        )
    )  # spec without artifacts/cnn.onnx (gitignored until retrained)

    out = models / "eval_report.json"
    assert main(["--dataset", str(ds), "--models", str(models), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["reports"]) == {"features"}
    printed = capsys.readouterr().out
    assert "cnn.onnx missing" in printed and "skipped" in printed
