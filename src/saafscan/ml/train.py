"""Training pipeline for Model A: engineered features → GBC → ONNX.

Runs on CPU in the serving environment (scikit-learn + skl2onnx); no GPU, no
torch. Two steps:

1. :func:`export_dataset` — turn a rendered dataset directory (see
   ``saafscan.synthetic.build_dataset``) into feature matrices per split. The
   split is read from the manifest, which was decided **by lighting condition**,
   never by random image, so no lit scene can leak across the boundary.
2. :func:`train_features_model` — fit a gradient-boosted classifier, calibrate
   it (temperature scaling + abstention threshold on the validation split), and
   export a CPU-only ONNX artifact plus its JSON spec.

The exported graph has exactly one output: the class-probability row. Label
outputs are stripped at export time so no consumer can ever mistake a class
index for a confidence.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier

from saafscan.ml.calibration import abstention_metrics, fit_abstention_policy
from saafscan.ml.features import FEATURE_NAMES, extract_features
from saafscan.roi import zones_from_strip
from saafscan.synthetic import TRAINING_CLASSES

# argmax index → class name. Must stay aligned with runtime.CLASSES.
CLASS_INDEX: dict[str, int] = {label: i for i, label in enumerate(TRAINING_CLASSES)}
_N_CLASSES = len(CLASS_INDEX)
_INCONCLUSIVE_IDX = CLASS_INDEX["inconclusive"]


def export_dataset(ds_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a rendered dataset directory into per-split feature matrices.

    Reads ``manifest.json`` and the ``.npy`` strip crops, re-deriving the zone
    colours through the exact same ``zones_from_strip`` → ``extract_features``
    path the serving pipeline uses — training features cannot drift from
    inference features.

    Args:
        ds_dir: Directory produced by ``synthetic.build_dataset``.

    Returns:
        Mapping of split name → ``(X float32 (n, 12), y int64 (n,))``. Splits
        with no samples are absent from the mapping.

    Raises:
        ValueError: If the manifest is missing, declares unexpected classes,
            or references a crop that fails to load.
    """
    ds_dir = Path(ds_dir)
    manifest_path = ds_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"no manifest.json in {ds_dir}")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("classes", ())) != TRAINING_CLASSES:
        raise ValueError(
            f"manifest classes {manifest.get('classes')} != expected {list(TRAINING_CLASSES)}"
        )

    rows: dict[str, list[tuple[np.ndarray, int]]] = {}
    for sample in manifest["samples"]:
        crop_path = ds_dir / sample["path"]
        if not crop_path.is_file():
            raise ValueError(f"missing crop for sample: {crop_path}")
        crop = np.load(crop_path)
        if crop.ndim != 3 or crop.shape[0] == 0:
            raise ValueError(f"corrupt crop {crop_path}: shape {crop.shape}")
        label = sample["label"]
        if label not in CLASS_INDEX:
            raise ValueError(f"unknown label {label!r} in manifest")
        features = extract_features(zones_from_strip(crop))
        rows.setdefault(sample["split"], []).append((features, CLASS_INDEX[label]))

    return {
        split: (
            np.stack([f for f, _ in items]).astype(np.float32),
            np.asarray([y for _, y in items], dtype=np.int64),
        )
        for split, items in rows.items()
    }


def _export_onnx(model: GradientBoostingClassifier, path: Path) -> None:
    """Export the fitted classifier to ONNX with probabilities as sole output."""
    import onnx

    initial = [("input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onx = convert_sklearn(
        model,
        initial_types=initial,
        target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    # Strip the int64 label output: the serving contract is "one input row in,
    # one probability row out". Leaving the label output would invite a caller
    # to read run()[0] as a confidence.
    float_outputs = [
        o for o in onx.graph.output if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    ]
    if not float_outputs:
        raise RuntimeError("skl2onnx export produced no float probability output")
    del onx.graph.output[:]
    onx.graph.output.extend(float_outputs)
    onnx.checker.check_model(onx)
    path.write_bytes(onx.SerializeToString())


def train_features_model(
    exported: dict[str, tuple[np.ndarray, np.ndarray]],
    models_dir: Path,
    name: str,
) -> dict[str, Any]:
    """Train Model A on engineered features and write ``<name>.onnx`` + ``.json``.

    Calibration (temperature, abstain threshold) is fitted on the validation
    split; ``metrics.held_out_lighting`` is measured on the test split, whose
    lighting conditions the model has never seen. The test split is never
    consulted for fitting.

    Args:
        exported: Output of :func:`export_dataset`.
        models_dir: Directory to write artifacts into (created if missing).
        name: Artifact basename (e.g. ``"features"``).

    Returns:
        The spec dict written to ``<name>.json``.

    Raises:
        ValueError: If the training split is missing or lacks a class.
    """
    if "train" not in exported:
        raise ValueError("exported dataset has no training split")
    x_train, y_train = exported["train"]
    if set(np.unique(y_train)) != set(range(_N_CLASSES)):
        raise ValueError(
            f"training split must contain all {_N_CLASSES} classes, "
            f"got {sorted(set(int(v) for v in np.unique(y_train)))}"
        )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0
    )
    model.fit(x_train, y_train)

    x_val, y_val = exported.get("val", (np.empty(0, np.float32), np.empty(0, np.int64)))
    # Shared calibration policy (calibration.fit_abstention_policy): temperature
    # + abstain threshold fitted on val only, never on the held-out test split.
    temperature, threshold = fit_abstention_policy(
        model.predict_proba(x_val) if len(y_val) else None, y_val, _N_CLASSES
    )

    metrics: dict[str, Any] = {"held_out_lighting": {}}
    if "test" in exported and len(exported["test"][1]):
        x_test, y_test = exported["test"]
        metrics["held_out_lighting"] = abstention_metrics(
            model.predict_proba(x_test),
            y_test,
            inconclusive_index=_INCONCLUSIVE_IDX,
            temperature=temperature,
            threshold=threshold,
        )

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    _export_onnx(model, models_dir / f"{name}.onnx")

    spec: dict[str, Any] = {
        "name": name,
        "kind": "features",
        "backend": "sklearn",
        "model": "GradientBoostingClassifier(n_estimators=200, max_depth=3)",
        "classes": list(CLASS_INDEX),
        "features": list(FEATURE_NAMES),
        "temperature": float(temperature),
        "abstain_threshold": float(threshold),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    (models_dir / f"{name}.json").write_text(json.dumps(spec, indent=2))
    return spec
