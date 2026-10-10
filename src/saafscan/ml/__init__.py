"""ML package: trained classifier as PRIMARY decision-maker, rule-table as fallback.

The rule-table in ``saafscan.classifier`` remains importable and is the total
fallback: if no trained artifact is present, or it fails to load, or the input
is unusable, the wrapper returns exactly the rule-table verdict. ML never
fabricates a confident verdict on ambiguous or out-of-distribution input —
low calibrated confidence abstains to ``inconclusive``.

Public names are re-exported lazily (PEP 562): importing a light submodule
such as ``saafscan.ml.preprocess`` must not pull ``runtime`` → ``features`` →
``color`` → ``cv2``, because the GPU training environment has no OpenCV.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ModelSpec",
    "active_model_name",
    "classify",
    "load_active_spec",
    "reset_cache",
    "verdict_from_probs",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        runtime = import_module("saafscan.ml.runtime")
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "calibration", "eval", "features", "preprocess", "runtime", "train"])
