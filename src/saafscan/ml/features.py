"""Engineered colour-feature vector for the ML classifier (Model A input).

A 12-dimension feature vector derived purely from the three measured reaction
zones relative to their printed nominal colours. Deterministic, interpretable,
and identical between training and inference — the classifier consumes the same
features the rule-table reasons about (ΔE per zone), plus raw colour and
cross-zone ratios that let a learned model separate cases the fixed thresholds
cannot.
"""

from __future__ import annotations

import numpy as np

from saafscan.color import zone_features
from saafscan.roi import ZONE_NOMINAL

# Fixed feature order. The ML model is only valid for this exact ordering.
FEATURE_NAMES: tuple[str, ...] = (
    "control_deltaE",
    "starch_deltaE",
    "detergent_deltaE",
    "control_r",
    "control_g",
    "control_b",
    "starch_r",
    "starch_g",
    "starch_b",
    "detergent_r",
    "detergent_g",
    "detergent_b",
)

_ZONE_ORDER = ("control", "starch", "detergent")
_REQUIRED = frozenset(_ZONE_ORDER)


def extract_features(zones: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Build the feature vector from measured zone colours.

    Args:
        zones: Mapping of zone name → measured RGB triple (0-255 floats).

    Returns:
        Float32 vector of length ``len(FEATURE_NAMES)``.

    Raises:
        ValueError: If any required reaction zone is missing.
    """
    missing = _REQUIRED - set(zones)
    if missing:
        raise ValueError(f"missing reaction zones: {sorted(missing)}")

    deltas = zone_features(zones, ZONE_NOMINAL)
    values: list[float] = []
    for name in _ZONE_ORDER:
        values.append(float(deltas.get(name, 0.0)))
    for name in _ZONE_ORDER:
        r, g, b = (float(c) for c in zones[name])
        values.extend((r, g, b))
    return np.asarray(values, dtype=np.float32)
