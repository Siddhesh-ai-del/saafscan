"""Classifier: zone colour deltas → Verdict. Deterministic rules, no ML.

Thresholds are calibrated against the printed reagent nominal colours
(roi.ZONE_NOMINAL). Scores are ΔE distances; ambiguous bands produce an
'inconclusive' verdict instead of a forced guess.
"""

from __future__ import annotations

import numpy as np

from saafscan.color import zone_features
from saafscan.roi import ZONE_NOMINAL
from saafscan.verdict import Verdict

# ΔE bands: below CLEAR the zone is at nominal; above FIRED the reaction is unambiguous.
CLEAR = 40.0
FIRED = 55.0
CONTROL_TOLERANCE = 45.0

_REQUIRED = ("control", "starch", "detergent")

_ADVICE = {
    "none": "No adulterant detected. Reagent zones match printed reference colours.",
    "starch": "Starch adulterant likely (blue-black starch-iodine complex). Do not use this sample; re-test with a fresh strip and report to the vendor.",
    "detergent": "Detergent adulterant likely (methylene blue decolourised). Do not consume; preserve the sample and report to the vendor or food-safety helpline.",
    "both": "Both starch and detergent reactions detected. Treat as unsafe; escalate with a fresh-strip confirmation and report.",
    "inconclusive": "Reaction colours are ambiguous — the strip may be expired, lighting too uneven, or the sample unusual. Re-test with a fresh strip in daylight.",
}


def _severity(margin: float) -> str:
    if margin >= 45.0:
        return "high"
    if margin >= 20.0:
        return "moderate"
    return "low"


def classify_zones(
    zones: dict[str, tuple[float, float, float]],
    nominal: dict[str, tuple[int, int, int]] | None = None,
) -> Verdict:
    """Classify measured zone colours. Missing/ambiguous input → inconclusive."""
    ref = nominal or ZONE_NOMINAL
    if not all(name in zones for name in _REQUIRED):
        return Verdict(
            adulterant="inconclusive",
            confidence=0.2,
            severity="low",
            advice=_ADVICE["inconclusive"],
            evidence={},
        )

    scores = zone_features(zones, ref)
    control = scores["control"]
    starch = scores["starch"]
    detergent = scores["detergent"]
    evidence = {f"{k}_deltaE": round(v, 2) for k, v in scores.items()}

    if control > CONTROL_TOLERANCE:
        # The control zone should never react — if it moved, the strip/image is bad.
        return Verdict(
            adulterant="inconclusive",
            confidence=0.3,
            severity="low",
            advice=_ADVICE["inconclusive"],
            evidence=evidence,
        )

    starch_fired = starch >= FIRED
    detergent_fired = detergent >= FIRED
    starch_ambig = CLEAR <= starch < FIRED
    detergent_ambig = CLEAR <= detergent < FIRED

    if starch_ambig or detergent_ambig:
        confidence = 0.35 + 0.1 * float(starch_ambig + detergent_ambig)
        return Verdict(
            adulterant="inconclusive",
            confidence=min(confidence, 0.5),
            severity="low",
            advice=_ADVICE["inconclusive"],
            evidence=evidence,
        )

    if starch_fired and detergent_fired:
        margin = min(starch, detergent) - FIRED
        return Verdict(
            adulterant="both",
            confidence=min(0.55 + margin / 90.0, 0.97),
            severity=_severity(margin),
            advice=_ADVICE["both"],
            evidence=evidence,
        )
    if starch_fired:
        margin = starch - FIRED
        return Verdict(
            adulterant="starch",
            confidence=min(0.55 + margin / 90.0, 0.97),
            severity=_severity(margin),
            advice=_ADVICE["starch"],
            evidence=evidence,
        )
    if detergent_fired:
        margin = detergent - FIRED
        return Verdict(
            adulterant="detergent",
            confidence=min(0.55 + margin / 90.0, 0.97),
            severity=_severity(margin),
            advice=_ADVICE["detergent"],
            evidence=evidence,
        )

    # Both reagent zones at nominal → nothing detected.
    margin = CLEAR - max(starch, detergent)
    return Verdict(
        adulterant="none",
        confidence=min(0.6 + margin / 100.0, 0.96),
        severity="none",
        advice=_ADVICE["none"],
        evidence=evidence,
    )


def classify(image: np.ndarray, nominal: dict | None = None) -> Verdict:
    """Convenience: extract zones from a full scene and classify them."""
    from saafscan.roi import extract_zones

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("classify() requires a non-empty image array")
    return classify_zones(extract_zones(image), nominal)
