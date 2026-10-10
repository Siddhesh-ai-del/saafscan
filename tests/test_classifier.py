"""RED tests: colour features and the adulterant classifier (incl. inconclusive)."""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.classifier import classify, classify_zones
from saafscan.color import delta_e, rgb_to_lab, zone_features
from saafscan.verdict import ADULTERANTS, Verdict


# --- colour primitives -----------------------------------------------------


def test_delta_e_of_identical_colours_is_zero() -> None:
    c = np.array([200.0, 120.0, 60.0])
    assert delta_e(c, c) == pytest.approx(0.0, abs=1e-6)


def test_delta_e_is_symmetric() -> None:
    a = np.array([210.0, 30.0, 30.0])
    b = np.array([30.0, 30.0, 210.0])
    assert delta_e(a, b) == pytest.approx(delta_e(b, a), rel=1e-9)


def test_delta_e_grows_with_difference() -> None:
    base = np.array([128.0, 128.0, 128.0])
    near = base + np.array([10.0, 0.0, 0.0])
    far = base + np.array([80.0, 0.0, 0.0])
    assert delta_e(base, near) < delta_e(base, far)


def test_rgb_to_lab_is_deterministic_and_in_range() -> None:
    lab1 = rgb_to_lab(np.array([180.0, 90.0, 40.0]))
    lab2 = rgb_to_lab(np.array([180.0, 90.0, 40.0]))
    assert np.allclose(lab1, lab2)
    # L in [0, 100]; a/b roughly [-128, 127]
    assert 0.0 <= lab1[0] <= 100.0
    assert abs(lab1[1]) <= 128.0 and abs(lab1[2]) <= 128.0


def test_zone_features_exposes_three_zone_scores() -> None:
    zones = {"control": (200, 60, 40), "starch": (40, 190, 210), "detergent": (60, 90, 200)}
    feats = zone_features(
        zones,
        nominal={"control": (205, 65, 45), "starch": (45, 185, 205), "detergent": (65, 95, 195)},
    )
    assert set(feats) == {"control", "starch", "detergent"}
    assert all(isinstance(v, float) for v in feats.values())


# --- verdict model ---------------------------------------------------------


def test_verdict_is_immutable_and_serialisable() -> None:
    v = Verdict(
        adulterant="starch", confidence=0.93, severity="moderate", advice="Avoid.", evidence={}
    )
    with pytest.raises(Exception):
        v.adulterant = "none"  # type: ignore[misc]
    data = v.to_dict()
    assert data["adulterant"] == "starch"
    assert data["confidence"] == pytest.approx(0.93)
    assert set(data) >= {"adulterant", "confidence", "severity", "advice", "evidence"}


def test_adulterant_labels_are_closed() -> None:
    assert set(ADULTERANTS) == {"none", "starch", "detergent", "both", "inconclusive"}


# --- classifier: golden zone signatures ------------------------------------


def _zones_for(label: str) -> dict[str, tuple[float, float, float]]:
    """Zone signature derived from the synthetic scene for a given label."""
    from saafscan.roi import extract_zones

    return {
        k: tuple(float(c) for c in v)
        for k, v in extract_zones(
            __import__("saafscan.synthetic", fromlist=["render_sample"]).render_sample(
                label, seed=7
            )
        ).items()
    }


@pytest.mark.parametrize(
    ("label", "expected"),
    [("none", "none"), ("starch", "starch"), ("detergent", "detergent"), ("both", "both")],
)
def test_classify_zones_names_the_right_adulterant(label: str, expected: str) -> None:
    verdict = classify_zones(_zones_for(label))
    assert verdict.adulterant == expected
    assert 0.0 <= verdict.confidence <= 1.0
    if expected != "none":
        assert verdict.confidence >= 0.7, f"low confidence for clear case: {verdict.confidence}"


def test_pure_returns_zero_or_low_severity() -> None:
    verdict = classify_zones(_zones_for("none"))
    assert verdict.severity in {"none", "low"}


@pytest.mark.parametrize("label", ["starch", "detergent", "both"])
def test_positive_verdicts_carry_actionable_advice(label: str) -> None:
    verdict = classify_zones(_zones_for(label))
    assert len(verdict.advice) > 15, "advice must be plain-language and actionable"
    assert verdict.evidence, "evidence (zone deltas) must be attached"


# --- classifier: inconclusive / OOD paths ----------------------------------


def test_empty_zones_are_inconclusive() -> None:
    verdict = classify_zones({})
    assert verdict.adulterant == "inconclusive"
    assert verdict.confidence <= 0.5


def test_partial_zones_are_inconclusive() -> None:
    verdict = classify_zones({"control": (200, 60, 40)})
    assert verdict.adulterant == "inconclusive"


def test_ambiguous_mixed_signal_is_inconclusive() -> None:
    """Zones drifting halfway between states must not be forced into a class."""
    zones = {"control": (200, 140, 120), "starch": (120, 160, 170), "detergent": (140, 150, 170)}
    verdict = classify_zones(zones)
    if verdict.adulterant != "inconclusive":
        # Accept only if the classifier is genuinely confident and explains why.
        assert verdict.confidence >= 0.7
        assert verdict.evidence


def test_noise_zones_do_not_crash_and_stay_bounded() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        zones = {
            "control": tuple(float(x) for x in rng.integers(0, 256, 3)),
            "starch": tuple(float(x) for x in rng.integers(0, 256, 3)),
            "detergent": tuple(float(x) for x in rng.integers(0, 256, 3)),
        }
        verdict = classify_zones(zones)
        assert verdict.adulterant in ADULTERANTS
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.severity in {"none", "low", "moderate", "high"}


def test_classify_rejects_missing_key_inputs() -> None:
    """classify() on malformed input must raise a clear error, not crash later."""
    with pytest.raises((KeyError, TypeError, ValueError)):
        classify(np.zeros((4, 4, 3), dtype=np.uint8), None)
