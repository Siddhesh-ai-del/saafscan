"""RED tests: strip detection and reaction-zone colour extraction."""

from __future__ import annotations

import numpy as np

from saafscan.color import delta_e
from saafscan.roi import ZONE_NOMINAL, detect_strip, extract_zones, warp_strip
from saafscan.synthetic import render_random_photo, render_sample
from tests.conftest import assert_is_image


def test_detect_strip_finds_the_reagent_strip() -> None:
    quad = detect_strip(render_sample("none", seed=7))
    assert quad is not None
    assert quad.shape == (4, 2)


def test_detect_strip_returns_none_without_a_strip() -> None:
    assert detect_strip(render_random_photo(seed=11)) is None


def test_detect_strip_survives_exposure_changes() -> None:
    base = render_sample("detergent", seed=7)
    for gain in (0.7, 1.15):
        adjusted = np.clip(base.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        assert detect_strip(adjusted) is not None, f"strip lost at gain={gain}"


def test_warp_strip_returns_canonical_strip_image() -> None:
    strip = warp_strip(render_sample("none", seed=7), detect_strip(render_sample("none", seed=7)))
    assert_is_image(strip, min_side=64)


def test_extract_zones_returns_three_named_zones() -> None:
    zones = extract_zones(render_sample("none", seed=7))
    assert set(zones) == set(ZONE_NOMINAL)
    for name, rgb in zones.items():
        assert isinstance(name, str)
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb), f"{name} out of range: {rgb}"


def test_pure_scene_zones_match_their_nominal_reagent_colours() -> None:
    """Control + both reagent zones must read close to printed reference."""
    zones = extract_zones(render_sample("none", seed=7))
    for name, rgb in zones.items():
        err = delta_e(
            np.asarray(rgb, dtype=np.float32), np.asarray(ZONE_NOMINAL[name], dtype=np.float32)
        )
        assert err < 40.0, f"{name}: deltaE={err:.1f} vs nominal {ZONE_NOMINAL[name]}, got {rgb}"


def test_starch_darkens_the_starch_zone_only() -> None:
    zones = extract_zones(render_sample("starch", seed=7))
    pure = extract_zones(render_sample("none", seed=7))
    starch_shift = delta_e(
        np.asarray(zones["starch"], np.float32), np.asarray(pure["starch"], np.float32)
    )
    control_shift = delta_e(
        np.asarray(zones["control"], np.float32), np.asarray(pure["control"], np.float32)
    )
    assert starch_shift > 50.0, "starch reaction did not visibly darken the starch zone"
    assert control_shift < 30.0, (
        f"control zone drifted too ({control_shift:.1f}) — zones not independent"
    )


def test_detergent_pales_the_detergent_zone_only() -> None:
    zones = extract_zones(render_sample("detergent", seed=7))
    pure = extract_zones(render_sample("none", seed=7))
    det_shift = delta_e(
        np.asarray(zones["detergent"], np.float32), np.asarray(pure["detergent"], np.float32)
    )
    control_shift = delta_e(
        np.asarray(zones["control"], np.float32), np.asarray(pure["control"], np.float32)
    )
    assert det_shift > 50.0, "detergent reaction did not visibly change the detergent zone"
    assert control_shift < 30.0, f"control zone drifted too ({control_shift:.1f})"


def test_zones_are_stable_across_exposure() -> None:
    """Exposure changes must not flip zone colours (lighting invariance)."""
    for label in ("none", "starch", "detergent"):
        base = render_sample(label, seed=7)
        adjusted = np.clip(base.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
        zones_a = extract_zones(base)
        zones_b = extract_zones(adjusted)
        for name in zones_a:
            err = delta_e(
                np.asarray(zones_a[name], np.float32), np.asarray(zones_b[name], np.float32)
            )
            assert err < 35.0, f"{label}/{name}: deltaE={err:.1f} under 0.75x exposure"
