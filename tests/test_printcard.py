"""RED tests: the printed colour calibration card the demo physically needs.

The card exists in the repo only as a canvas inside the synthetic renderer —
there has never been anything printable. Without one, "photograph a blank strip
next to the card and check detection" cannot happen before the reagents arrive,
so the geometry the detector actually samples has to be reproduced exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.calib import (
    CARD_NOMINAL,
    _looks_like_card,
    canonical_card,
    detect_card,
    extract_patches,
)
from saafscan.printcard import PHYSICAL_SIZE_MM, render_card


def test_patches_measured_on_the_printed_artwork_match_the_nominal_colours() -> None:
    card = render_card(CARD_NOMINAL["width"], CARD_NOMINAL["height"])
    measured = extract_patches(card)
    for i, patch in enumerate(CARD_NOMINAL["patches"]):
        np.testing.assert_allclose(measured[i], patch["rgb"], atol=2.0, err_msg=patch["name"])


def test_the_printed_artwork_satisfies_the_card_validator() -> None:
    """The same gate detect_card applies before it accepts a quad."""
    card = render_card(CARD_NOMINAL["width"], CARD_NOMINAL["height"])
    assert _looks_like_card(card)


def test_white_is_the_brightest_patch_and_black_the_darkest() -> None:
    """_looks_like_card's second condition, stated in its own terms."""
    measured = extract_patches(render_card(CARD_NOMINAL["width"], CARD_NOMINAL["height"]))
    lums = measured @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    assert lums[0] >= lums.max() - 8.0
    assert lums[2] <= lums.min() + 8.0


def test_aspect_ratio_stays_inside_the_detection_gate() -> None:
    """detect_card rejects quads outside 1.0-2.4; the artwork must not drift."""
    w, h = PHYSICAL_SIZE_MM
    aspect = w / h
    assert 1.0 <= aspect <= 2.4
    assert aspect == pytest.approx(CARD_NOMINAL["width"] / CARD_NOMINAL["height"], rel=0.01)


def test_a_card_printed_at_real_size_is_still_detected_in_a_scene() -> None:
    """End to end: upscale to print resolution, composite, run detect_card."""
    card = render_card(900, 600)
    scene = np.full((1600, 1200, 3), 143, dtype=np.uint8)
    scene[400 : 400 + card.shape[0], 300 : 300 + card.shape[1]] = card

    quad = detect_card(scene)
    assert quad is not None
    warped = canonical_card(scene, quad)
    assert _looks_like_card(warped)


def test_render_card_refuses_a_size_the_detector_cannot_use() -> None:
    with pytest.raises(ValueError):
        render_card(0, 100)
    with pytest.raises(ValueError):
        render_card(100, 0)


def test_render_card_returns_a_contiguous_uint8_rgb_array() -> None:
    card = render_card(828, 552)  # 70 x 46.7 mm at 300 dpi
    assert card.dtype == np.uint8
    assert card.ndim == 3 and card.shape[2] == 3
    assert card.flags["C_CONTIGUOUS"]
