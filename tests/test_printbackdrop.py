"""The mid-grey shooting backdrop everything else is photographed on.

The backdrop is a constraint, not a preference: below 50 the calibration
card's black frame stops being distinguishable from the surface it lies on
and card detection fails outright; above 170 the white reagent strip merges
into it. Light kraft or tan paper reads ~190 once the colour cast is
corrected and lands outside the band — which is exactly why a sheet you can
print beats hunting for a suitable table at the venue.

The numbers here were measured through the pipeline, not guessed: rendering
a backdrop at 40 makes detect_card return None, and the range it fails in
matches the lower bound this module enforces.
"""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.calib import detect_card, white_balance
from saafscan.printbackdrop import (
    BACKDROP_RGB,
    LUMA_MAX,
    LUMA_MIN,
    backdrop_luma,
    in_range,
    render_sheet,
)
from saafscan.printcard import PHYSICAL_SIZE_MM as CARD_MM
from saafscan.printcard import render_card
from saafscan.printstrip import STRIP_PHYSICAL_MM, render_strip
from saafscan.roi import detect_strip


def _shoot(surface_rgb: tuple[int, int, int], dpi: int = 150) -> np.ndarray:
    """A backdrop with the calibration card and a reagent strip laid on it."""
    mm = dpi / 25.4
    width, height = int(210 * mm), int(297 * mm)
    image = np.full((height, width, 3), surface_rgb, dtype=np.uint8)

    card_w, card_h = int(CARD_MM[0] * mm), int(CARD_MM[1] * mm)
    image[120 : 120 + card_h, 150 : 150 + card_w] = render_card(card_w, card_h)

    strip_w, strip_h = int(STRIP_PHYSICAL_MM[1] * mm), int(STRIP_PHYSICAL_MM[0] * mm)
    image[700 : 700 + strip_h, 300 : 300 + strip_w] = render_strip(strip_w, strip_h)
    return image


def test_the_printed_backdrop_sits_inside_the_measured_range() -> None:
    """The value on the sheet must land where the pipeline can still see."""
    assert in_range(backdrop_luma(np.asarray(BACKDROP_RGB, np.float32)))
    assert LUMA_MIN == 50.0
    assert LUMA_MAX == 170.0


def test_it_sits_near_the_middle_of_the_band() -> None:
    """Both failure modes bite in opposite directions.

    Dark loses the card's black frame; light loses the white strip. Sitting
    at the centre buys the same headroom against a printer that lays down
    ink darker than asked and lighting brighter than the last session.
    """
    luma = backdrop_luma(np.asarray(BACKDROP_RGB, np.float32))
    centre = (LUMA_MIN + LUMA_MAX) / 2.0
    assert abs(luma - centre) < 15.0, f"{luma:.1f} sits {abs(luma - centre):.1f} off centre"


def test_a_shoot_on_this_backdrop_finds_both_the_card_and_the_strip() -> None:
    """The whole point: the background must not swallow either object."""
    image = _shoot(BACKDROP_RGB)

    card = detect_card(image)
    assert card is not None, "card lost against its own backdrop"

    balanced = white_balance(image, card)
    assert detect_strip(balanced) is not None, "white strip merged into the surface"


def test_too_dark_a_backdrop_swallows_the_card() -> None:
    """Why the lower bound exists — this is the failure it prevents.

    A near-black surface and the card's black frame are the same colour, so
    detect_card cannot find an edge to lock onto.
    """
    assert detect_card(_shoot((40, 40, 40))) is None


def test_too_light_a_backdrop_is_declared_out_of_range() -> None:
    """The upper bound: kraft and tan paper read ~190 and must be refused."""
    assert not in_range(backdrop_luma(np.asarray((200, 196, 190), np.float32)))


def test_backdrop_luma_measures_the_surface_not_the_things_on_it() -> None:
    """Taken after white-balance, as the specification requires.

    Median rather than mean so the card and the strip — together a minority
    of the frame, and at opposite ends of the scale — cannot pull the reading
    towards themselves.
    """
    image = _shoot(BACKDROP_RGB)
    measured = backdrop_luma(image)
    expected = backdrop_luma(np.asarray(BACKDROP_RGB, np.float32))

    assert measured == pytest.approx(expected, abs=2.0)


def test_the_sheet_is_a4() -> None:
    """Same page as the strip template and the calibration card."""
    sheet = render_sheet(dpi=150)
    assert sheet.shape[0] == pytest.approx(1754, abs=2)  # 297 mm
    assert sheet.shape[1] == pytest.approx(1240, abs=2)  # 210 mm


def test_the_sheet_is_uniform_its_whole_area() -> None:
    """Any variation is the printer's problem, never ours.

    A stray mark inside the sheet would read as a surface feature and drag
    the measured luma somewhere it does not belong.
    """
    sheet = render_sheet(dpi=100)
    np.testing.assert_array_equal(sheet, np.full_like(sheet, BACKDROP_RGB[0]))


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        render_sheet(0)
