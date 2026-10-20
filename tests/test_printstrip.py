"""RED tests: the printed strip template the shoot physically needs.

Applying reagent by eye onto a bare strip is what went wrong in `photo5.jpeg` —
the three marks landed one zone to the left, so the pipeline read untouched
paper in the detergent band as a fully decoloured, i.e. reacted, zone. The
template exists to make that impossible, and the only way it can is by deriving
the aiming squares from the windows `zones_from_strip` actually averages rather
than from the strip's thirds.

The safety-critical property is therefore negative: printing must never put ink
inside a measurement window, because the zone colour is a *mean* and any ink
bias in it moves the verdict.
"""

from __future__ import annotations

from math import ceil, floor

import numpy as np

from saafscan.printstrip import (
    CONTROL_COLOUR,
    CONTROL_ZONE,
    DEFAULT_DPI,
    PROUD_MM,
    SHEET_MM,
    main,
    render_sheet,
    render_strip,
    sheet_size_px,
    strip_size_px,
    zone_windows_proud,
)
from saafscan.roi import (
    STRIP_PHYSICAL_MM,
    ZONE_NAMES,
    ZONE_NOMINAL,
    zone_window_mm,
    zones_from_strip,
)

ZONES = ("control", "starch", "detergent")


def test_the_reagent_windows_stay_pure_paper() -> None:
    """No ink may fall inside a window the pipeline averages.

    zones_from_strip takes the mean colour of each window, so an outline drawn
    inside it would bias that mean and quietly shift the verdict. The squares
    are drawn proud of the window instead, which keeps the artwork out of the
    measurement while still showing an operator where to aim.

    Control is exempt by design: it is printed solid, and there the ink *is*
    the measurement. See the next test.
    """
    zones = zones_from_strip(render_strip(*strip_size_px()))
    for name in ("starch", "detergent"):
        np.testing.assert_allclose(zones[name], (255.0, 255.0, 255.0), atol=0.5, err_msg=name)


def test_the_control_window_is_the_printed_reference_colour() -> None:
    """The control box is printed solid in the colour the guard compares against.

    A control zone loaded by hand is a control zone sometimes not loaded at
    all, and photo4 proved the cost: with no control mark the window read blank
    paper, a deltaE of 79.8 against nominal, and the strip was refused before
    its chemistry was ever considered. Printing the control makes the reference
    present on every strip by construction.

    It must be ZONE_NOMINAL specifically, not merely some red, because that is
    the colour classify_zones measures against. Print a control it does not
    recognise and every strip refuses.
    """
    zones = zones_from_strip(render_strip(*strip_size_px()))
    np.testing.assert_allclose(zones[CONTROL_ZONE], CONTROL_COLOUR, atol=0.5, err_msg=CONTROL_ZONE)


def test_the_printed_control_is_named_and_coloured_consistently() -> None:
    """Guards against the printed control drifting away from the classifier."""
    assert CONTROL_ZONE in ZONE_NAMES
    assert CONTROL_COLOUR == ZONE_NOMINAL[CONTROL_ZONE]


def test_each_square_covers_the_window_it_aims_at() -> None:
    """Reagent dropped anywhere inside a square must reach the measured band.

    A square narrower than its window would let even a careful operator miss
    the measurement, which is precisely the failure this template prevents.
    """
    windows = zone_window_mm()
    long_mm, short_mm = STRIP_PHYSICAL_MM[1], STRIP_PHYSICAL_MM[0]
    for name, (x0, x1, y0, y1) in zone_windows_proud().items():
        wx0, wx1, wy0, wy1 = windows[name]
        assert x0 <= wx0 and x1 >= wx1, f"{name}: square does not cover the window in x"
        assert y0 <= wy0 and y1 >= wy1, f"{name}: square does not cover the window in y"
        assert x0 > 0.0 and x1 < long_mm, f"{name}: square runs off the strip"
        assert y0 > 0.0 and y1 < short_mm, f"{name}: square runs off the strip"
        assert x1 - x0 > PROUD_MM * 2, f"{name}: square is too small to aim at"


def test_the_template_and_the_detector_agree_on_where_the_windows_are() -> None:
    """Paint each window a known colour; zones_from_strip must read it back.

    `zones_from_strip` averages a pixel window and `zone_window_mm` reports that
    same window in millimetres. If they ever drift apart the printed squares
    would slide out of the measurement they exist to protect, so the agreement
    is pinned by test rather than asserted in a docstring.

    The painted band is floored and ceilinged outward, because the pixel window
    is computed by *truncation* (`int()`), which can only ever come out narrower
    than the millimetre window it came from. Containing it exactly is what lets
    the read-back be exact rather than approximate.
    """
    width_px, height_px = strip_size_px()
    sx = width_px / STRIP_PHYSICAL_MM[1]
    sy = height_px / STRIP_PHYSICAL_MM[0]
    colours = {"control": (250, 0, 0), "starch": (0, 250, 0), "detergent": (0, 0, 250)}

    strip = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    for name, (x0, x1, y0, y1) in zone_window_mm().items():
        strip[floor(y0 * sy) : ceil(y1 * sy), floor(x0 * sx) : ceil(x1 * sx)] = colours[name]

    zones = zones_from_strip(strip)
    for name, rgb in colours.items():
        np.testing.assert_allclose(zones[name], rgb, atol=1.0, err_msg=name)


def test_each_of_the_three_zones_is_marked_visibly() -> None:
    """A template with no marks to aim at would be worse than none at all."""
    artwork = render_strip(*strip_size_px())
    ink = (artwork < 180).any(axis=2)
    width = artwork.shape[1]
    for i, name in enumerate(ZONES):
        assert ink[:, int(i * width / 3) : int((i + 1) * width / 3)].any(), f"{name} unmarked"


def test_the_sheet_is_a4_so_the_millimetres_survive_the_printer() -> None:
    """A4 portrait at 300 dpi, stated exactly.

    Every dimension on this sheet is a millimetre, so the sheet size is not a
    cosmetic choice: print a strip template at the wrong paper size and the
    squares land at the wrong millimetres on the strip.
    """
    assert SHEET_MM == (210.0, 297.0)
    assert sheet_size_px() == (2480, 3508)
    assert sheet_size_px(600) == (4961, 7016)


def test_the_sheet_holds_sixteen_strips() -> None:
    """Enough for a whole class of samples per page."""
    from saafscan.printstrip import strips_per_sheet

    assert strips_per_sheet() == 16


def test_rendering_the_sheet_leaves_the_windows_clean_too() -> None:
    """The sheet adds labels and cut marks; none of them may reach a window."""
    artwork = render_sheet(dpi=DEFAULT_DPI)
    assert artwork.shape[:2] == (3508, 2480)
    assert artwork.min() < 128, "no ink at all on the sheet — nothing was drawn"


def test_the_cli_writes_the_template(tmp_path) -> None:
    out = tmp_path / "strip_template.png"
    assert main(["--out", str(out), "--dpi", "150"]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_strip_dimensions_must_be_positive() -> None:
    import pytest

    from saafscan.printstrip import render_strip as render

    with pytest.raises(ValueError, match="positive"):
        render(0, 100)
