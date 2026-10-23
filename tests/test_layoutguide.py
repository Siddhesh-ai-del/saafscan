"""A worked example of where everything goes before anyone takes a photo.

The point of these tests is that the guide is *checked*, not illustrated. It
would be easy to draw a plausible-looking arrangement that the pipeline itself
would reject — a strip stood on end, a card touching the frame edge, a label
plate wide and bright enough to be detected instead of the strip it points at.
Each of those is asserted against here, using the same detect_card and
detect_strip the live demo runs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from saafscan.calib import detect_card
from saafscan.layoutguide import (
    _RULES,
    CARD_ORIGIN_MM,
    FIELD_MM,
    STRIP_ORIGIN_MM,
    layout,
    overlaps,
    placements,
    render_example,
)
from saafscan.printbackdrop import BACKDROP_RGB, backdrop_luma, in_range
from saafscan.printcard import PHYSICAL_SIZE_MM as CARD_MM
from saafscan.printcard import render_card
from saafscan.printstrip import STRIP_PHYSICAL_MM, render_strip
from saafscan.roi import STRIP_ASPECT_RANGE, detect_strip

SIZE = (1600, 1200)


def _centre(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    xs, ys = quad[:, 0], quad[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_the_example_shows_the_backdrop_it_is_asking_you_to_print() -> None:
    """A guide on any other surface would teach the wrong background."""
    guide = render_example(*SIZE)
    np.testing.assert_array_equal(guide[5, 5], np.asarray(BACKDROP_RGB))
    assert in_range(backdrop_luma(guide))


def test_the_two_objects_keep_their_real_sizes_relative_to_each_other() -> None:
    """True scale, because the guide is also teaching how far apart they sit."""
    card = placements(*SIZE)["card"]
    strip = placements(*SIZE)["strip"]

    card_w, card_h = card[2] - card[0], card[3] - card[1]
    strip_w, strip_h = strip[2] - strip[0], strip[3] - strip[1]

    assert card_w / card_h == pytest.approx(CARD_MM[0] / CARD_MM[1], abs=0.05)
    assert strip_w / strip_h == pytest.approx(STRIP_PHYSICAL_MM[1] / STRIP_PHYSICAL_MM[0], abs=0.05)


def test_the_strip_is_laid_lengthways_not_stood_on_end() -> None:
    """The single most important instruction on the sheet.

    detect_strip refuses a lengthways strip because warping it across its
    short axis would report three identical slices as three different
    reaction zones — a refusal is recoverable, a confident wrong verdict is
    not. The two size constants are ordered differently upstream (card is
    width-then-height, strip is short-then-long), which is exactly how a
    vertical strip sneaks in.
    """
    x0, y0, x1, y1 = placements(*SIZE)["strip"]
    assert x1 - x0 > y1 - y0, "the example must never show a strip stood on end"


def test_everything_fits_inside_the_frame_with_room_around_it() -> None:
    """Detection wants grey all around: the strip is compared against a ring
    of the surface immediately outside it."""
    width, height = SIZE
    for name, rect in placements(*SIZE).items():
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, name


def test_the_card_and_the_strip_do_not_touch() -> None:
    card = placements(*SIZE)["card"]
    strip = placements(*SIZE)["strip"]
    assert not overlaps(card, strip)


def test_the_objects_sit_inside_the_detectors_own_gates() -> None:
    """Not merely 'plausible' — inside the numbers the detectors enforce.

    Area fractions are what stop a setup being rejected for zooming too far
    in or too far out, so they belong on the guide rather than being
    discovered at the venue.
    """
    width, height = SIZE
    image_area = width * height
    rect = placements(*SIZE)

    strip = rect["strip"]
    strip_area = (strip[2] - strip[0]) * (strip[3] - strip[1]) / image_area
    assert 0.015 <= strip_area <= 0.2, f"strip occupies {strip_area:.4f} of frame"

    lo, hi = STRIP_ASPECT_RANGE
    strip_aspect = (strip[2] - strip[0]) / (strip[3] - strip[1])
    assert lo <= strip_aspect <= hi, f"strip aspect {strip_aspect:.2f}"

    card = rect["card"]
    card_area = (card[2] - card[0]) * (card[3] - card[1]) / image_area
    assert 0.015 <= card_area <= 0.5, f"card occupies {card_area:.4f} of frame"

    card_aspect = (card[2] - card[0]) / (card[3] - card[1])
    assert 1.0 <= card_aspect <= 2.4, f"card aspect {card_aspect:.2f}"


@pytest.mark.parametrize("annotate", [False, True])
def test_detection_finds_the_objects_where_the_guide_put_them(annotate: bool) -> None:
    """The test that makes the guide trustworthy.

    Not merely 'something was detected' — the returned quads must be the card
    and the strip at the positions shown. A label plate picked up instead of
    the strip, or a rule of text mistaken for the card, would pass a weaker
    assertion while teaching a setup that does not work.
    """
    guide = render_example(*SIZE, annotate=annotate)
    width, height = SIZE
    tol_x, tol_y = 0.03 * width, 0.03 * height

    card_quad = detect_card(guide)
    assert card_quad is not None, "card not found"
    cx, cy = _centre(_bbox(card_quad))
    ex, ey = _centre(placements(*SIZE)["card"])
    assert abs(cx - ex) < tol_x and abs(cy - ey) < tol_y, (
        f"card found at {(cx, cy)}, not {(ex, ey)}"
    )

    strip_quad = detect_strip(guide)
    assert strip_quad is not None, "strip not found"
    sx, sy = _centre(_bbox(strip_quad))
    ex, ey = _centre(placements(*SIZE)["strip"])
    assert abs(sx - ex) < tol_x and abs(sy - ey) < tol_y, (
        f"strip found at {(sx, sy)}, not {(ex, ey)}"
    )


def test_no_label_is_drawn_over_the_object_it_points_at() -> None:
    """Ink on the card's frame or the strip's paper is ink the detectors have
    to look through — so every label keeps the guide's declared gap."""
    plan = layout(*SIZE)
    gap = plan["gap"]
    for label_name, object_name in (("card_label", "card"), ("strip_label", "strip")):
        label = plan[label_name]
        obj = plan[object_name]
        assert not overlaps(label, obj), f"{label_name} drawn on the {object_name}"
        assert label[3] <= obj[1] - gap, f"{label_name} left short of its {object_name}'s clearance"


def test_no_caption_falls_off_the_sheet() -> None:
    """A caption clipped by the print is a rule the person never read, and a
    caption wider than the sheet is worse than none at all."""
    plan = layout(*SIZE)
    width, height = SIZE

    captions: list[tuple[str, tuple[int, int, int, int]]] = [
        ("title", plan["title"]),
        ("card_label", plan["card_label"]),
        ("strip_label", plan["strip_label"]),
    ]
    captions.extend((f"rule {n}", rect) for n, rect in enumerate(plan["rules"], start=1))

    for name, (x0, y0, x1, y1) in captions:
        assert 0 <= x0 < x1 <= width, f"{name} does not fit the sheet's width"
        assert 0 <= y0 < y1 <= height, f"{name} does not fit the sheet's height"


def test_a_canvas_too_small_for_the_rules_says_so_rather_than_clipping() -> None:
    """Better an error at the laptop than a missing rule at the venue."""
    with pytest.raises(ValueError, match="too small"):
        layout(400, 300)


def test_every_rule_is_on_the_sheet_and_not_running_off_the_bottom() -> None:
    """A caption clipped by the print is a rule the person never read.

    Counted as dark plates below the strip rather than read back out of the
    module, so a rule that silently fell off the canvas fails here too.
    """
    guide = render_example(*SIZE, annotate=True)
    below = guide[placements(*SIZE)["strip"][3] :, :, :].astype(np.int16)
    # Anything that is not the backdrop — a plate and the white lettering on
    # it count as one shape, where thresholding on darkness alone would split
    # each caption into a blob per letter.
    marked = (np.abs(below - 110).max(axis=2) > 40).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(marked, connectivity=8)
    assert count - 1 == len(_RULES), f"found {count - 1} rule plates, expected {len(_RULES)}"
    for _, y, w, h, _ in stats[1:]:
        assert w > 100, "a rule rendered as something other than a caption plate"
        assert y + h < below.shape[0], "a rule ran off the bottom of the sheet"


def test_a_strip_stood_on_end_is_refused_rather_than_misread() -> None:
    """Why rule 2 exists, stated as the pipeline's own behaviour."""
    width, height = SIZE
    image = np.full((height, width, 3), BACKDROP_RGB[0], dtype=np.uint8)

    sy = height / FIELD_MM[1]
    long_px = int(round(STRIP_PHYSICAL_MM[1] * sy))
    short_px = int(round(STRIP_PHYSICAL_MM[0] * sy))
    x0, y0 = int(width * 0.4), int(height * 0.3)
    image[y0 : y0 + long_px, x0 : x0 + short_px] = render_strip(short_px, long_px)

    assert detect_strip(image) is None, "a vertical strip must be refused, not measured"


def test_the_card_alone_would_still_be_recognised() -> None:
    """Isolates card detection from strip detection."""
    width, height = SIZE
    image = np.full((height, width, 3), BACKDROP_RGB[0], dtype=np.uint8)
    card = placements(width, height)["card"]
    image[card[1] : card[3], card[0] : card[2]] = render_card(card[2] - card[0], card[3] - card[1])

    assert detect_card(image) is not None


def test_guide_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        placements(0, 100)


def test_the_origins_are_inside_the_field_they_reference() -> None:
    """Guards the millimetre constants from drifting off the sheet."""
    assert CARD_ORIGIN_MM[0] + CARD_MM[0] <= FIELD_MM[0]
    assert CARD_ORIGIN_MM[1] + CARD_MM[1] <= FIELD_MM[1]
    assert STRIP_ORIGIN_MM[0] + STRIP_PHYSICAL_MM[1] <= FIELD_MM[0]
    assert STRIP_ORIGIN_MM[1] + STRIP_PHYSICAL_MM[0] <= FIELD_MM[1]
