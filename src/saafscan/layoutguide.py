"""A worked example of where everything goes before anyone takes a photo.

Composition is where this demo quietly fails. The strip must lie lengthways —
`detect_strip` refuses one stood on end because warping it across its short
axis would report three identical slices as three different reaction zones,
and a refusal is recoverable while a confident wrong verdict is not. The strip
must also sit more than 25 grey levels above a ring of the surface around it,
and both objects have to keep clear of each other and of the frame edge.

So this renders an example that is *checked*, not illustrated: the tests push
it through the same detect_card and detect_strip the live demo uses and assert
they return the objects where this guide put them. A picture of a good setup
that the pipeline itself would reject teaches the wrong lesson.

Geometry lives in :func:`layout` and :func:`placements`, separate from the
painting, so every rule on the sheet can be asserted against the detectors'
own gates rather than eyeballed.

    python -m saafscan.layoutguide --out layout_guide.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.calib import detect_card
from saafscan.printbackdrop import BACKDROP_RGB, backdrop_luma, in_range
from saafscan.printcard import PHYSICAL_SIZE_MM as CARD_MM
from saafscan.printcard import render_card
from saafscan.printstrip import STRIP_PHYSICAL_MM, render_strip
from saafscan.roi import detect_strip

Rect = tuple[int, int, int, int]

# The slice of surface the example shows, in millimetres. Chosen so both
# objects land comfortably inside the detectors' area-fraction gates: the strip
# occupies 1600/30000 = 0.053 of it against a 0.015-0.2 window, and the card
# 3750/30000 = 0.125 against 0.015-0.5.
FIELD_MM: tuple[float, float] = (200.0, 150.0)

# Origins in millimetres from the field's top-left. The card sits upper-left
# and the strip lower-centre, the arrangement render_training_scene uses and
# therefore the one detection is known to be tuned for. The strip sits low
# enough to leave the rules a row each at the bottom of the sheet.
CARD_ORIGIN_MM: tuple[float, float] = (20.0, 25.0)
STRIP_ORIGIN_MM: tuple[float, float] = (60.0, 95.0)

# The two size constants are ordered differently upstream — printcard's is
# (width, height), roi's is (short, long) — so each is unpacked by name here
# rather than passed through by position. Passing STRIP_PHYSICAL_MM straight
# into a (width, height) slot draws the strip stood on end.
CARD_SIZE_MM: tuple[float, float] = (CARD_MM[0], CARD_MM[1])
STRIP_SIZE_MM: tuple[float, float] = (STRIP_PHYSICAL_MM[1], STRIP_PHYSICAL_MM[0])

# Annotations are drawn on dark chips: a chip is dark, so it can never pass
# detect_strip's brightness gate, and wide, so it can never satisfy
# detect_card's 1.0-2.4 aspect gate. Staying out of both gates is the reason
# the labels do not get detected instead of the objects they point at.
# (printstrip.INK is the same 110 grey as the backdrop, so unbacked text here
# would be invisible anyway.)
_CHIP_BG = (28, 28, 30)
_CHIP_FG = (255, 255, 255)
_GAP_MM = 4.0
_RULE_GAP_PX = 6
_EDGE_MARGIN_PX = 12
_MIN_SCALE = 0.5

_TITLE = "HOW TO PLACE IT"
_CARD_LABEL = "CALIBRATION CARD - any way up"
_STRIP_LABEL = "STRIP - long edge LEFT-TO-RIGHT"
_RULES = (
    "1.  Grey fills the whole picture - surface luma 50-170 after white-balance",
    "2.  STRIP HORIZONTAL: long edge left-to-right, never stood on end",
    "3.  CALIBRATION CARD fully inside the frame, grey all around it",
    "4.  Card and strip do not touch or overlap each other",
    "5.  Keep your distance: the strip must stay under 1/5 of the picture area",
)


def overlaps(a: Rect, b: Rect) -> bool:
    """True when two rects share any pixel."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def placements(width_px: int, height_px: int) -> dict[str, Rect]:
    """Pixel rects ``(x0, y0, x1, y1)`` for the card and the strip.

    Exposed so the guide's geometry can be checked against the detectors'
    gates directly, rather than inferred from pixels after the fact.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("guide dimensions must be positive pixels")
    sx, sy = width_px / FIELD_MM[0], height_px / FIELD_MM[1]

    def rect(origin: tuple[float, float], size: tuple[float, float]) -> Rect:
        x0, y0 = int(round(origin[0] * sx)), int(round(origin[1] * sy))
        return x0, y0, x0 + int(round(size[0] * sx)), y0 + int(round(size[1] * sy))

    return {
        "card": rect(CARD_ORIGIN_MM, CARD_SIZE_MM),
        "strip": rect(STRIP_ORIGIN_MM, STRIP_SIZE_MM),
    }


def _pads(scale: float) -> tuple[int, int]:
    return max(4, int(round(8 * scale))), max(3, int(round(5 * scale)))


def _text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return tw, th


def _chip_rect(text: str, origin: tuple[int, int], scale: float, thickness: int) -> Rect:
    """Bounding rect a chip of `text` would paint at `origin`."""
    tw, th = _text_size(text, scale, thickness)
    pad_x, pad_y = _pads(scale)
    x0, y0 = origin
    return x0, y0, x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y


def _chip_height(text: str, scale: float, thickness: int) -> int:
    rect = _chip_rect(text, (0, 0), scale, thickness)
    return rect[3] - rect[1]


def _label_rect(object_rect: Rect, text: str, gap: int, scale: float, thickness: int) -> Rect:
    """Chip sitting `gap` above the object it names, clear of it by that gap.

    Placed above rather than beside because the strip is 80 mm long and there
    is no column beside it wide enough for the label at a readable size — and
    because ink landing on the card's frame or the strip's paper is ink the
    detectors would have to look through.
    """
    x0, y0 = object_rect[0], object_rect[1]
    return _chip_rect(
        text, (x0, max(0, y0 - gap - _chip_height(text, scale, thickness))), scale, thickness
    )


def _scale_for(width_px: int) -> float:
    return max(width_px / 900.0, _MIN_SCALE)


def _thickness(scale: float) -> int:
    return max(int(round(scale)), 1)


def _rule_rects(width_px: int, rules_y: int, scale: float, thickness: int) -> list[Rect]:
    sx = width_px / FIELD_MM[0]
    step = _chip_height(_RULES[0], scale, thickness) + _RULE_GAP_PX
    x0 = int(round(8 * sx))
    return [
        _chip_rect(rule, (x0, rules_y + index * step), scale, thickness)
        for index, rule in enumerate(_RULES)
    ]


def _rules_fit(width_px: int, height_px: int, rules_y: int, scale: float) -> bool:
    for _, _, x1, y1 in _rule_rects(width_px, rules_y, scale, _thickness(scale)):
        if x1 > width_px - _EDGE_MARGIN_PX or y1 > height_px - _EDGE_MARGIN_PX:
            return False
    return True


def _fit_scale(width_px: int, height_px: int, rules_y: int) -> float:
    """Largest caption size at which every rule still lands inside the sheet.

    Shrunk until it fits rather than solved from an assumed character width:
    plate height comes from getTextSize, and measuring beats predicting it.
    The first attempt here was predicted, and produced a caption 1922 px wide
    on a 1600 px sheet with the last two rules below the canvas altogether.
    """
    scale = _scale_for(width_px)
    while True:
        if _rules_fit(width_px, height_px, rules_y, scale):
            return scale
        if scale <= _MIN_SCALE:
            raise ValueError("canvas is too small to hold the five rules — make it taller or wider")
        scale = max(_MIN_SCALE, round(scale * 0.9, 3))


def layout(width_px: int, height_px: int) -> dict[str, object]:
    """Every rect the guide paints, in pixels — geometry without drawing.

    Keeping this separate from :func:`render_example` is what lets the tests
    check that each label sits clear of the objects and that all five rules
    land inside the sheet, instead of only noticing a clipped caption once it
    has been printed.
    """
    rects = placements(width_px, height_px)
    sx, sy = width_px / FIELD_MM[0], height_px / FIELD_MM[1]
    gap = int(round(_GAP_MM * sy))

    rules_y = rects["strip"][3] + gap
    scale = _fit_scale(width_px, height_px, rules_y)
    thickness = _thickness(scale)

    title = _chip_rect(_TITLE, (int(8 * sx), int(4 * sy)), scale * 1.25, thickness)
    labels = {
        "card_label": _label_rect(rects["card"], _CARD_LABEL, gap, scale, thickness),
        "strip_label": _label_rect(rects["strip"], _STRIP_LABEL, gap, scale, thickness),
    }

    return {
        **rects,
        "title": title,
        **labels,
        "rules": _rule_rects(width_px, rules_y, scale, thickness),
        "scale": scale,
        "thickness": thickness,
        "gap": gap,
    }


def _paint_chip(image: np.ndarray, text: str, rect: Rect, scale: float, thickness: int) -> None:
    """Dark label plate with white text, filling `rect` exactly."""
    x0, y0, x1, y1 = rect
    cv2.rectangle(image, (x0, y0), (x1, y1), _CHIP_BG, -1, cv2.LINE_AA)
    pad_x, pad_y = _pads(scale)
    _, th = _text_size(text, scale, thickness)
    cv2.putText(
        image,
        text,
        (x0 + pad_x, y0 + pad_y + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _CHIP_FG,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _paint_arrow(image: np.ndarray, label: Rect, object_rect: Rect) -> None:
    """Stem from the label down onto the object, stopping short of it.

    Stops short deliberately: a dot on the card's black frame would lighten
    the very border _looks_like_card compares against the interior.
    """
    stem = object_rect[0] + 24
    cv2.line(image, (stem, label[3] + 4), (stem, object_rect[1] - 8), _CHIP_FG, 2, cv2.LINE_AA)
    cv2.circle(image, (stem, object_rect[1] - 8), 4, _CHIP_FG, -1, cv2.LINE_AA)


def render_example(width_px: int, height_px: int, *, annotate: bool = True) -> np.ndarray:
    """The worked example: backdrop, card and strip, optionally labelled.

    Scale is true — card and strip keep the sizes they really have relative to
    each other and to the surface, so the picture teaches the spacing as well
    as the arrangement.
    """
    plan = layout(width_px, height_px)
    guide = np.full((height_px, width_px, 3), BACKDROP_RGB[0], dtype=np.uint8)

    cx0, cy0, cx1, cy1 = plan["card"]
    guide[cy0:cy1, cx0:cx1] = render_card(cx1 - cx0, cy1 - cy0)

    sx0, sy0, sx1, sy1 = plan["strip"]
    guide[sy0:sy1, sx0:sx1] = render_strip(sx1 - sx0, sy1 - sy0)

    if not annotate:
        return np.ascontiguousarray(guide)

    scale, thickness = plan["scale"], plan["thickness"]
    _paint_chip(guide, _TITLE, plan["title"], scale * 1.25, thickness)
    _paint_chip(guide, _CARD_LABEL, plan["card_label"], scale, thickness)
    _paint_chip(guide, _STRIP_LABEL, plan["strip_label"], scale, thickness)
    _paint_arrow(guide, plan["card_label"], plan["card"])
    _paint_arrow(guide, plan["strip_label"], plan["strip"])
    for rule, rect in zip(_RULES, plan["rules"], strict=True):
        _paint_chip(guide, rule, rect, scale, thickness)

    return np.ascontiguousarray(guide)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.layoutguide",
        description="Render a worked example of how to arrange the shot.",
    )
    parser.add_argument("--out", type=Path, default=Path("layout_guide.png"))
    parser.add_argument("--width", type=int, default=1600, help="output width in pixels")
    parser.add_argument("--height", type=int, default=1200, help="output height in pixels")
    args = parser.parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        print("--width and --height must be positive", file=sys.stderr)
        return 2

    try:
        guide = render_example(args.width, args.height)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    cv2.imwrite(str(args.out), cv2.cvtColor(guide, cv2.COLOR_RGB2BGR))

    card = detect_card(guide)
    strip = detect_strip(guide)
    luma = backdrop_luma(guide)
    card_found, strip_found = card is not None, strip is not None
    print(f"wrote {args.out} ({args.width} x {args.height} px)")
    print(
        f"this example itself: card={'found' if card_found else 'NOT FOUND'}, "
        f"strip={'found' if strip_found else 'NOT FOUND'}, surface luma {luma:.1f} "
        f"({'in range' if in_range(luma) else 'OUT OF RANGE'})"
    )
    return 0 if card_found and strip_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
