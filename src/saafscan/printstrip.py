"""Print-ready artwork for the reagent strips.

Applying reagent by eye onto a bare strip is what went wrong in
`photo5.jpeg`: the three marks landed one zone to the left, so untouched paper
in the detergent band read back as a fully decoloured — i.e. *reacted* — zone,
and the rule table answered `both` with 0.71 confidence for a strip that had
never held detergent. Nobody can place drops to ±4 mm by eye, so the placement
has to be printed rather than remembered.

Two of the boxes are dotted squares, one per reagent, at the millimetres
:func:`saafscan.roi.zones_from_strip` actually averages. They sit half a
millimetre *proud* of each measurement window: reagent applied inside a square
is guaranteed to cover the band, while no ink falls inside it to bias the mean
colour the verdict is computed from.

The third box, the control, is printed solid instead — see ``CONTROL_COLOUR``.
It holds no reagent at all. It is a reference patch, so it carries the
classifier's own nominal colour by construction and needs no dye.

Print at **100 % / actual size** and disable "fit to page" — every dimension
here is a millimetre, and a scaled print moves the squares off the strip.

    python -m saafscan.printstrip --out strip_template.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.roi import (
    STRIP_PHYSICAL_MM,
    ZONE_NAMES,
    ZONE_NOMINAL,
    zone_window_mm,
)

# A4 portrait, mm.
SHEET_MM: tuple[float, float] = (210.0, 297.0)
MARGIN_MM = 12.0
LABEL_MM = 6.0  # strip label above each strip
SLOT_GAP_MM = 5.0  # cutting slack between slots
PROUD_MM = 0.5  # how far each square sits outside its measurement window
CUT_OUTLINE_MM = 0.4  # cut line offset, so the kept strip carries no ink
LINE_MM = 0.35
DASH_MM = 2.0
GAP_MM = 1.6
DEFAULT_DPI = 300
INK = np.array([110, 110, 110], dtype=np.uint8)

# The control box is printed solid rather than left as an aiming outline. It is
# the reference classify_zones measures against, so it is filled with that
# reference rather than a colour chosen here: print a control the guard does not
# recognise and every strip is refused as inconclusive. Left as an outline it is
# also a zone an operator must remember to load, and photo4 priced that — with no
# control mark the window read blank paper, deltaE 79.8 against nominal, and the
# strip was refused before its chemistry was ever considered.
CONTROL_ZONE = ZONE_NAMES[0]
CONTROL_COLOUR = ZONE_NOMINAL[CONTROL_ZONE]

_HEADER = (
    "PRINT AT 100% / ACTUAL SIZE  -  turn 'fit to page' OFF",
    "80 x 20 mm each  -  cut on the outlines  -  the RED box is a printed "
    "reference: leave it dry  -  apply reagent only in the other two boxes",
)


def zone_windows_proud() -> dict[str, tuple[float, float, float, float]]:
    """The aiming squares: each measurement window grown by ``PROUD_MM``.

    Growing rather than insetting is the whole design. An operator aims *inside*
    a square, so the square must enclose the band it protects — a square smaller
    than the window would let even a careful person miss the measurement.
    """
    return {
        name: (x0 - PROUD_MM, x1 + PROUD_MM, y0 - PROUD_MM, y1 + PROUD_MM)
        for name, (x0, x1, y0, y1) in zone_window_mm().items()
    }


def sheet_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """Sheet dimensions in pixels, as ``(width, height)``."""
    width_mm, height_mm = SHEET_MM
    return (int(round(width_mm / 25.4 * dpi)), int(round(height_mm / 25.4 * dpi)))


def strip_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """One strip's dimensions in pixels, as ``(width, height)``."""
    long_mm, short_mm = STRIP_PHYSICAL_MM[1], STRIP_PHYSICAL_MM[0]
    return (int(round(long_mm / 25.4 * dpi)), int(round(short_mm / 25.4 * dpi)))


def grid() -> tuple[int, int]:
    """Columns and rows of strips that fit inside the printable area."""
    slot_w = STRIP_PHYSICAL_MM[1] + SLOT_GAP_MM
    slot_h = STRIP_PHYSICAL_MM[0] + LABEL_MM + SLOT_GAP_MM
    cols = int((SHEET_MM[0] - 2.0 * MARGIN_MM) // slot_w)
    rows = int((SHEET_MM[1] - 2.0 * MARGIN_MM) // slot_h)
    return cols, rows


def strips_per_sheet() -> int:
    cols, rows = grid()
    return cols * rows


def _fill(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float, rgb: tuple[int, int, int]
) -> None:
    """Paint an axis-aligned block, clamped to the canvas."""
    height, width = image.shape[:2]
    left = max(0, min(int(round(x0)), width - 1))
    right = max(left + 1, min(int(round(x1)), width))
    top = max(0, min(int(round(y0)), height - 1))
    bottom = max(top + 1, min(int(round(y1)), height))
    image[top:bottom, left:right] = rgb


def _stroke(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    """Draw an axis-aligned mark in the template ink."""
    _fill(image, x0, y0, x1, y1, INK)


def _dotted_box(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, sx: float) -> None:
    """A dashed rectangle whose corners close properly.

    Dashes are clamped to the box on both axes, so the last one meets the
    corner instead of overrunning it — a stray mark outside the box is harmless
    to the measurement but it is the sort of thing that makes a person doubt
    where to aim.
    """
    line = max(int(round(LINE_MM * sx)), 1)
    dash = max(int(round(DASH_MM * sx)), 2)
    step = dash + max(int(round(GAP_MM * sx)), 1)
    left, right = int(round(x0)), int(round(x1))
    top, bottom = int(round(y0)), int(round(y1))
    half = line // 2
    for y in (top + half, bottom - half):
        for x in range(left, right, step):
            _stroke(image, x, y - half, min(x + dash, right), y + half + 1)
    for x in (left + half, right - half):
        for y in range(top, bottom, step):
            _stroke(image, x - half, y, x + half + 1, min(y + dash, bottom))


def render_strip(width_px: int, height_px: int) -> np.ndarray:
    """One strip's artwork: white paper, two dotted aiming squares, one solid one.

    Nothing else is drawn on the strip itself. Labels and cut outlines belong to
    :func:`render_sheet` because they would be ink sitting in the paper that the
    milk can dissolve and the camera can then read back as reaction colour.

    The control box is filled solid and the two reagent boxes are not, which is
    the whole asymmetry: reagent goes in the empty ones, nothing goes in the red
    one, and the red one is already exactly what the classifier expects to see.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("strip dimensions must be positive pixels")

    strip = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    sx = width_px / STRIP_PHYSICAL_MM[1]
    sy = height_px / STRIP_PHYSICAL_MM[0]
    for name, (x0, x1, y0, y1) in zone_windows_proud().items():
        # Fill the proud box rather than the window. Half a millimetre of
        # misalignment still lands entirely on solid colour, so the reference
        # stays stable; past that the reading degrades sharply, which is the
        # guard working — a strip that far out of position cannot have its
        # other two zones trusted either.
        if name == CONTROL_ZONE:
            _fill(strip, x0 * sx, y0 * sy, x1 * sx, y1 * sy, CONTROL_COLOUR)
        _dotted_box(strip, x0 * sx, y0 * sy, x1 * sx, y1 * sy, sx)
    return np.ascontiguousarray(strip)


def _label(
    image: np.ndarray, text: str, cx: int, baseline: int, scale: float, thickness: int
) -> None:
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _put_text(image, text, (max(0, cx - width // 2), baseline), scale, thickness)


def _put_text(
    image: np.ndarray, text: str, origin: tuple[int, int], scale: float, thickness: int
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        tuple(int(c) for c in INK),
        thickness,
        lineType=cv2.LINE_AA,
    )


def render_sheet(dpi: int = DEFAULT_DPI) -> np.ndarray:
    """An A4 page of strip templates, RGB uint8.

    Each slot carries a cut outline offset clear of the strip and a zone label
    above it, both of which stay out of the paper itself.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    width_px, height_px = sheet_size_px(dpi)
    sheet = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    px_per_mm = dpi / 25.4
    cols, rows = grid()

    slot_w = STRIP_PHYSICAL_MM[1] + SLOT_GAP_MM
    slot_h = STRIP_PHYSICAL_MM[0] + LABEL_MM + SLOT_GAP_MM
    origin_x = (SHEET_MM[0] - cols * slot_w) / 2.0
    origin_y = (SHEET_MM[1] - rows * slot_h) / 2.0

    long_px, short_px = strip_size_px(dpi)
    strip = render_strip(long_px, short_px)
    label_scale = 0.8 * dpi / DEFAULT_DPI
    label_thickness = max(int(round(dpi / DEFAULT_DPI)), 1)
    outline = max(int(round(CUT_OUTLINE_MM * px_per_mm)), 1)
    centres = {name: (w[0] + w[1]) / 2.0 for name, w in zone_window_mm().items()}

    for index in range(strips_per_sheet()):
        row, col = divmod(index, cols)
        left = int(round((origin_x + col * slot_w) * px_per_mm))
        top = int(round((origin_y + row * slot_h + LABEL_MM) * px_per_mm))
        sheet[top : top + short_px, left : left + long_px] = strip

        _stroke(sheet, left - outline, top - outline, left + long_px + outline, top)
        _stroke(
            sheet,
            left - outline,
            top + short_px,
            left + long_px + outline,
            top + short_px + outline,
        )
        _stroke(sheet, left - outline, top - outline, left, top + short_px + outline)
        _stroke(
            sheet, left + long_px, top - outline, left + long_px + outline, top + short_px + outline
        )

        baseline = top - int(round(1.6 * px_per_mm))
        for name in ZONE_NAMES:
            cx = left + int(round(centres[name] * px_per_mm))
            _label(sheet, name, cx, baseline, label_scale, label_thickness)

    # Well inside the printable area: printers commonly refuse the first ~5 mm,
    # and losing the instruction block would be a silent failure at the venue.
    _put_text(
        sheet,
        _HEADER[0],
        (int(round(MARGIN_MM * px_per_mm)), int(round(15.0 * px_per_mm))),
        0.72 * dpi / DEFAULT_DPI,
        label_thickness,
    )
    _put_text(
        sheet,
        _HEADER[1],
        (int(round(MARGIN_MM * px_per_mm)), int(round(19.5 * px_per_mm))),
        0.48 * dpi / DEFAULT_DPI,
        label_thickness,
    )
    return np.ascontiguousarray(sheet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.printstrip",
        description="Render the reagent-strip template for printing.",
    )
    parser.add_argument("--out", type=Path, default=Path("strip_template.png"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="output resolution")
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        print("--dpi must be positive", file=sys.stderr)
        return 2

    sheet = render_sheet(args.dpi)
    cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    width_px, height_px = sheet_size_px(args.dpi)
    cols, rows = grid()
    print(f"wrote {args.out} ({width_px} x {height_px} px, A4 portrait, {args.dpi} dpi)")
    print(
        f"{strips_per_sheet()} strips per sheet ({cols} x {rows}), each "
        f"{STRIP_PHYSICAL_MM[1]:g} x {STRIP_PHYSICAL_MM[0]:g} mm"
    )
    print("PRINT AT 100% / ACTUAL SIZE with 'fit to page' OFF — scaled print moves the squares")
    print("cut on the outlines; leave the red reference box dry; dip the whole strip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
