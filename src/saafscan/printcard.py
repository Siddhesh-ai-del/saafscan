"""Print-ready artwork for the physical colour calibration card.

The card previously existed only as a canvas inside the synthetic renderer, so
there was nothing to print — and without a printed card the real-photo demo
cannot be tested before the reagents arrive. This reproduces the exact patch
geometry `extract_patches` samples, at any requested resolution, so the printed
card and the detector agree by construction rather than by luck.

Print it on **matte** photo paper. Glossy stock catches glare and breaks both
card and strip detection. Cut along the black frame; the black border is part
of the detector's dark-frame check, so do not trim it away.

    python -m saafscan.printcard --out print_card.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.calib import CARD_NOMINAL

# Physical size, mm. The aspect matches CARD_NOMINAL (276 x 184 = 1.5), which
# detect_card requires to be within its 1.0-2.4 gate. Absolute size is not
# gated, only the aspect and the patch contrast, but 75 x 50 mm keeps every
# patch large enough to resolve in an ordinary phone photograph.
PHYSICAL_SIZE_MM: tuple[float, float] = (75.0, 50.0)
DEFAULT_DPI = 300


def _sampling_bounds() -> tuple[float, float, float, float]:
    """The patch field bounds, mirroring calib.extract_patches exactly.

    Anything drawn outside these bounds is frame, which is what makes the
    printed card pass `_looks_like_card`'s dark-frame check.
    """
    nominal_w = float(CARD_NOMINAL["width"])
    nominal_h = float(CARD_NOMINAL["height"])
    border = float(CARD_NOMINAL["border"])
    inset = max(int(0.12 * nominal_h), int(border) + 2)
    return border + 4.0, nominal_w - border - 4.0, float(inset), nominal_h - float(inset)


def render_card(width_px: int, height_px: int) -> np.ndarray:
    """Calibration card artwork at the requested pixel size, RGB uint8.

    Black frame, six nominal patches in the row-major order extract_patches
    expects: white, grey, black, then red, green, amber.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("card dimensions must be positive pixels")

    nominal_w = float(CARD_NOMINAL["width"])
    nominal_h = float(CARD_NOMINAL["height"])
    x0, x1, y0, y1 = _sampling_bounds()
    sx, sy = width_px / nominal_w, height_px / nominal_h

    # The frame is the canvas itself: everything outside the patch field.
    card = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cols, rows = 3, 2
    cell_w, cell_h = (x1 - x0) / cols, (y1 - y0) / rows
    for index, patch in enumerate(CARD_NOMINAL["patches"]):
        row, col = divmod(index, cols)
        cx0 = int(round((x0 + col * cell_w) * sx))
        cx1 = int(round((x0 + (col + 1) * cell_w) * sx))
        cy0 = int(round((y0 + row * cell_h) * sy))
        cy1 = int(round((y0 + (row + 1) * cell_h) * sy))
        card[cy0:cy1, cx0:cx1] = patch["rgb"]
    return np.ascontiguousarray(card)


def print_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    width_mm, height_mm = PHYSICAL_SIZE_MM
    return (
        int(round(width_mm / 25.4 * dpi)),
        int(round(height_mm / 25.4 * dpi)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.printcard",
        description="Render the colour calibration card for printing.",
    )
    parser.add_argument("--out", type=Path, default=Path("print_card.png"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="output resolution")
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        print("--dpi must be positive", file=sys.stderr)
        return 2

    width_px, height_px = print_size_px(args.dpi)
    card = render_card(width_px, height_px)
    cv2.imwrite(str(args.out), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))

    width_mm, height_mm = PHYSICAL_SIZE_MM
    print(f"wrote {args.out} ({width_px} x {height_px} px, {width_mm:g} x {height_mm:g} mm)")
    print("print at 100% scale on MATTE photo paper — glossy stock causes glare")
    print("cut along the black frame, keep the border, make 3 copies plus spares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
