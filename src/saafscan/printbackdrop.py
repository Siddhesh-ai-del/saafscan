"""Print-ready mid-grey backdrop: the surface the card and strip lie on.

The backdrop is a constraint, not a preference. Below 50 luma the calibration
card's black frame stops being distinguishable from the surface it lies on and
card detection fails outright; above 170 the white reagent strip merges into
it. Light kraft or tan paper reads ~190 once the colour cast is corrected and
lands outside the band — which is why a sheet you can print beats hoping the
venue has a suitable table.

    python -m saafscan.printbackdrop --out backdrop.png
    python -m saafscan.printbackdrop --check photo.jpg

A flat colour carries no geometry, so print scale does not matter; what does
matter is the value that comes out of the printer. Photograph the printed
sheet and pass it to ``--check`` — if it lands outside the band, reprint
darker or lighter and check again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.calib import white_balance

# Measured through the pipeline, not guessed: a backdrop rendered at 40 makes
# detect_card return None, and the range the detection fails across matches
# this lower bound.
LUMA_MIN = 50.0
LUMA_MAX = 170.0

# Centre of that band, so a printer laying ink down darker than asked and a
# room lit brighter than the last session both have the same headroom.
BACKDROP_RGB: tuple[int, int, int] = (110, 110, 110)

SHEET_MM: tuple[float, float] = (210.0, 297.0)  # A4 portrait
DEFAULT_DPI = 300

# Rec.601 luma, the same weighting the band above was measured with.
_WEIGHTS = (0.299, 0.587, 0.114)


def backdrop_luma(image: np.ndarray) -> float:
    """Luma of a surface, after white-balance as the specification requires.

    Accepts either a single RGB triple — the colour going onto the sheet — or
    a full photograph, which is balanced through the calibration card first so
    its reading is comparable to the band it is checked against.

    Photographs are reduced by median rather than mean: the card and the strip
    sitting on the backdrop are a minority of the frame and sit at opposite
    ends of the scale, so a mean would be dragged by whichever happened to
    cover more of the picture.

    Raises:
        ValueError: If ``image`` is neither a triple nor a photograph.
    """
    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim == 1 and pixels.shape[0] == 3:
        pass  # a colour: nothing to balance and nothing else in frame
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        pixels = white_balance(np.clip(pixels, 0, 255).astype(np.uint8)).astype(np.float32)
    else:
        raise ValueError("expected an RGB triple or an HxWx3 photograph")

    r, g, b = _WEIGHTS
    luma = r * pixels[..., 0] + g * pixels[..., 1] + b * pixels[..., 2]
    return float(np.median(luma))


def in_range(luma: float) -> bool:
    """Whether a measured surface luma can still support both detections."""
    return LUMA_MIN <= luma <= LUMA_MAX


def sheet_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """A4 portrait in pixels at ``dpi``."""
    width_mm, height_mm = SHEET_MM
    return int(round(width_mm / 25.4 * dpi)), int(round(height_mm / 25.4 * dpi))


def render_sheet(dpi: int = DEFAULT_DPI) -> np.ndarray:
    """An A4 page of uniform backdrop, RGB uint8.

    Unbroken on purpose: any mark printed inside the sheet is a surface
    feature the measurement would take seriously.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    width_px, height_px = sheet_size_px(dpi)
    return np.full((height_px, width_px, 3), BACKDROP_RGB[0], dtype=np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.printbackdrop",
        description="Render the mid-grey shooting backdrop, or check a photo of it.",
    )
    parser.add_argument("--out", type=Path, default=Path("backdrop.png"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="output resolution")
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        metavar="PHOTO",
        help="measure a photographed backdrop instead of rendering one",
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        if not args.check.is_file():
            print(f"no such file: {args.check}", file=sys.stderr)
            return 2
        bgr = cv2.imread(str(args.check), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"could not decode {args.check} as an image", file=sys.stderr)
            return 2
        measured = backdrop_luma(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ok = in_range(measured)
        band = f"{LUMA_MIN:g}-{LUMA_MAX:g}"
        print(
            f"surface luma {measured:.1f} (band {band}) -> {'IN RANGE' if ok else 'OUT OF RANGE'}"
        )
        if not ok:
            direction = "lighter" if measured > LUMA_MAX else "darker"
            print(f"reprint the sheet {direction} and check again")
        return 0 if ok else 1

    if args.dpi <= 0:
        print("--dpi must be positive", file=sys.stderr)
        return 2

    sheet = render_sheet(args.dpi)
    cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    width_px, height_px = sheet_size_px(args.dpi)
    luma = backdrop_luma(np.asarray(BACKDROP_RGB, np.float32))
    print(f"wrote {args.out} ({width_px} x {height_px} px, A4 portrait, {args.dpi} dpi)")
    print(f"flat {BACKDROP_RGB[0]} grey, luma {luma:.1f} (band {LUMA_MIN:g}-{LUMA_MAX:g})")
    print("print scale does not matter for a flat colour — use NORMAL quality, matte stock")
    print("glossy stock glares; a heavy-ink 'high quality' pass can band and streak")
    print("lay the calibration card and the strips on the grey, then photograph")
    print("check the photo: python -m saafscan.printbackdrop --check your_photo.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
