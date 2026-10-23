"""Printable colour targets: prove the imaging half before any chemistry.

A printed target is not an adulterant result and must never be presented as
one. What it buys is separation of the two ways this demo can fail. A real
strip that misreads could be the camera, the white balance, the strip
detector, the zone extraction — or the chemistry. A printed target has no
chemistry to go wrong, so if a printed `detergent` target photographs and
reads back as detergent, the whole imaging chain is proven and any later
failure is unambiguously the chemistry.

Two things make the target trustworthy rather than merely pretty:

* it is drawn by :mod:`saafscan.printstrip`, not from a second set of
  coordinates, so a printed target is the *same physical object* as the real
  strip it stands in for — control patch, window geometry, and all;
* its colours come from :func:`saafscan.synthetic.zone_colours`, the same
  function that renders the training strip. Train on it, print it, and the
  classifier can only ever be tested on the object it was taught.

    python -m saafscan.caltarget --out assets/caltargets.png

Print at 100 % / actual size, same as the real template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.printstrip import DEFAULT_DPI, sheet_size_px
from saafscan.printstrip import render_sheet as _template_sheet
from saafscan.printstrip import render_strip as _template_strip
from saafscan.synthetic import TRAINING_CLASSES, zone_colours

# One target per class the model can emit.
TARGETS: tuple[str, ...] = TRAINING_CLASSES


def render_strip(width_px: int, height_px: int, label: str) -> np.ndarray:
    """A printed target for ``label``, on the real strip's geometry."""
    return _template_strip(width_px, height_px, fills=zone_colours(label))


def target_page() -> list[tuple[str, dict[str, tuple[int, int, int]]]]:
    """The sheet spec: one captioned, coloured strip per class.

    Separate from :func:`render_sheet` so the header choice can be checked
    against it without OCR.
    """
    return [(f"CAL {label.upper()}", zone_colours(label)) for label in TARGETS]


def render_sheet(dpi: int = DEFAULT_DPI) -> np.ndarray:
    """An A4 page carrying one target per class, captioned above each strip."""
    return _template_sheet(dpi, strips=target_page())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.caltarget",
        description="Render printed colour targets, one per verdict class.",
    )
    parser.add_argument("--out", type=Path, default=Path("caltargets.png"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="output resolution")
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        print("--dpi must be positive", file=sys.stderr)
        return 2

    sheet = render_sheet(args.dpi)
    cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    width_px, height_px = sheet_size_px(args.dpi)
    print(f"wrote {args.out} ({width_px} x {height_px} px, A4 portrait, {args.dpi} dpi)")
    print(f"{len(TARGETS)} targets: {', '.join(TARGETS)}")
    print("PRINT AT 100% / ACTUAL SIZE with 'fit to page' OFF")
    print(
        "WARNING: these are printed colour references, NOT milk samples.\n"
        "Photograph them to calibrate the camera and white balance only.\n"
        "Never ingest them into data/realset — they contain no chemistry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
