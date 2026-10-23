"""Printable colour targets: prove the imaging half before any chemistry.

A printed target is not an adulterant result and must never be presented as
one. What it buys is separation of the two ways this demo can fail. A real
strip that misreads could be the camera, the white balance, the strip
detector, the zone extraction — or the chemistry. A printed target has no
chemistry to go wrong, so if a printed `detergent` target photographs and
reads back as detergent, the whole imaging chain is proven and any later
failure is unambiguously the chemistry.

The targets are drawn by :mod:`saafscan.printstrip`, not from a second set of
coordinates, so a printed target is the *same physical object* as the real
strip it stands in for — control patch, window geometry, and all.

    python -m saafscan.caltarget --out assets/caltargets.png

Print at 100 % / actual size, same as the real template.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from saafscan.classifier import CLEAR, FIRED
from saafscan.color import delta_e
from saafscan.printstrip import (
    CONTROL_COLOUR,
    DEFAULT_DPI,
    sheet_size_px,
)
from saafscan.printstrip import (
    render_sheet as _template_sheet,
)
from saafscan.printstrip import (
    render_strip as _template_strip,
)
from saafscan.roi import ZONE_NAMES, ZONE_NOMINAL
from saafscan.synthetic import REACTION_COLOURS, TRAINING_CLASSES

# One target per class the model can emit.
TARGETS: tuple[str, ...] = TRAINING_CLASSES

# Ambiguous chemistry is a *band*, CLEAR..FIRED, not a threshold. Aim the
# ambiguous target at the middle of it so it stays as far from either edge as
# an ambiguous strip can be — one sitting a whisker under FIRED would pass by
# luck rather than test the refusal we actually care about.
AMBIGUOUS_DELTA_E = (CLEAR + FIRED) / 2.0

_BISECTIONS = 40


def ambiguous_colour(name: str) -> tuple[int, int, int]:
    """A colour sitting ``AMBIGUOUS_DELTA_E`` from ``name``'s nominal.

    Bisected on the real :func:`saafscan.color.delta_e` rather than blended on
    RGB, because ΔE is not linear in RGB: an evenly spaced blend lands nowhere
    near the band it is aiming at.
    """
    reacted = REACTION_COLOURS["both"].get(name)
    if reacted is None:
        raise ValueError(f"no reaction model for zone: {name!r}")
    start = np.asarray(ZONE_NOMINAL[name], dtype=np.float64)
    end = np.asarray(reacted, dtype=np.float64)

    lo, hi = 0.0, 1.0
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2.0
        if delta_e(start + (end - start) * mid, start) < AMBIGUOUS_DELTA_E:
            lo = mid
        else:
            hi = mid
    blend = (start + (end - start) * (lo + hi) / 2.0).round()
    return tuple(int(c) for c in blend)


def zone_colours(label: str) -> dict[str, tuple[int, int, int]]:
    """The three zone colours a printed target for ``label`` carries.

    ``inconclusive`` is not a reaction, it is the classifier refusing to guess,
    so it cannot be drawn as one. It is drawn as *both* reagent zones partway
    reacted, which is what an expired strip or a half-dipped sample looks like.
    """
    if label not in TARGETS:
        raise ValueError(f"unknown target: {label!r}")

    reacted = REACTION_COLOURS.get(label, {})
    ambiguous = label == "inconclusive"
    return {
        name: (
            CONTROL_COLOUR
            if name == "control"
            else ambiguous_colour(name)
            if ambiguous
            else reacted.get(name, ZONE_NOMINAL[name])
        )
        for name in ZONE_NAMES
    }


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
