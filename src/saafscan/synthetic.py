"""Deterministic synthetic SaafScan scenes for demo mode and tests.

Renders an RGB scene containing the printed calibration card and the reagent
strip under mild lighting variation, plus an adversarial photo-like image with
neither (the 'judge uploads a selfie' case).

⚠️ Synthetic scenes exist for Round-1 screenshots and dry-run rehearsal only.
The 23 Oct final demo must run on real reagent strips (see README).
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from saafscan.calib import CARD_NOMINAL
from saafscan.roi import ZONE_NOMINAL

SampleLabel = Literal["none", "starch", "detergent", "both"]
ADULTERANTS: tuple[str, ...] = ("none", "starch", "detergent", "both")

SCENE_SIZE = (640, 480)  # (width, height)

# Reaction outcomes: how each reagent zone looks after the reaction.
_REACTED = {
    "starch": {"starch": (45, 40, 95)},  # blue-black starch-iodine complex
    "detergent": {"detergent": (214, 216, 220)},  # methylene blue fully reduced (near-colourless)
    "both": {"starch": (45, 40, 95), "detergent": (214, 216, 220)},
    "none": {},
}


def _card_canvas(rng: np.random.Generator) -> np.ndarray:
    """Render the calibration card exactly to nominal spec, no lighting."""
    w, h, border = CARD_NOMINAL["width"], CARD_NOMINAL["height"], CARD_NOMINAL["border"]
    card = np.full((h, w, 3), 18, dtype=np.uint8)  # black frame
    x0, y0 = border, border
    cell_w = (w - 2 * border) / 3.0
    cell_h = (h - 2 * border) / 2.0
    for i, patch in enumerate(CARD_NOMINAL["patches"]):
        row, col = divmod(i, 3)
        cx0 = int(x0 + col * cell_w + 3)
        cx1 = int(x0 + (col + 1) * cell_w - 3)
        cy0 = int(y0 + row * cell_h + 3)
        cy1 = int(y0 + (row + 1) * cell_h - 3)
        card[cy0:cy1, cx0:cx1] = np.array(patch["rgb"], dtype=np.uint8)
    return card


def _strip_canvas(label: str) -> np.ndarray:
    """Render the strip with nominal or reacted zone colours, no lighting."""
    from saafscan.roi import STRIP_SIZE

    w, h = STRIP_SIZE
    strip = np.full((h, w, 3), 238, dtype=np.uint8)  # white paper
    band = w / 3.0
    reacted = _REACTED[label]
    for i, name in enumerate(("control", "starch", "detergent")):
        colour = reacted.get(name, ZONE_NOMINAL[name])
        cx0 = int(i * band + 0.15 * band)
        cx1 = int((i + 1) * band - 0.15 * band)
        strip[int(0.25 * h) : int(0.75 * h), cx0:cx1] = np.array(colour, dtype=np.uint8)
    return strip


def _paste(scene: np.ndarray, canvas: np.ndarray, x: int, y: int, angle: float) -> None:
    """Rotate `canvas` by `angle` degrees and paste centered at (x, y)."""
    h, w = canvas.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    hh, ww = scene.shape[:2]
    x0, y0 = max(x - w // 2, 0), max(y - h // 2, 0)
    x1, y1 = min(x + w // 2, ww), min(y + h // 2, hh)
    sx0, sy0 = x0 - (x - w // 2), y0 - (y - h // 2)
    region = rotated[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    scene[y0:y1, x0:x1] = region


def render_sample(label: str, seed: int = 0) -> np.ndarray:
    """Render a complete synthetic scene for an adulterant label (RGB uint8).

    Deterministic per (label, seed). Applies mild per-channel lighting and a
    soft vignette so detection/white-balance are exercised, not assumed.
    """
    if label not in ADULTERANTS:
        raise ValueError(f"unknown sample label: {label!r}")
    rng = np.random.default_rng(seed)
    w, h = SCENE_SIZE
    base = np.full((h, w, 3), 143, dtype=np.int16)
    for ch in range(3):
        base[:, :, ch] += rng.integers(-6, 7, size=(h, w), dtype=np.int16)
    scene = np.clip(base, 0, 255).astype(np.uint8)

    _paste(
        scene,
        _card_canvas(rng),
        x=int(w * 0.30),
        y=int(h * 0.30),
        angle=float(rng.uniform(-2.5, 2.5)),
    )
    _paste(
        scene,
        _strip_canvas(label),
        x=int(w * 0.55),
        y=int(h * 0.74),
        angle=float(rng.uniform(-3.0, 3.0)),
    )

    # Lighting: mild per-channel gain + soft vignette (keeps anchors usable
    # across the full exposure range the tests exercise: 0.55x – 1.3x).
    gains = rng.uniform(0.95, 1.03, size=3)
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vignette = (1.0 - 0.06 * np.clip(radius, 0, 1) ** 2)[:, :, None]
    out = scene.astype(np.float32) * gains[None, None, :] * vignette
    noise = rng.normal(0, 2.0, size=out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def render_random_photo(seed: int = 0) -> np.ndarray:
    """Photo-like RGB image with no card and no strip (must be rejected)."""
    rng = np.random.default_rng(seed)
    w, h = SCENE_SIZE
    small = rng.integers(40, 210, size=(h // 8, w // 8, 3), dtype=np.uint8)
    photo = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    photo = cv2.GaussianBlur(photo, (31, 31), 0)
    # Saturated colour casts keep it far from white-paper strip statistics.
    tint = rng.uniform(0.6, 1.4, size=3)
    photo = np.clip(photo.astype(np.float32) * tint[None, None, :], 25, 200).astype(np.uint8)
    return photo


def sample_jpeg(label: str, seed: int = 0) -> bytes:
    """Deterministic JPEG bytes of a synthetic scene (demo-mode asset)."""
    payload = cv2.imencode(
        ".jpg", render_sample(label, seed)[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )[1]
    return payload.tobytes()
