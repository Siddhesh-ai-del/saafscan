"""Strip detection and reaction-zone colour extraction."""

from __future__ import annotations

import cv2
import numpy as np

from saafscan.calib import _quad_geometry, _warp, detect_card, order_quad, white_balance

# Nominal (unreacted) reagent colours, RGB, matching the synthetic renderer.
ZONE_NOMINAL: dict[str, tuple[int, int, int]] = {
    "control": (205, 65, 45),
    "starch": (225, 185, 105),
    "detergent": (70, 95, 200),
}

STRIP_SIZE = (360, 90)  # canonical (width, height)
_ZONE_SPAN = 0.55  # fraction of each band sampled (inner region)

# Long/short aspect accepted by detect_strip. Anything outside this window is
# never considered a strip: below it a bright rectangle looks too square to be
# one, above it the object is too narrow to carry three readable zones.
STRIP_ASPECT_RANGE = (2.5, 9.0)

# Cut size for real paper strips, (width, height) in mm. Recorded here, next to
# the gate it has to satisfy, because the previous spec of 8x80 mm was aspect
# 10 — outside STRIP_ASPECT_RANGE — so every strip cut to the documented size
# was rejected by the detector. Long/short must also track STRIP_SIZE's 4:1,
# since that is the proportion zones_from_strip splits into three zones.
STRIP_PHYSICAL_MM = (20.0, 80.0)


def detect_strip(image: np.ndarray) -> np.ndarray | None:
    """Find the elongated bright reagent strip. Returns ordered quad or None.

    The strip is white paper: bright relative to the scene, low saturation,
    rectangular, and elongated. Colourful or textured photo regions that are
    merely bright are rejected by the saturation/texture/rectness gates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    thr = max(110, float(gray.mean()) + 20.0)
    bright = (gray > thr).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    img_area = float(h_img * w_img)
    whiteness = ((gray > thr) & (sat < 60)).astype(np.uint8)  # bright + unsaturated = paper
    best = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        quad = order_quad(poly.reshape(4, 2).astype(np.float32))
        aspect, solidity, area_frac = _quad_geometry(quad, img_area)
        lo, hi = STRIP_ASPECT_RANGE
        if not (lo <= aspect <= hi and solidity > 0.8 and 0.015 <= area_frac <= 0.2):
            continue
        rect_mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(rect_mask, [quad.astype(np.int32)], 255)
        inside = rect_mask > 0
        # A reagent strip is mostly white paper; the coloured reaction zones are
        # a minority of its area. Colourful photo regions fail this gate.
        paper_frac = float(whiteness[inside].mean())
        if paper_frac < 0.45:
            continue
        interior_mean = float(gray[inside].mean())
        ring = cv2.dilate(rect_mask, np.ones((15, 15), np.uint8)) & ~rect_mask
        ring_mean = float(gray[ring > 0].mean()) if (ring > 0).any() else 0.0
        if interior_mean < ring_mean + 25.0:
            continue
        area = area_frac * img_area
        if best is None or area > best[0]:
            best = (area, quad)
    return best[1] if best else None


def warp_strip(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-corrected strip at canonical size."""
    return _warp(image, quad, STRIP_SIZE)


def zones_from_strip(strip: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Mean RGB of the three reaction zones in a canonical strip image."""
    width = strip.shape[1]
    band = width / 3.0
    names = ("control", "starch", "detergent")
    out: dict[str, tuple[float, float, float]] = {}
    h = strip.shape[0]
    y0, y1 = int(0.3 * h), int(0.7 * h)
    for i, name in enumerate(names):
        cx0 = int(i * band + (1 - _ZONE_SPAN) / 2 * band)
        cx1 = int((i + 1) * band - (1 - _ZONE_SPAN) / 2 * band)
        out[name] = tuple(float(c) for c in strip[y0:y1, cx0:cx1].reshape(-1, 3).mean(axis=0))
    return out


def extract_zones(image: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Full chain: white-balance → strip detection → zone colours.

    Raises ValueError when no strip (or card) can be found.
    """
    quad = detect_card(image)
    balanced = white_balance(image, quad) if quad is not None else image
    strip_quad = detect_strip(balanced)
    if strip_quad is None:
        raise ValueError("no reagent strip detected")
    return zones_from_strip(warp_strip(balanced, strip_quad))


def extract_strip(image: np.ndarray) -> np.ndarray:
    """Full chain: white-balance → strip detection → canonical strip crop.

    Returns the perspective-warped, white-balanced strip at STRIP_SIZE — the
    exact tensor the classifier consumes. Raises ValueError when no strip (or
    card) can be found.
    """
    quad = detect_card(image)
    balanced = white_balance(image, quad) if quad is not None else image
    strip_quad = detect_strip(balanced)
    if strip_quad is None:
        raise ValueError("no reagent strip detected")
    return warp_strip(balanced, strip_quad)
