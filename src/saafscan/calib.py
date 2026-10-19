"""Calibration card: detection, perspective warp, patch extraction, white-balance.

All images in SaafScan are RGB uint8 arrays. Functions are pure — inputs are
never mutated; corrections always return new arrays.
"""

from __future__ import annotations

import cv2
import numpy as np

# Printed reference card: black frame + 2x3 grid of nominal colour patches.
CARD_NOMINAL: dict = {
    "width": 276,
    "height": 184,
    "border": 14,
    "patches": [
        {"name": "white", "rgb": (240, 240, 240)},
        {"name": "grey", "rgb": (128, 128, 128)},
        {"name": "black", "rgb": (25, 25, 25)},
        {"name": "red", "rgb": (200, 40, 40)},
        {"name": "green", "rgb": (40, 170, 60)},
        {"name": "amber", "rgb": (230, 160, 60)},
    ],
}

PatchRef = dict[str, object]


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
        dtype=np.float32,
    )


def _quad_geometry(quad: np.ndarray, img_area: float) -> tuple[float, float, float]:
    """Return (aspect, solidity, area_frac) of a quad in an image."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    long_, short_ = max(w, h), max(min(w, h), 1e-6)
    rect = cv2.minAreaRect(quad.astype(np.float32))
    rect_area = max(rect[1][0] * rect[1][1], 1e-6)
    quad_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    return long_ / short_, quad_area / rect_area, quad_area / max(img_area, 1e-6)


def _warp(image: np.ndarray, quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp ``quad`` onto a ``size`` rectangle.

    Corner order is **honoured, not re-derived**: the quad must already be
    ordered top-left, top-right, bottom-right, bottom-left. detect_card
    deliberately returns a *rotated* order so that a card photographed upside
    down or on its side still maps to a correctly-oriented card image —
    re-ordering by image position here would silently undo that.

    Every quad reaching this function comes from a detector that has already
    called :func:`order_quad` once.
    """
    width, height = size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def _looks_like_card(warped: np.ndarray) -> bool:
    """Validate a candidate: dark frame, lighter interior, brightest=white, darkest=black."""
    h, w = warped.shape[:2]
    ring = max(int(0.08 * h), 3)
    border = np.concatenate(
        [
            warped[:ring].reshape(-1, 3),
            warped[-ring:].reshape(-1, 3),
            warped[:, :ring].reshape(-1, 3),
            warped[:, -ring:].reshape(-1, 3),
        ]
    ).mean(axis=0)
    interior = warped[ring:-ring, ring:-ring].reshape(-1, 3).mean(axis=0)
    if float(border.mean()) > float(interior.mean()) - 20.0:
        return False  # frame must be clearly darker than the patch field
    # White patch must be (near-)brightest and black patch (near-)darkest.
    patches = extract_patches(warped)
    lums = patches @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return bool(lums[0] >= lums.max() - 8.0 and lums[2] <= lums.min() + 8.0)


def _card_orientations(quad: np.ndarray) -> list[np.ndarray]:
    """The same rectangle as seen from each of four viewing angles.

    ``order_quad`` labels corners by their position in the *image*, so a card
    photographed upside down or turned on its side yields the right four points
    in the wrong order. The warp then rotates the patch layout, white lands
    where black should be, and the card is refused. Shifting the corner order by
    one is precisely that rotation, so every candidate is checked in all four.
    """
    return [np.roll(quad, -shift, axis=0) for shift in range(4)]


def detect_card(image: np.ndarray) -> np.ndarray | None:
    """Find the black-framed calibration card. Returns ordered quad or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Adaptive darkness threshold: the frame must stay "dark" under any exposure.
    _, mask = cv2.threshold(gray, max(40.0, float(gray.mean()) - 45.0), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    img_area = float(h_img * w_img)
    candidates = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        quad = order_quad(poly.reshape(4, 2).astype(np.float32))
        aspect, solidity, area_frac = _quad_geometry(quad, img_area)
        if not (1.0 <= aspect <= 2.4 and solidity > 0.85 and 0.015 <= area_frac <= 0.5):
            continue
        candidates.append((area_frac, quad))
    for _, quad in sorted(candidates, key=lambda c: -c[0]):
        for oriented in _card_orientations(quad):
            warped = _warp(image, oriented, (CARD_NOMINAL["width"], CARD_NOMINAL["height"]))
            if _looks_like_card(warped):
                return oriented
    return None


def canonical_card(image: np.ndarray, quad: np.ndarray | None = None) -> np.ndarray:
    """Perspective-corrected card image at nominal size."""
    if quad is None:
        quad = detect_card(image)
        if quad is None:
            raise ValueError("no calibration card detected")
    return _warp(image, quad, (CARD_NOMINAL["width"], CARD_NOMINAL["height"]))


def extract_patches(card: np.ndarray) -> np.ndarray:
    """Mean RGB of each nominal patch cell (row-major, CARD_NOMINAL order)."""
    h, w = card.shape[:2]
    border = CARD_NOMINAL["border"]
    inset = max(int(0.12 * h), border + 2)
    x0, x1 = border + 4, w - border - 4
    y0, y1 = inset, h - inset
    cell_w, cell_h = (x1 - x0) / 3.0, (y1 - y0) / 2.0
    out = np.zeros((len(CARD_NOMINAL["patches"]), 3), dtype=np.float32)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx0 = int(x0 + col * cell_w + 0.2 * cell_w)
            cx1 = int(x0 + (col + 1) * cell_w - 0.2 * cell_w)
            cy0 = int(y0 + row * cell_h + 0.2 * cell_h)
            cy1 = int(y0 + (row + 1) * cell_h - 0.2 * cell_h)
            out[idx] = card[cy0:cy1, cx0:cx1].reshape(-1, 3).mean(axis=0)
            idx += 1
    return out


def _nominal_patch_rgb() -> np.ndarray:
    return np.array([p["rgb"] for p in CARD_NOMINAL["patches"]], dtype=np.float32)


def white_balance(image: np.ndarray, quad: np.ndarray | None = None) -> np.ndarray:
    """Per-channel linear correction anchored on the white/grey/black patches.

    Returns a new array; the input is never mutated. Degenerate inputs (no
    contrast between anchors) are returned unchanged.
    """
    if quad is None:
        quad = detect_card(image)
        if quad is None:
            return image.copy()
    card = _warp(image, quad, (CARD_NOMINAL["width"], CARD_NOMINAL["height"]))
    measured = extract_patches(card)  # white=0, grey=1, black=2
    nominal = _nominal_patch_rgb()
    x = measured[:3]  # (3 anchors, 3 channels)
    y = nominal[:3]
    out = image.astype(np.float32)
    for ch in range(3):
        xs, ys = x[:, ch], y[:, ch]
        span = xs.max() - xs.min()
        if span < 20.0:
            continue  # channel has no usable anchor contrast
        a, b = np.polyfit(xs, ys, 1)
        if not (0.2 <= a <= 5.0):
            continue
        out[:, :, ch] = out[:, :, ch] * a + b
    return np.clip(out, 0, 255).astype(np.uint8)
