"""Colour primitives: sRGB → CIE Lab, ΔE (CIE76), zone feature extraction.

Pure functions only — no I/O, no globals, immutable in/out.
"""

from __future__ import annotations

import cv2
import numpy as np

RGB = tuple[float, float, float]


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB triple (0-255) to CIE Lab. Deterministic."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab.reshape(3).astype(np.float32)


def delta_e(c1: np.ndarray, c2: np.ndarray) -> float:
    """CIE76 ΔE between two sRGB colours. Symmetric, zero iff identical."""
    lab1 = rgb_to_lab(c1)
    lab2 = rgb_to_lab(c2)
    return float(np.linalg.norm(lab1 - lab2))


def zone_features(
    zones: dict[str, RGB],
    nominal: dict[str, RGB],
) -> dict[str, float]:
    """ΔE of each measured zone against its printed nominal reagent colour."""
    return {
        name: delta_e(
            np.asarray(rgb, np.float32), np.asarray(nominal.get(name, (0, 0, 0)), np.float32)
        )
        for name, rgb in zones.items()
    }
