"""Shared CNN preprocessing — NumPy only, importable without OpenCV.

The GPU training environment (Python 3.14, torch via a separate interpreter)
has no ``cv2``. This module deliberately imports nothing beyond NumPy so both
the training script and the CPU inference path can share the *exact* same
tensor transformation — any drift between train and serve would silently
corrupt the CNN's held-out accuracy.

The input is the canonical white-balanced strip crop (``STRIP_SIZE`` =
360x90 RGB uint8) produced by ``saafscan.roi.extract_strip``. ResNet18's
adaptive average pool accepts any spatial size, so the crop is fed at native
resolution — no resizing, and therefore no interpolation mismatch between the
training and inference code paths.
"""

from __future__ import annotations

import numpy as np

# ImageNet channel statistics — ResNet18 is finetuned from ImageNet weights, so
# its expected input distribution is these means/stds applied to [0, 1] pixels.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def strip_to_chw(crop: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 strip crop to a normalised CHW float32 tensor.

    Args:
        crop: (height, width, 3) RGB uint8 strip crop, canonical STRIP_SIZE.

    Returns:
        (3, height, width) float32, ImageNet-normalised.

    Raises:
        ValueError: If ``crop`` is not an (H, W, 3) array.
    """
    if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB crop, got shape {getattr(crop, 'shape', None)}"
        )
    x = crop.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
