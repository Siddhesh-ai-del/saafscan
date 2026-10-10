"""Shared fixtures: synthetic SaafScan scenes (colour card + reagent strip).

All synthetic scenes are RGB uint8 images, as produced by
``saafscan.synthetic.render_sample``.
"""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.synthetic import ADULTERANTS, render_random_photo, render_sample


@pytest.fixture(scope="session")
def scenes() -> dict[str, np.ndarray]:
    """One deterministic scene per adulterant label."""
    return {label: render_sample(label, seed=7) for label in ADULTERANTS}


@pytest.fixture(scope="session")
def pure_scene() -> np.ndarray:
    return render_sample("none", seed=7)


@pytest.fixture(scope="session")
def starch_scene() -> np.ndarray:
    return render_sample("starch", seed=7)


@pytest.fixture(scope="session")
def detergent_scene() -> np.ndarray:
    return render_sample("detergent", seed=7)


@pytest.fixture(scope="session")
def both_scene() -> np.ndarray:
    return render_sample("both", seed=7)


@pytest.fixture(scope="session")
def photo_scene() -> np.ndarray:
    """A photo-like image with no colour card and no strip (adversarial)."""
    return render_random_photo(seed=11)


def assert_is_image(img: np.ndarray, *, min_side: int = 200) -> None:
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3, f"expected HxWx3, got {img.shape}"
    assert img.dtype == np.uint8
    assert min(img.shape[0], img.shape[1]) >= min_side
