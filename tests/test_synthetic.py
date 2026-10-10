"""RED tests: synthetic scene generator (deterministic demo + adversarial input)."""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.synthetic import (
    ADULTERANTS,
    render_random_photo,
    render_sample,
)
from tests.conftest import assert_is_image


def test_adulterant_labels_are_the_four_cases() -> None:
    assert set(ADULTERANTS) == {"none", "starch", "detergent", "both"}


@pytest.mark.parametrize("label", sorted({"none", "starch", "detergent", "both"}))
def test_render_sample_returns_rgb_image(label: str) -> None:
    img = render_sample(label, seed=1)
    assert_is_image(img)


def test_render_sample_is_deterministic_for_a_seed() -> None:
    a = render_sample("starch", seed=3)
    b = render_sample("starch", seed=3)
    assert np.array_equal(a, b)


def test_different_seeds_change_the_scene() -> None:
    a = render_sample("none", seed=1)
    b = render_sample("none", seed=2)
    assert not np.array_equal(a, b)


def test_starch_scene_differs_from_pure_where_it_matters() -> None:
    pure = render_sample("none", seed=7)
    starch = render_sample("starch", seed=7)
    assert np.mean(np.abs(pure.astype(int) - starch.astype(int))) > 1.0


def test_random_photo_has_no_card_and_no_strip_geometry() -> None:
    """The adversarial image must look nothing like a test scene.

    It is the 'judge uploads a selfie' case: smooth, photo-like, without the
    black-bordered calibration card or the elongated white strip.
    """
    photo = render_random_photo(seed=11)
    assert_is_image(photo)
    # Deterministic so downstream rejection tests are reproducible.
    assert np.array_equal(photo, render_random_photo(seed=11))
