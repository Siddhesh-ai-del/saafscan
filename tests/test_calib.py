"""RED tests: calibration card detection, warp, and white-balance correction."""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.calib import (
    CARD_NOMINAL,
    canonical_card,
    detect_card,
    extract_patches,
    white_balance,
)
from saafscan.synthetic import render_random_photo, render_sample
from tests.conftest import assert_is_image


def test_detect_card_finds_quad_on_synthetic_scene() -> None:
    quad = detect_card(render_sample("none", seed=7))
    assert quad is not None
    assert quad.shape == (4, 2)
    assert quad.dtype == np.float32
    # Corners must lie inside the image.
    assert quad[:, 0].min() >= 0 and quad[:, 1].min() >= 0


def test_detect_card_orders_corners_clockwise_from_top_left() -> None:
    quad = detect_card(render_sample("starch", seed=7))
    assert quad is not None
    # Sum of x+y is minimal at top-left, maximal at bottom-right.
    sums = quad.sum(axis=1)
    assert sums.argmin() == 0
    assert sums.argmax() == 2
    # Difference y-x is minimal at top-right, maximal at bottom-left.
    diffs = quad[:, 1] - quad[:, 0]
    assert diffs.argmin() == 1
    assert diffs.argmax() == 3


def test_detect_card_returns_none_without_a_card() -> None:
    assert detect_card(render_random_photo(seed=11)) is None


def test_detect_card_survives_brighter_and_darker_exposure() -> None:
    base = render_sample("none", seed=7)
    for gain in (0.7, 1.15):
        adjusted = np.clip(base.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        assert detect_card(adjusted) is not None, f"card lost at gain={gain}"


def test_canonical_card_is_a_fixed_size_rgb_image() -> None:
    card = canonical_card(render_sample("none", seed=7))
    assert_is_image(card, min_side=64)
    assert card.shape[:2] == (CARD_NOMINAL["height"], CARD_NOMINAL["width"])


def test_extract_patches_returns_nominal_colour_references() -> None:
    card = canonical_card(render_sample("none", seed=7))
    patches = extract_patches(card)
    assert patches.shape == (len(CARD_NOMINAL["patches"]), 3)
    # Every patch must be reasonably close to its printed nominal value.
    nominal = np.array([p["rgb"] for p in CARD_NOMINAL["patches"]], dtype=np.float32)
    errors = np.abs(patches - nominal).mean(axis=1)
    assert errors.max() < 18.0, f"patch colour error too high: {errors}"


def test_white_balance_restores_a_darkened_scene() -> None:
    """A 0.7x exposure must be pushed back near nominal after white-balance.

    This is the 'venue lighting' defence: the printed white/grey/black patches
    anchor the per-channel gain so verdicts do not depend on the room.
    """
    base = render_sample("none", seed=7)
    dark = np.clip(base.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)

    quad = detect_card(dark)
    assert quad is not None
    corrected = white_balance(dark, quad)

    patches = extract_patches(canonical_card(corrected))
    nominal = np.array([p["rgb"] for p in CARD_NOMINAL["patches"]], dtype=np.float32)
    errors = np.abs(patches - nominal).mean(axis=1)
    assert errors.max() < 18.0, f"white-balance failed, errors: {errors}"


def test_white_balance_is_idempotent_enough_on_already_lit_scene() -> None:
    base = render_sample("none", seed=7)
    quad = detect_card(base)
    assert quad is not None
    once = white_balance(base, quad)
    twice = white_balance(once, detect_card(once))
    drift = np.abs(once.astype(np.float32) - twice.astype(np.float32)).mean()
    assert drift < 6.0, f"double white-balance drifted by {drift:.2f}"


def test_white_balance_does_not_mutate_input() -> None:
    base = render_sample("none", seed=7)
    before = base.copy()
    quad = detect_card(base)
    assert quad is not None
    white_balance(base, quad)
    assert np.array_equal(base, before), "input image was mutated"


@pytest.mark.parametrize("gain", [0.55, 0.7, 0.85, 1.0, 1.15, 1.3])
def test_white_balance_stabilises_white_patch_across_exposures(gain: float) -> None:
    base = render_sample("none", seed=7)
    adjusted = np.clip(base.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    quad = detect_card(adjusted)
    assert quad is not None, f"card detection failed at gain={gain}"
    corrected = white_balance(adjusted, quad)
    patches = extract_patches(canonical_card(corrected))
    nominal = np.array([p["rgb"] for p in CARD_NOMINAL["patches"]], dtype=np.float32)
    errors = np.abs(patches - nominal).mean(axis=1)
    assert errors.max() < 20.0, f"gain={gain}: errors {errors}"
