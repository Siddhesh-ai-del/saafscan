"""RED tests: synthetic demo images served for rehearsal / Round-1 screenshots."""

from __future__ import annotations

import pytest

from saafscan.synthetic import ADULTERANTS, render_sample, sample_jpeg


@pytest.mark.parametrize("label", sorted({"none", "starch", "detergent", "both"}))
def test_sample_jpeg_is_valid_jpeg_bytes(label: str) -> None:
    payload = sample_jpeg(label, seed=7)
    assert isinstance(payload, bytes)
    assert payload[:2] == b"\xff\xd8", "not a JPEG SOI marker"
    assert payload[-2:] == b"\xff\xd9", "not a JPEG EOI marker"


def test_sample_jpeg_is_deterministic() -> None:
    assert sample_jpeg("starch", seed=7) == sample_jpeg("starch", seed=7)


def test_sample_jpeg_decodes_back_to_the_rendered_scene() -> None:
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(sample_jpeg("none", seed=7), np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # JPEG is lossy — allow modest mean error against the RGB-rendered scene.
    original_rgb = render_sample("none", seed=7)
    error = np.abs(decoded[:, :, ::-1].astype(int) - original_rgb.astype(int)).mean()
    assert error < 6.0, f"roundtrip error too high: {error}"


def test_all_labels_available_for_demo_mode() -> None:
    for label in ADULTERANTS:
        assert sample_jpeg(label, seed=1)
