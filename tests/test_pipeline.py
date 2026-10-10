"""RED tests: analysis pipeline (calib → roi → classifier) and its failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from saafscan.pipeline import PipelineError, analyze
from saafscan.synthetic import render_random_photo, render_sample


@pytest.mark.parametrize("label", ["none", "starch", "detergent", "both"])
def test_analyze_returns_correct_verdict_on_synthetic_scenes(label: str) -> None:
    result = analyze(render_sample(label, seed=7))
    assert result.verdict.adulterant == label
    assert 0.0 <= result.verdict.confidence <= 1.0
    assert result.timing_ms >= 0.0


def test_analyze_attaches_debug_overlay_data() -> None:
    result = analyze(render_sample("starch", seed=7))
    assert result.card_quad.shape == (4, 2)
    assert result.strip_quad.shape == (4, 2)
    assert set(result.zones) == {"control", "starch", "detergent"}


def test_analyze_rejects_random_photo_with_clear_error() -> None:
    """The 'judge uploads a selfie' case must fail loudly, not diagnose."""
    with pytest.raises(PipelineError) as exc:
        analyze(render_random_photo(seed=11))
    message = str(exc.value).lower()
    assert "card" in message or "not appear" in message


def test_analyze_rejects_empty_and_tiny_images() -> None:
    with pytest.raises(PipelineError):
        analyze(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(PipelineError):
        analyze(np.zeros((8, 8, 3), dtype=np.uint8))


def test_analyze_is_deterministic() -> None:
    a = analyze(render_sample("detergent", seed=7))
    b = analyze(render_sample("detergent", seed=7))
    assert a.verdict.adulterant == b.verdict.adulterant
    assert a.verdict.confidence == pytest.approx(b.verdict.confidence)


def test_analyze_does_not_mutate_input() -> None:
    img = render_sample("starch", seed=7)
    before = img.copy()
    analyze(img)
    assert np.array_equal(img, before)


def test_analyze_runs_fast_enough_for_a_live_demo() -> None:
    """Deck promises a verdict well under 10 s — budget 2 s on CPU."""
    import time

    img = render_sample("both", seed=7)
    analyze(img)  # warm-up
    start = time.perf_counter()
    analyze(img)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 2000, f"analysis took {elapsed_ms:.0f}ms"
