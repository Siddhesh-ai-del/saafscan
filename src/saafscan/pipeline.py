"""Analysis pipeline: card → white-balance → strip → zones → verdict.

Orchestrates the pure CV modules. Raises PipelineError with a human-readable
message on any unusable input — never emits a verdict for a bad image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from saafscan.calib import detect_card, white_balance
from saafscan.ml.runtime import classify as ml_classify
from saafscan.roi import detect_strip, warp_strip, zones_from_strip
from saafscan.verdict import Verdict

MIN_SIDE = 40


class PipelineError(ValueError):
    """Input cannot be analysed (no card, no strip, malformed image)."""


@dataclass(frozen=True)
class Analysis:
    verdict: Verdict
    card_quad: np.ndarray
    strip_quad: np.ndarray
    zones: dict[str, tuple[float, float, float]]
    timing_ms: float


def analyze(image: np.ndarray) -> Analysis:
    """Run the full detection pipeline on an RGB uint8 image."""
    started = time.perf_counter()
    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise PipelineError("Expected an RGB image (height x width x 3).")
    if min(image.shape[0], image.shape[1]) < MIN_SIDE:
        raise PipelineError("Image is too small to contain a test strip. Please retake the photo.")

    card_quad = detect_card(image)
    if card_quad is None:
        raise PipelineError(
            "This does not appear to be a SaafScan test image — no colour calibration card was found."
        )

    balanced = white_balance(image, card_quad)
    strip_quad = detect_strip(balanced)
    if strip_quad is None:
        raise PipelineError(
            "Calibration card found but no reagent strip. Place the strip next to the card and retake."
        )

    # Canonical white-balanced crop: the CNN's input, and the exact pixels the
    # zone means below are measured from — model and rules see the same paper.
    # (Same chain extract_zones() runs, without re-running card/strip detection.)
    crop = warp_strip(balanced, strip_quad)
    zones = zones_from_strip(crop)
    # ML-primary wrapper: trained model if available, else the rule-table.
    # Never fabricates a verdict — bad/ambiguous input falls back or abstains.
    verdict = ml_classify(zones, strip=crop)
    timing_ms = (time.perf_counter() - started) * 1000.0
    return Analysis(
        verdict=verdict,
        card_quad=card_quad,
        strip_quad=strip_quad,
        zones=zones,
        timing_ms=timing_ms,
    )


# Re-export for convenience (tests import from pipeline directly).
__all__ = ["Analysis", "PipelineError", "analyze", "warp_strip"]
