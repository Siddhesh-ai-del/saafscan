"""RED tests: per-stage pipeline diagnostics for real photographs.

The app's analyze() stops at the first failure with a one-line message. That is
correct behaviour for users, but when a real strip photo is rejected on 24 Oct
it says only *that* the card or strip was missing — not how close detection
came. These tests pin a diagnostic report that records what each stage saw.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from saafscan.diagnostics import (
    STAGES,
    Report,
    _draw_quad,
    format_report,
    inspect_image,
    main,
    overlay,
)
from saafscan.roi import detect_strip
from saafscan.synthetic import render_random_photo, render_sample


def test_all_stages_pass_on_a_known_good_synthetic_scene() -> None:
    report = inspect_image(render_sample("starch", seed=7))
    assert [s.name for s in report.stages] == [
        "image",
        "card",
        "balance",
        "strip",
        "zones",
        "verdict",
    ]
    assert all(s.ok for s in report.stages)
    assert report.passed
    assert report.error is None
    assert report.verdict is not None
    assert report.verdict.adulterant == "starch"


def test_a_rejected_photo_names_the_stage_that_failed() -> None:
    """The 'judge uploads a selfie' case: only the stages that ran are recorded."""
    report = inspect_image(render_random_photo(seed=11))
    assert not report.passed
    assert [s.name for s in report.stages] == ["image", "card"]
    assert report.stages[-1].ok is False
    assert "card" in report.error.lower()


def test_a_rejected_photo_never_carries_a_verdict() -> None:
    report = inspect_image(render_random_photo(seed=11))
    assert report.verdict is None
    assert report.zones is None


def test_zones_stage_reports_measured_rgb_and_delta_e() -> None:
    report = inspect_image(render_sample("detergent", seed=7))
    zones = next(s for s in report.stages if s.name == "zones")
    assert set(zones.metrics) == {
        *(f"{name}_rgb_{axis}" for name in ("control", "starch", "detergent") for axis in "rgb"),
        "control_de",
        "starch_de",
        "detergent_de",
    }
    assert all(v >= 0.0 for v in zones.metrics.values())


def test_card_stage_reports_geometry_and_patch_accuracy() -> None:
    report = inspect_image(render_sample("none", seed=7))
    card = next(s for s in report.stages if s.name == "card")
    assert set(card.metrics) == {
        "quad_area_frac",
        "white_de",
        "grey_de",
        "black_de",
        "red_de",
        "green_de",
        "amber_de",
    }
    # The printed card's nominal patches must survive the render closely.
    assert card.metrics["white_de"] < 20.0


def test_balance_stage_quantifies_the_correction_applied() -> None:
    report = inspect_image(render_sample("starch", seed=7))
    balance = next(s for s in report.stages if s.name == "balance")
    assert {"mean_shift_r", "mean_shift_g", "mean_shift_b"} <= set(balance.metrics)


def test_overlay_keeps_the_input_geometry_and_marks_what_it_found() -> None:
    image = render_sample("starch", seed=7)
    report = inspect_image(image)
    drawn = overlay(image, report)
    assert drawn.shape == image.shape
    assert not np.array_equal(drawn, image)  # quads and labels were painted


def test_overlay_survives_a_photo_where_nothing_was_detected() -> None:
    image = render_random_photo(seed=11)
    report = inspect_image(image)
    drawn = overlay(image, report)
    assert drawn.shape == image.shape
    assert drawn.dtype == np.uint8


def test_annotations_are_drawn_as_rgb_and_not_reversed() -> None:
    """Regression: a channel reversal painted the amber card outline blue.

    The overlay canvas is RGB, so a colour must be written through unchanged.
    A black canvas makes the drawn pixels unambiguous, unlike a real scene
    where the synthetic card already contains an amber patch.
    """
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    quad = np.array([[5, 5], [35, 5], [35, 35], [5, 35]], dtype=np.float32)
    _draw_quad(canvas, quad, (200, 10, 5), "quad")

    def has_colour(frame: np.ndarray, rgb: tuple[int, int, int]) -> bool:
        """Any pixel equal to the triple — not any single channel equal."""
        return bool((frame == np.array(rgb, dtype=np.uint8)).all(axis=-1).any())

    assert has_colour(canvas, (200, 10, 5))
    assert not has_colour(canvas, (5, 10, 200))


def test_report_is_immutable() -> None:
    report = inspect_image(render_sample("none", seed=7))
    assert isinstance(report, Report)
    with pytest.raises(AttributeError):
        report.error = "nope"  # type: ignore[misc]


def test_format_report_is_a_readable_multi_line_summary() -> None:
    text = format_report(inspect_image(render_sample("both", seed=7)))
    assert text.count("\n") >= 5
    for stage in ("image", "card", "balance", "strip", "zones", "verdict"):
        assert stage in text


def test_format_report_states_the_failure_in_words() -> None:
    text = format_report(inspect_image(render_random_photo(seed=11)))
    assert "card" in text.lower()
    assert "not run" in text.lower()


def test_inspect_image_does_not_mutate_its_input() -> None:
    image = render_sample("starch", seed=7)
    before = image.copy()
    inspect_image(image)
    assert np.array_equal(image, before)


# --------------------------------------------------------------------------
# failure paths: what a rejected real photo must still tell the team
# --------------------------------------------------------------------------
def test_image_stage_rejects_a_non_rgb_image() -> None:
    report = inspect_image(np.zeros((64, 64), dtype=np.uint8))
    assert [s.name for s in report.stages] == ["image"]
    assert report.stages[0].ok is False
    assert "rgb" in report.error.lower()


def test_image_stage_rejects_an_image_below_the_minimum_size() -> None:
    report = inspect_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert [s.name for s in report.stages] == ["image"]
    assert "at least" in report.stages[0].detail
    assert report.verdict is None


def test_image_stage_rejects_a_value_that_is_not_an_image() -> None:
    report = inspect_image(None)  # type: ignore[arg-type]
    assert not report.passed
    assert [s.name for s in report.stages] == ["image"]


def test_strip_failure_is_reported_while_the_card_still_counts() -> None:
    """Card in frame, strip missing or out of shot — the 24 Oct misfire case."""
    image = render_sample("starch", seed=7)
    quad = detect_strip(image)
    assert quad is not None  # baseline: the scene does contain a strip

    # Remove the strip the way a real misfire would: out of frame, or not laid
    # down. The card must still be reported as found, and nothing downstream.
    masked = image.copy()
    cv2.fillPoly(masked, [np.round(quad).astype(np.int32)], (120, 120, 120))

    report = inspect_image(masked)
    assert [s.name for s in report.stages] == ["image", "card", "balance", "strip"]
    assert report.stage("card").ok is True
    assert report.stage("strip").ok is False
    assert report.strip_quad is None  # nothing detected is recorded as nothing
    assert report.zones is None
    assert report.verdict is None
    assert report.error is not None and "strip" in report.error.lower()


# --------------------------------------------------------------------------
# command line: this is what actually gets run on 24 Oct
# --------------------------------------------------------------------------
def test_cli_exits_zero_and_prints_every_stage(tmp_path: Path, capsys) -> None:
    photo = tmp_path / "good.png"
    cv2.imwrite(str(photo), cv2.cvtColor(render_sample("starch", seed=7), cv2.COLOR_RGB2BGR))
    assert main([str(photo)]) == 0
    output = capsys.readouterr().out
    for stage in STAGES:
        assert stage in output
    assert "PASSED" in output


def test_cli_writes_an_overlay_and_exits_non_zero_when_rejected(tmp_path: Path) -> None:
    photo = tmp_path / "selfie.png"
    cv2.imwrite(str(photo), cv2.cvtColor(render_random_photo(seed=11), cv2.COLOR_RGB2BGR))
    out = tmp_path / "debug"
    assert main([str(photo), "--out", str(out)]) == 1
    assert (out / "selfie_overlay.png").is_file()


def test_cli_reports_a_missing_or_undecodable_file(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path / "nope.jpg")]) == 2
    assert "no such file" in capsys.readouterr().err

    garbage = tmp_path / "junk.png"
    garbage.write_bytes(b"not an image at all")
    assert main([str(garbage)]) == 2
    assert "could not decode" in capsys.readouterr().err
