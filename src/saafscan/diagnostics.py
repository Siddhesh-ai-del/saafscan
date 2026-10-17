"""Per-stage pipeline diagnostics for real photographs.

`analyze()` stops at the first failure with a one-line message. That is right
for the app: a judge gets one clear sentence, never a fabricated verdict. But it
is useless when a real strip photo is rejected — it says the card (or the
strip) was missing without saying how close detection came, so the fix is
guesswork.

This module runs every stage it can, records what each stage actually saw, and
annotates the photograph with the quads it found. It is a diagnostic tool for
the team, not part of the request path.

Honest by construction: no stage invents a measurement. A stage that could not
run reports no metrics and is not recorded as having passed, and the overlay
paints only what was genuinely detected.

    python -m saafscan.diagnostics photo.jpg --out debug/
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np

from saafscan.calib import (
    CARD_NOMINAL,
    _quad_geometry,
    canonical_card,
    detect_card,
    extract_patches,
)
from saafscan.calib import white_balance as _white_balance
from saafscan.color import delta_e, zone_features
from saafscan.ml.runtime import classify as ml_classify
from saafscan.pipeline import MIN_SIDE
from saafscan.roi import ZONE_NOMINAL, detect_strip, warp_strip, zones_from_strip
from saafscan.verdict import Verdict

STAGES: tuple[str, ...] = ("image", "card", "balance", "strip", "zones", "verdict")

# Annotation colours, RGB like the rest of the codebase (converted on draw).
_CARD_COLOR = (230, 160, 60)
_STRIP_COLOR = (40, 170, 60)
_TEXT_COLOR = (240, 240, 240)
_PANEL_COLOR = (31, 58, 48)


@dataclass(frozen=True)
class Stage:
    """One pipeline stage and what it measured. Metrics are frozen."""

    name: str
    ok: bool
    detail: str
    metrics: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class Report:
    """Outcome of running the pipeline stage by stage.

    `stages` holds only the stages that actually ran, in order, ending at the
    first failure. Later stages are absent rather than guessed at.
    """

    stages: tuple[Stage, ...]
    card_quad: np.ndarray | None
    strip_quad: np.ndarray | None
    zones: Mapping[str, tuple[float, float, float]] | None
    verdict: Verdict | None
    error: str | None

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.stages) and all(s.ok for s in self.stages)

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)


def _freeze(metrics: dict[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(sorted(metrics.items())))


def inspect_image(image: np.ndarray) -> Report:
    """Run the detection pipeline stage by stage and record what each saw."""
    card_quad: np.ndarray | None = None
    strip_quad: np.ndarray | None = None
    zones: Mapping[str, tuple[float, float, float]] | None = None
    verdict: Verdict | None = None

    # -- image ---------------------------------------------------------------
    shape = image.shape if isinstance(image, np.ndarray) else None
    rgb = shape is not None and len(shape) == 3 and shape[2] == 3
    big_enough = rgb and min(shape[0], shape[1]) >= MIN_SIDE
    if not rgb:
        detail, metrics, error = (
            "not an RGB image (height x width x 3)",
            {},
            "expected an RGB image",
        )
    elif not big_enough:
        detail = f"only {shape[1]}x{shape[0]} px; at least {MIN_SIDE} px on each side is needed"
        metrics, error = {"height": float(shape[0]), "width": float(shape[1])}, detail
    else:
        detail = f"{shape[1]}x{shape[0]} px RGB"
        metrics, error = {"height": float(shape[0]), "width": float(shape[1])}, None
    stages = [Stage("image", big_enough, detail, _freeze(metrics))]
    if not big_enough:
        return Report(tuple(stages), None, None, None, None, error)

    # -- card ----------------------------------------------------------------
    card_quad = detect_card(image)
    if card_quad is None:
        stages.append(
            Stage(
                "card",
                False,
                "no quad passed the card geometry gates (area, aspect, patch contrast)",
                MappingProxyType({}),
            )
        )
        return Report(tuple(stages), None, None, None, None, "no colour calibration card was found")
    patches = extract_patches(canonical_card(image, card_quad))
    nominal = {p["name"]: p["rgb"] for p in CARD_NOMINAL["patches"]}
    card_metrics = {
        "quad_area_frac": float(_quad_geometry(card_quad, shape[0] * shape[1])[2]),
        **{
            f"{name}_de": delta_e(patches[i], np.asarray(nominal[name], np.float32))
            for i, name in enumerate(nominal)
        },
    }
    stages.append(Stage("card", True, "calibration card detected", _freeze(card_metrics)))

    # -- balance -------------------------------------------------------------
    balanced = _white_balance(image, card_quad)
    shift = balanced.reshape(-1, 3).mean(axis=0) - image.reshape(-1, 3).mean(axis=0)
    stages.append(
        Stage(
            "balance",
            True,
            "per-channel correction anchored on the printed white/grey/black patches",
            _freeze({f"mean_shift_{c}": float(shift[i]) for i, c in enumerate("rgb")}),
        )
    )

    # -- strip ---------------------------------------------------------------
    strip_quad = detect_strip(balanced)
    if strip_quad is None:
        stages.append(
            Stage(
                "strip",
                False,
                "no elongated bright quad passed the strip gates "
                "(aspect, solidity, paper fraction, contrast over background)",
                MappingProxyType({}),
            )
        )
        return Report(
            tuple(stages),
            card_quad,
            None,
            None,
            None,
            "calibration card found but no reagent strip was detected",
        )
    img_area = float(shape[0] * shape[1])
    stages.append(
        Stage(
            "strip",
            True,
            "reagent strip detected",
            _freeze({"quad_area_frac": float(_quad_geometry(strip_quad, img_area)[2])}),
        )
    )

    # -- zones ---------------------------------------------------------------
    crop = warp_strip(balanced, strip_quad)
    measured = zones_from_strip(crop)
    zones = MappingProxyType(dict(measured))
    deltas = zone_features(measured, ZONE_NOMINAL)
    zone_metrics = {
        f"{name}_rgb_{axis}": float(measured[name][i])
        for name in measured
        for i, axis in enumerate("rgb")
    }
    zone_metrics.update({f"{name}_de": float(deltas[name]) for name in deltas})
    stages.append(Stage("zones", True, "reaction-zone colours measured", _freeze(zone_metrics)))

    # -- verdict -------------------------------------------------------------
    verdict = ml_classify(measured, strip=crop)
    stages.append(
        Stage(
            "verdict",
            True,
            f"{verdict.adulterant} at confidence {verdict.confidence:.2f}",
            _freeze({"confidence": float(verdict.confidence)}),
        )
    )
    return Report(tuple(stages), card_quad, strip_quad, zones, verdict, None)


# --------------------------------------------------------------------------
# annotated overlay
# --------------------------------------------------------------------------
def _draw_quad(canvas: np.ndarray, quad: np.ndarray, rgb: tuple[int, int, int], label: str) -> None:
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(canvas, [pts], True, rgb, 3, cv2.LINE_AA)
    x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
    _put(canvas, label, x, max(y - 8, 12), rgb)


def _put(canvas: np.ndarray, text: str, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    scale = max(canvas.shape[1] / 1600.0, 0.45)
    thickness = max(int(round(scale * 2)), 1)
    cv2.putText(
        canvas,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        rgb,
        thickness,
        cv2.LINE_AA,
    )


def overlay(image: np.ndarray, report: Report) -> np.ndarray:
    """Copy of the photo with detected quads and the measured numbers painted on.

    Draws only what was genuinely detected; a stage that found nothing is
    reported in the text panel, never as a shape on the image.
    """
    canvas = np.ascontiguousarray(image.copy())
    if report.card_quad is not None:
        _draw_quad(canvas, report.card_quad, _CARD_COLOR, "card")
    if report.strip_quad is not None:
        _draw_quad(canvas, report.strip_quad, _STRIP_COLOR, "strip")

    lines = [f"{s.name}: {'ok' if s.ok else 'FAIL'} - {s.detail}" for s in report.stages]
    if report.zones is not None:
        lines += [
            f"{name} RGB {tuple(round(v) for v in rgb)} dE {report.stage('zones').metrics[f'{name}_de']:.1f}"
            for name, rgb in report.zones.items()
        ]
    if report.error is not None:
        lines.append(f"ERROR: {report.error}")

    scale = max(canvas.shape[1] / 1600.0, 0.45)
    line_h = int(round(20 * scale))
    panel_h = line_h * len(lines) + 12
    cv2.rectangle(
        canvas,
        (0, canvas.shape[0] - panel_h),
        (canvas.shape[1], canvas.shape[0]),
        _PANEL_COLOR,
        -1,
    )
    for i, line in enumerate(lines):
        _put(canvas, line, 8, canvas.shape[0] - panel_h + line_h * (i + 1) + 4, _TEXT_COLOR)
    return canvas


# --------------------------------------------------------------------------
def format_report(report: Report) -> str:
    """Plain-text summary: one block per stage that ran, then the ones that did not."""
    lines = []
    for stage in report.stages:
        lines.append(f"{'ok  ' if stage.ok else 'FAIL'}  {stage.name:<8} {stage.detail}")
        for key, value in stage.metrics.items():
            lines.append(f"        {key:<18} {value:>9.2f}")
    ran = {s.name for s in report.stages}
    lines += [
        f"skip  {name:<8} not run (a previous stage failed)" for name in STAGES if name not in ran
    ]
    lines.append(
        f"\n{'PASSED' if report.passed else 'REJECTED'}: {report.error or 'analysis complete'}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.diagnostics",
        description="Explain, stage by stage, what the pipeline saw in a photograph.",
    )
    parser.add_argument("image", type=Path, help="photo to inspect (jpg/png)")
    parser.add_argument(
        "--out", type=Path, default=None, help="directory for the annotated overlay"
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"no such file: {args.image}", file=sys.stderr)
        return 2
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not decode {args.image} as an image", file=sys.stderr)
        return 2

    report = inspect_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(format_report(report))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / f"{args.image.stem}_overlay.png"
        cv2.imwrite(
            str(target),
            cv2.cvtColor(overlay(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), report), cv2.COLOR_RGB2BGR),
        )
        print(f"\noverlay written to {target}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
