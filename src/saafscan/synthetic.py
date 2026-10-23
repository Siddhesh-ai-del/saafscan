"""Deterministic synthetic SaafScan scenes for demo mode and tests.

Renders an RGB scene containing the printed calibration card and the reagent
strip under mild lighting variation, plus an adversarial photo-like image with
neither (the 'judge uploads a selfie' case).

⚠️ Synthetic scenes exist for Round-1 screenshots and dry-run rehearsal only.
The 24 Oct final demo must run on real reagent strips (see README).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from saafscan.calib import CARD_NOMINAL
from saafscan.roi import ZONE_NOMINAL

SampleLabel = Literal["none", "starch", "detergent", "both"]
ADULTERANTS: tuple[str, ...] = ("none", "starch", "detergent", "both")

# ML training classes: the four reactions plus a learned 'abstain' class so the
# classifier can express uncertainty instead of forcing a guess.
TRAINING_CLASSES: tuple[str, ...] = ("none", "starch", "detergent", "both", "inconclusive")

SCENE_SIZE = (640, 480)  # (width, height)

# Hard cap on total rotation applied to a training scene. The strip detector
# tolerates roughly ±4°; beyond that a correctly-rendered strip is not found,
# which would corrupt the training set with undetectable samples.
MAX_SCENE_SKEW = 3.5

# The reagent strip is pasted into a training scene at this multiple of its
# canonical STRIP_SIZE. Reacted (esp. starch) zones darken part of the strip's
# area, so a little headroom keeps the bright-paper area fraction above the
# detector's 0.015 floor without pushing it near the 0.2 ceiling.
# 1.15 centres the strip's area-fraction (~0.12) inside the detector's
# [0.015, 0.2] window across all 36 lighting conditions: at scale 1.0 a few
# dark starch strips fell under the 0.015 floor, while at 1.35 bright strips
# hit the 0.2 ceiling. Measured 0/540 geometry rejections at 1.15.
TRAINING_STRIP_SCALE = 1.15

# Reaction outcomes: how each reagent zone looks after the reaction.
REACTION_COLOURS = {
    "starch": {"starch": (45, 40, 95)},  # blue-black starch-iodine complex
    "detergent": {"detergent": (214, 216, 220)},  # methylene blue fully reduced (near-colourless)
    "both": {"starch": (45, 40, 95), "detergent": (214, 216, 220)},
    "none": {},
}


def _card_canvas(rng: np.random.Generator) -> np.ndarray:
    """Render the calibration card exactly to nominal spec, no lighting."""
    w, h, border = CARD_NOMINAL["width"], CARD_NOMINAL["height"], CARD_NOMINAL["border"]
    card = np.full((h, w, 3), 18, dtype=np.uint8)  # black frame
    x0, y0 = border, border
    cell_w = (w - 2 * border) / 3.0
    cell_h = (h - 2 * border) / 2.0
    for i, patch in enumerate(CARD_NOMINAL["patches"]):
        row, col = divmod(i, 3)
        cx0 = int(x0 + col * cell_w + 3)
        cx1 = int(x0 + (col + 1) * cell_w - 3)
        cy0 = int(y0 + row * cell_h + 3)
        cy1 = int(y0 + (row + 1) * cell_h - 3)
        card[cy0:cy1, cx0:cx1] = np.array(patch["rgb"], dtype=np.uint8)
    return card


def _strip_canvas(label: str) -> np.ndarray:
    """Render the strip with nominal or reacted zone colours, no lighting."""
    from saafscan.roi import STRIP_SIZE

    w, h = STRIP_SIZE
    strip = np.full((h, w, 3), 238, dtype=np.uint8)  # white paper
    band = w / 3.0
    reacted = REACTION_COLOURS[label]
    for i, name in enumerate(("control", "starch", "detergent")):
        colour = reacted.get(name, ZONE_NOMINAL[name])
        cx0 = int(i * band + 0.15 * band)
        cx1 = int((i + 1) * band - 0.15 * band)
        strip[int(0.25 * h) : int(0.75 * h), cx0:cx1] = np.array(colour, dtype=np.uint8)
    return strip


def _paste(scene: np.ndarray, canvas: np.ndarray, x: int, y: int, angle: float) -> None:
    """Rotate `canvas` by `angle` degrees and paste centered at (x, y)."""
    h, w = canvas.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    hh, ww = scene.shape[:2]
    x0, y0 = max(x - w // 2, 0), max(y - h // 2, 0)
    x1, y1 = min(x + w // 2, ww), min(y + h // 2, hh)
    sx0, sy0 = x0 - (x - w // 2), y0 - (y - h // 2)
    region = rotated[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    scene[y0:y1, x0:x1] = region


def render_sample(label: str, seed: int = 0) -> np.ndarray:
    """Render a complete synthetic scene for an adulterant label (RGB uint8).

    Deterministic per (label, seed). Applies mild per-channel lighting and a
    soft vignette so detection/white-balance are exercised, not assumed.
    """
    if label not in ADULTERANTS:
        raise ValueError(f"unknown sample label: {label!r}")
    rng = np.random.default_rng(seed)
    w, h = SCENE_SIZE
    base = np.full((h, w, 3), 143, dtype=np.int16)
    for ch in range(3):
        base[:, :, ch] += rng.integers(-6, 7, size=(h, w), dtype=np.int16)
    scene = np.clip(base, 0, 255).astype(np.uint8)

    _paste(
        scene,
        _card_canvas(rng),
        x=int(w * 0.30),
        y=int(h * 0.30),
        angle=float(rng.uniform(-2.5, 2.5)),
    )
    _paste(
        scene,
        _strip_canvas(label),
        x=int(w * 0.55),
        y=int(h * 0.74),
        angle=float(rng.uniform(-3.0, 3.0)),
    )

    # Lighting: mild per-channel gain + soft vignette (keeps anchors usable
    # across the full exposure range the tests exercise: 0.55x – 1.3x).
    gains = rng.uniform(0.95, 1.03, size=3)
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vignette = (1.0 - 0.06 * np.clip(radius, 0, 1) ** 2)[:, :, None]
    out = scene.astype(np.float32) * gains[None, None, :] * vignette
    noise = rng.normal(0, 2.0, size=out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def render_random_photo(seed: int = 0) -> np.ndarray:
    """Photo-like RGB image with no card and no strip (must be rejected)."""
    rng = np.random.default_rng(seed)
    w, h = SCENE_SIZE
    small = rng.integers(40, 210, size=(h // 8, w // 8, 3), dtype=np.uint8)
    photo = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    photo = cv2.GaussianBlur(photo, (31, 31), 0)
    # Saturated colour casts keep it far from white-paper strip statistics.
    tint = rng.uniform(0.6, 1.4, size=3)
    photo = np.clip(photo.astype(np.float32) * tint[None, None, :], 25, 200).astype(np.uint8)
    return photo


def sample_jpeg(label: str, seed: int = 0) -> bytes:
    """Deterministic JPEG bytes of a synthetic scene (demo-mode asset)."""
    payload = cv2.imencode(
        ".jpg", render_sample(label, seed)[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )[1]
    return payload.tobytes()


# --- Lighting conditions for ML training ---------------------------------------
#
# A LightingCondition captures everything about *how* a scene was lit: exposure,
# per-channel white-balance drift (colour temperature), vignette strength, sensor
# noise, and a little perspective skew. Training/test splits are made over these
# CONDITIONS — never over individual images — so the held-out accuracy measures
# generalisation to unseen lighting rather than memorisation of near-duplicates.


@dataclass(frozen=True)
class LightingCondition:
    """A deterministic lighting regime. `seed` fully determines every field."""

    seed: int
    key: str
    exposure: tuple[float, float, float]  # per-channel (R, G, B) gain
    vignette: float  # 0..1 corner darkening strength
    noise: float  # additive Gaussian sigma
    skew: float  # degrees of perspective/rotation applied to pasted objects
    blur: float  # Gaussian blur sigma (0 = sharp)

    @property
    def split(self) -> str:
        return split_for_condition(self)


def sample_lighting(seed: int) -> LightingCondition:
    """Deterministically build a LightingCondition from an integer seed."""
    rng = np.random.default_rng(10_000 + seed)
    # Colour temperature: warm (R up / B down) ↔ cool (B up / R down), plus exposure.
    warmth = rng.uniform(-0.18, 0.18)
    exposure = (
        float(np.clip(1.0 + warmth + rng.uniform(-0.05, 0.05), 0.55, 1.35)),
        float(np.clip(1.0 + rng.uniform(-0.04, 0.04), 0.6, 1.3)),
        float(np.clip(1.0 - warmth + rng.uniform(-0.05, 0.05), 0.55, 1.35)),
    )
    return LightingCondition(
        seed=seed,
        key=f"L{seed:03d}",
        exposure=exposure,
        vignette=float(rng.uniform(0.0, 0.14)),
        noise=float(rng.uniform(0.5, 4.0)),
        skew=float(rng.uniform(-3.0, 3.0)),
        blur=float(rng.uniform(0.0, 0.9)),
    )


# Fixed seed → split bucket, stable across runs and independent of any catalogue
# a caller assembles. ~70% train / ~20% val / ~10% test.
_SPLIT_BY_SEED: dict[int, str] = {}
for _seed in range(10):
    if _seed < 7:
        _SPLIT_BY_SEED[_seed] = "train"
    elif _seed < 9:
        _SPLIT_BY_SEED[_seed] = "val"
    else:
        _SPLIT_BY_SEED[_seed] = "test"

# A default catalogue of 36 conditions (seeds 0..35), reused across the
# default bucketing above so the split is leak-free and evenly populated.
LIGHTING_CONDITIONS: tuple[LightingCondition, ...] = tuple(sample_lighting(s) for s in range(36))


def split_for_condition(condition: LightingCondition) -> str:
    """Which split a condition belongs to: train / val / test.

    Deterministic in ``condition.seed`` alone, so it is stable across processes
    and does not depend on which catalogue the caller assembled.
    """
    return _SPLIT_BY_SEED[condition.seed % 10]


def zone_colours(label: str) -> dict[str, tuple[int, int, int]]:
    """Zone colours for one strip class — the single source of truth.

    Renders *both* the synthetic training strip and the printed calibration
    target, so the classifier can only ever be trained on exactly the object
    it will later be photographed as. The two once diverged: synthetic drew
    solid bands on a 238-grey page while the printer produced proud boxes and
    dotted rules on white paper, and the model learned the drawing rather than
    the chemistry — 1.000 on held-out synthetic lighting against 2/5 on the
    real printed strips.

    Raises:
        ValueError: If ``label`` is not a class the model can emit.
    """
    if label not in TRAINING_CLASSES:
        raise ValueError(f"unknown strip label: {label!r}")
    if label == "inconclusive":
        # Partial reaction: starch halfway to fully reacted, which measures
        # 45.4 ΔE — inside the CLEAR..FIRED band the rule table calls
        # ambiguous. Detergent is left unreacted, so one zone is doubtful and
        # one is not: the shape a real half-dipped strip actually has.
        return {
            "control": ZONE_NOMINAL["control"],
            "starch": tuple(
                int(round(0.5 * a + 0.5 * b))
                for a, b in zip(
                    ZONE_NOMINAL["starch"],
                    REACTION_COLOURS["both"]["starch"],
                    strict=True,
                )
            ),
            "detergent": ZONE_NOMINAL["detergent"],
        }
    reacted = REACTION_COLOURS.get(label, {})
    return {
        name: reacted.get(name, ZONE_NOMINAL[name]) for name in ("control", "starch", "detergent")
    }


def render_training_scene(label: str, condition: LightingCondition, seed: int = 0) -> np.ndarray:
    """Render a scene for `label` under a specific lighting `condition`.

    Deterministic per (label, condition, seed). Exercises the full pipeline
    (card detection, white balance, strip detection) under the given lighting.
    """
    if label not in TRAINING_CLASSES:
        raise ValueError(f"unknown training label: {label!r}")
    rng = np.random.default_rng(_stable_seed(label, condition.seed, seed))
    w, h = SCENE_SIZE
    scene = np.full((h, w, 3), 150, dtype=np.uint8)

    # Total rotation is the *condition* skew plus only a small per-scene jitter,
    # clamped to ±MAX_SCENE_SKEW. Stacking full per-scene skew on the condition
    # skew pushed strips past the detector's rotation tolerance (51/144 scenes
    # failed), so the condition dominates and the jitter stays bounded.
    def _clamped_angle(per_scene: float) -> float:
        return float(np.clip(condition.skew + per_scene, -MAX_SCENE_SKEW, MAX_SCENE_SKEW))

    _paste(
        scene,
        _card_canvas(rng),
        x=int(w * 0.32) + rng.integers(-6, 7),
        y=int(h * 0.34) + rng.integers(-6, 7),
        angle=_clamped_angle(float(rng.uniform(-1.0, 1.0))),
    )
    # The strip is pasted ~1.35x its canonical size. A starch reaction darkens a
    # third of the strip's area toward blue-black, which eats into the bright-
    # paper area fraction the detector gates on; without this headroom ~3% of
    # starch scenes fell below the 0.015 area floor (a label-dependent blind spot
    # that would have under-represented the starch class in training).
    _paste(
        scene,
        cv2.resize(
            training_strip_canvas(label),
            None,
            fx=TRAINING_STRIP_SCALE,
            fy=TRAINING_STRIP_SCALE,
            interpolation=cv2.INTER_LINEAR,
        ),
        x=int(w * 0.58) + rng.integers(-6, 7),
        y=int(h * 0.66) + rng.integers(-6, 7),
        angle=_clamped_angle(float(rng.uniform(-1.0, 1.0))),
    )

    return _apply_lighting(scene, condition, rng)


def training_strip_canvas(label: str) -> np.ndarray:
    """The strip as the printer draws it, in that class's colours.

    Delegated to :func:`saafscan.printstrip.render_strip` rather than drawn
    here, because the classifier must be trained on the object it will
    actually be shown: proud boxes, dotted rules, white paper. Hand-drawing
    bands instead taught it to recognise a strip no printer can make, which
    is why it abstained on every real one.
    """
    from saafscan.printstrip import render_strip as print_strip
    from saafscan.roi import STRIP_SIZE

    return print_strip(*STRIP_SIZE, fills=zone_colours(label))


def _apply_lighting(
    scene: np.ndarray, condition: LightingCondition, rng: np.random.Generator
) -> np.ndarray:
    """Apply exposure, vignette, blur and noise from a LightingCondition."""
    h, w = scene.shape[:2]
    out = scene.astype(np.float32) * np.array(condition.exposure, dtype=np.float32)[None, None, :]

    if condition.vignette > 0.0:
        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        vignette = (1.0 - condition.vignette * np.clip(radius, 0, 1) ** 2)[:, :, None]
        out = out * vignette

    out = np.clip(out, 0, 255).astype(np.uint8)
    if condition.blur > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), condition.blur)
    if condition.noise > 0.0:
        noise = rng.normal(0, condition.noise, size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _stable_seed(*parts: int | str) -> int:
    """Deterministically combine ints/strings into one RNG seed (FNV-1a)."""
    h = 1469598103934665603
    for part in parts:
        if isinstance(part, str):
            data: bytes = part.encode("utf-8")
        else:
            data = int(part).to_bytes(8, "little", signed=False)
        for byte in data:
            h ^= byte
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % (2**32)


def build_dataset(
    out_dir: Path,
    conditions: list[LightingCondition] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """Render every (condition × seed × class) scene, extract its strip crop, and
    write a manifest.

    Each sample's crop is the strip region warped to the canonical STRIP_SIZE
    and white-balanced via the printed card — the exact tensor the classifier
    consumes, stored losslessly as ``.npy`` under ``out_dir/crops/``. The
    manifest records label, condition key, split and per-sample seed so the
    eval split can be reconstructed exactly. Returns the manifest dict (also
    saved to disk).
    """
    from saafscan.roi import extract_strip

    conds = conditions if conditions is not None else list(LIGHTING_CONDITIONS)
    out_dir = Path(out_dir)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    skipped = 0
    for condition in conds:
        split = split_for_condition(condition)
        for label in TRAINING_CLASSES:
            for seed in seeds:
                img = render_training_scene(label, condition, seed)
                try:
                    crop = extract_strip(img)  # white-balanced canonical strip crop
                except ValueError:
                    # The CV front-end could not locate the strip in this render
                    # (~1% of scenes, almost all dark starch reactions whose blue-
                    # black zone shrinks the bright-paper area below the detector
                    # floor). A scene the pipeline cannot even find must not become
                    # a training target, so skip it and count it for the record.
                    skipped += 1
                    continue
                fname = f"{condition.key}_{label}_{seed}.npy"
                np.save(crop_dir / fname, crop)
                samples.append(
                    {
                        "label": label,
                        "condition_key": condition.key,
                        "split": split,
                        "seed": seed,
                        "path": f"crops/{fname}",
                    }
                )

    manifest = {
        "version": 1,
        "classes": list(TRAINING_CLASSES),
        "n_conditions": len(conds),
        "seeds": list(seeds),
        "skipped_undetectable": skipped,
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
