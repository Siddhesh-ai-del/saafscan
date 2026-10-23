# IRL printed colour targets — photographed

**These are printed colour references cut from `assets/caltargets.png`.
There is no milk, no reagent, and nothing here was ever applied to anything.
The folder name says which swatch the printed sheet depicts — it is not a test
result, and this directory must never be ingested as chemistry.**

The printed sheet says the same thing on every target:
*"COLOUR REFERENCE ONLY: no milk, no reagent, nothing to apply — never
ingest these."*

## Why they exist

Phase one of the two-phase calibration: print the colour-rendered strips,
photograph them, and put them through the live pipeline **before** any
chemistry is involved. That separates "the imaging half cannot read a colour"
from "the reaction did not happen" — two failures that look identical once
reagent is on the paper.

## What is here

Sorted 23 Oct 2026 from five frames shot 18:48–18:52.

| folder | frame | middle zone (measured) | detergent chroma |
|---|---|---|---|
| `starch/` | `IMG_20261023_184835.jpg` | `(99, 111, 156)` blue-dominant = printed navy | 110 |
| `none/` | `IMG_20261023_184951.jpg` | `(254, 213, 148)` warm = printed amber | 107 |
| `detergent/` | `IMG_20261023_185104.jpg` | `(253, 231, 193)` warm = printed amber | 6 |
| `inconclusive/` | `IMG_20261023_185228.jpg` | `(193, 163, 153)` warm, low chroma = printed brown | 27 |
| `_duplicate/` | `IMG_20261023_184951 (1).jpg` | pixel-identical to the `none/` frame | — |

The label was read from the middle zone's **hue direction** — blue-dominant
means the printed navy, warm means amber or brown — and then cross-checked
against the captioned sheet in `assets/caltargets.png`. Detergent-zone chroma
separates reacted (near-colourless, chroma ~0) from unreacted (blue, chroma
~110). A classifier fitted to these same four frames agrees 4/4, which is
expected rather than independent; the captioned sheet is the check that counts.

**`both` was never photographed** — four of the five targets are present.

## Cropping: not needed

Every frame clears every gate with room to spare, measured through the real
`pipeline.analyze`:

| metric | measured | detector gate |
|---|---|---|
| surface luma | 117.6 – 124.8 | 50 – 170 |
| strip aspect | 3.92 – 4.15 | 2.5 – 9.0 |
| strip share of frame | 3.4 – 3.7 % | 1.5 – 20 % |
| nearest frame edge | ≥ 411 px | fully in frame |

Nothing was cropped, rotated or recoloured. Adjusting pixels until the
classifier agrees would be adjusting the answer, not the photograph — if a
frame needs help, it gets re-shot in better light.

## What the drift says

The control zone is the same red on all five targets, so its measured ΔE
against `roi.ZONE_NOMINAL["control"]` is the imaging error in one number:

| frame | control ΔE | outcome |
|---|---|---|
| `inconclusive/` | 22.7 | passes |
| `starch/` | 25.2 | passes |
| `none/` | 27.7 | passes |
| `detergent/` | **46.3** | **over `CONTROL_TOLERANCE` (45) — refused** |

That refusal is correct behaviour: `ml.runtime.classify` returns
`inconclusive` at confidence 0.3 rather than diagnose from a control zone it
cannot trust.

Read as printed-swatch fidelity, not as adulterant accuracy on milk:
`starch/` is read correctly, `none/` and `inconclusive/` are misread as
`starch`, and `detergent/` is refused. Detection, white-balance and zone
extraction are all sound — it is the measured colours that drift from
nominal, which is precisely what phase one exists to surface.

Re-check any frame with:

    python -m saafscan.diagnostics Synthetic_IRL_test_image/starch/IMG_20261023_184835.jpg

which reports card patch ΔE, the white-balance shift applied, strip geometry,
each zone's RGB and ΔE against nominal, and the verdict — `PASSED` only if the
whole chain ran.
