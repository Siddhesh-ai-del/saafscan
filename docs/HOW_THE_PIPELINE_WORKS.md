# How the pipeline works

A stage-by-stage walkthrough of a single reading, from photograph to evidence record.
Each stage either passes the frame forward or raises a structured refusal. Nothing in
this path guesses.

## At a glance

```
photograph
  -> 1. calibration card detection
  -> 2. white-balance correction
  -> 3. strip detection
  -> 4. reaction-zone extraction
  -> 5. deltaE against printed nominal references
  -> 6. classifier (ML primary, rule table always reachable)
  -> 7. verdict, or an explicit refusal
  -> 8. append-only evidence record -> PDF + QR report
```

Median time through stages 1–7 is **6.0 ms** measured locally, **19.3 ms** end-to-end
over HTTP. Nothing here depends on a network.

## 1. Calibration card detection

`src/saafscan/calib.py`

The card is a black-bordered quadrilateral printed alongside the strip. The detector
locates that border, then computes a perspective transform to a canonical top-down view.

The card is the whole reason colour can be trusted. A phone camera under a kitchen
light, a desk lamp and daylight produces three different white points from the same
paper. Without a reference in frame, "pink" is not a measurable quantity.

If no card is found the pipeline raises `PipelineError` and the API returns **HTTP 422**
with a human-readable message. It does not proceed and it does not guess.

## 2. White-balance correction

`src/saafscan/calib.py`

Printed white, grey and black reference patches on the card give three known points.
The frame is corrected so those patches land on their nominal values, which puts every
downstream colour measurement on a common footing.

This is also where the **control-zone guard** begins to matter. Drift between the
photographed reference patches and their printed nominal is measured as ΔE. Past a
tolerance of **45**, the frame is no longer trustworthy and the pipeline refuses rather
than diagnosing chemistry from an untrustworthy image.

## 3. Strip detection

`src/saafscan/roi.py`

The strip is found as an elongated bright quad and warped to a canonical rectangle.
Two gates apply:

| Gate | Accepted band | Rejects |
|---|---|---|
| Aspect ratio | 2.5 – 9.0 | A strip laid sideways, or a fragment |
| Area fraction | 1.5% – 20% of frame | Too distant to resolve, or not a strip |

A strip laid the wrong way is **refused, never misread**. The detector also enforces
clearance from the frame edge so a strip cropped by the photograph border cannot pass.

## 4. Reaction-zone extraction

`src/saafscan/roi.py`

Within the canonical strip rectangle, three zones are located: **control**, **starch**
and **detergent**. Each zone's colour is averaged over its region.

The control zone is the strip's own internal check. A strip that has not been dipped
properly shows no control reaction, and no amount of image processing recovers that.

## 5. ΔE comparison

`src/saafscan/color.py`

Each zone colour is converted to CIELAB and compared against the printed nominal
reference for that reagent as a ΔE distance.

Note what is *not* done: the image is never adjusted to make a result come out a
particular way. In validation, photographs are used exactly as captured — no cropping,
no pixel adjustment. Recolouring a frame until a classifier agrees with the expected
label adjusts the answer, not the photograph.

## 6. Classification

`src/saafscan/classifier.py`, `src/saafscan/ml/`

Two implementations sit behind one interface:

- **Rule table** — fixed thresholds, `CLEAR = 40` and `FIRED = 55`, mapping zone
  ΔE to a verdict. Deterministic, interpretable, always reachable as a fallback.
- **Gradient booster** — a 12-feature model (3 ΔE values plus 9 RGB values), a
  179 KB artifact calibrated for confidence. A 45 MB ResNet18 can be served the same
  way but is not the default.

`SAAFSCAN_CLASSIFIER=rules` forces the rule table at runtime.

An important behavioural detail: a **detergent positive is near-colourless**, because
methylene blue is *reduced* by the reagent. Blue means negative. A blue detergent zone
is the absence of the reaction, not its presence.

## 7. Verdict or refusal

`src/saafscan/verdict.py`

Two independent refusal mechanisms exist and must not be conflated:

| Mechanism | Trigger | What it means |
|---|---|---|
| **Control-zone guard** | Reference ΔE above 45 | The *image* cannot be trusted |
| **Model abstention** | Confidence below 0.55 | The *model* declines to call it |

Either produces `inconclusive` with a stated reason. Neither produces a fabricated
adulterant name.

The distinction matters when reading results: a frame refused by the control guard
tells you to reshoot; an abstention tells you the chemistry was genuinely ambiguous.

## 8. Evidence and reporting

`src/saafscan/evidence.py`, `src/saafscan/report.py`

A verdict appends a row to a local SQLite store. The store exposes **no update and no
delete API**, so a logged reading cannot be quietly rewritten after the fact.

From that record, `report.py` renders a PDF with a QR code. Scanning the code resolves
back to the stored record, so a reading taken in the field can be checked later against
what was actually captured.

## Failure modes

| Situation | Response |
|---|---|
| No calibration card in frame | HTTP 422, human-readable message, no evidence entry |
| Strip rejected by aspect or area gate | HTTP 422, no evidence entry |
| Reference patches drift past tolerance | Refusal — reshoot required |
| Zone colours ambiguous | `inconclusive`, confidence at or below 0.5 |
| Oversized or non-image upload | HTTP 413 or 422, no evidence entry |

## Debugging a rejected photograph

When a frame is refused, run the diagnostics tool. It executes every stage it can and
records what each one saw, which localises the failure to a specific stage rather than
leaving you guessing:

```bash
.venv/bin/python -m saafscan.diagnostics photo.jpg --out debug/
```

Output lands in `debug/`, one artefact per stage.

## Further reading

- [README](../README.md) — measured results and architecture
- [CONTRIBUTING](../CONTRIBUTING.md) — the constraints any change must respect
