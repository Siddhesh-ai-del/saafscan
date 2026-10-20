# SaafScan

**10-second adulterant detection using any phone camera.** Tech4Food 2026 · theme **Detect**.

Dip the printed reagent strip into a sample → photograph it against the printed colour
calibration card → SaafScan auto-detects the strip, corrects lighting and perspective,
measures each reaction zone, and returns **adulterant type + confidence + severity** in
well under 10 seconds. Fully offline. Evidence is logged to a local append-only store and
can be exported as a PDF + QR report.

> **Prototype status.** This repository is the Round-1 prototype. Colour thresholds are
> validated against **synthetic** reagent scenes. See
> [Physical kit for 23 Oct](#physical-kit-for-23-oct) — the final demo must run on real
> reagent strips.

## Quickstart (cold start = one command)

```bash
bash start_demo.sh
```

Then open **http://127.0.0.1:8000**.

- **Live camera** (where the browser allows it) or **file upload** of a strip photo.
- **Demo mode**: buttons render the three synthetic samples (`pure` / `starch` /
  `detergent`) and push them through the *real* HTTP pipeline — this is the rehearsal and
  Round-1 screenshot path.
- **History** shows the evidence log; every entry exports a **PDF + QR report**.

Manual startup:

```bash
uv venv --python 3.12 .venv
uv sync --python .venv/bin/python
.venv/bin/uvicorn saafscan.api:app --host 127.0.0.1 --port 8000
```

## How it works

```
phone camera / upload
   → calibration card detection (black-bordered quad, perspective warp)
   → white-balance correction anchored on printed white/grey/black patches
   → strip detection (elongated bright quad, perspective warp)
   → reaction-zone colour extraction (control · starch · detergent)
   → ΔE vs printed nominal references → classifier
   → verdict {adulterant, confidence, severity, advice, evidence}
   → append-only SQLite evidence log → PDF + QR report
```

Failure modes are explicit:

- **No calibration card** → rejected with *"This does not appear to be a SaafScan test image"* (HTTP 422), never a fabricated verdict.
- **Ambiguous zone colours** → verdict `inconclusive` with confidence ≤ 0.5.
- **Oversized / non-image / empty uploads** → HTTP 422/413, no evidence entry created.

## Development

```bash
.venv/bin/pytest --cov=saafscan --cov-report=term-missing   # ≥80% gate
.venv/bin/ruff check src tests                                # lint gate
.venv/bin/python -m pytest tests/test_e2e.py                 # demo gauntlet
```

Layout:

| Path | Role |
|---|---|
| `src/saafscan/calib.py` | card detection, perspective warp, white-balance |
| `src/saafscan/roi.py` | strip detection, zone extraction |
| `src/saafscan/color.py` | ΔE / Lab colour primitives (pure functions) |
| `src/saafscan/classifier.py` | rules → `Verdict` (incl. `inconclusive`) |
| `src/saafscan/verdict.py` | immutable `Verdict` model |
| `src/saafscan/pipeline.py` | orchestration + `PipelineError` |
| `src/saafscan/evidence.py` | append-only SQLite store (no update/delete API) |
| `src/saafscan/report.py` | PDF + QR export |
| `src/saafscan/synthetic.py` | deterministic synthetic demo scenes |
| `src/saafscan/api.py` | FastAPI app + static SPA |
| `frontend/` | single-page UI (camera, verdict, history, report) |
| `tests/` | TDD suites incl. `test_e2e.py` demo gauntlet |
| `screenshots/` | Round-1 deck images |

### Model training & evaluation (Phase 7)

The served classifier is **ML-primary** with the rule-table always reachable as
fallback. Two candidates were trained on the same lighting-split dataset and
compared on the **held-out-lighting** split (the manifest decides splits by
lighting condition, never by random image):

| Model | Train command | Held-out acc | Coverage | Δ vs rules |
|---|---|---|---|---|
| **A — 12 colour/ΔE features → GBC (ships default)** | `scripts/train_features.py` | 1.000 | 0.800 | +0.000 |
| B — ResNet18 finetune, AMP/CUDA | `scripts/train_cnn.py` | 1.000 | 0.800 | +0.000 |

They tie on every measured metric (accuracy, coverage, confusion, full
accuracy-vs-coverage curve); A wins the documented tie-breaks: abstention
threshold **0.55 vs 0.999** (operating margin under real-phone shift),
temperature **1.21 vs 0.81** (softening vs sharpening), and **179 KB vs 45 MB**
artifacts. B stays servable behind `SAAFSCAN_ML_MODEL=cnn`.

```bash
.venv/bin/python scripts/train_features.py                  # Model A (+ dataset render, CPU ~15 s)
python scripts/train_cnn.py                                 # Model B (needs torch+CUDA; --cpu works)
.venv/bin/python -m saafscan.ml.eval --dataset data/trainset \
    --models models --out models/eval_report.json           # both models vs the rule-table baseline

SAAFSCAN_ML_MODEL=cnn .venv/bin/pytest                       # serve Model B instead
SAAFSCAN_CLASSIFIER=rules .venv/bin/pytest                   # pure rule-table fallback
```

`data/trainset/` (51 MB) and `models/cnn.onnx` (45 MB) are gitignored as
regenerable — first Model B run needs `uv pip install --target .mltools onnx`
(for the export gate). Temperature scaling, the abstention threshold (low
confidence → `inconclusive`, never a fabricated verdict) and every held-out
number come from `saafscan.ml.calibration`, shared by both trainers, so the
delta is computed by identical code.

## Diagnosing a rejected photograph

`analyze()` stops at the first failure with one sentence — correct for a user,
useless when a real strip photo is rejected. `saafscan.diagnostics` runs every
stage it can and records what each one saw:

```bash
.venv/bin/python -m saafscan.diagnostics photo.jpg --out debug/
```

Exit code `0` if the photo analysed, `1` if it was rejected. `debug/` gets an
annotated copy: the detected card and strip quads outlined, and a panel listing
every measurement.

```
ok    image    640x480 px RGB
ok    card     calibration card detected
        white_de                 4.43     <- printed card accuracy
ok    strip    reagent strip detected
ok    zones    reaction-zone colours measured
        starch_de               98.73     <- reacted zone vs unreacted nominal
FAIL  strip    no elongated bright quad passed the strip gates
skip  zones    not run (a previous stage failed)
```

Read it like this:

- **`card` fails** — the printed card is wrong, too small, or the photo is not of
  a SaafScan scene. Re-print on matte paper, get closer, more even light.
- **`white_de`/`grey_de`/`black_de` large** — the printed card's own patches are
  off nominal, so white-balance is anchoring on a bad reference. Re-print.
- **`strip` fails with the card found** — the strip is out of frame, too dark,
  or sitting on a bright surface so its contrast over the background is lost.
- **`zones` ΔE values** — a reacted zone should sit *far* from its unreacted
  nominal (starch reading ~99 above is the blue-black complex, which is
  expected). A zone hovering just above zero is the failure to worry about: the
  chemistry did not visibly react.

## Turning a shoot into training data

Round 1 said the model would be retrained on real strips before the final.
`synthetic.build_dataset` can only render synthetic scenes; `ml.train` reads
`manifest.json` + `.npy` crops and never asks where a crop came from. `ingest`
is the piece in between: photograph → canonical crop → manifest row.

Name the shots `SESSION_LABEL_N` and point it at the folder:

```bash
photos/S001_none_0.jpg
photos/S001_starch_1.jpg
photos/S002_detergent_0.jpg

.venv/bin/python -m saafscan.ingest photos/ --dataset data/realset
# 3 taken in, 0 refused, 0 already present -> 3 samples
```

Each shot is warped to the same `STRIP_SIZE` crop and white-balanced through
the printed card, so training features come from the identical chain the
request path uses. A shot the detector cannot use is reported with the stage
that gave up — `card` or `strip` — never silently dropped, so a bad frame is
caught while you are still standing at the bench.

**Split is decided by session, not by photograph.** All frames from one sitting
go to one split, the same way the synthetic set splits by lighting condition.
Splitting 40 photos of a single desk lamp across train and test would score
beautifully and mean nothing.

Train on it the same way as synthetic:

```bash
.venv/bin/python scripts/train_features.py --dataset data/realset
```

`ml.train` refuses a training split missing any of the five classes, so a
partial shoot fails loudly rather than producing a model that maps argmax to
the wrong labels.

## Physical kit for 23 Oct

> ⚠️ **Synthetic demo mode is for Round-1 screenshots and dry-run rehearsal only.**
> The **23 Oct final at PCCOE must run on real reagent strips** with real samples.
> The synthetic path exists so the UI, pipeline, latency and screenshots can be built and
> rehearsed before the consumables arrive — it is **not** the demo.

Procurement list (non-code items the live demo needs):

| # | Item | Purpose | Notes |
|---|---|---|---|
| 1 | **Iodine solution** (tincture/Lugol's, dilute) | starch test — blue-black complex | pharmacy / lab supplier; keep away from eyes; carry in a sealed small bottle |
| 2 | **Methylene blue** solution | detergent test — decolourisation | lab chemical supplier; small quantity only |
| 3 | **Strips / paper material** (filter paper or chromatography paper, cut to strip size) | reagent carrier | cut **20 × 80 mm (4:1)** — the detector's aspect gate is 2.5–9.0 and `STRIP_SIZE` is 360×90, so 4:1 matches the trained geometry; 8×80 is aspect 10 and is rejected. **Print `assets/strip_template.png` at 100% to get sixteen at a time**, each with a printed red control reference plus two dotted reagent boxes — leave the red box dry. Lay the strip **lengthways, long edge left-to-right**: a strip laid end-to-end down the frame is refused rather than misread. ~20 strips for the day |
| 4 | **Printed colour calibration card** | white-balance + ΔE reference | print on matte photo paper, never glossy (glare breaks detection); ≥2 spares. **Which way up it lies no longer matters** — detection checks all four rotations, so a judge can pick it up and put it down at any angle |
| 5 | **Samples**: pure milk, milk + starch (known amount), milk + detergent, + one venue beverage | judge-picks demo | prepare fresh on 23 Oct in labelled cups; bring iodine/starch control |
| 6 | **Second phone (backup)** | camera failure fallback | pre-tested with the app before leaving |
| 7 | **Mid-grey shooting backdrop** (grey cloth, grey T-shirt, charcoal/olive notebook) | card + strip detection | Surface luma must land in **50–170 after white-balance**. Below 50 the card's black frame merges into the background; above 170 the white strip merges into it. Light kraft/tan paper reads ~190 once the colour cast is corrected and fails — this range was measured through the pipeline, not guessed. Bring one to the venue rather than hoping PCCOE has a suitable table |
| 8 | Extra: zip bags, dropper/pipette, gloves, printed verdict QR card, power bank | logistics | demo hygiene + 3-hour buffer |

**Also carry:** a recorded demo video as fallback, laptop with `.venv` pre-built (offline),
and a printed one-page verdict sample for judges who don't scan QR.

**Shooting plan (18–19 Oct).** Photograph all five classes — plain milk, milk +
starch, milk + detergent, both, and one deliberately under-reacted strip for
`inconclusive` — across at least three lighting setups (desk lamp, window
light, corridor/flash). Each setup is its own session: `S001`, `S002`, `S003`.
Five to eight frames per class per session is enough to retire the synthetic
disclaimer in the deck: *"trained on N real strip photographs across M lighting
conditions, held out by session."* Rehearse the shoot on a blank paper strip
first, with `diagnostics` open, so the first real reagent frames are not also
the first time the pipeline sees your room.

**Where each sample must land — do not do this by eye.** Generate the template:

    python -m saafscan.printstrip --out strip_template.png

`assets/strip_template.png` already contains it: sixteen strips per A4. The
**red box is a printed reference patch — leave it dry.** Only the two dotted
boxes take reagent, one each. Print at **100 % / actual size** with "fit to page"
off, cut on the outlines, then dip the whole strip. One sheet is a whole day of
samples.

The strip is read as three equal thirds — 0–26.7, 26.7–53.3 and 53.3–80 mm on an
80 mm strip — and each third is sampled only across its **middle 55 %**, so the
measurement bands are 5.8–20.7, 32.7–47.3 and 59.3–74.0 mm. The printed squares
sit 0.5 mm *proud* of those bands, which is the whole point: reagent inside a
square is guaranteed to cover the window, while no ink falls inside it to bias
the mean colour the verdict is computed from. `saafscan.printstrip` derives them
from `roi.zone_window_mm`, so the artwork cannot drift away from the detector.

The red box is the exception, and is filled solid because there the ink *is* the
measurement. It carries `ZONE_NOMINAL["control"]` — the exact colour
`classify_zones` measures the control band against — so the reference is present
on every strip by construction. Loading it by hand is what `photo4.jpeg` priced:
with no control mark the window read blank paper, ΔE 79.8 against nominal, and
the strip was refused before its chemistry was ever considered. Printing it also
means no reagent is spent on the one zone that never reacts.

Colour placed at 10 / 25 / 42 mm reads back **one zone shifted**: the marking
meant for the detergent zone falls inside the starch band, and the detergent
band reads untouched paper — which is indistinguishable from a fully
decolourised, i.e. reacted, zone. This is what `photo5.jpeg` did: colouring only
the left half of the strip made the rule table report `both` at 0.71 from a
strip that had no detergent on it. Blank paper in a loaded zone cannot be told
apart from a positive, so zone placement is an operator responsibility, not
something the pipeline can detect.

## Security & privacy

- No personal data leaves the machine; telemetry: none.
- Evidence store is local SQLite, append-only (no update/delete API by design).
- Uploads are size- and type-validated at the boundary; failures return structured errors
  (no stack traces to clients).

## License

MIT (prototype; not a certified food-safety device — see DISCLAIMER below).

**DISCLAIMER:** SaafScan is a student prototype for educational demonstration. It is not a
certified diagnostic or food-testing device. Do not use it as the sole basis for consuming
or discarding food.
