# SaafScan

![License](https://img.shields.io/badge/License-Apache_2.0-0F172A?style=for-the-badge&labelColor=0F172A&logo=apache&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-0F172A?style=for-the-badge&labelColor=0F172A&logo=rocket&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-FastAPI-0F172A?style=for-the-badge&labelColor=0F172A&logo=fastapi&logoColor=white)
![Language](https://img.shields.io/badge/Language-Python-0F172A?style=for-the-badge&labelColor=0F172A&logo=python&logoColor=white)

**Ten-second food-adulterant screening from any phone camera, fully offline, with an auditable evidence trail.**

SaafScan reads a printed reagent strip from a single photograph. It locates the calibration card, corrects the lighting, finds the strip, measures each reaction zone and returns an adulterant verdict with confidence, severity and advice — in well under ten seconds, on a laptop CPU, with no network. When an image cannot support a verdict, it says so and stops.

Tech4Food 2026 · theme **Detect** · final round 24 October 2026, PCCOE.

---

## Overview

Point-of-use screening has historically forced a choice between a fast answer you cannot trust and a lab answer you cannot afford to wait for. SaafScan is built to remove that trade-off for the two adulterants a paper reagent test can actually detect: starch and detergent.

| Principle | What it means in practice |
|---|---|
| **Refuse rather than guess** | Two independent refusal paths — a control-zone guard and a genuine model abstention. Neither fabricates a verdict. |
| **Evidence you can audit** | Every reading appends to a local SQLite store with no update or delete API, then exports as a PDF + QR report. |
| **Offline by construction** | No cloud, no telemetry, no network at inference time. Nothing leaves the device. |
| **Costs nothing to run** | The kit is printed paper and two common reagents. Target budget: £0 beyond consumables. |

## Why SaafScan

The problem is large, measured and current.

- The World Health Organization fact sheet of 4 June 2026 records **866 million** foodborne illnesses and **1.52 million** deaths annually, an economic loss of **US$310 billion**, affecting roughly **1 in 9** people worldwide.
- India's FSSAI reported for FY26 that **14,668 of 39,351** milk samples tested failed to conform — a **37.3%** failure rate across the samples examined.

Both figures describe detection that happens *after* the product has moved. A distributor pulling a whole product line because one batch failed is the expensive version of a slow test. SaafScan exists so a reading can be taken at the point of use, in seconds, by someone without a laboratory, and still leave a record that survives scrutiny.

## What it does today

Shipped and covered by the test suite:

- Card detection with perspective correction, and white-balance anchored on printed reference patches.
- Strip detection with an aspect and area gate; a strip laid the wrong way is **refused, never misread**.
- Reaction-zone colour extraction and ΔE comparison against printed nominal references.
- A rule-table classifier and a 12-feature gradient-boosting model, with the rule table always reachable as fallback.
- Two distinct refusal mechanisms: the control-zone guard, and model abstention below a calibrated confidence threshold.
- Local append-only evidence log with PDF + QR report export.
- A web interface supporting live camera capture and file upload.

## Measured results

All figures below are reproducible from `models/eval_report.json`, `deck/bench.json` and the test suite.

| Metric | Value | Note |
|---|---|---|
| Held-out accuracy | **0.933** | 45 held-out samples, split by lighting condition |
| Coverage | **0.733** | 33 of 45 answered |
| Selective accuracy | **1.000** | on the 33 it commits to, it is right every time |
| Rule-table baseline | **1.000** | the same split |
| Delta, model minus rules | **−0.067** | see below |
| Median pipeline time | **6.0 ms** | measured with `deck/bench.py` |
| HTTP end-to-end | **19.3 ms** | median |
| Cold start to first verdict | **0.42 s** | median |
| Test suite | **291 passing, 90% coverage** | ruff clean |

**The model does not beat the rule table, and the project does not claim it does.** Of the 12 samples the model declines, 9 are genuinely labelled inconclusive and 3 are one starch and two detergent it refused to call rather than guess. The gradient booster ships as the default on artifact size — 179 KB against 45 MB for the ResNet18 — and on abstention headroom, not on accuracy.

## Project status

Prototype. The core pipeline, evidence store, classifiers and reporting path are complete and tested. The imaging half has been validated against photographed printed colour targets: every frame cleared every detector gate, with surface luma measured at 117.6–124.8 inside the accepted 50–170 band.

Capture and training on real reagent strips is in progress.

### Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Core pipeline: card, white-balance, strip, zones | Complete |
| 2 | Rule-table classifier, verdict model, refusal paths | Complete |
| 3 | Append-only evidence store and PDF + QR reporting | Complete |
| 4 | Web interface: live camera, upload, history, report | Complete |
| 5 | Synthetic training set, split by lighting condition | Complete |
| 6 | Gradient booster and ResNet18, calibration, evaluation | Complete |
| 7 | Printed-target imaging validation | Complete |
| 8 | Real reagent strip capture, ingest, retrain | In progress |
| 9 | Final-round demonstration, 24 October 2026 | Scheduled |

## Installation

Requires Python 3.12.

```bash
uv venv --python 3.12 .venv
uv sync --python .venv/bin/python
bash start_demo.sh
```

Then open **http://127.0.0.1:8000**. `127.0.0.1` matters: browsers only expose the camera in a secure context, and localhost qualifies.

Manual start:

```bash
.venv/bin/uvicorn saafscan.api:app --host 127.0.0.1 --port 8000
```

To drive a phone against the laptop without a network, reverse the port over USB:

```bash
adb reverse tcp:8000 tcp:8000
```

## Architecture

```
phone camera or upload
  -> calibration card detection (black-bordered quad, perspective warp)
  -> white-balance anchored on printed white/grey/black patches
  -> strip detection (elongated bright quad, perspective warp)
  -> reaction-zone extraction (control, starch, detergent)
  -> deltaE against printed nominal references
  -> classifier (ML primary, rule table always reachable)
  -> verdict {adulterant, confidence, severity, advice, evidence}
  -> append-only SQLite evidence log -> PDF + QR report
```

Failure modes are explicit and structured:

- **No calibration card** — rejected with HTTP 422 and a human-readable message. Never a fabricated verdict.
- **Ambiguous zone colours** — verdict `inconclusive` with confidence at or below 0.5.
- **Oversized, non-image or empty uploads** — HTTP 422 or 413, with no evidence entry created.

### Repository layout

```
src/saafscan/
  calib.py        card detection, perspective warp, white-balance
  roi.py          strip detection and zone extraction
  color.py        Lab and deltaE primitives
  classifier.py   rule table -> Verdict
  verdict.py      immutable Verdict model
  pipeline.py     orchestration and PipelineError
  evidence.py     append-only SQLite store
  report.py       PDF and QR export
  api.py          FastAPI application and static frontend
  ingest.py       photograph -> canonical crop -> manifest row
  synthetic.py    deterministic synthetic demo scenes
  ml/             training, calibration, runtime, evaluation
tests/            TDD suites, including the demo gauntlet
scripts/          dataset rendering and model training
deck/             presentation build, verification gate and charts
assets/           generated print targets and reference artwork
frontend/         single-page interface
docs/             documentation index
```

## Working with real strips

Synthetic scenes exist so the interface, pipeline, latency and screenshots could be built before consumables arrived. They are a rehearsal path, not the demonstration.

Photographs are named by session and label — `S001_none_0.jpg` — and ingested into a trainable dataset:

```bash
.venv/bin/python -m saafscan.ingest photos/ --dataset data/realset
.venv/bin/python scripts/train_features.py --dataset data/realset
```

**Splits are decided by session, never by photograph.** All frames from one sitting go to one split. Assigning forty photographs of a single desk lamp across train and test would score beautifully and mean nothing.

Three generated print artefacts support the shoot, each verified through the same detectors the live path runs:

| Artefact | Command | Purpose |
|---|---|---|
| Strip template | `python -m saafscan.printstrip --out strip_template.png` | Sixteen strips per A4. Red box is a printed reference; leave it dry. |
| Backdrop | `python -m saafscan.printbackdrop --out backdrop.png` | Flat grey 110. `--check photo.jpg` refuses a sheet outside the 50–170 luma band. |
| Frame guide | `python -m saafscan.layoutguide --out layout_guide.png` | True-scale worked example; the CLI re-detects its own output and fails if the pipeline would reject it. |

When a photograph is rejected, `saafscan.diagnostics` runs every stage it can and records what each one saw:

```bash
.venv/bin/python -m saafscan.diagnostics photo.jpg --out debug/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions follow test-driven development with an 80% coverage floor, conventional commits and ruff-clean source.

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities privately through GitHub security advisories. No personal data leaves the machine and there is no telemetry.

## Support

See [SUPPORT.md](SUPPORT.md). There is no service-level agreement at this stage.

## Code of conduct

Participants are expected to read and follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Licensed under the [Apache License 2.0](LICENSE).

---

**Disclaimer.** SaafScan is a student prototype built for educational demonstration. It is not a certified diagnostic or food-testing device, and its readings are screening signals rather than legal or laboratory test results. Do not use it as the sole basis for consuming or discarding food.
