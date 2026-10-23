# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

No versions have been released. All work is tracked under **Unreleased**.

## [Unreleased]

### Added

- Core pipeline: calibration card detection, perspective correction, white-balance
  anchored on printed reference patches, strip detection with aspect and area gates,
  reaction-zone extraction and ΔE comparison against printed nominal references.
- Two classifiers behind a single interface: a rule table and a 12-feature gradient
  booster, with the rule table always reachable as fallback.
- Two independent refusal paths: a control-zone guard (ΔE above 45) and genuine model
  abstention below a calibrated confidence threshold.
- Append-only SQLite evidence store with no update or delete API.
- PDF and QR evidence report export.
- FastAPI application with a single-page frontend supporting live camera capture and
  file upload.
- Deterministic synthetic demo scenes so the interface, pipeline, latency and
  screenshots could be built before consumables arrived.
- Deterministic dataset rendering and gradient-booster training scripts.
- Training and evaluation that hold out an entire lighting condition, with a classifier
  comparison report.
- Photographic ingest: session-labelled frames become a trainable dataset, with splits
  decided by session and never by photograph.
- Generated print artefacts — strip template, printable backdrop with a refuse-gate
  luma check, and a frame guide that re-detects its own output.
- Diagnostics tool that runs every pipeline stage it can on a rejected photograph and
  records what each stage saw.
- Presentation deck built entirely by script, gated by a 29-check verification pass that
  asserts the source template is unchanged before and after the build.
- Printed colour target validation: five photographed frames each cleared every
  detector gate, with surface luma measured at 117.6–124.8 inside the accepted
  50–170 band.
- Repository documentation set: contributing, security, support, code of conduct,
  changelog and documentation index.

### Changed

- Model artefacts regressed from 1.000 held-out accuracy to 0.933 after the training
  set was widened from 18 renderings to 132 across five lighting conditions. The
  project reports 0.933 rather than the flattering figure from the narrow set.
- Documentation, deck and model tables corrected to the measured values: accuracy
  0.933, coverage 0.733, answered/declined 33/12, and a model-minus-rules delta of
  −0.067.
- Public messaging removed the claim that the model matches or beats the rule table.
  The gradient booster is presented as shipping on artifact size — 179 KB against
  45 MB for the ResNet18 — and on abstention headroom.

### Known limitations

- Training and calibration are synthetic. The imaging half has been validated against
  photographed printed colour targets; capture and training on real reagent strips is
  in progress.
- The model does not outperform the rule table on the held-out split. The project does
  not claim that it does.
