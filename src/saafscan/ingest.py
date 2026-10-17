"""Ingest real strip photographs into a trainable dataset.

Round 1 promised the model would be retrained on real strip photos before the
final. ``build_dataset`` only ever calls the synthetic renderer, while
``ml.train.load_dataset`` reads manifest.json plus .npy crops without caring
how a crop was produced — so this module is the missing link between a
photograph taken on a phone and a dataset the trainers accept.

Splits are assigned by **shooting session, never by photograph**. The synthetic
set splits by lighting condition precisely so no condition appears in two
split; photographs taken of the same lamp in one sitting have to be grouped the
same way, or the held-out accuracy measures nothing at all.

Shoot-day naming convention::

    photos/S001_none_0.jpg
            └─┬─┘
              session and label, parsed from the filename when not given

    python -m saafscan.ingest photos/ --dataset data/realset
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

from saafscan.roi import extract_strip
from saafscan.synthetic import _SPLIT_BY_SEED, TRAINING_CLASSES

DEFAULT_DATASET_DIR = Path("data/realset")
MANIFEST_VERSION = 1
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class IngestError(ValueError):
    """A photograph cannot become a training sample."""


def split_for_session(session: str) -> str:
    """Which split an entire shooting session belongs to.

    Derived from a stable CRC of the session name rather than ``hash()``, which
    Python randomises per process and which would reshuffle every split on each
    run. Reuses the synthetic set's 7/2/1 train/val/test table so both datasets
    split on identical terms.
    """
    return _SPLIT_BY_SEED[zlib.crc32(session.encode("utf-8")) % 10]


def _parse_name(photo: Path) -> tuple[str, str]:
    """Session and label out of ``S001_none_0.jpg``."""
    parts = photo.stem.split("_")
    session = parts[0] if parts else photo.stem
    label = parts[1] if len(parts) > 1 else ""
    return session, label


def _result(
    ingested: bool,
    *,
    label: str | None = None,
    session: str | None = None,
    path: str | None = None,
    reason: str | None = None,
) -> dict:
    return {
        "ingested": ingested,
        "label": label,
        "session": session,
        "split": split_for_session(session) if session else None,
        "path": path,
        "reason": reason,
    }


def _read_rgb(photo: Path):
    if not photo.is_file():
        return None, f"no such file: {photo}"
    bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, f"could not decode {photo.name} as an image"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None


def ingest_photo(
    photo: Path,
    label: str,
    dataset_dir: Path,
    session: str,
    strict: bool = True,
) -> dict:
    """Turn one photograph into a canonical strip crop plus a manifest row.

    The crop written is the exact tensor the classifier consumes — white-
    balanced, warped to STRIP_SIZE — taken through the same ``extract_strip``
    chain the request path uses.

    On a photo the detector cannot use, ``strict=True`` (the default) raises so
    a mistake is loud. ``strict=False`` reports instead, which is what a batch
    shoot needs: one unusable frame must not abort two hundred good ones.
    """
    if label not in TRAINING_CLASSES:
        raise IngestError(f"unknown label {label!r}; expected one of {list(TRAINING_CLASSES)}")

    image, read_error = _read_rgb(photo)
    if image is None:
        if strict:
            raise IngestError(read_error)
        return _result(False, label=label, session=session, reason=read_error)

    try:
        crop = extract_strip(image)
    except ValueError:
        # Diagnostics is only worth running on the failure path: it re-runs
        # detection to say *which* stage gave up, which is the information the
        # person holding the phone needs while they are still shooting.
        from saafscan.diagnostics import inspect_image

        report = inspect_image(image)
        reason = report.error or "the photograph could not be analysed"
        if strict:
            raise IngestError(reason) from None
        return _result(False, label=label, session=session, reason=reason)

    dataset_dir = Path(dataset_dir)
    crop_dir = dataset_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"crops/{photo.stem}.npy"
    np.save(dataset_dir / rel_path, crop)
    return _result(True, label=label, session=session, path=rel_path)


def _empty_manifest() -> dict:
    return {
        "version": MANIFEST_VERSION,
        "classes": list(TRAINING_CLASSES),
        "origin": "real photographs",
        "sessions": [],
        "samples": [],
    }


def _load_manifest(dataset_dir: Path) -> dict:
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        return _empty_manifest()
    manifest = json.loads(path.read_text())
    if tuple(manifest.get("classes", ())) != TRAINING_CLASSES:
        raise IngestError(
            f"{path} declares classes {manifest.get('classes')}, "
            f"expected {list(TRAINING_CLASSES)} — refusing to mix datasets"
        )
    return manifest


def _write_manifest(dataset_dir: Path, manifest: dict) -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def ingest_directory(
    directory: Path,
    dataset_dir: Path,
    label: str | None = None,
    session: str | None = None,
    strict: bool = False,
) -> dict:
    """Ingest every photograph in a directory.

    Labels and sessions come from the filenames unless supplied. Returns how
    many were taken in and how many were refused, with a reason for each
    refusal — a rejected shot is reported, never silently dropped.
    """
    if label is not None and label not in TRAINING_CLASSES:
        raise IngestError(f"unknown label {label!r}; expected one of {list(TRAINING_CLASSES)}")

    directory = Path(directory)
    photos = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        raise IngestError(f"no photographs found in {directory}")

    dataset_dir = Path(dataset_dir)
    manifest = _load_manifest(dataset_dir)
    known = {sample["path"] for sample in manifest["samples"]}

    ingested = rejected = duplicates = 0
    reasons: list[str] = []
    for photo in photos:
        photo_session, photo_label = _parse_name(photo)
        effective_label = label or photo_label
        if effective_label not in TRAINING_CLASSES:
            # A filename the operator mis-typed is a rejected shot, not a
            # reason to abort the whole batch they are standing there shooting.
            rejected += 1
            reasons.append(f"{photo.name}: unknown label {effective_label!r}")
            continue
        result = ingest_photo(
            photo,
            label=effective_label,
            dataset_dir=dataset_dir,
            session=session or photo_session,
            strict=strict,
        )
        if not result["ingested"]:
            rejected += 1
            reasons.append(f"{photo.name}: {result['reason']}")
            continue
        if result["path"] in known:
            duplicates += 1
            continue
        known.add(result["path"])
        manifest["samples"].append(
            {
                "label": result["label"],
                "session": result["session"],
                "split": result["split"],
                "path": result["path"],
            }
        )
        ingested += 1

    manifest["sessions"] = sorted({sample["session"] for sample in manifest["samples"]})
    _write_manifest(dataset_dir, manifest)
    return {
        "ingested": ingested,
        "rejected": rejected,
        "duplicates": duplicates,
        "reasons": reasons,
        "total_samples": len(manifest["samples"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saafscan.ingest",
        description="Ingest real strip photographs into a training dataset.",
    )
    parser.add_argument("directory", type=Path, help="folder of shots, named S001_none_0.jpg")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--label", default=None, help="override the label parsed from filenames")
    parser.add_argument(
        "--session", default=None, help="override the session parsed from filenames"
    )
    args = parser.parse_args(argv)

    try:
        summary = ingest_directory(
            args.directory, args.dataset, label=args.label, session=args.session
        )
    except IngestError as exc:
        print(f"ingest: {exc}", file=sys.stderr)
        return 2

    print(
        f"{summary['ingested']} taken in, {summary['rejected']} refused, "
        f"{summary['duplicates']} already present -> {summary['total_samples']} samples"
    )
    for reason in summary["reasons"]:
        print(f"  refused {reason}", file=sys.stderr)
    if summary["rejected"] and not summary["ingested"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
