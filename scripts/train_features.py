#!/usr/bin/env python
"""Build the training dataset (if needed) and train Model A (features → GBC → ONNX).

Reproduce from scratch:

    .venv/bin/python scripts/train_features.py
    .venv/bin/python -m saafscan.ml.eval --dataset data/trainset \
        --models models --out models/eval_report.json

Dataset: 36 lighting conditions × 5 classes × 3 seeds, split BY LIGHTING
CONDITION (condition.seed % 10), never random images. Renders are skipped if
the CV front-end cannot find the strip (counted in the manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saafscan.ml.train import export_dataset, train_features_model  # noqa: E402
from saafscan.synthetic import build_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/trainset", help="dataset dir")
    parser.add_argument("--models", default="models", help="output models dir")
    parser.add_argument("--name", default="features", help="artifact basename")
    parser.add_argument(
        "--force-render", action="store_true", help="re-render even if manifest exists"
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    manifest_file = dataset_dir / "manifest.json"
    if args.force_render or not manifest_file.is_file():
        t0 = time.time()
        manifest = build_dataset(dataset_dir)
        print(
            f"rendered {len(manifest['samples'])} samples "
            f"({manifest['skipped_undetectable']} skipped as undetectable) "
            f"in {time.time() - t0:.1f}s -> {manifest_file}"
        )
    else:
        manifest = json.loads(manifest_file.read_text())
        print(f"reusing existing dataset: {len(manifest['samples'])} samples ({manifest_file})")

    splits: dict[str, int] = {}
    for s in manifest["samples"]:
        splits[s["split"]] = splits.get(s["split"], 0) + 1
    print(f"split sizes: {splits} (by lighting condition, never random)")

    t0 = time.time()
    exported = export_dataset(dataset_dir)
    spec = train_features_model(exported, models_dir=Path(args.models), name=args.name)
    print(
        f"trained {args.name} in {time.time() - t0:.1f}s "
        f"(n_train={spec['n_train']}, T={spec['temperature']:.3f}, "
        f"tau={spec['abstain_threshold']:.3f})"
    )
    held = spec["metrics"].get("held_out_lighting") or {}
    if held:
        print(
            f"held-out lighting: acc={held['accuracy']:.3f} "
            f"coverage={held['coverage']:.3f} abstained={held['abstained']}/{held['n']}"
        )
    print(f"artifacts: {args.models}/{args.name}.onnx + {args.models}/{args.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
