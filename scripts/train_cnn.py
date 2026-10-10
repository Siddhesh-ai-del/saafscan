#!/usr/bin/env python
"""Train Model B (ResNet18 finetune, CUDA AMP) and export a CPU-servable ONNX.

Reproduce from scratch (CUDA environment — torch/torchvision, no OpenCV):

    <cuda-env-python> scripts/train_cnn.py
    .venv/bin/python -m saafscan.ml.eval --dataset data/trainset \
        --models models --out models/eval_report.json

Dataset: rendered by scripts/train_features.py into data/trainset, split BY
LIGHTING CONDITION in the manifest (never random images). Preprocessing is the
shared NumPy-only saafscan.ml.preprocess.strip_to_chw, so training and serving
see byte-identical tensors; calibration (temperature + abstention) and the
held-out metrics come from saafscan.ml.calibration — the exact code Model A
uses — so the delta report compares like with like.

GPU is training-only: the exported graph (softmax in-graph, single float
"probabilities" output, dynamic batch axis) is served CPU-only by onnxruntime,
so the demo never depends on CUDA. --cpu runs everywhere but is slow (smoke).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Allow running as a plain script without installing the package. .mltools holds
# the onnx package used only for the export gate (checker + reference run).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_MLTOOLS = _ROOT / ".mltools"
if _MLTOOLS.is_dir():
    sys.path.insert(0, str(_MLTOOLS))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:  # pragma: no cover - depends on the interpreter
    raise SystemExit(
        "Model B training needs the CUDA environment (torch + torchvision).\n"
        "Run instead:  <cuda-env-python> scripts/train_cnn.py"
    ) from exc

from saafscan.ml.calibration import (  # noqa: E402
    abstention_metrics,
    accuracy_and_nll,
    fit_abstention_policy,
)
from saafscan.ml.preprocess import strip_to_chw  # noqa: E402

# Serving contract (runtime.CLASSES / TRAINING_CLASSES): a manifest with a
# different class order would silently mislabel every exported probability row.
EXPECTED_CLASSES = ("none", "starch", "detergent", "both", "inconclusive")
SPLITS = ("train", "val", "test")


def load_split(
    dataset_dir: Path, manifest: dict, split: str, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack one split's canonical crops (3, 90, 360) float32 and class indices."""
    xs, ys = [], []
    for sample in manifest["samples"]:
        if sample["split"] != split:
            continue
        crop = np.load(dataset_dir / sample["path"])  # (90, 360, 3) uint8
        xs.append(strip_to_chw(crop))  # identical transform to the serving path
        ys.append(classes.index(sample["label"]))
    if not xs:
        return np.empty((0, 3, 90, 360), np.float32), np.empty(0, np.int64)
    return np.stack(xs), np.asarray(ys, np.int64)


def build_model(n_classes: int) -> nn.Module:
    """ImageNet-pretrained ResNet18 with a fresh classification head.

    The stem and adaptive average pool accept the native 360×90 crop directly —
    no resize, so no interpolation mismatch between training and serving.
    """
    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


@torch.no_grad()
def predict_probs(
    net: nn.Module, x: np.ndarray, device: torch.device, batch: int, use_amp: bool
) -> np.ndarray:
    """Softmax probabilities for a (n, 3, 90, 360) array; empty input → (0, C)."""
    net.eval()
    rows = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch])).to(device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = net(xb)
        rows.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if not rows:
        return np.empty((0, net.fc.out_features), np.float32)
    return np.concatenate(rows)


def _snapshot(net: nn.Module) -> dict[str, torch.Tensor]:
    """Detach a CPU copy of the weights (best-checkpoint bookkeeping)."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _train_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    """One pass over the training split; returns mean cross-entropy loss."""
    net.train()
    loss_sum, seen = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss = criterion(net(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(yb)
        seen += len(yb)
    return loss_sum / max(seen, 1)


def fit(
    net: nn.Module,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    """Finetune with AMP; keep the best-val epoch (accuracy, then NLL tie-break).

    Discriminative learning rates: a fresh head learns fast, the pretrained
    backbone moves slowly. Selection is on the val split only — the held-out
    lighting-condition test split is never touched during training.
    """
    x_train, y_train = train
    x_val, y_val = val
    use_amp = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in net.named_parameters() if not n.startswith("fc.")],
                "lr": lr,
            },
            {"params": net.fc.parameters(), "lr": lr * 10},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    best_score = (-1.0, -float("inf"))  # (val_acc, -val_nll): max acc, then calibration
    best_state = _snapshot(net)
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(net, loader, optimizer, scaler, criterion, device, use_amp)
        scheduler.step()
        probs = predict_probs(net, x_val, device, batch, use_amp)
        val_acc, val_nll = accuracy_and_nll(probs, y_val)
        if (val_acc, -val_nll) > best_score:
            best_score, best_state = (val_acc, -val_nll), _snapshot(net)
        print(
            f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_nll:.4f}  val_acc={val_acc:.3f}"
        )
    net.load_state_dict(best_state)
    return {"val_acc": best_score[0]}


class _SoftmaxWrapper(nn.Module):
    """Put softmax INSIDE the exported graph.

    The served tensor is then already a probability row: applying the fitted
    temperature to it in numpy is exactly the calibration maths the spec asks
    for, and no consumer can mistake raw logits for a confidence.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


def export_onnx(net: nn.Module, path: Path, input_shape: tuple[int, ...]) -> None:
    """Export opset 17, single float output, dynamic batch — then gate it.

    The gate fails the script if the graph has an extra/int output or if a
    reference run does not produce rows summing to 1, so a broken artifact
    never reaches models/.
    """
    import onnx
    from onnx.reference import ReferenceEvaluator

    net = net.cpu().eval()
    dummy = torch.zeros((1, *input_shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _SoftmaxWrapper(net),
            dummy,
            str(path),
            opset_version=17,
            dynamo=False,  # legacy exporter: no onnxscript dependency
            input_names=["input"],
            output_names=["probabilities"],
            dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    outputs = list(graph.graph.output)
    if len(outputs) != 1 or outputs[0].name != "probabilities":
        raise RuntimeError(f"expected one 'probabilities' output, got {[o.name for o in outputs]}")
    if outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("probability output must be float32 — never int labels")
    (row,) = ReferenceEvaluator(graph).run(None, {"input": dummy.numpy()})
    if not np.allclose(row.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"exported rows do not sum to 1: {row.sum(axis=1)}")


def _calibrate(
    probs: dict[str, np.ndarray],
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    classes: list[str],
) -> tuple[float, float, dict]:
    """Fit (T, tau) on val only, score the held-out-lighting test split."""
    temperature, tau = fit_abstention_policy(probs["val"], splits["val"][1], len(classes))
    held = abstention_metrics(
        probs["test"],
        splits["test"][1],
        inconclusive_index=classes.index("inconclusive"),
        temperature=temperature,
        threshold=tau,
    )
    return temperature, tau, held


def _print_summary(spec: dict, classes: list[str], seconds: float) -> None:
    held = spec["metrics"]["held_out_lighting"]
    print(
        f"trained {spec['name']} on {spec['device']} in {seconds:.1f}s "
        f"(n_train={spec['n_train']}, T={spec['temperature']:.3f}, "
        f"tau={spec['abstain_threshold']:.3f})"
    )
    print(
        f"held-out lighting: acc={held['accuracy']:.3f} "
        f"coverage={held['coverage']:.3f} abstained={held['abstained']}/{held['n']}"
    )
    if held["confusion"]:
        print("confusion (rows=true, cols=pred):")
        for label, row in zip(classes, held["confusion"], strict=True):
            print(f"  {label:>13}: {row}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", default="data/trainset", help="dataset dir (train_features.py)"
    )
    parser.add_argument("--models", default="models", help="output models dir")
    parser.add_argument("--name", default="cnn", help="artifact basename")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="backbone lr (head is 10x)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — train Model B on a CUDA machine or pass --cpu.")
    device = torch.device(args.device)
    if device.type == "cpu":
        print("WARNING: training on CPU (slow) — results are for smoke testing only.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_dir = Path(args.dataset)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    classes = list(manifest["classes"])
    if tuple(classes) != EXPECTED_CLASSES:
        raise SystemExit(
            f"manifest class order {classes} != serving order {list(EXPECTED_CLASSES)}"
        )
    splits = {name: load_split(dataset_dir, manifest, name, classes) for name in SPLITS}
    print(
        "split sizes (by lighting condition, never random): "
        + ", ".join(f"{name}={len(splits[name][1])}" for name in SPLITS)
    )

    started = time.time()
    net = build_model(len(classes)).to(device)
    fit(
        net,
        splits["train"],
        splits["val"],
        device,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
    use_amp = device.type == "cuda"
    probs = {
        name: predict_probs(net, splits[name][0], device, args.batch, use_amp) for name in SPLITS
    }
    temperature, tau, held = _calibrate(probs, splits, classes)

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / f"{args.name}.onnx"
    export_onnx(net, onnx_path, tuple(splits["train"][0].shape[1:]))
    spec = {
        "name": args.name,
        "kind": "cnn",
        "backend": "torch",
        "model": f"resnet18 (ImageNet pretrained) finetune, AMP={use_amp}",
        "classes": classes,
        "temperature": temperature,
        "abstain_threshold": tau,
        "n_train": int(len(splits["train"][1])),
        "n_val": int(len(splits["val"][1])),
        "n_test": int(len(splits["test"][1])),
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "device": str(device),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": {"held_out_lighting": held},
    }
    spec_path = models_dir / f"{args.name}.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    _print_summary(spec, classes, time.time() - started)
    print(f"artifacts: {onnx_path} + {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
