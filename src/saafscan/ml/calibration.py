"""Calibration primitives: temperature scaling, abstention, confusion matrix.

Pure NumPy, deterministic, no model loading. These are the honesty layer of the
ML swap: scores are temperature-calibrated so a reported confidence of 0.9 means
90% empirical accuracy, and inputs that stay ambiguous after calibration abstain
to ``inconclusive`` rather than forcing a guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability distribution by ``temperature``.

    ``T > 1`` softens an overconfident model; ``T < 1`` sharpens an
    under-confident one. Implemented on log-probabilities so the result is a
    proper normalised distribution regardless of input shape.

    Args:
        probs: Array of shape (..., n_classes), each row summing to ~1.
        temperature: Strictly positive scaling factor.

    Returns:
        Rescaled probabilities, same shape, each row summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Fit a single scalar temperature by minimising negative log-likelihood.

    Uses golden-section search over ``T`` on a fixed bracket, which is
    deterministic and needs no external optimiser. Returns a temperature in
    [0.1, 10.0]; falls back to 1.0 (identity) if the search degenerates.

    Args:
        probs: Shape (n_samples, n_classes) of uncalibrated probabilities.
        labels: Shape (n_samples,) of true class indices.
        n_classes: Number of classes.

    Returns:
        The fitted temperature scalar.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(f"probs must be (n, {n_classes}), got {probs.shape}")
    if len(labels) != len(probs):
        raise ValueError("probs and labels length mismatch")

    clipped = np.clip(probs, 1e-12, 1.0)
    log_probs = np.log(clipped)
    target = log_probs[np.arange(len(labels)), labels]

    def nll(temperature: float) -> float:
        scaled = log_probs / temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        z = np.exp(scaled).sum(axis=1)
        return float(-np.mean(target / temperature - np.log(z)))

    lo, hi = 0.1, 10.0
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = nll(d)
    best = (a + b) / 2.0
    if not np.isfinite(best) or best < lo or best > hi:
        return 1.0
    return float(best)


def select_abstain_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    min_precision: float = 1.0,
) -> float:
    """Choose the lowest confidence threshold meeting a precision target.

    Scans candidate thresholds (the observed max-probability values) and picks
    the one that accepts the most samples while every accepted prediction is
    correct, up to ``min_precision``. This is the abstention contract: below
    the returned threshold the wrapper must return ``inconclusive``.

    Args:
        probs: Shape (n_samples, n_classes), already temperature-calibrated.
        labels: Shape (n_samples,) true class indices.
        n_classes: Number of classes.
        min_precision: Required fraction of accepted predictions that are correct.

    Returns:
        Threshold in (0, 1]; 1.0 when no threshold accepts anything.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(f"probs must be (n, {n_classes}), got {probs.shape}")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    candidates = np.unique(conf)
    best_tau, best_accepted = 1.0, 0
    for tau in candidates:
        accepted = conf >= tau
        n_acc = int(accepted.sum())
        if n_acc == 0:
            continue
        precision = float(correct[accepted].mean())
        if precision >= min_precision and n_acc > best_accepted:
            best_accepted = n_acc
            best_tau = float(tau)
    return float(np.clip(best_tau, 1e-6, 1.0))


def fit_abstention_policy(
    probs: np.ndarray | None,
    labels: np.ndarray | None,
    n_classes: int,
    *,
    min_precision: float = 0.98,
    floor: float = 0.55,
) -> tuple[float, float]:
    """Fit the shipped ``(temperature, abstain_threshold)`` pair on a calib split.

    One policy for every trainer (features and CNN): temperature by
    minimum-NLL golden-section search, then the least restrictive threshold
    reaching ``min_precision`` on the calibration split, clipped to
    ``[floor, 0.999]`` — a verdict on a coin flip is not a verdict, and a
    threshold of 1.0 would abstain on everything.

    Args:
        probs: Raw model probabilities, shape (n, n_classes), or None.
        labels: True class indices, or None/empty when nothing to calibrate.
        n_classes: Class count.
        min_precision: Target precision among accepted predictions.
        floor: Policy minimum for the abstain threshold.

    Returns:
        ``(temperature, abstain_threshold)``. With no calibration data:
        ``(1.0, floor)`` — nothing was fitted, and the fallback says so.
    """
    if probs is None or labels is None or len(labels) == 0:
        return 1.0, float(floor)
    raw = np.asarray(probs, dtype=np.float64)
    temperature = fit_temperature(raw, labels, n_classes)
    threshold = select_abstain_threshold(
        apply_temperature(raw, temperature), labels, n_classes, min_precision=min_precision
    )
    return float(temperature), float(np.clip(threshold, floor, 0.999))


def predict_with_abstention(
    probs: np.ndarray, threshold: float, inconclusive_index: int
) -> np.ndarray:
    """Map calibrated probabilities to class indices under the serving rule.

    Mirrors ``runtime.verdict_from_probs`` exactly: the argmax is issued as a
    verdict only when its confidence reaches ``threshold`` and the top class is
    not the inconclusive class; otherwise the sample is assigned
    ``inconclusive_index`` (an abstention). Keeping one rule for serving and
    evaluation is what makes the reported coverage a truthful description of
    what the demo does.

    Args:
        probs: Shape (n_samples, n_classes), temperature-calibrated.
        threshold: Abstention threshold in (0, 1].
        inconclusive_index: Column index of the inconclusive class, or -1 when
            the label set has no inconclusive class (threshold-only abstention).

    Returns:
        Shape (n_samples,) int64 class indices; abstentions carry
        ``inconclusive_index``.

    Raises:
        ValueError: If ``probs`` is not 2-D, or ``inconclusive_index`` is not
            -1 or a valid column index.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, n_classes), got shape {probs.shape}")
    n_classes = probs.shape[1]
    if not -1 <= inconclusive_index < n_classes:
        raise ValueError(
            f"inconclusive_index {inconclusive_index} outside valid range [-1, {n_classes - 1}]"
        )
    if probs.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    accepted = (conf >= threshold) & (pred != inconclusive_index)
    return np.where(accepted, pred, inconclusive_index).astype(np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[list[int]]:
    """Row-normalised-free count confusion matrix, rows = true, cols = predicted.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        n_classes: Number of classes.

    Returns:
        Nested list ``cm[i][j]`` = count of true-class-``i`` predicted as ``j``.

    Raises:
        ValueError: If any index is outside ``[0, n_classes)`` — a negative
            prediction would silently land in the wrong column via Python's
            wrap-around indexing.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
            raise ValueError(
                f"{name} indices must be in [0, {n_classes}), "
                f"got range [{int(arr.min())}, {int(arr.max())}]"
            )
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t)][int(p)] += 1
    return cm


def accuracy_and_nll(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean accuracy and negative log-likelihood of a probability matrix.

    The checkpoint pair used for model selection (see ``scripts/train_cnn.py``):
    pick the highest accuracy, break ties with the lowest NLL, so the epoch
    that sharpens the *right* answers wins over one that merely repeats them.

    Args:
        probs: Shape (n, n_classes), raw or temperature-calibrated.
        labels: Shape (n,) true class indices.

    Returns:
        ``(accuracy, nll)``; ``(0.0, inf)`` on an empty split — no evidence
        beats every real score once inverted, and never NaN (a NaN would
        silently poison checkpoint comparisons and JSON reports).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return 0.0, float("inf")
    acc = float((probs.argmax(axis=1) == labels).mean())
    picked = np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1.0)
    return acc, float(-np.log(picked).mean())


def abstention_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    inconclusive_index: int,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    """Held-out accuracy/coverage under the exact serving abstention rule.

    Temperature-scales ``probs`` and applies :func:`predict_with_abstention`,
    then reports accuracy over ALL samples — an abstention counts as a
    prediction of the inconclusive class, which is right only when that is
    the truth — plus coverage (fraction receiving a real verdict) and the
    confusion matrix. Both trainers and the delta report use this function,
    so their numbers are computed by identical code.

    Args:
        probs: Raw model probabilities, shape (n, n_classes).
        labels: True class indices, shape (n,).
        inconclusive_index: Column index of the inconclusive class (-1 if the
            label set has none; such abstentions are excluded from the
            confusion matrix, which has no column for them).
        temperature: Fitted temperature to apply first.
        threshold: Abstain threshold.

    Returns:
        ``{"n", "accuracy", "coverage", "abstained", "confusion"}`` in plain
        Python types (JSON-serialisable); zeros, never NaN, on an empty split.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "coverage": 0.0, "abstained": 0, "confusion": []}
    preds = predict_with_abstention(
        apply_temperature(probs, temperature), threshold, inconclusive_index
    )
    # Accepted = a real food verdict was issued; representable = the confusion
    # matrix has a column for the prediction (always true when the label set
    # has an inconclusive class — abstentions then land in its column).
    accepted = preds >= 0 if inconclusive_index < 0 else preds != inconclusive_index
    representable = preds >= 0
    return {
        "n": int(n),
        "accuracy": float((preds == labels).mean()),
        "coverage": float(accepted.mean()),
        "abstained": int((~accepted).sum()),
        "confusion": (
            confusion_matrix(labels[representable], preds[representable], probs.shape[1])
            if representable.any()
            else []
        ),
    }


def accuracy_at_threshold(
    probs: np.ndarray, labels: np.ndarray, n_classes: int, threshold: float
) -> dict[str, float]:
    """Accuracy and coverage when abstaining below ``threshold``.

    Args:
        probs: Shape (n_samples, n_classes), calibrated.
        labels: True class indices.
        n_classes: Number of classes.
        threshold: Abstention threshold.

    Returns:
        ``{"coverage": accepted/n, "accuracy": correct/accepted}``; accuracy is
        0.0 when nothing is accepted.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    accepted = conf >= threshold
    n = len(labels)
    n_acc = int(accepted.sum())
    coverage = float(n_acc / n) if n else 0.0
    accuracy = float((pred[accepted] == labels[accepted]).mean()) if n_acc else 0.0
    return {"coverage": coverage, "accuracy": accuracy}
