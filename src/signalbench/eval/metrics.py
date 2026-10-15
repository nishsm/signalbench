"""Classification statistics with a fixed three-class label space."""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from signalbench import LABELS


def metrics(y, pred) -> dict:
    """Compute accuracy, macro-F1 and per-class precision/recall."""
    precision, recall, _, support = precision_recall_fscore_support(
        y, pred, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "classes": {
            label: {"precision": p, "recall": r, "support": int(n)}
            for label, p, r, n in zip(LABELS, precision, recall, support, strict=True)
        },
    }


def bootstrap(y, pred, seed: int = 42, n: int = 1000) -> dict:
    """Paired, class-stratified bootstrap percentile intervals."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in LABELS]
    draws = []
    for _ in range(n):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        m = metrics(y[ids], pred[ids])
        draws.append([m["accuracy"], m["macro_f1"]])
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        f"{k}_{bound}": float(ci[i, j])
        for j, k in enumerate(["accuracy", "macro_f1"])
        for i, bound in enumerate(["low", "high"])
    }


def mcnemar(y, a, b) -> dict:
    """Exact two-sided McNemar test on paired predictions."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    ab = int(np.sum((a == y) & (b != y)))
    ba = int(np.sum((b == y) & (a != y)))
    return {
        "a_only_correct": ab,
        "b_only_correct": ba,
        "p_value": float(binomtest(ab, ab + ba).pvalue) if ab + ba else 1.0,
    }


def calibration(y, probabilities, bins: int = 10) -> tuple[float, list[dict]]:
    """Equal-width ECE; probabilities must follow LABELS order."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative N x 3 probabilities")
    if not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probabilities must sum to one")
    confidence = p.max(axis=1)
    correct = np.asarray(LABELS)[p.argmax(axis=1)] == np.asarray(y)
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for i in range(bins):
        mask = bucket == i
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            rows.append({"confidence": conf, "accuracy": acc, "count": int(mask.sum())})
    return float(ece), rows
