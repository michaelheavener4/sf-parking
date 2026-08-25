"""Compatibility API for the authoritative V4 transition tournament.

Keep the historical helper functions importable because older tests and tools
use this module as a library, while command-line execution delegates to V4.
"""
from __future__ import annotations

import numpy as np


def prior_correct_probability(p, sample_prior, natural_prior):
    """Undo prior shift caused by deliberate rare-event oversampling."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    if not (0 < sample_prior < 1 and 0 < natural_prior < 1):
        return p
    odds = p / (1 - p)
    sample_odds = sample_prior / (1 - sample_prior)
    natural_odds = natural_prior / (1 - natural_prior)
    corrected = odds * (natural_odds / sample_odds)
    return np.clip(corrected / (1 + corrected), 1e-9, 1 - 1e-9)


def binary_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y), "positives": int(y.sum()), "tp": tp, "fp": fp,
        "fn": fn, "tn": tn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
    }


def transition_metrics(y, lag, pred, threshold):
    d = np.asarray(y, float) - np.asarray(lag, float)
    mask = np.abs(d) >= threshold - 1e-12
    if not mask.any():
        return {"n": 0, "mae": None, "direction_accuracy": None, "mean_abs_delta": None}
    pd = np.asarray(pred, float) - np.asarray(lag, float)
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(np.asarray(pred)[mask] - np.asarray(y)[mask]))),
        "direction_accuracy": float(np.mean((d[mask] > 0) == (pd[mask] > 0))),
        "mean_abs_delta": float(np.mean(np.abs(d[mask]))),
    }


def metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


try:
    from scripts.train_event_driven_v4 import main
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from train_event_driven_v4 import main


if __name__ == "__main__":
    raise SystemExit(main())
