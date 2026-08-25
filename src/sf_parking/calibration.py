"""Small, dependency-light probability calibration utilities.

Calibration is fitted only on held-out historical data.  The production
contract is deliberately simple: a score in [0,1] becomes an empirical
probability of a clearly-defined event such as availability >= threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: tuple[float, ...]
    y: tuple[float, ...]

    def predict(self, scores: np.ndarray | list[float]) -> np.ndarray:
        a = np.asarray(scores, dtype=float)
        if not self.x:
            return np.clip(a, 0.0, 1.0)
        return np.interp(a, np.asarray(self.x), np.asarray(self.y), left=self.y[0], right=self.y[-1])

    def to_dict(self) -> dict:
        return {"x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(tuple(float(v) for v in d["x"]), tuple(float(v) for v in d["y"]))


def fit_isotonic(scores: np.ndarray, events: np.ndarray) -> IsotonicCalibrator:
    """Fit monotone P(event|score) with the pool-adjacent-violators algorithm."""
    s = np.asarray(scores, dtype=float)
    e = np.asarray(events, dtype=float)
    mask = np.isfinite(s) & np.isfinite(e)
    s, e = s[mask], e[mask]
    if len(s) == 0:
        return IsotonicCalibrator((), ())
    order = np.argsort(s, kind="mergesort")
    s, e = s[order], e[order]

    # Collapse equal score values first.
    uniq, inverse = np.unique(s, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=e)
    means = sums / counts

    # PAVA blocks: start as singleton bins, merge whenever monotonicity fails.
    levels = means.tolist(); weights = counts.astype(float).tolist()
    starts = list(range(len(levels)))
    ends = list(range(len(levels)))
    i = 0
    while i < len(levels) - 1:
        if levels[i] <= levels[i + 1] + 1e-15:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        total_y = levels[i] * weights[i] + levels[i + 1] * weights[i + 1]
        levels[i] = total_y / total_w
        weights[i] = total_w
        ends[i] = ends[i + 1]
        del levels[i + 1]; del weights[i + 1]; del starts[i + 1]; del ends[i + 1]
        if i > 0:
            i -= 1
    fitted = np.empty(len(uniq), dtype=float)
    for level, start, end in zip(levels, starts, ends):
        fitted[start:end + 1] = level
    return IsotonicCalibrator(tuple(uniq.tolist()), tuple(np.clip(fitted, 0.0, 1.0).tolist()))


def save_calibrator(cal: IsotonicCalibrator, path: Path, *, event: str, training_window: tuple[str, str]) -> None:
    payload = {"version": "isotonic_v1", "event": event, "training_window": list(training_window), "calibrator": cal.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibrator(path: Path) -> IsotonicCalibrator:
    return IsotonicCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8"))["calibrator"])
