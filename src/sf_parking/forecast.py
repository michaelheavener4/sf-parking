"""Leakage-safe supervised forecasting primitives for parking state.

The primary production target is a future paid-occupancy probability. The
module supports an optional LightGBM implementation for serious training and
ships a dependency-free deterministic logistic model for tests/smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, isnan
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ForecastRow:
    timestamp: datetime
    post_id: str
    target: float
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: tuple[ForecastRow, ...]
    validation: tuple[ForecastRow, ...]
    test: tuple[ForecastRow, ...]


def temporal_split(
    rows: Iterable[ForecastRow],
    *,
    train_until: datetime,
    validation_until: datetime,
) -> TemporalSplit:
    """Split strictly by time; no random shuffling is permitted."""
    ordered = sorted(rows, key=lambda r: (r.timestamp, r.post_id))
    train = tuple(r for r in ordered if r.timestamp < train_until)
    validation = tuple(
        r for r in ordered if train_until <= r.timestamp < validation_until
    )
    test = tuple(r for r in ordered if r.timestamp >= validation_until)
    if not train:
        raise ValueError("training split is empty")
    if not validation:
        raise ValueError("validation split is empty")
    if not test:
        raise ValueError("test split is empty")
    return TemporalSplit(train, validation, test)


def assert_no_future_leakage(rows: Sequence[ForecastRow], *, cutoff: datetime) -> None:
    if any(r.timestamp >= cutoff for r in rows):
        raise ValueError("forecast rows contain data at or after the cutoff")


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    ys, ps = list(y_true), list(y_prob)
    if len(ys) != len(ps) or not ys:
        raise ValueError("y_true and y_prob must have equal non-zero length")
    return sum((p - y) ** 2 for y, p in zip(ys, ps)) / len(ys)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    ys, ps = list(y_true), list(y_pred)
    if len(ys) != len(ps) or not ys:
        raise ValueError("y_true and y_pred must have equal non-zero length")
    return sum(abs(p - y) for y, p in zip(ys, ps)) / len(ys)


def _clean_feature(x: float) -> float:
    return 0.0 if isnan(x) else x


class LogisticFallback:
    """Small dependency-free logistic regression for smoke tests and fallback use."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 250, l2: float = 1e-3) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = exp(-x)
            return 1.0 / (1.0 + z)
        z = exp(x)
        return z / (1.0 + z)

    def fit(self, rows: Sequence[ForecastRow]) -> "LogisticFallback":
        if not rows:
            raise ValueError("cannot train on empty rows")
        width = len(rows[0].features)
        if width == 0 or any(len(r.features) != width for r in rows):
            raise ValueError("feature width mismatch")
        self.weights = [0.0] * width
        self.bias = 0.0
        n = float(len(rows))
        for _ in range(self.epochs):
            grad_w = [0.0] * width
            grad_b = 0.0
            for row in rows:
                z = self.bias + sum(
                    w * _clean_feature(x)
                    for w, x in zip(self.weights, row.features)
                )
                p = self._sigmoid(z)
                err = p - row.target
                grad_b += err
                for i, x in enumerate(row.features):
                    grad_w[i] += err * _clean_feature(x)
            self.bias -= self.learning_rate * grad_b / n
            for i in range(width):
                self.weights[i] -= self.learning_rate * (
                    (grad_w[i] / n) + self.l2 * self.weights[i]
                )
        return self

    def predict_proba(self, rows: Sequence[ForecastRow]) -> list[float]:
        if not self.weights:
            raise RuntimeError("model is not fitted")
        return [
            self._sigmoid(
                self.bias
                + sum(w * _clean_feature(x) for w, x in zip(self.weights, row.features))
            )
            for row in rows
        ]


class LightGBMForecastModel:
    """Optional high-performance tree forecaster."""

    def __init__(self, **params: object) -> None:
        self.params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 42,
            **params,
        }
        self._model = None

    def fit(self, rows: Sequence[ForecastRow]) -> "LightGBMForecastModel":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is optional; install requirements-ml.txt to use the production forecaster"
            ) from exc
        if not rows:
            raise ValueError("cannot train on empty rows")
        x = [list(r.features) for r in rows]
        y = [r.target for r in rows]
        dataset = lgb.Dataset(x, label=y, free_raw_data=True)
        self._model = lgb.train(self.params, dataset, num_boost_round=250)
        return self

    def predict_proba(self, rows: Sequence[ForecastRow]) -> list[float]:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return [
            float(min(1.0, max(0.0, p)))
            for p in self._model.predict([list(r.features) for r in rows])
        ]
