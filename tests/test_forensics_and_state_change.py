from datetime import date

import numpy as np
import pandas as pd

from scripts.forensic_paid_state import quantiles
from scripts.train_state_change import metric, transition


def test_quantiles_are_finite_and_monotonic():
    q = quantiles(np.array([0.0, 0.1, 0.2, 0.5, 1.0]))
    vals = list(q.values())
    assert vals == sorted(vals)
    assert all(np.isfinite(vals))


def test_transition_metrics_uses_actual_delta_threshold():
    y = np.array([0.9, 0.7, 0.5, 0.8])
    lag = np.array([0.8, 0.7, 0.6, 0.5])
    pred = np.array([0.9, 0.8, 0.55, 0.6])
    r = transition(y, lag, pred, 0.10)
    assert r["n"] == 2
    assert 0 <= r["direction_accuracy"] <= 1


def test_metric_zero_error():
    y = np.array([0.1, 0.5, 0.9])
    assert metric(y, y) == {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
