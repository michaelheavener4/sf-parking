import numpy as np

from scripts.forensic_paid_state import quantiles
from scripts.train_state_change import metric, transition


def test_quantiles_are_finite_and_monotonic():
    q = quantiles(np.array([0.0, 0.1, 0.2, 0.5, 1.0]))
    vals = list(q.values())
    assert vals == sorted(vals)
    assert all(np.isfinite(vals))


def test_transition_metrics_uses_actual_delta_threshold():
    y = np.array([0.95, 0.70, 0.45, 0.80])
    lag = np.array([0.80, 0.70, 0.60, 0.50])
    pred = np.array([0.90, 0.80, 0.55, 0.60])
    r = transition(y, lag, pred, 0.10)
    assert r["n"] == 3
    assert 0 <= r["direction_accuracy"] <= 1


def test_metric_zero_error():
    y = np.array([0.1, 0.5, 0.9])
    assert metric(y, y) == {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
