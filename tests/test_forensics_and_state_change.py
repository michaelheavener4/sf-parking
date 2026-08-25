import numpy as np

from scripts.forensic_paid_state import exact_hour_state_audit, quantiles
from scripts.train_state_change import metric, transition
from scripts.train_event_driven import prior_correct_probability, binary_metrics


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def run(self, sql, **kwargs):
        self.queries.append(sql)
        return self.rows


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


def test_prior_correction_moves_oversampled_probability_toward_natural_rate():
    raw = np.array([0.5])
    corrected = prior_correct_probability(raw, sample_prior=0.20, natural_prior=0.01)
    assert corrected[0] < raw[0]
    assert 0 < corrected[0] < 1


def test_binary_metrics_counts_events():
    result = binary_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_exact_hour_forensics_requires_t_minus_one_hour_pair():
    rows = [
        ("A", "2026-08-20 10:00:00", 0.60, 2, "2026-08-20", 10,
         "2026-08-20 09:00:00", 0.55, 1),
        ("A", "2026-08-20 12:00:00", 0.20, 4, "2026-08-20", 12,
         "2026-08-20 11:00:00", 0.30, 3),
    ]
    conn = FakeConn(rows)
    result = exact_hour_state_audit(conn, "2026-08-20 00:00:00", "2026-08-20 23:00:00")
    assert result["n_exact_hour_pairs"] == 2
    assert result["transition_counts"]["0.1"] == 1
    assert result["transition_counts"]["0.25"] == 0
    assert "slot_start - INTERVAL '1 hour'" in conn.queries[0]
    assert "lag(" not in conn.queries[0].lower()
