import numpy as np
import pandas as pd

from scripts.train_spatial_paid_state import evaluate, target_diagnostics, transition_delta_metrics


def frame():
    return pd.DataFrame(
        {
            "target": [0.90, 0.82, 0.65, 0.40, 0.10],
            "lag1_availability": [0.90, 0.90, 0.80, 0.70, 0.40],
            "slot_start": pd.date_range("2026-08-24", periods=5, freq="h", tz="UTC"),
        }
    )


def test_target_diagnostics_exposes_transition_distribution():
    d = target_diagnostics(frame())
    assert d["n"] == 5
    assert d["fraction_abs_delta_ge_0_10"] == 0.6
    assert d["fraction_exactly_unchanged"] == 0.2
    assert d["mean_abs_delta"] > 0


def test_transition_metrics_measures_direction_and_mae():
    df = frame()
    pred = np.array([0.91, 0.80, 0.60, 0.45, 0.05])
    result = transition_delta_metrics(df, pred, 0.10)
    assert result["n"] == 3
    assert 0 <= result["direction_accuracy"] <= 1
    assert result["mae"] >= 0


def test_evaluate_contains_multiple_transition_regimes():
    df = frame()
    result = evaluate(df, df["lag1_availability"].to_numpy())
    assert result["transition_05"]["n"] >= result["transition_10"]["n"]
    assert result["transition_10"]["n"] >= result["transition_25"]["n"]
