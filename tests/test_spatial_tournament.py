from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_spatial_paid_state import evaluate, mask_for


def frame():
    return pd.DataFrame({
        "target": [0.10, 0.40, 0.60, 0.80, 0.95, 0.50],
        "lag1_availability": [0.30, 0.55, 0.62, 0.82, 0.94, 0.90],
        "slot_start": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
    })


def test_regime_masks_partition_availability():
    df = frame()
    masks = [mask_for(df, n) for n in ("0_30", "30_50", "50_70", "70_90", "90_100")]
    assert np.all(np.sum(np.vstack(masks), axis=0) == 1)


def test_transition_mask_is_based_on_prior_observation():
    df = frame()
    mask = mask_for(df, "transition")
    assert mask.tolist() == [True, True, False, False, False, True]


def test_evaluate_returns_overall_and_difficult_regimes():
    df = frame()
    result = evaluate(df, df.target.to_numpy())
    assert result["overall"]["mae"] == 0.0
    assert result["overall"]["rmse"] == 0.0
    assert result["n"] == len(df)
    assert result["0_30"]["n"] == 1
    assert result["transition"]["n"] == 3
