#!/usr/bin/env python3
"""Train the first genuinely ground-truth-calibrated parking model.

Target: SFpark sensor-measured hourly occupancy (historical 2011-2013).
Baseline: exact one-hour persistence of measured occupancy.
Features: only information available before the target hour:
  - prior measured occupancy
  - prior payment-session starts
  - prior rolling payment intensity
  - prior same-day-slot payment intensity
  - rate / rate type
  - hour-of-day / day-of-week

This is deliberately a calibration experiment, not the production model. Its
job is to establish whether transaction behavior contains information beyond
physical occupancy persistence. Once that is established, the same feature
contract can be reused against modern data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "fused_sensor_calibration_v1.json"

FEATURES = [
    "lag_occupancy_total",
    "lag_payment_session_starts",
    "prior_3h_mean_payment_starts",
    "prior_same_slot_payment_mean",
    "rate",
    "hour_of_day",
    "dow",
]


def metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
    }


def load(conn, max_rows: int):
    return conn.run(
        """
        WITH x AS (
            SELECT
                local_hour,
                street_block,
                occupancy_total,
                lag_occupancy_total,
                lag_payment_session_starts,
                prior_3h_mean_payment_starts,
                prior_same_slot_payment_mean,
                rate,
                EXTRACT(HOUR FROM local_hour)::int AS hour_of_day,
                EXTRACT(ISODOW FROM local_hour)::int AS dow
            FROM v_fusion_historical_calibration_hourly
        )
        SELECT * FROM x
        WHERE occupancy_total IS NOT NULL
          AND lag_occupancy_total IS NOT NULL
        ORDER BY local_hour, street_block
        LIMIT :k
        """,
        k=max_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    c = connect()
    try:
        rows = load(c, args.max_rows)
    finally:
        c.close()

    if len(rows) < 1000:
        raise RuntimeError(
            f"Only {len(rows):,} usable calibration rows found. "
            "Import both SFpark sensor-hourly and smart-payment historical CSVs first."
        )

    arr = np.asarray(rows, dtype=object)
    # columns: local_hour, street_block, y, lag_y, lag_pay, roll_pay, slot_pay, rate, hour, dow
    n = len(arr)
    cut = max(int(n * (1.0 - args.test_fraction)), 1)
    y = arr[:, 2].astype(float)
    persistence = arr[:, 3].astype(float)

    x = np.column_stack([
        np.nan_to_num(arr[:, 3].astype(float), nan=0.5),
        np.nan_to_num(arr[:, 4].astype(float), nan=0.0),
        np.nan_to_num(arr[:, 5].astype(float), nan=0.0),
        np.nan_to_num(arr[:, 6].astype(float), nan=0.0),
        np.nan_to_num(arr[:, 7].astype(float), nan=0.0),
        arr[:, 8].astype(int),
        arr[:, 9].astype(int),
    ])

    # Preserve chronological evaluation: train precedes test.
    x_train, x_test = x[:cut], x[cut:]
    y_train, y_test = y[:cut], y[cut:]
    p_test = persistence[cut:]

    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="regression_l1",
            verbosity=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
            loss="absolute_error",
        )

    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 0.0, 1.0)
    pm = metric(y_test, p_test)
    mm = metric(y_test, pred)

    result = {
        "version": 1,
        "target": "SFpark sensor-measured hourly total occupancy",
        "features": FEATURES,
        "rows": {"total": n, "train": len(y_train), "test": len(y_test)},
        "chronology": "ordered by local_hour; test is strictly after train",
        "persistence": pm,
        "fused_model": mm,
        "improvement_over_persistence": (pm["mae"] - mm["mae"]) / pm["mae"] if pm["mae"] else None,
        "promotion": "candidate" if mm["mae"] < pm["mae"] else "retained_only",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
