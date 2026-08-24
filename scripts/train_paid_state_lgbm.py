"""Train the first LightGBM forecaster on materialized paid-state history."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date

import pandas as pd

from sf_parking.database import connect

FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "current_availability", "transaction_count"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    print("🌉 SF PARKING — FIRST LIGHTGBM PAID-STATE FORECAST")
    print("[1/5] Loading the materialized hourly paid-state table.")
    conn = connect()
    try:
        bounds = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly")[0]
        if bounds[0] is None or bounds[1] is None:
            raise SystemExit("parking_state_hourly is empty; run build_hourly_state.py first")
        print(f"      Earliest state: {bounds[0]}")
        print(f"      Latest state:   {bounds[1]}")

        print("[2/5] Building a one-hour-ahead training table.")
        print("      Missing next-hour state is treated as zero paid occupancy probability.")
        rows = conn.run("""
            WITH base AS (
                SELECT
                    s.post_id,
                    s.slot_start,
                    s.local_hour,
                    EXTRACT(ISODOW FROM s.local_date)::int AS iso_weekday,
                    s.transaction_count,
                    s.paid_availability_probability AS current_availability,
                    COALESCE(n.paid_availability_probability, 1.0) AS target_availability
                FROM parking_state_hourly s
                LEFT JOIN parking_state_hourly n
                  ON n.post_id = s.post_id
                 AND n.slot_start = s.slot_start + INTERVAL '1 hour'
            )
            SELECT *,
                   SIN(2 * PI() * local_hour / 24.0) AS hour_sin,
                   COS(2 * PI() * local_hour / 24.0) AS hour_cos,
                   SIN(2 * PI() * (iso_weekday - 1) / 7.0) AS weekday_sin,
                   COS(2 * PI() * (iso_weekday - 1) / 7.0) AS weekday_cos
            FROM base
            ORDER BY slot_start, post_id
            LIMIT :limit_rows
        """, limit_rows=args.max_rows)
    finally:
        conn.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "post_id", "slot_start", "local_hour", "iso_weekday",
            "transaction_count", "current_availability", "target_availability",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        ],
    )
    if df.empty:
        raise SystemExit("No training rows were produced")

    df["slot_start"] = pd.to_datetime(df["slot_start"], utc=True)
    latest = df["slot_start"].max()
    test_cut = latest - pd.Timedelta(days=args.test_days)
    val_cut = test_cut - pd.Timedelta(days=args.validation_days)

    train = df[df.slot_start < val_cut]
    validation = df[(df.slot_start >= val_cut) & (df.slot_start < test_cut)]
    test = df[df.slot_start >= test_cut]

    print(f"[3/5] Chronological split.")
    print(f"      Train:      {len(train):,} rows")
    print(f"      Validation: {len(validation):,} rows")
    print(f"      Test:       {len(test):,} rows")
    if train.empty or validation.empty or test.empty:
        raise SystemExit("Chronological split produced an empty partition")

    print("[4/5] Training LightGBM.")
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        train[FEATURES],
        (train.target_availability >= 0.5).astype(int),
        eval_set=[(validation[FEATURES], (validation.target_availability >= 0.5).astype(int))],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    print("[5/5] Evaluating the held-out future period.")
    pred = model.predict_proba(test[FEATURES])[:, 1]
    y = (test.target_availability >= 0.5).astype(int).to_numpy()
    brier = float(((pred - y) ** 2).mean())
    mae = float(abs(pred - y).mean())
    result = {
        "rows": len(df),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "test_start": test.slot_start.min().isoformat(),
        "test_end": test.slot_start.max().isoformat(),
        "brier": round(brier, 6),
        "mae_binary_target": round(mae, 6),
        "positive_rate": round(float(y.mean()), 6),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators_)),
    }
    print(json.dumps(result, indent=2))
    print("✅ COMPLETE")
    try:
        subprocess.run(["pbcopy"], input=json.dumps(result, indent=2), text=True, check=True)
        print("📋 Results copied to macOS clipboard.")
    except (OSError, subprocess.CalledProcessError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
