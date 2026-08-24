"""Train a leakage-safe LightGBM forecaster on materialized paid-state history.

The target is the paid-availability probability of the current completed hour.
At forecast time T, only states strictly before T are used as features. This is
an inferred paid-use target, not physical parking occupancy ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from datetime import timedelta

import pandas as pd

from sf_parking.database import connect

FEATURES = [
    "lag1_availability",
    "lag2_availability",
    "lag24_availability",
    "rolling3_availability",
    "rolling24_availability",
    "lag1_transactions",
    "lag24_transactions",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_ms",
]


def _heartbeat(message: str, stop: threading.Event) -> None:
    started = time.monotonic()
    faces = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌")
    i = 0
    while not stop.wait(5):
        elapsed = int(time.monotonic() - started)
        print(
            f"{faces[i % len(faces)]} Still working after "
            f"{elapsed // 60:02d}:{elapsed % 60:02d}. {message}",
            flush=True,
        )
        i += 1


def _run_query_with_heartbeat(conn, sql: str, params: dict[str, object], message: str):
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat, args=(message, stop), daemon=True)
    thread.start()
    try:
        return conn.run(sql, **params)
    finally:
        stop.set()
        thread.join(timeout=0.2)


def _metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - y_true
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(math.sqrt((err * err).mean())),
        "mse": float((err * err).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--max-rows-per-split", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.test_days <= 0 or args.validation_days <= 0:
        raise SystemExit("test/validation days must be positive")

    started = time.monotonic()
    print("🌉 SF PARKING — LEAKAGE-SAFE PAID-STATE LIGHTGBM")
    print("════════════════════════════════════════════════════════════════════")
    print("[1/5] Opening the materialized hourly state table.")
    print("      This table already contains the bounded paid-use state, so we do")
    print("      not need to walk the raw transaction history again.")

    conn = connect()
    try:
        bounds = conn.run(
            "SELECT min(slot_start), max(slot_start) FROM parking_state_hourly"
        )[0]
        if bounds[0] is None or bounds[1] is None:
            raise SystemExit("parking_state_hourly is empty; build it first")

        latest = bounds[1]
        test_cut = latest - timedelta(days=args.test_days)
        validation_cut = test_cut - timedelta(days=args.validation_days)

        print(f"      First completed state: {bounds[0]}")
        print(f"      Last completed state:  {latest}")
        print(f"      Validation starts:     {validation_cut}")
        print(f"      Test starts:           {test_cut}")
        print("      ✅ Chronological boundaries established.")

        print("\n[2/5] Building strictly prior-state features.")
        print("      The target is the state of hour T.")
        print("      Every feature is from T-1h, T-2h, T-24h, or earlier.")
        print("      There is deliberately NO feature from the hour being predicted.")
        print("      This prevents the model from seeing the answer before it predicts.")

        query = """
        WITH base AS (
            SELECT
                s.post_id,
                s.slot_start,
                s.paid_availability_probability AS target_availability,
                s.meter_type,
                EXTRACT(ISODOW FROM s.local_date)::int AS iso_weekday,
                s.transaction_count,
                LAG(s.paid_availability_probability, 1) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                ) AS lag1_availability,
                LAG(s.paid_availability_probability, 2) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                ) AS lag2_availability,
                LAG(s.paid_availability_probability, 24) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                ) AS lag24_availability,
                AVG(s.paid_availability_probability) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) AS rolling3_availability,
                AVG(s.paid_availability_probability) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
                ) AS rolling24_availability,
                LAG(s.transaction_count, 1) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                ) AS lag1_transactions,
                LAG(s.transaction_count, 24) OVER (
                    PARTITION BY s.post_id ORDER BY s.slot_start
                ) AS lag24_transactions
            FROM parking_state_hourly s
        )
        SELECT
            post_id,
            slot_start,
            target_availability,
            lag1_availability,
            lag2_availability,
            lag24_availability,
            rolling3_availability,
            rolling24_availability,
            lag1_transactions,
            lag24_transactions,
            SIN(2 * PI() * EXTRACT(HOUR FROM (slot_start AT TIME ZONE 'America/Los_Angeles')) / 24.0) AS hour_sin,
            COS(2 * PI() * EXTRACT(HOUR FROM (slot_start AT TIME ZONE 'America/Los_Angeles')) / 24.0) AS hour_cos,
            SIN(2 * PI() * (iso_weekday - 1) / 7.0) AS weekday_sin,
            COS(2 * PI() * (iso_weekday - 1) / 7.0) AS weekday_cos,
            CASE WHEN meter_type = 'MS' THEN 1.0 ELSE 0.0 END AS is_ms
        FROM base
        WHERE slot_start >= :start_slot
          AND slot_start < :end_slot
          AND lag1_availability IS NOT NULL
        ORDER BY abs(hashtext(post_id || slot_start::text)), slot_start, post_id
        LIMIT :limit_rows
        """

        def load_split(start, end, label):
            if start >= end:
                raise SystemExit(f"{label} interval is empty")
            print(f"      Loading {label} rows: {start} → {end}")
            rows = _run_query_with_heartbeat(
                conn,
                query,
                {
                    "start_slot": start,
                    "end_slot": end,
                    "limit_rows": args.max_rows_per_split,
                },
                f"PostgreSQL is building leakage-safe {label} rows from prior hours.",
            )
            print(f"      ✅ Loaded {len(rows):,} {label} rows.")
            return rows

        train_rows = load_split(bounds[0], validation_cut, "training")
        validation_rows = load_split(validation_cut, test_cut, "validation")
        test_rows = load_split(test_cut, latest + timedelta(hours=1), "test")

    finally:
        conn.close()

    columns = ["post_id", "slot_start", "target_availability", *FEATURES]
    train = pd.DataFrame(train_rows, columns=columns)
    validation = pd.DataFrame(validation_rows, columns=columns)
    test = pd.DataFrame(test_rows, columns=columns)

    if train.empty or validation.empty or test.empty:
        raise SystemExit("chronological split produced an empty partition")

    for df in (train, validation, test):
        df["slot_start"] = pd.to_datetime(df["slot_start"], utc=True)
        for feature in FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df["target_availability"] = pd.to_numeric(df["target_availability"], errors="coerce")
        df.dropna(subset=["target_availability"], inplace=True)

    print("\n[3/5] Checking the learning target.")
    print("      This is a continuous [0,1] paid-availability estimate.")
    print("      We will NOT threshold it at 0.5 and pretend it is physical truth.")
    print(f"      Train:       {len(train):,}")
    print(f"      Validation:  {len(validation):,}")
    print(f"      Test:        {len(test):,}")
    print(f"      Test target mean:   {test.target_availability.mean():.4f}")
    print(f"      Test target median: {test.target_availability.median():.4f}")

    print("\n[4/5] Training LightGBM regression.")
    print("      The model learns the current completed hour's paid-availability")
    print("      state from prior history, time-of-week, and meter type.")

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        train[FEATURES],
        train["target_availability"],
        eval_set=[(validation[FEATURES], validation["target_availability"])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    print("\n[5/5] Evaluating against simple baselines.")
    pred = pd.Series(model.predict(test[FEATURES]), index=test.index).clip(0.0, 1.0)
    y = test["target_availability"]
    model_metrics = _metrics(y, pred)

    persistence = test["lag1_availability"].clip(0.0, 1.0)
    persistence_metrics = _metrics(y, persistence)

    hour_means = train.groupby(
        train["slot_start"].dt.tz_convert("America/Los_Angeles").dt.hour
    )["target_availability"].mean()
    test_hours = test["slot_start"].dt.tz_convert("America/Los_Angeles").dt.hour
    hourly_pred = test_hours.map(hour_means).fillna(train["target_availability"].mean())
    hourly_metrics = _metrics(y, hourly_pred)

    result = {
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_start": test.slot_start.min().isoformat(),
        "test_end": test.slot_start.max().isoformat(),
        "target": "paid_availability_probability",
        "model": model_metrics,
        "persistence_baseline": persistence_metrics,
        "hour_climatology_baseline": hourly_metrics,
        "improvement_vs_persistence_mae": round(
            persistence_metrics["mae"] - model_metrics["mae"], 6
        ),
        "improvement_vs_hour_mae": round(
            hourly_metrics["mae"] - model_metrics["mae"], 6
        ),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators_)),
    }

    print(json.dumps(result, indent=2))
    print(f"\n✅ COMPLETE — elapsed {int(time.monotonic() - started)}s")
    try:
        subprocess.run(
            ["pbcopy"], input=json.dumps(result, indent=2), text=True, check=True
        )
        print("📋 Results copied to macOS clipboard.")
    except (OSError, subprocess.CalledProcessError):
        print("⚠️ Could not copy results to clipboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
