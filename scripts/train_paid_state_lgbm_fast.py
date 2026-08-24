"""Fast, leakage-safe LightGBM experiment on materialized paid-state history.

This version samples target rows first and then performs indexed prior-state
lookups. It intentionally avoids whole-table PostgreSQL window functions.
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
    "lag1_availability", "lag2_availability", "lag24_availability",
    "rolling3_availability", "rolling24_availability",
    "lag1_transactions", "lag24_transactions",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_ms",
]


def heartbeat(message: str, stop: threading.Event) -> None:
    start = time.monotonic()
    faces = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌")
    i = 0
    while not stop.wait(5):
        elapsed = int(time.monotonic() - start)
        print(f"{faces[i % len(faces)]} Still working after {elapsed//60:02d}:{elapsed%60:02d}. {message}", flush=True)
        i += 1


def runq(conn, sql: str, params: dict[str, object], message: str):
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(message, stop), daemon=True)
    thread.start()
    try:
        return conn.run(sql, **params)
    finally:
        stop.set()
        thread.join(timeout=0.2)


def metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    e = p - y
    return {
        "mae": float(e.abs().mean()),
        "rmse": float(math.sqrt((e * e).mean())),
        "mse": float((e * e).mean()),
    }


QUERY = """
WITH sampled AS (
    SELECT
        s.post_id, s.slot_start,
        s.paid_availability_probability AS target_availability,
        s.meter_type,
        EXTRACT(ISODOW FROM s.local_date)::int AS iso_weekday
    FROM parking_state_hourly s
    WHERE s.slot_start >= :start_slot
      AND s.slot_start < :end_slot
      AND MOD(ABS(hashtext(s.post_id || '|' || s.slot_start::text)), 1000) < :sample_threshold
    ORDER BY s.slot_start, s.post_id
    LIMIT :limit_rows
)
SELECT
    t.post_id,
    t.slot_start,
    t.target_availability,
    p1.paid_availability_probability AS lag1_availability,
    p2.paid_availability_probability AS lag2_availability,
    p24.paid_availability_probability AS lag24_availability,
    r3.rolling3_availability,
    r24.rolling24_availability,
    p1.transaction_count AS lag1_transactions,
    p24.transaction_count AS lag24_transactions,
    SIN(2 * PI() * EXTRACT(HOUR FROM (t.slot_start AT TIME ZONE 'America/Los_Angeles')) / 24.0) AS hour_sin,
    COS(2 * PI() * EXTRACT(HOUR FROM (t.slot_start AT TIME ZONE 'America/Los_Angeles')) / 24.0) AS hour_cos,
    SIN(2 * PI() * (t.iso_weekday - 1) / 7.0) AS weekday_sin,
    COS(2 * PI() * (t.iso_weekday - 1) / 7.0) AS weekday_cos,
    CASE WHEN t.meter_type = 'MS' THEN 1.0 ELSE 0.0 END AS is_ms
FROM sampled t
LEFT JOIN parking_state_hourly p1
  ON p1.post_id = t.post_id
 AND p1.slot_start = t.slot_start - INTERVAL '1 hour'
LEFT JOIN parking_state_hourly p2
  ON p2.post_id = t.post_id
 AND p2.slot_start = t.slot_start - INTERVAL '2 hours'
LEFT JOIN parking_state_hourly p24
  ON p24.post_id = t.post_id
 AND p24.slot_start = t.slot_start - INTERVAL '24 hours'
LEFT JOIN LATERAL (
    SELECT AVG(x.paid_availability_probability) AS rolling3_availability
    FROM parking_state_hourly x
    WHERE x.post_id = t.post_id
      AND x.slot_start >= t.slot_start - INTERVAL '3 hours'
      AND x.slot_start < t.slot_start
) r3 ON TRUE
LEFT JOIN LATERAL (
    SELECT AVG(x.paid_availability_probability) AS rolling24_availability
    FROM parking_state_hourly x
    WHERE x.post_id = t.post_id
      AND x.slot_start >= t.slot_start - INTERVAL '24 hours'
      AND x.slot_start < t.slot_start
) r24 ON TRUE
WHERE p1.slot_start IS NOT NULL
ORDER BY t.slot_start, t.post_id
"""


def load_split(conn, start, end, label, limit_rows, sample_threshold):
    print(f"      Loading {label}: {start} → {end}")
    rows = runq(
        conn, QUERY,
        {
            "start_slot": start, "end_slot": end,
            "limit_rows": limit_rows, "sample_threshold": sample_threshold,
        },
        f"PostgreSQL is selecting {label} targets, then fetching only their prior-state history.",
    )
    print(f"      ✅ {label} rows loaded: {len(rows):,}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--max-rows-per-split", type=int, default=100_000)
    parser.add_argument("--sample-threshold", type=int, default=10,
                        help="Hash threshold out of 1000; 10 ≈ 1%% deterministic sample")
    args = parser.parse_args()

    started = time.monotonic()
    print("🌉 SF PARKING — FAST LEAKAGE-SAFE LIGHTGBM")
    print("════════════════════════════════════════════════════════════════════")
    print("[1/5] Checking the materialized state table.")

    conn = connect()
    try:
        bounds = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly")[0]
        if bounds[0] is None or bounds[1] is None:
            raise SystemExit("parking_state_hourly is empty")
        latest = bounds[1]
        test_cut = latest - timedelta(days=args.test_days)
        val_cut = test_cut - timedelta(days=args.validation_days)
        print(f"      first={bounds[0]}")
        print(f"      validation={val_cut}")
        print(f"      test={test_cut}")
        print(f"      latest={latest}")

        print("\n[2/5] Building bounded training examples.")
        print("      I am deliberately NOT asking PostgreSQL to sort/window all 30M rows.")
        train_rows = load_split(conn, bounds[0], val_cut, "training", args.max_rows_per_split, args.sample_threshold)
        val_rows = load_split(conn, val_cut, test_cut, "validation", args.max_rows_per_split, args.sample_threshold)
        test_rows = load_split(conn, test_cut, latest + timedelta(hours=1), "test", args.max_rows_per_split, args.sample_threshold)
    finally:
        conn.close()

    cols = ["post_id", "slot_start", "target_availability", *FEATURES]
    train = pd.DataFrame(train_rows, columns=cols)
    val = pd.DataFrame(val_rows, columns=cols)
    test = pd.DataFrame(test_rows, columns=cols)
    if train.empty or val.empty or test.empty:
        raise SystemExit("one chronological split is empty")

    for df in (train, val, test):
        df["slot_start"] = pd.to_datetime(df["slot_start"], utc=True)
        for f in FEATURES:
            df[f] = pd.to_numeric(df[f], errors="coerce")
        df["target_availability"] = pd.to_numeric(df["target_availability"], errors="coerce")
        df.dropna(subset=["target_availability"], inplace=True)

    print("\n[3/5] Inspecting the target before training.")
    print(f"      train={len(train):,} validation={len(val):,} test={len(test):,}")
    print(f"      test mean={test.target_availability.mean():.4f}")
    print(f"      test median={test.target_availability.median():.4f}")

    print("\n[4/5] Training LightGBM regression.")
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.04,
        num_leaves=31, min_child_samples=200, random_state=42, verbosity=-1,
    )
    model.fit(
        train[FEATURES], train.target_availability,
        eval_set=[(val[FEATURES], val.target_availability)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    print("\n[5/5] Comparing ML with simple baselines.")
    y = test.target_availability
    pred = pd.Series(model.predict(test[FEATURES]), index=test.index).clip(0, 1)
    persistence = test.lag1_availability.clip(0, 1)
    hour_means = train.groupby(train.slot_start.dt.tz_convert("America/Los_Angeles").dt.hour).target_availability.mean()
    hours = test.slot_start.dt.tz_convert("America/Los_Angeles").dt.hour
    hour_pred = hours.map(hour_means).fillna(train.target_availability.mean())

    result = {
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_start": test.slot_start.min().isoformat(),
        "test_end": test.slot_start.max().isoformat(),
        "target": "paid_availability_probability",
        "model": metrics(y, pred),
        "persistence_baseline": metrics(y, persistence),
        "hour_climatology_baseline": metrics(y, hour_pred),
        "improvement_vs_persistence_mae": round(metrics(y, persistence)["mae"] - metrics(y, pred)["mae"], 6),
        "improvement_vs_hour_mae": round(metrics(y, hour_pred)["mae"] - metrics(y, pred)["mae"], 6),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators_)),
    }
    print(json.dumps(result, indent=2))
    print(f"\n✅ COMPLETE — elapsed {int(time.monotonic() - started)}s")
    try:
        subprocess.run(["pbcopy"], input=json.dumps(result, indent=2), text=True, check=True)
        print("📋 Results copied to macOS clipboard.")
    except (OSError, subprocess.CalledProcessError):
        print("⚠️ Clipboard copy failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
