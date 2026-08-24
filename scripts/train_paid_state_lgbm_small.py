"""Small, indexed, leakage-safe LightGBM paid-state experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd

from sf_parking.database import connect

FEATURES = [
    "lag1_availability",
    "lag24_availability",
    "lag1_transactions",
    "lag24_transactions",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_ms",
]


def heartbeat(message: str, stop: threading.Event) -> None:
    started = time.monotonic()
    icons = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌")
    i = 0
    while not stop.wait(5):
        elapsed = int(time.monotonic() - started)
        print(f"{icons[i % len(icons)]} Still working after {elapsed // 60:02d}:{elapsed % 60:02d}. {message}", flush=True)
        i += 1


def run_query(conn, sql: str, params: dict[str, object], message: str):
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(message, stop), daemon=True)
    thread.start()
    try:
        return conn.run(sql, **params)
    finally:
        stop.set()
        thread.join(timeout=0.2)


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - y
    return {"mae": float(err.abs().mean()), "rmse": float(math.sqrt((err * err).mean()))}


def load_days(conn, days: list[date], rows_per_day: int, label: str) -> pd.DataFrame:
    target_sql = """
    SELECT post_id, slot_start, paid_availability_probability, meter_type, local_hour, local_date
    FROM parking_state_hourly
    WHERE local_date = :day
    ORDER BY post_id, slot_start
    LIMIT :limit_rows
    """
    targets: list[tuple[object, ...]] = []
    for i, day in enumerate(days, 1):
        print(f"      [{label} {i}/{len(days)}] selecting targets for {day}", flush=True)
        rows = run_query(conn, target_sql, {"day": day, "limit_rows": rows_per_day}, f"PostgreSQL is selecting {label} targets for {day}.")
        targets.extend(rows)
        print(f"        ✅ {len(rows):,} target rows", flush=True)

    if not targets:
        return pd.DataFrame()

    # pg8000/native uses autocommit in this project, so ON COMMIT DROP would
    # remove the helper table immediately after CREATE. Keep it session-local
    # and drop it explicitly after each feature query.
    conn.run("DROP TABLE IF EXISTS _ml_targets")
    conn.run("CREATE TEMP TABLE _ml_targets (post_id text, slot_start timestamptz, target double precision, meter_type text, local_hour int, local_date date)")
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in targets:
        writer.writerow(row)
    conn.run("COPY _ml_targets (post_id, slot_start, target, meter_type, local_hour, local_date) FROM STDIN WITH (FORMAT csv)", stream=[buf.getvalue().encode("utf-8")])

    feature_sql = """
    SELECT
        t.post_id,
        t.slot_start,
        t.target AS target_availability,
        p1.paid_availability_probability AS lag1_availability,
        p24.paid_availability_probability AS lag24_availability,
        p1.transaction_count AS lag1_transactions,
        p24.transaction_count AS lag24_transactions,
        SIN(2 * PI() * t.local_hour / 24.0) AS hour_sin,
        COS(2 * PI() * t.local_hour / 24.0) AS hour_cos,
        SIN(2 * PI() * (EXTRACT(ISODOW FROM t.local_date) - 1) / 7.0) AS weekday_sin,
        COS(2 * PI() * (EXTRACT(ISODOW FROM t.local_date) - 1) / 7.0) AS weekday_cos,
        CASE WHEN t.meter_type = 'MS' THEN 1.0 ELSE 0.0 END AS is_ms
    FROM _ml_targets t
    INNER JOIN parking_state_hourly p1
      ON p1.post_id = t.post_id
     AND p1.slot_start = t.slot_start - INTERVAL '1 hour'
    INNER JOIN parking_state_hourly p24
      ON p24.post_id = t.post_id
     AND p24.slot_start = t.slot_start - INTERVAL '24 hours'
    """
    rows = run_query(conn, feature_sql, {}, f"PostgreSQL is fetching only T-1h and T-24h features for {label} targets.")
    conn.run("DROP TABLE IF EXISTS _ml_targets")
    return pd.DataFrame(rows, columns=["post_id", "slot_start", "target_availability", *FEATURES])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-per-split", type=int, default=3)
    parser.add_argument("--rows-per-day", type=int, default=5000)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--validation-days", type=int, default=7)
    args = parser.parse_args()

    started = time.monotonic()
    print("🌉 SF PARKING — SMALL LEAKAGE-SAFE LIGHTGBM")
    print("════════════════════════════════════════════════════════════════════")
    print("[1/6] Opening the hourly state table.")
    print("      This experiment uses indexed day slices and point lookups only.")

    conn = connect()
    try:
        bounds = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly")[0]
        latest = bounds[1]
        test_cut = latest - timedelta(days=args.test_days)
        validation_cut = test_cut - timedelta(days=args.validation_days)

        train_days = [validation_cut.date() - timedelta(days=i) for i in range(args.days_per_split, 0, -1)]
        val_days = [test_cut.date() - timedelta(days=i) for i in range(args.days_per_split, 0, -1)]
        test_days = [latest.date() - timedelta(days=i) for i in range(args.days_per_split - 1, -1, -1)]

        print(f"      Latest completed state: {latest}")
        print(f"      Training sample days:   {train_days}")
        print(f"      Validation days:        {val_days}")
        print(f"      Test days:              {test_days}")

        print("\n[2/6] Loading training examples.")
        train = load_days(conn, train_days, args.rows_per_day, "TRAIN")
        print(f"      ✅ Training rows ready: {len(train):,}")

        print("\n[3/6] Loading validation examples.")
        validation = load_days(conn, val_days, args.rows_per_day, "VAL")
        print(f"      ✅ Validation rows ready: {len(validation):,}")

        print("\n[4/6] Loading held-out future examples.")
        test = load_days(conn, test_days, args.rows_per_day, "TEST")
        print(f"      ✅ Test rows ready: {len(test):,}")
    finally:
        conn.close()

    for df in (train, validation, test):
        df["slot_start"] = pd.to_datetime(df["slot_start"], utc=True)
        for c in [*FEATURES, "target_availability"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["target_availability", *FEATURES], inplace=True)

    if train.empty or validation.empty or test.empty:
        raise SystemExit("one or more sample splits are empty")

    print("\n[5/6] Training LightGBM regression.")
    print("      The model sees only T-1h, T-24h, time-of-day, weekday, and meter type.")
    import lightgbm as lgb

    model = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42, verbosity=-1)
    model.fit(train[FEATURES], train["target_availability"], eval_set=[(validation[FEATURES], validation["target_availability"])], eval_metric="l2", callbacks=[lgb.early_stopping(30, verbose=False)])

    print("\n[6/6] Comparing the model with simple baselines.")
    pred = pd.Series(model.predict(test[FEATURES]), index=test.index).clip(0, 1)
    y = test["target_availability"]
    persistence = test["lag1_availability"].clip(0, 1)
    hour_mean = train.groupby(train["slot_start"].dt.tz_convert("America/Los_Angeles").dt.hour)["target_availability"].mean()
    hours = test["slot_start"].dt.tz_convert("America/Los_Angeles").dt.hour
    climatology = hours.map(hour_mean).fillna(train["target_availability"].mean())

    m_model, m_persist, m_hour = metrics(y, pred), metrics(y, persistence), metrics(y, climatology)
    result = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "model": m_model,
        "persistence": m_persist,
        "hour_climatology": m_hour,
        "mae_gain_vs_persistence": round(m_persist["mae"] - m_model["mae"], 6),
        "mae_gain_vs_climatology": round(m_hour["mae"] - m_model["mae"], 6),
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
