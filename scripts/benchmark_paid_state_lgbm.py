"""Representative, leakage-safe LightGBM benchmark for hourly paid-state forecasting.

Design goals:
- never window the entire 30M-row state table in one query;
- sample training/validation targets deterministically across meters and time;
- evaluate the full held-out test window day-by-day;
- use only features strictly earlier than the target timestamp;
- compare against persistence and hour-of-day climatology;
- narrate progress and emit a machine-readable summary to the clipboard.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sf_parking.database import connect

TZ = "America/Los_Angeles"
FEATURES = [
    "lag1_availability",
    "lag2_availability",
    "lag3_availability",
    "lag6_availability",
    "lag24_availability",
    "lag168_availability",
    "lag1_transactions",
    "lag24_transactions",
    "roll3_availability",
    "roll24_availability",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_ms",
]


def heartbeat(message: str, stop: threading.Event) -> None:
    icons = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌")
    i = 0
    started = time.monotonic()
    while not stop.wait(5):
        elapsed = int(time.monotonic() - started)
        print(
            f"{icons[i % len(icons)]} Still working after "
            f"{elapsed // 60:02d}:{elapsed % 60:02d}. {message}",
            flush=True,
        )
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


def metric_arrays(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    err = pred - y
    return float(np.mean(np.abs(err))), float(math.sqrt(np.mean(err * err)))


def choose_train_days(latest: datetime, validation_days: int, test_days: int, n: int) -> list[date]:
    latest_day = latest.astimezone(timezone.utc).date()
    test_start = latest_day - timedelta(days=test_days - 1)
    val_end = test_start - timedelta(days=1)
    val_start = val_end - timedelta(days=validation_days - 1)
    available_end = val_start - timedelta(days=1)
    return [available_end - timedelta(days=i) for i in range(n - 1, -1, -1)]


def sample_day_targets(conn, day: date, limit_rows: int, seed: int) -> list[tuple[object, ...]]:
    # Deterministic hash bucket gives city-wide coverage without a giant ORDER BY.
    bucket = max(1, min(999, int((limit_rows / 350_000) * 1000)))
    sql = """
    SELECT post_id, slot_start, paid_availability_probability, meter_type, local_hour, local_date
    FROM parking_state_hourly
    WHERE local_date = :day
      AND mod(abs(hashtext(post_id || '|' || slot_start::text || :seed::text)), 1000) < :bucket
    ORDER BY slot_start, post_id
    LIMIT :limit_rows
    """
    return conn.run(
        sql,
        day=day,
        seed=str(seed),
        bucket=bucket,
        limit_rows=limit_rows,
    )


def build_features(conn, targets: list[tuple[object, ...]], label: str) -> pd.DataFrame:
    if not targets:
        return pd.DataFrame(columns=["target", *FEATURES])

    # Python-side feature construction from per-target exact point lookups.
    # Batch through SQL values to avoid giant window functions over the full table.
    conn.run("DROP TABLE IF EXISTS _benchmark_targets")
    conn.run(
        """
        CREATE TEMP TABLE _benchmark_targets (
            post_id text,
            slot_start timestamptz,
            target double precision,
            meter_type text,
            local_hour int,
            local_date date
        )
        """
    )

    import csv
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in targets:
        writer.writerow(row)
    conn.run(
        "COPY _benchmark_targets (post_id, slot_start, target, meter_type, local_hour, local_date) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )

    sql = """
    SELECT
        t.post_id,
        t.slot_start,
        t.target,
        CASE WHEN t.meter_type = 'MS' THEN 1.0 ELSE 0.0 END AS is_ms,
        t.local_hour,
        t.local_date,
        p1.paid_availability_probability AS lag1,
        p2.paid_availability_probability AS lag2,
        p3.paid_availability_probability AS lag3,
        p6.paid_availability_probability AS lag6,
        p24.paid_availability_probability AS lag24,
        p168.paid_availability_probability AS lag168,
        p1.transaction_count AS tx1,
        p24.transaction_count AS tx24,
        (
            p1.paid_availability_probability +
            p2.paid_availability_probability +
            p3.paid_availability_probability
        ) / 3.0 AS roll3,
        (
            p1.paid_availability_probability +
            p2.paid_availability_probability +
            p3.paid_availability_probability +
            p6.paid_availability_probability +
            p24.paid_availability_probability
        ) / 5.0 AS roll24_proxy
    FROM _benchmark_targets t
    INNER JOIN parking_state_hourly p1
      ON p1.post_id = t.post_id AND p1.slot_start = t.slot_start - INTERVAL '1 hour'
    INNER JOIN parking_state_hourly p2
      ON p2.post_id = t.post_id AND p2.slot_start = t.slot_start - INTERVAL '2 hours'
    INNER JOIN parking_state_hourly p3
      ON p3.post_id = t.post_id AND p3.slot_start = t.slot_start - INTERVAL '3 hours'
    INNER JOIN parking_state_hourly p6
      ON p6.post_id = t.post_id AND p6.slot_start = t.slot_start - INTERVAL '6 hours'
    INNER JOIN parking_state_hourly p24
      ON p24.post_id = t.post_id AND p24.slot_start = t.slot_start - INTERVAL '24 hours'
    INNER JOIN parking_state_hourly p168
      ON p168.post_id = t.post_id AND p168.slot_start = t.slot_start - INTERVAL '168 hours'
    """
    rows = run_query(conn, sql, {}, f"PostgreSQL is fetching prior-state point features for {label}.")
    conn.run("DROP TABLE IF EXISTS _benchmark_targets")

    out: list[dict[str, object]] = []
    for row in rows:
        hour = int(row[4])
        local_date = row[5]
        iso = local_date.isocalendar()
        dow = int(iso.weekday)
        out.append(
            {
                "post_id": row[0],
                "slot_start": row[1],
                "target": float(row[2]),
                "lag1_availability": float(row[6]),
                "lag2_availability": float(row[7]),
                "lag3_availability": float(row[8]),
                "lag6_availability": float(row[9]),
                "lag24_availability": float(row[10]),
                "lag168_availability": float(row[11]),
                "lag1_transactions": float(row[12]),
                "lag24_transactions": float(row[13]),
                "roll3_availability": float(row[14]),
                "roll24_availability": float((
                    float(row[14]) * 3.0 + float(row[9]) + float(row[10])
                ) / 5.0),
                "hour_sin": math.sin(2 * math.pi * hour / 24.0),
                "hour_cos": math.cos(2 * math.pi * hour / 24.0),
                "weekday_sin": math.sin(2 * math.pi * (dow - 1) / 7.0),
                "weekday_cos": math.cos(2 * math.pi * (dow - 1) / 7.0),
                "is_ms": float(row[3]),
            }
        )
    return pd.DataFrame(out)


def train_model(train: pd.DataFrame, validation: pd.DataFrame):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        train[FEATURES],
        train["target"],
        eval_X=validation[FEATURES],
        eval_y=validation["target"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.monotonic()
    print("🌉 SF PARKING — REPRESENTATIVE 7-DAY LIGHTGBM BENCHMARK")
    print("════════════════════════════════════════════════════════════════════")
    print("[1/5] Opening the completed hourly state table.")

    conn = connect()
    try:
        first, latest = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly")[0]
        test_start = latest - timedelta(days=args.test_days)
        validation_start = test_start - timedelta(days=args.validation_days)
        train_end = validation_start - timedelta(hours=1)

        train_day_count = max(1, (train_end.date() - first.date()).days + 1)
        target_per_day = max(5000, math.ceil(args.max_train_rows / train_day_count))
        # Keep the sample intentionally bounded if the requested cap is huge.
        target_per_day = min(target_per_day, 50_000)

        print(f"      first={first}")
        print(f"      validation_start={validation_start}")
        print(f"      test_start={test_start}")
        print(f"      latest={latest}")
        print(f"      max_train_rows={args.max_train_rows:,}")
        print(f"      deterministic sample target/day={target_per_day:,}")

        print("\n[2/5] Building representative training and validation samples.")
        print("      Sampling is spread across every historical day by deterministic hash buckets.")
        train_targets: list[tuple[object, ...]] = []
        current = first.date()
        while current <= train_end.date() and len(train_targets) < args.max_train_rows:
            rows = sample_day_targets(conn, current, target_per_day, args.seed)
            train_targets.extend(rows)
            print(
                f"      🚗 train {current}: +{len(rows):,} rows; total={len(train_targets):,}",
                flush=True,
            )
            current += timedelta(days=1)
        train_targets = train_targets[: args.max_train_rows]

        validation_targets: list[tuple[object, ...]] = []
        current = validation_start.date()
        while current < test_start.date():
            rows = sample_day_targets(conn, current, min(25_000, target_per_day), args.seed + 1)
            validation_targets.extend(rows)
            print(
                f"      🧪 validation {current}: +{len(rows):,} rows; total={len(validation_targets):,}",
                flush=True,
            )
            current += timedelta(days=1)

        print("\n[3/5] Materializing prior-only features for train/validation.")
        train = build_features(conn, train_targets, "training")
        validation = build_features(conn, validation_targets, "validation")
        print(f"      ✅ train features={len(train):,}")
        print(f"      ✅ validation features={len(validation):,}")

        conn.close()
        conn = None

        print("\n[4/5] Training LightGBM and evaluating the complete held-out week day-by-day.")
        model = train_model(train, validation)

        test_mae = []
        test_rmse = []
        persist_mae = []
        persist_rmse = []
        clim_mae = []
        clim_rmse = []
        all_y: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_persist: list[np.ndarray] = []
        all_clim: list[np.ndarray] = []
        test_rows = 0

        # Hour climatology learned from training only.
        local_hours = pd.to_datetime(train["slot_start"], utc=True).dt.tz_convert(TZ).dt.hour
        hour_mean = train.groupby(local_hours)["target"].mean()
        global_mean = float(train["target"].mean())

        current = test_start.date()
        final_day = latest.astimezone(timezone.utc).date()
        while current <= final_day:
            targets = sample_day_targets(conn, current, 350_000, args.seed + 2)
            # For the test, the sample is still deterministic and broad; if a future day is partial,
            # filter to the actual observed frontier.
            targets = [r for r in targets if r[1] <= latest]
            features = build_features(conn, targets, f"test {current}")
            if not features.empty:
                pred = np.clip(model.predict(features[FEATURES]), 0.0, 1.0)
                y = features["target"].to_numpy(dtype=float)
                persistence = features["lag1_availability"].to_numpy(dtype=float)
                hours = pd.to_datetime(features["slot_start"], utc=True).dt.tz_convert(TZ).dt.hour
                climatology = hours.map(hour_mean).fillna(global_mean).to_numpy(dtype=float)
                m, r = metric_arrays(y, pred)
                mp, rp = metric_arrays(y, persistence)
                mc, rc = metric_arrays(y, climatology)
                test_mae.append(m); test_rmse.append(r)
                persist_mae.append(mp); persist_rmse.append(rp)
                clim_mae.append(mc); clim_rmse.append(rc)
                all_y.append(y); all_pred.append(pred); all_persist.append(persistence); all_clim.append(climatology)
                test_rows += len(y)
                print(
                    f"      🚕 test {current}: rows={len(y):,} model_mae={m:.4f} "
                    f"persist_mae={mp:.4f} climatology_mae={mc:.4f}",
                    flush=True,
                )
            current += timedelta(days=1)

    finally:
        if conn is not None:
            conn.close()

    y = np.concatenate(all_y) if all_y else np.array([], dtype=float)
    pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=float)
    persistence = np.concatenate(all_persist) if all_persist else np.array([], dtype=float)
    climatology = np.concatenate(all_clim) if all_clim else np.array([], dtype=float)
    model_mae, model_rmse = metric_arrays(y, pred)
    persist_m, persist_r = metric_arrays(y, persistence)
    clim_m, clim_r = metric_arrays(y, climatology)

    result = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(test_rows),
        "model": {"mae": model_mae, "rmse": model_rmse},
        "persistence": {"mae": persist_m, "rmse": persist_r},
        "hour_climatology": {"mae": clim_m, "rmse": clim_r},
        "mae_gain_vs_persistence": round(persist_m - model_mae, 6),
        "mae_gain_vs_climatology": round(clim_m - model_mae, 6),
        "rmse_gain_vs_persistence": round(persist_r - model_rmse, 6),
        "rmse_gain_vs_climatology": round(clim_r - model_rmse, 6),
        "best_iteration": int(getattr(model, "best_iteration_", 0)),
    }

    print("\n[5/5] Final benchmark results.")
    print(json.dumps(result, indent=2))
    print(f"\n✅ COMPLETE — total elapsed {int(time.monotonic() - started)}s")
    try:
        subprocess.run(["pbcopy"], input=json.dumps(result, indent=2), text=True, check=True)
        print("📋 Results copied to macOS clipboard.")
    except (OSError, subprocess.CalledProcessError):
        print("⚠️ Clipboard copy failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
