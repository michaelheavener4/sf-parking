"""Benchmark a first-principles two-state parking dynamics model.

This is intentionally not a machine-learning tournament. It tests whether a
simple arrival/departure hazard model can improve on one-hour persistence.
All fitted rates are learned only from each fold's training window and all
prediction features come from T-1 or earlier.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sf_parking.database import connect
from sf_parking.dynamics import availability_forecast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "first_principles_dynamics_v1.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def local_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, TZ)


def local_window(start_day: date, end_day: date) -> tuple[datetime, datetime]:
    return local_midnight(start_day).astimezone(UTC), local_midnight(end_day + timedelta(days=1)).astimezone(UTC)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def transition_metrics(y: np.ndarray, lag: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    d = y - lag
    mask = np.abs(d) >= threshold - 1e-12
    if not mask.any():
        return {"n": 0, "mae": None, "direction_accuracy": None, "mean_abs_delta": None}
    pd = pred - lag
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(pred[mask] - y[mask]))),
        "direction_accuracy": float(np.mean((d[mask] > 0) == (pd[mask] > 0))),
        "mean_abs_delta": float(np.mean(np.abs(d[mask]))),
    }


def make_folds(first: datetime, latest: datetime, train_days: int, validation_days: int, test_days: int, max_folds: int):
    first_day = first.astimezone(TZ).date()
    last_day = latest.astimezone(TZ).date()
    end = last_day
    result = []
    while end >= first_day and len(result) < max_folds:
        test_start = end - timedelta(days=test_days - 1)
        val_end = test_start - timedelta(days=1)
        val_start = val_end - timedelta(days=validation_days - 1)
        train_end = val_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)
        if train_start < first_day:
            break
        result.append({
            "train": local_window(train_start, train_end),
            "validation": local_window(val_start, val_end),
            "test": local_window(test_start, end),
            "local_days": {
                "train": [str(train_start), str(train_end)],
                "validation": [str(val_start), str(val_end)],
                "test": [str(test_start), str(end)],
            },
        })
        end -= timedelta(days=test_days)
    result = list(reversed(result))
    if result:
        result[-1]["test"] = (result[-1]["test"][0], min(result[-1]["test"][1], latest + timedelta(microseconds=1)))
    return result


def learn_rates(conn, train_start: datetime, train_end: datetime) -> dict[str, object]:
    duration_rows = conn.run("""
        SELECT post_id,
               AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0) AS mean_duration_h,
               COUNT(*)::double precision AS sessions,
               COUNT(*)::double precision / GREATEST(EXTRACT(EPOCH FROM (:end - :start)) / 3600.0, 1.0) AS baseline_arrival_h
        FROM meter_transactions
        WHERE session_start >= :start
          AND session_end IS NOT NULL
          AND session_end <= :end
          AND session_end > session_start
        GROUP BY post_id
    """, start=train_start, end=train_end)
    durations = {
        str(r[0]): {"mean_duration_h": float(r[1]), "sessions": float(r[2]), "baseline_arrival_h": float(r[3])}
        for r in duration_rows
        if r[1] is not None and float(r[1]) > 0
    }
    hour_rows = conn.run("""
        SELECT EXTRACT(HOUR FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS h,
               EXTRACT(ISODOW FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS dow,
               COUNT(*)::double precision AS arrivals
        FROM meter_transactions
        WHERE session_start >= :start
          AND session_start < :end
        GROUP BY 1, 2
    """, start=train_start, end=train_end)
    days = max(1.0, (train_end - train_start).total_seconds() / 86400.0)
    total_arrivals = sum(float(r[2]) for r in hour_rows)
    baseline = max(1e-9, total_arrivals / (days * 168.0))
    seasonality = {
        (int(r[0]), int(r[1])): max(0.05, float(r[2]) / max(days * baseline, 1e-9))
        for r in hour_rows
    }
    global_duration = conn.run("""
        SELECT AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0)
        FROM meter_transactions
        WHERE session_start >= :start
          AND session_end IS NOT NULL
          AND session_end <= :end
          AND session_end > session_start
    """, start=train_start, end=train_end)[0][0]
    global_duration_h = float(global_duration) if global_duration is not None and float(global_duration) > 0 else 1.5
    return {"post": durations, "seasonality": seasonality, "global_duration_h": global_duration_h, "baseline_hourly_rate": baseline}


def sample_targets(conn, start: datetime, end: datetime, limit_rows: int, seed: int):
    # Target contains y(T), but every predictor is explicitly from T-1.
    return conn.run("""
        SELECT
            t.post_id,
            t.slot_start,
            t.paid_availability_probability AS target_availability,
            p.paid_availability_probability AS lag1_availability,
            p.transaction_count AS lag1_transactions,
            t.local_hour,
            t.local_date
        FROM parking_state_hourly t
        JOIN parking_state_hourly p
          ON p.post_id = t.post_id
         AND p.slot_start = t.slot_start - INTERVAL '1 hour'
        WHERE t.slot_start >= :start
          AND t.slot_start < :end
        ORDER BY hashtext(t.post_id || '|' || t.slot_start::text || :seed::text), t.post_id, t.slot_start
        LIMIT :limit_rows
    """, start=start, end=end, seed=str(seed), limit_rows=limit_rows)


def predict(targets, rates: dict[str, object], horizon_hours: float) -> pd.DataFrame:
    rows = []
    post_rates = rates["post"]
    seasonality = rates["seasonality"]
    global_duration = float(rates["global_duration_h"])
    global_lambda = float(rates["baseline_hourly_rate"])
    for post_id, slot, target, lag_availability, lag_tx1, hour, local_date in targets:
        params = post_rates.get(str(post_id))
        if params is None:
            duration_h = global_duration
            post_lambda = global_lambda
        else:
            duration_h = 0.8 * float(params["mean_duration_h"]) + 0.2 * global_duration
            post_lambda = 0.8 * float(params["baseline_arrival_h"]) + 0.2 * global_lambda
        season = float(seasonality.get((int(hour), int(pd.Timestamp(local_date).isoweekday())), 1.0))
        recent = max(0.0, float(lag_tx1))
        lam = 0.55 * post_lambda * season + 0.45 * recent
        pred = availability_forecast(float(lag_availability), lam, duration_h, horizon_hours)
        rows.append((str(post_id), slot, float(target), float(lag_availability), float(pred), lam, duration_h, recent))
    return pd.DataFrame(rows, columns=["post_id", "slot_start", "target_availability", "lag1_availability", "predicted_availability", "arrival_rate_h", "duration_h", "recent_tx1"])


def run_fold(conn, fold, args, idx):
    rates = learn_rates(conn, *fold["train"])
    targets = sample_targets(conn, *fold["test"], args.max_test_rows, 4000 + idx)
    df = predict(targets, rates, args.horizon_hours)
    if df.empty:
        raise RuntimeError(f"fold {idx}: empty test sample")
    keep = np.isfinite(df["lag1_availability"].to_numpy(float)) & np.isfinite(df["target_availability"].to_numpy(float))
    df = df.loc[keep].reset_index(drop=True)
    y = df.target_availability.to_numpy(float)
    lag = df.lag1_availability.to_numpy(float)
    pred = df.predicted_availability.to_numpy(float)
    return {
        "fold": idx,
        "local_days": fold["local_days"],
        "rows": int(len(df)),
        "dynamics": metric(y, pred),
        "persistence": metric(y, lag),
        "transitions": {str(t): transition_metrics(y, lag, pred, t) for t in (0.05, 0.10, 0.25)},
        "mean_arrival_rate_h": float(df.arrival_rate_h.mean()),
        "mean_duration_h": float(df.duration_h.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=6)
    ap.add_argument("--validation-days", type=int, default=1)
    ap.add_argument("--test-days", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=10)
    ap.add_argument("--max-test-rows", type=int, default=75000)
    ap.add_argument("--horizon-hours", type=float, default=1.0)
    args = ap.parse_args()

    print("🚦 SF PARKING — FIRST-PRINCIPLES OCCUPANCY DYNAMICS V1")
    conn = connect()
    try:
        first, latest = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly WHERE slot_start <= NOW()")[0]
        fs = make_folds(first, latest, args.train_days, args.validation_days, args.test_days, args.max_folds)
        print(f"Local data: {first.astimezone(TZ).date()} → {latest.astimezone(TZ).date()}; folds={len(fs)}")
        reports = []
        for i, fold in enumerate(fs, 1):
            print(f"\n[Fold {i}/{len(fs)}] {fold['local_days']}")
            reports.append(run_fold(conn, fold, args, i))
    finally:
        conn.close()

    w = np.array([r["rows"] for r in reports], float)
    dm = float(np.average([r["dynamics"]["mae"] for r in reports], weights=w))
    pm = float(np.average([r["persistence"]["mae"] for r in reports], weights=w))
    result = {
        "version": 1,
        "model": "two_state_arrival_departure_hazard",
        "ground_truth": "exact same-post T-1 hour; targets use y(T), predictors use only T-1 and training-window transaction history",
        "folds": reports,
        "aggregate": {"test_rows": int(sum(w)), "persistence_mae": pm, "dynamics_mae": dm, "improvement_over_persistence": (pm - dm) / pm if pm else None},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\nFINAL:")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())