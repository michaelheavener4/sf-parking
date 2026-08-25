"""Clean causal two-state parking dynamics benchmark.

V2 deliberately uses only information available at T-1:
- exact T-1 availability state
- exact T-1 transaction count
- post-specific arrival rate learned from training history
- hour-of-week arrival seasonality learned from training history
- post/global mean completed-session duration learned from training history

It compares the dynamics forecast directly against exact one-hour persistence.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from sf_parking.database import connect
from sf_parking.dynamics import availability_forecast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "first_principles_dynamics_v2.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def local_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, TZ)


def local_window(start_day: date, end_day: date) -> tuple[datetime, datetime]:
    return local_midnight(start_day).astimezone(UTC), local_midnight(end_day + timedelta(days=1)).astimezone(UTC)


def folds(first: datetime, latest: datetime, train_days: int, test_days: int, max_folds: int):
    first_day = first.astimezone(TZ).date()
    last_day = latest.astimezone(TZ).date()
    end = last_day
    out = []
    while end >= first_day and len(out) < max_folds:
        test_start = end - timedelta(days=test_days - 1)
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)
        if train_start < first_day:
            break
        out.append({
            "train": local_window(train_start, train_end),
            "test": local_window(test_start, end),
            "local_days": {"train": [str(train_start), str(train_end)], "test": [str(test_start), str(end)]},
        })
        end -= timedelta(days=test_days)
    return list(reversed(out))


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(math.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def learn_rates(conn, start: datetime, end: datetime) -> dict[str, object]:
    hours = max((end - start).total_seconds() / 3600.0, 1.0)
    post_rows = conn.run("""
        SELECT post_id,
               COUNT(*)::double precision AS sessions,
               AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0) AS mean_duration_h
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_start < CAST(:end AS timestamptz)
          AND session_end IS NOT NULL
          AND session_end > session_start
        GROUP BY post_id
    """, start=start, end=end)
    post = {
        str(r[0]): {
            "arrival_h": float(r[1]) / hours,
            "duration_h": float(r[2]),
        }
        for r in post_rows
        if r[1] is not None and r[2] is not None and float(r[2]) > 0
    }

    hist = conn.run("""
        SELECT EXTRACT(ISODOW FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS dow,
               EXTRACT(HOUR FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS hour,
               COUNT(*)::double precision AS arrivals
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_start < CAST(:end AS timestamptz)
        GROUP BY 1, 2
    """, start=start, end=end)

    total_sessions = sum(float(r[2]) for r in hist)
    baseline_hour = total_sessions / hours
    seasonality = {
        (int(r[0]), int(r[1])): max(0.05, float(r[2]) / max(baseline_hour, 1e-9))
        for r in hist
    }
    global_duration = conn.run("""
        SELECT AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0)
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_start < CAST(:end AS timestamptz)
          AND session_end IS NOT NULL
          AND session_end > session_start
    """, start=start, end=end)[0][0]
    global_duration = float(global_duration) if global_duration is not None and float(global_duration) > 0 else 1.5
    return {"post": post, "seasonality": seasonality, "global_duration_h": global_duration, "baseline_hour": baseline_hour}


def sample_targets(conn, start: datetime, end: datetime, limit_rows: int, seed: int):
    return conn.run("""
        SELECT t.post_id,
               t.slot_start,
               t.paid_availability_probability AS target_availability,
               p.paid_availability_probability AS prev_availability,
               p.transaction_count AS prev_transaction_count,
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


def predict(rows, rates, horizon_hours: float):
    preds = []
    y = []
    lag = []
    for r in rows:
        post_id = str(r[0])
        target = float(r[2])
        prev = float(r[3])
        tx_prev = max(0.0, float(r[4] or 0.0))
        hour = int(r[5])
        local_date = r[6]
        dow = int(local_date.isoweekday())
        params = rates["post"].get(post_id)
        if params:
            post_arrival = float(params["arrival_h"])
            duration = float(params["duration_h"])
        else:
            post_arrival = float(rates["baseline_hour"])
            duration = float(rates["global_duration_h"])
        seasonal = float(rates["seasonality"].get((dow, hour), 1.0))
        long_rate = post_arrival * seasonal
        arrival_rate = 0.70 * long_rate + 0.30 * tx_prev
        pred = availability_forecast(prev, arrival_rate, duration, horizon_hours)
        y.append(target)
        lag.append(prev)
        preds.append(pred)
    return np.asarray(y), np.asarray(lag), np.asarray(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=6)
    ap.add_argument("--test-days", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=2)
    ap.add_argument("--max-test-rows", type=int, default=75000)
    ap.add_argument("--horizon-hours", type=float, default=1.0)
    args = ap.parse_args()

    print("🚦 SF PARKING — FIRST-PRINCIPLES OCCUPANCY DYNAMICS V2")
    conn = connect()
    try:
        first, latest = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly WHERE slot_start <= NOW()")[0]
        fs = folds(first, latest, args.train_days, args.test_days, args.max_folds)
        print(f"Local data: {first.astimezone(TZ).date()} → {latest.astimezone(TZ).date()}; folds={len(fs)}")
        reports = []
        for i, fold in enumerate(fs, 1):
            print(f"\n[Fold {i}/{len(fs)}] {fold['local_days']}")
            rates = learn_rates(conn, *fold["train"])
            rows = sample_targets(conn, *fold["test"], args.max_test_rows, 9000 + i)
            if not rows:
                raise RuntimeError(f"fold {i}: no exact T-1 target pairs")
            y, lag, pred = predict(rows, rates, args.horizon_hours)
            dm = metric(y, pred)
            pm = metric(y, lag)
            print(f"    rows={len(y):,} dynamics_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}")
            reports.append({"fold": i, "local_days": fold["local_days"], "rows": len(y), "dynamics": dm, "persistence": pm})
    finally:
        conn.close()

    w = np.asarray([r["rows"] for r in reports], float)
    dm = float(np.average([r["dynamics"]["mae"] for r in reports], weights=w))
    pm = float(np.average([r["persistence"]["mae"] for r in reports], weights=w))
    result = {
        "version": 2,
        "model": "two_state_arrival_departure_hazard_v2",
        "causality": "target uses T-1 availability and T-1 transaction count only; rates learned from earlier training window",
        "aggregate": {"test_rows": int(w.sum()), "persistence_mae": pm, "dynamics_mae": dm, "improvement_over_persistence": (pm - dm) / pm if pm else None, "promotion": "candidate" if dm < pm else "retained_only"},
        "folds": reports,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL:")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
