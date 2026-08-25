"""Research-grade blockface parking dynamics benchmark.

Principles:
1. Ground truth is reconstructed from raw paid sessions, not the existing
   paid_availability_probability heuristic.
2. Unit of analysis is blockface, because transaction-derived occupancy is
   more reliable at an aggregated spatial granularity than at a single meter.
3. Forecast is made from an explicit as-of time and uses only information
   available at or before that time.
4. Active sessions at the as-of time are projected with an empirical
   conditional survival function. New arrivals during the forecast horizon
   are generated from a time-varying arrival intensity and the same survival
   distribution.
5. Capacity is the number of parking spaces mapped to the blockface.

This is intentionally a white-box benchmark before ML.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "blockface_transaction_dynamics_v1.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def local_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, TZ)


def local_window(a: date, b: date) -> tuple[datetime, datetime]:
    return local_midnight(a).astimezone(UTC), local_midnight(b + timedelta(days=1)).astimezone(UTC)


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
            "local_days": {
                "train": [str(train_start), str(train_end)],
                "test": [str(test_start), str(end)],
            },
        })
        end -= timedelta(days=test_days)
    return list(reversed(out))


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
    }


def build_targets(conn, start: datetime, end: datetime, limit_rows: int, seed: int):
    # Target is a blockface/time pair. Capacity is derived from mapped parking
    # spaces, while state is reconstructed from meter sessions mapped to the
    # same blockface.
    return conn.run(
        """
        WITH capacities AS (
            SELECT blockface_id, COUNT(DISTINCT parking_space_id)::int AS capacity
            FROM parking_spaces
            WHERE blockface_id IS NOT NULL
            GROUP BY blockface_id
        ),
        slots AS (
            SELECT DISTINCT
                pm.blockface_id,
                s.slot_start,
                (s.slot_start AT TIME ZONE 'America/Los_Angeles')::date AS local_date,
                EXTRACT(HOUR FROM (s.slot_start AT TIME ZONE 'America/Los_Angeles'))::int AS local_hour
            FROM parking_meters pm
            JOIN parking_state_hourly s ON s.post_id = pm.post_id
            WHERE s.slot_start >= :start
              AND s.slot_start < :end
              AND pm.blockface_id IS NOT NULL
        )
        SELECT x.blockface_id,
               x.slot_start,
               c.capacity,
               x.local_date,
               x.local_hour
        FROM slots x
        JOIN capacities c ON c.blockface_id = x.blockface_id
        WHERE c.capacity > 0
        ORDER BY hashtext(x.blockface_id::text || '|' || x.slot_start::text || :seed::text),
                 x.blockface_id, x.slot_start
        LIMIT :limit_rows
        """,
        start=start,
        end=end,
        seed=str(seed),
        limit_rows=limit_rows,
    )


def reconstruct_observed(conn, targets, start: datetime, end: datetime):
    """Compute raw transaction-implied occupancy at target times and at T-1h."""
    conn.run("DROP TABLE IF EXISTS _bf_targets")
    conn.run(
        """
        CREATE TEMP TABLE _bf_targets(
            blockface_id text,
            slot_start timestamptz,
            capacity int,
            local_date date,
            local_hour int
        )
        """
    )
    import csv
    from io import StringIO
    buf = StringIO()
    csv.writer(buf, lineterminator="\n").writerows(targets)
    conn.run(
        "COPY _bf_targets(blockface_id,slot_start,capacity,local_date,local_hour)
         FROM STDIN WITH(FORMAT csv)",
        stream=[buf.getvalue().encode()],
    )

    rows = conn.run(
        """
        WITH mapped AS (
            SELECT t.blockface_id, t.session_start, t.session_end
            FROM meter_transactions t
            JOIN parking_meters pm ON pm.post_id = t.post_id
            WHERE pm.blockface_id IS NOT NULL
              AND t.session_end IS NOT NULL
              AND t.session_end > t.session_start
              AND t.session_start < CAST(:end AS timestamptz)
              AND t.session_end > CAST(:start AS timestamptz) - INTERVAL '1 hour'
        )
        SELECT z.blockface_id,
               z.slot_start,
               z.capacity,
               COUNT(m.session_start) FILTER (
                   WHERE m.session_start <= z.slot_start
                     AND m.session_end > z.slot_start
               )::int AS active_now,
               COUNT(m.session_start) FILTER (
                   WHERE m.session_start <= z.slot_start - INTERVAL '1 hour'
                     AND m.session_end > z.slot_start - INTERVAL '1 hour'
               )::int AS active_prev
        FROM _bf_targets z
        LEFT JOIN mapped m ON m.blockface_id = z.blockface_id
        GROUP BY z.blockface_id, z.slot_start, z.capacity
        """,
        start=start,
        end=end,
    )
    conn.run("DROP TABLE IF EXISTS _bf_targets")
    return rows


def learn_rates(conn, train_start: datetime, train_end: datetime):
    """Learn blockface arrival intensity and empirical duration survival."""
    arrivals = conn.run(
        """
        SELECT pm.blockface_id,
               EXTRACT(ISODOW FROM (t.session_start AT TIME ZONE 'America/Los_Angeles'))::int AS dow,
               EXTRACT(HOUR FROM (t.session_start AT TIME ZONE 'America/Los_Angeles'))::int AS hour,
               COUNT(*)::double precision AS n
        FROM meter_transactions t
        JOIN parking_meters pm ON pm.post_id = t.post_id
        WHERE pm.blockface_id IS NOT NULL
          AND t.session_start >= CAST(:start AS timestamptz)
          AND t.session_start < CAST(:end AS timestamptz)
        GROUP BY 1,2,3
        """,
        start=train_start,
        end=train_end,
    )
    train_hours = max((train_end - train_start).total_seconds() / 3600.0, 1.0)
    intensity = {
        (str(r[0]), int(r[1]), int(r[2])): float(r[3])
        / max(train_hours / 168.0, 1.0 / 24.0)
        for r in arrivals
    }

    durations = conn.run(
        """
        SELECT pm.blockface_id,
               EXTRACT(EPOCH FROM (t.session_end-t.session_start))/3600.0 AS duration_h
        FROM meter_transactions t
        JOIN parking_meters pm ON pm.post_id = t.post_id
        WHERE pm.blockface_id IS NOT NULL
          AND t.session_start >= CAST(:start AS timestamptz)
          AND t.session_start < CAST(:end AS timestamptz)
          AND t.session_end IS NOT NULL
          AND t.session_end > t.session_start
          AND EXTRACT(EPOCH FROM (t.session_end-t.session_start)) > 0
        """,
        start=train_start,
        end=train_end,
    )
    by_block = {}
    global_durations = []
    for r in durations:
        d = max(float(r[1]), 1e-6)
        by_block.setdefault(str(r[0]), []).append(d)
        global_durations.append(d)

    baseline = {}
    for r in arrivals:
        key = (int(r[1]), int(r[2]))
        baseline.setdefault(key, 0.0)
        baseline[key] += float(r[3])

    return intensity, by_block, np.asarray(global_durations, float), baseline


def survival(durations: np.ndarray, age_h: np.ndarray, extra_h: float) -> np.ndarray:
    """Conditional empirical survival P(D > age+extra | D > age)."""
    if len(durations) == 0:
        return np.exp(-extra_h / 1.5)
    out = np.empty(len(age_h), float)
    for i, age in enumerate(age_h):
        denom = max(np.sum(durations > age), 1)
        out[i] = np.sum(durations > age + extra_h) / denom
    return out


def forecast_block(conn, rows, rates, horizon_minutes: int, asof_backfill_hours: int = 24):
    """Forecast blockface occupancy fraction at target slots.

    Each target is predicted from sessions known at target-1h. Existing active
    sessions survive according to empirical conditional duration survival.
    New arrivals during the 1h forecast are integrated on 10-minute cohorts.
    """
    intensity, by_block, global_durations, baseline = rates
    out = []

    # Materialize target keys to retrieve all as-of sessions efficiently.
    conn.run("DROP TABLE IF EXISTS _bf_pred")
    conn.run(
        """
        CREATE TEMP TABLE _bf_pred(
            blockface_id text,
            target_slot timestamptz,
            capacity int,
            local_date date,
            local_hour int
        )
        """
    )
    import csv
    from io import StringIO
    buf = StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    conn.run(
        "COPY _bf_pred(blockface_id,target_slot,capacity,local_date,local_hour)
         FROM STDIN WITH(FORMAT csv)",
        stream=[buf.getvalue().encode()],
    )

    active = conn.run(
        """
        SELECT p.blockface_id, p.target_slot, p.capacity,
               EXTRACT(EPOCH FROM (p.target_slot - t.session_start))/3600.0 AS age_h
        FROM _bf_pred p
        JOIN meter_transactions t
          ON t.session_start <= p.target_slot - INTERVAL '1 hour'
         AND t.session_end > p.target_slot - INTERVAL '1 hour'
        JOIN parking_meters pm ON pm.post_id=t.post_id
         AND pm.blockface_id=p.blockface_id
        WHERE t.session_start >= p.target_slot - CAST(:lookback AS interval)
          AND t.session_end IS NOT NULL
        """,
        lookback=f"{asof_backfill_hours} hours",
    )
    active_by = {}
    for r in active:
        active_by.setdefault((str(r[0]), r[1]), []).append(float(r[3]))

    for blockface_id, target_slot, capacity, local_date, local_hour in rows:
        key = (str(blockface_id), target_slot)
        ages = np.asarray(active_by.get(key, []), float)
        durs = np.asarray(by_block.get(str(blockface_id), global_durations), float)
        if len(ages):
            survive_existing = float(np.sum(survival(durs, ages, 1.0)))
        else:
            survive_existing = 0.0

        arrival_sum = 0.0
        base_rate = float(baseline.get((int(local_date.isoweekday()), int(local_hour)), 0.0))
        if not base_rate:
            base_rate = float(np.mean(list(intensity.values()))) if intensity else 0.0
        post_rate = float(intensity.get((str(blockface_id), int(local_date.isoweekday()), int(local_hour)), 0.0))
        hourly_rate = post_rate if post_rate > 0 else base_rate

        # 10-minute arrival cohorts inside the forecast hour.
        for k in range(6):
            offset_h = (k + 0.5) / 6.0
            remaining = 1.0 - offset_h
            cohort = hourly_rate / 6.0
            # New arrivals get at most the remaining horizon to survive.
            if len(durs):
                surv = float(np.mean(durs > remaining))
            else:
                surv = math.exp(-remaining / 1.5)
            arrival_sum += cohort * surv

        expected_active = survive_existing + arrival_sum
        occupancy = min(1.0, max(0.0, expected_active / max(int(capacity), 1)))
        availability = 1.0 - occupancy
        out.append((float(blockface_id) if isinstance(blockface_id, (int, float)) else str(blockface_id), target_slot, availability))

    conn.run("DROP TABLE IF EXISTS _bf_pred")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=6)
    ap.add_argument("--test-days", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=2)
    ap.add_argument("--max-test-rows", type=int, default=50000)
    args = ap.parse_args()
    print("🚦 SF PARKING — BLOCKFACE TRANSACTION DYNAMICS V1")
    conn = connect()
    try:
        first, latest = conn.run(
            "SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()"
        )[0]
        fs = folds(first, latest, args.train_days, args.test_days, args.max_folds)
        reports = []
        for i, f in enumerate(fs, 1):
            print(f"\n[Fold {i}/{len(fs)}] {f['local_days']}")
            rates = learn_rates(conn, *f["train"])
            targets = build_targets(conn, *f["test"], args.max_test_rows, 17000 + i)
            observed = reconstruct_observed(conn, targets, *f["test"])
            obs_map = {(str(r[0]), r[1]): (float(r[2]), int(r[3]), int(r[4])) for r in observed}
            pred = forecast_block(conn, targets, rates, 60)
            y = []
            lag = []
            for blockface_id, target_slot, capacity, *_ in targets:
                cur = obs_map[(str(blockface_id), target_slot)]
                y.append(1.0 - min(1.0, cur[0] / max(capacity, 1)))
                lag.append(1.0 - min(1.0, cur[1] / max(capacity, 1)))
            p = np.asarray([x[2] for x in pred], float)
            y = np.asarray(y, float)
            lag = np.asarray(lag, float)
            dm, pm = metric(y, p), metric(y, lag)
            print(f"    rows={len(y):,} dynamics_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}")
            reports.append({"fold": i, "local_days": f["local_days"], "rows": len(y), "dynamics": dm, "persistence": pm})
    finally:
        conn.close()

    w = np.asarray([r["rows"] for r in reports], float)
    dm = float(np.average([r["dynamics"]["mae"] for r in reports], weights=w))
    pm = float(np.average([r["persistence"]["mae"] for r in reports], weights=w))
    result = {
        "version": 1,
        "model": "blockface_transaction_session_dynamics",
        "ground_truth": "transaction-implied active-session occupancy / mapped blockface capacity",
        "research_basis": [
            "Yang & Qian 2017: transaction-based occupancy estimation requires behavioral/payment modeling and has a spatial granularity tradeoff",
            "Xiao et al. 2018: finite-capacity queueing models can forecast parking availability in San Francisco",
            "Tavafoghi et al. 2019: non-homogeneous arrivals and time-varying service times support model-based real-time occupancy forecasts",
        ],
        "aggregate": {
            "test_rows": int(w.sum()),
            "persistence_mae": pm,
            "dynamics_mae": dm,
            "improvement_over_persistence": (pm - dm) / pm if pm else None,
            "promotion": "candidate" if dm < pm else "retained_only",
        },
        "folds": reports,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nFINAL:")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
