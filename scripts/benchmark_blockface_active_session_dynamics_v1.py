"""Causal blockface active-session dynamics benchmark V1.

Scientific target: active paid sessions per blockface at target T.
Forecast origin: T-1 hour.
Predictors: only session starts observed by T-1 plus training-window duration
survival and time-varying historical arrival intensity. session_end is never
used for a still-open target-time session; it is used only for training
survival and held-out labels.

This intentionally avoids a capacity/availability target because the current
source data does not provide a reliable blockface capacity denominator.
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
OUT = ROOT / "models" / "blockface_active_session_dynamics_v1.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def lm(d: date) -> datetime:
    return datetime.combine(d, time.min, TZ)


def win(a: date, b: date) -> tuple[datetime, datetime]:
    return lm(a).astimezone(UTC), lm(b + timedelta(days=1)).astimezone(UTC)


def complete_folds(first, latest, train_days=6, test_days=1, max_folds=2):
    first_local = first.astimezone(TZ).date()
    latest_local = latest.astimezone(TZ)
    end = latest_local.date() if latest_local.hour >= 23 else latest_local.date() - timedelta(days=1)
    out = []
    while end >= first_local and len(out) < max_folds:
        test_start = end - timedelta(days=test_days - 1)
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)
        if train_start < first_local:
            break
        out.append({
            "train": win(train_start, train_end),
            "test": win(test_start, end),
            "local_days": {"train": [str(train_start), str(train_end)], "test": [str(test_start), str(end)]},
        })
        end -= timedelta(days=test_days)
    return list(reversed(out))


def mapping_sql() -> str:
    return (
        "SELECT DISTINCT post_id::text AS post_id, blockface_id::text AS blockface_id "
        "FROM parking_meters WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL"
    )


def metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
    }


def targets(conn, start, end, k, seed):
    return conn.run(f"""
        WITH mapping AS ({mapping_sql()}), slots AS (
            SELECT DISTINCT m.blockface_id, s.slot_start
            FROM mapping m
            JOIN parking_state_hourly s ON s.post_id::text = m.post_id
            WHERE s.slot_start >= :start AND s.slot_start < :end
        )
        SELECT blockface_id, slot_start
        FROM slots
        ORDER BY hashtext(blockface_id || '|' || slot_start::text || :seed::text), blockface_id, slot_start
        LIMIT :k
    """, start=start, end=end, seed=str(seed), k=k)


def labels(conn, ts):
    conn.run("DROP TABLE IF EXISTS _bfv5_labels")
    conn.run("CREATE TEMP TABLE _bfv5_labels(blockface_id text, slot_start timestamptz)")
    import csv
    from io import StringIO
    b = StringIO(); csv.writer(b, lineterminator="\n").writerows(ts)
    conn.run("COPY _bfv5_labels(blockface_id,slot_start) FROM STDIN WITH(FORMAT csv)", stream=[b.getvalue().encode()])
    rows = conn.run(f"""
        WITH mapping AS ({mapping_sql()}), sessions AS (
            SELECT DISTINCT t.transmission_id, t.post_id::text AS post_id, t.session_start, t.session_end
            FROM meter_transactions t
            WHERE t.session_end IS NOT NULL AND t.session_end > t.session_start
        ), mapped AS (
            SELECT s.transmission_id, m.blockface_id, s.session_start, s.session_end
            FROM sessions s JOIN mapping m ON m.post_id = s.post_id
        )
        SELECT z.blockface_id, z.slot_start,
               COUNT(DISTINCT m.transmission_id) FILTER (
                   WHERE m.session_start <= z.slot_start AND m.session_end > z.slot_start
               )::int AS active,
               COUNT(DISTINCT m.transmission_id) FILTER (
                   WHERE m.session_start <= z.slot_start - INTERVAL '1 hour'
                     AND m.session_end > z.slot_start - INTERVAL '1 hour'
               )::int AS active_prev
        FROM _bfv5_labels z
        LEFT JOIN mapped m ON m.blockface_id = z.blockface_id
        GROUP BY z.blockface_id, z.slot_start
    """)
    conn.run("DROP TABLE IF EXISTS _bfv5_labels")
    return rows


def learn(conn, start, end):
    # Arrival intensity by blockface × local dow × local hour, normalized by
    # the actual number of occurrences of that weekday/hour in the train window.
    hist = conn.run(f"""
        WITH mapping AS ({mapping_sql()}), x AS (
            SELECT m.blockface_id,
                   (t.session_start AT TIME ZONE 'America/Los_Angeles')::date AS local_date,
                   EXTRACT(ISODOW FROM (t.session_start AT TIME ZONE 'America/Los_Angeles'))::int AS dow,
                   EXTRACT(HOUR FROM (t.session_start AT TIME ZONE 'America/Los_Angeles'))::int AS hour,
                   t.transmission_id
            FROM meter_transactions t
            JOIN mapping m ON m.post_id = t.post_id::text
            WHERE t.session_start >= :start AND t.session_start < :end
        )
        SELECT blockface_id, dow, hour, COUNT(DISTINCT transmission_id)::float AS events,
               COUNT(DISTINCT local_date)::float AS day_count
        FROM x GROUP BY 1,2,3
    """, start=start, end=end)
    rates = {
        (str(r[0]), int(r[1]), int(r[2])): float(r[3]) / max(float(r[4]), 1.0)
        for r in hist
    }
    global_hours = max((end - start).total_seconds() / 3600.0, 1.0)
    global_rate = sum(float(r[3]) for r in hist) / global_hours if hist else 0.0

    drows = conn.run(f"""
        WITH mapping AS ({mapping_sql()})
        SELECT m.blockface_id,
               EXTRACT(EPOCH FROM (t.session_end - t.session_start))/3600.0 AS duration_h
        FROM meter_transactions t JOIN mapping m ON m.post_id=t.post_id::text
        WHERE t.session_start >= :start AND t.session_start < :end
          AND t.session_end IS NOT NULL AND t.session_end > t.session_start
    """, start=start, end=end)
    by_block = {}; all_d = []
    for r in drows:
        d = max(float(r[1]), 1e-6)
        by_block.setdefault(str(r[0]), []).append(d)
        all_d.append(d)
    return rates, global_rate, by_block, np.asarray(all_d, float)


def survival_prob(durs, age, extra):
    if len(durs) == 0:
        return math.exp(-extra / 1.5)
    eligible = durs[durs > age]
    if len(eligible) == 0:
        return 0.0
    return float(np.mean(eligible > age + extra))


def forecast(conn, ts, rates, global_rate, by_block, global_durs):
    conn.run("DROP TABLE IF EXISTS _bfv5_pred")
    conn.run("CREATE TEMP TABLE _bfv5_pred(blockface_id text,target_slot timestamptz)")
    import csv
    from io import StringIO
    b = StringIO(); csv.writer(b, lineterminator="\n").writerows(ts)
    conn.run("COPY _bfv5_pred(blockface_id,target_slot) FROM STDIN WITH(FORMAT csv)", stream=[b.getvalue().encode()])

    starts = conn.run(f"""
        WITH mapping AS ({mapping_sql()})
        SELECT p.blockface_id, p.target_slot,
               EXTRACT(EPOCH FROM ((p.target_slot - INTERVAL '1 hour') - t.session_start))/3600.0 AS age_h
        FROM _bfv5_pred p
        JOIN mapping m ON m.blockface_id=p.blockface_id
        JOIN meter_transactions t ON t.post_id::text=m.post_id
        WHERE t.session_start <= p.target_slot - INTERVAL '1 hour'
          AND t.session_start > p.target_slot - INTERVAL '73 hours'
    """)
    by_target = {}
    for r in starts:
        by_target.setdefault((str(r[0]), r[1]), []).append(float(r[2]))

    out = []
    for bf, slot in ts:
        ages = by_target.get((str(bf), slot), [])
        durs = np.asarray(by_block.get(str(bf), global_durs), float)
        existing = sum(survival_prob(durs, age, 1.0) for age in ages)
        local = slot.astimezone(TZ)
        lam = float(rates.get((str(bf), local.isoweekday(), local.hour), global_rate))
        # One-hour expected arrivals × probability a new session survives to T.
        new = 0.0
        for j in range(12):
            age = (j + 0.5) / 12.0
            surv = float(np.mean(durs > age)) if len(durs) else math.exp(-age / 1.5)
            new += (lam / 12.0) * surv
        out.append(max(0.0, existing + new))
    conn.run("DROP TABLE IF EXISTS _bfv5_pred")
    return np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-days', type=int, default=6)
    ap.add_argument('--test-days', type=int, default=1)
    ap.add_argument('--max-folds', type=int, default=2)
    ap.add_argument('--max-test-rows', type=int, default=50000)
    a = ap.parse_args()

    print('🚦 SF PARKING — CAUSAL BLOCKFACE ACTIVE-SESSION DYNAMICS V1')
    c = connect(); reps = []
    try:
        first, latest = c.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly WHERE slot_start <= NOW()")[0]
        fs = complete_folds(first, latest, a.train_days, a.test_days, a.max_folds)
        if not fs:
            raise RuntimeError('No complete local-day folds available')
        for i, f in enumerate(fs, 1):
            print(f"\n[Fold {i}/{len(fs)}] {f['local_days']}")
            ts = targets(c, *f['test'], a.max_test_rows, 41000+i)
            lab = labels(c, ts)
            obs = {(str(r[0]), r[1]): (int(r[2]), int(r[3])) for r in lab}
            rates, gr, bb, gd = learn(c, *f['train'])
            p = forecast(c, ts, rates, gr, bb, gd)
            y = np.asarray([obs[(str(bf), slot)][0] for bf, slot in ts], float)
            lag = np.asarray([obs[(str(bf), slot)][1] for bf, slot in ts], float)
            dm, pm = metric(y, p), metric(y, lag)
            print(f"    rows={len(y):,} causal_active_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}")
            print(f"    target_mean={y.mean():.6f} target_max={y.max():.0f} persistence_mean={lag.mean():.6f} prediction_mean={p.mean():.6f}")
            reps.append({'fold': i, 'local_days': f['local_days'], 'rows': len(y), 'causal_dynamics': dm, 'persistence': pm,
                          'target_mean': float(y.mean()), 'target_max': float(y.max()), 'prediction_mean': float(p.mean()),
                          'persistence_mean': float(lag.mean())})
    finally:
        c.close()

    w = np.asarray([r['rows'] for r in reps], float)
    dm = float(np.average([r['causal_dynamics']['mae'] for r in reps], weights=w))
    pm = float(np.average([r['persistence']['mae'] for r in reps], weights=w))
    res = {
        'version': 1,
        'model': 'causal_blockface_active_session_dynamics',
        'ground_truth': 'active paid session count per blockface',
        'aggregate': {
            'test_rows': int(w.sum()),
            'persistence_mae': pm,
            'causal_active_mae': dm,
            'improvement_over_persistence': (pm-dm)/pm if pm else None,
            'promotion': 'candidate' if np.isfinite(dm) and dm < pm else 'retained_only',
        },
        'folds': reps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print('\nFINAL:')
    print(json.dumps(res['aggregate'], indent=2))
    print(f'Report: {OUT}')

if __name__ == '__main__':
    main()
