"""Runnable V4 research runner for the causal blockface dynamics benchmark.

V4 fixes three benchmark-harness issues:
- canonical text ID types at every SQL boundary;
- direct execution from repo root;
- partial latest local days are excluded from rolling-origin test folds.
"""
from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_blockface_transaction_dynamics_v3",
    HERE / "benchmark_blockface_transaction_dynamics_v3.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load V3 benchmark module")
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


def normalized_mapping_sql() -> str:
    return (
        "SELECT DISTINCT post_id::text AS post_id, "
        "blockface_id::text AS blockface_id "
        "FROM parking_meters "
        "WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL"
    )


def normalized_targets(conn, start, end, k, seed):
    return conn.run(
        f"""
        WITH mapping AS ({normalized_mapping_sql()}), capacity AS (
            SELECT blockface_id::text AS blockface_id,
                   COUNT(DISTINCT parking_space_id)::int AS capacity
            FROM parking_spaces
            WHERE blockface_id IS NOT NULL
            GROUP BY blockface_id
        ), slots AS (
            SELECT DISTINCT
                m.blockface_id,
                s.slot_start,
                (s.slot_start AT TIME ZONE 'America/Los_Angeles')::date AS local_date,
                EXTRACT(HOUR FROM (s.slot_start AT TIME ZONE 'America/Los_Angeles'))::int AS local_hour
            FROM mapping m
            JOIN parking_state_hourly s
              ON s.post_id::text = m.post_id
            WHERE s.slot_start >= :start
              AND s.slot_start < :end
        )
        SELECT s.blockface_id,
               s.slot_start,
               c.capacity,
               s.local_date,
               s.local_hour
        FROM slots s
        JOIN capacity c
          ON c.blockface_id = s.blockface_id
        WHERE c.capacity > 0
        ORDER BY hashtext(
                     s.blockface_id::text || '|' || s.slot_start::text || :seed::text
                 ),
                 s.blockface_id,
                 s.slot_start
        LIMIT :k
        """,
        start=start,
        end=end,
        seed=str(seed),
        k=k,
    )


def complete_day_folds(first, latest, train_days, test_days, max_folds):
    """Use only the latest fully observed local day.

    parking_state_hourly is hourly. A local day is considered complete when the
    latest observed local slot is hour 23 or later. Otherwise, move back one
    local day so the benchmark never treats an in-progress day as a test day.
    """
    latest_local = latest.astimezone(v3.TZ)
    if latest_local.hour < 23:
        adjusted_latest = latest_local - timedelta(days=1)
    else:
        adjusted_latest = latest_local
    result = v3.folds(first, adjusted_latest, train_days, test_days, max_folds)
    skipped = latest_local.date() if adjusted_latest.date() != latest_local.date() else None
    if skipped:
        print(f"  excluding partial local day {skipped}; latest observed local hour={latest_local.hour:02d}")
    return result


v3.mapping_sql = normalized_mapping_sql
v3.targets = normalized_targets
v3.folds = complete_day_folds


if __name__ == "__main__":
    raise SystemExit(v3.main())
