"""Runnable V4 wrapper for the causal blockface dynamics benchmark.

Normalizes IDs at the SQL boundary and fixes the capacity CTE so blockface_id
has one canonical text type throughout the target-building query. The module
is loaded by file path so `python3 scripts/...` works directly from repo root.
"""
from __future__ import annotations

import importlib.util
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


v3.mapping_sql = normalized_mapping_sql
v3.targets = normalized_targets


if __name__ == "__main__":
    raise SystemExit(v3.main())
