"""Runnable V4 research runner for the causal blockface dynamics benchmark.

V4 fixes benchmark-harness issues:
- canonical text ID types at every SQL boundary;
- direct execution from repo root;
- partial latest local days are excluded from rolling-origin test folds;
- the original fold generator is preserved before V4 overrides it;
- blockface identity and capacity use the same canonical parking_meters relation.

Capacity uses distinct parking_space_id when populated; otherwise each distinct
mapped post_id contributes one capacity unit. This matches the actual source
grain instead of assuming parking_space_id is populated.
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

ORIGINAL_FOLDS = v3.folds


def normalized_mapping_sql() -> str:
    return (
        "SELECT DISTINCT post_id::text AS post_id, "
        "blockface_id::text AS blockface_id "
        "FROM parking_meters "
        "WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL"
    )


def normalized_targets(conn, start, end, k, seed):
    rows = conn.run(
        f"""
        WITH mapping AS ({normalized_mapping_sql()}), capacity AS (
            SELECT blockface_id::text AS blockface_id,
                   COUNT(DISTINCT COALESCE(parking_space_id::text, post_id::text))::int AS capacity
            FROM parking_meters
            WHERE blockface_id IS NOT NULL
              AND post_id IS NOT NULL
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
    if rows:
        return rows

    diag = conn.run(
        f"""
        WITH mapping AS ({normalized_mapping_sql()}),
        mapped_slots AS (
            SELECT DISTINCT m.blockface_id, s.slot_start
            FROM mapping m
            JOIN parking_state_hourly s ON s.post_id::text = m.post_id
            WHERE s.slot_start >= :start AND s.slot_start < :end
        ),
        capacity AS (
            SELECT blockface_id::text AS blockface_id,
                   COUNT(DISTINCT COALESCE(parking_space_id::text, post_id::text))::int AS capacity
            FROM parking_meters
            WHERE blockface_id IS NOT NULL AND post_id IS NOT NULL
            GROUP BY blockface_id
        )
        SELECT
            (SELECT COUNT(*) FROM mapping)::int AS mapping_rows,
            (SELECT COUNT(DISTINCT blockface_id) FROM mapping)::int AS mapped_blockfaces,
            (SELECT COUNT(*) FROM mapped_slots)::int AS mapped_slots,
            (SELECT COUNT(*) FROM capacity WHERE capacity > 0)::int AS capacity_blockfaces,
            (SELECT COUNT(*) FROM mapped_slots ms JOIN capacity c USING(blockface_id) WHERE c.capacity > 0)::int AS eligible_slots
        """,
        start=start,
        end=end,
    )[0]
    raise RuntimeError(
        "No eligible blockface targets for test window "
        f"{start.isoformat()} → {end.isoformat()}. "
        f"diagnostic={{'mapping_rows':{diag[0]},'mapped_blockfaces':{diag[1]},"
        f"'mapped_slots':{diag[2]},'capacity_blockfaces':{diag[3]},'eligible_slots':{diag[4]}}}"
    )


def complete_day_folds(first, latest, train_days, test_days, max_folds):
    """Use only complete local days, without recursively calling the override."""
    latest_local = latest.astimezone(v3.TZ)
    if latest_local.hour < 23:
        adjusted_latest = latest_local - timedelta(days=1)
    else:
        adjusted_latest = latest_local

    result = ORIGINAL_FOLDS(first, adjusted_latest, train_days, test_days, max_folds)
    if adjusted_latest.date() != latest_local.date():
        print(
            f"  excluding partial local day {latest_local.date()}; "
            f"latest observed local hour={latest_local.hour:02d}"
        )
    if not result:
        raise RuntimeError(
            "No complete local-day folds available. "
            f"first={first.astimezone(v3.TZ).date()}, "
            f"latest_complete={adjusted_latest.astimezone(v3.TZ).date()}"
        )
    return result


v3.mapping_sql = normalized_mapping_sql
v3.targets = normalized_targets
v3.folds = complete_day_folds


if __name__ == "__main__":
    raise SystemExit(v3.main())
