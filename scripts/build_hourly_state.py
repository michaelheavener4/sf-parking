"""Materialize bounded paid-occupancy state one local day at a time."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "db/queries/build_hourly_state.sql"
SF_TZ = ZoneInfo("America/Los_Angeles")
MAX_PROB = 0.98
P90_MINUTES = 120.0


def utc_bounds(day: date) -> tuple[datetime, datetime]:
    local = datetime(day.year, day.month, day.day, tzinfo=SF_TZ)
    return local.astimezone(UTC), (local + timedelta(days=1)).astimezone(UTC)


def ensure_table(conn) -> None:
    conn.run("""
    CREATE TABLE IF NOT EXISTS parking_state_hourly (
        post_id text NOT NULL,
        slot_start timestamptz NOT NULL,
        local_date date NOT NULL,
        local_hour smallint NOT NULL,
        meter_type text,
        transaction_count integer NOT NULL,
        paid_overlap_minutes double precision NOT NULL,
        paid_occupancy_probability double precision NOT NULL,
        paid_availability_probability double precision NOT NULL,
        source text NOT NULL DEFAULT 'derived_from_sfmta_transactions',
        generated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (post_id, slot_start)
    )
    """)
    conn.run("CREATE INDEX IF NOT EXISTS idx_state_hourly_slot ON parking_state_hourly(slot_start)")
    conn.run("CREATE INDEX IF NOT EXISTS idx_state_hourly_post_slot ON parking_state_hourly(post_id, slot_start DESC)")
    conn.run("COMMIT")


def write_day(conn, day: date, query: str) -> int:
    start, end = utc_bounds(day)
    rows = conn.run(query, day_start=start, day_end=end, max_prob=MAX_PROB, p90_minutes=P90_MINUTES)
    conn.run("DELETE FROM parking_state_hourly WHERE local_date = :day", day=day)
    if not rows:
        conn.run("COMMIT")
        return 0

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        post_id, slot, local_date, local_hour, meter_type, tx_count, minutes, prob = row
        writer.writerow([post_id, slot, local_date, local_hour, meter_type, tx_count, minutes, prob, 1.0 - prob])
    buffer.seek(0)

    conn.run(
        "COPY parking_state_hourly "
        "(post_id, slot_start, local_date, local_hour, meter_type, transaction_count, "
        "paid_overlap_minutes, paid_occupancy_probability, paid_availability_probability) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=[buffer.getvalue().encode("utf-8")],
    )
    conn.run("COMMIT")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    print("🌉 SF PARKING — HOURLY PAID-STATE MATERIALIZER")
    print("[1/3] Creating the derived hourly state table.")
    conn = connect()
    try:
        ensure_table(conn)
        query = QUERY.read_text(encoding="utf-8")
        print("      ✅ Table ready.")
        print("[2/3] Rebuilding one local day at a time.")
        total = 0
        days = (end - start).days + 1
        day = start
        for i in range(days):
            rows = write_day(conn, day, query)
            total += rows
            print(f"      🚗 [{i + 1}/{days}] {day}: {rows:,} state rows; total={total:,}", flush=True)
            day += timedelta(days=1)
        print("[3/3] Verifying the derived table.")
        row = conn.run("SELECT count(*), min(slot_start), max(slot_start), min(paid_occupancy_probability), max(paid_occupancy_probability) FROM parking_state_hourly")[0]
        print(f"      rows={row[0]:,}")
        print(f"      first_slot={row[1]}")
        print(f"      last_slot={row[2]}")
        print(f"      probability_range={row[3]}..{row[4]}")
        print("✅ COMPLETE")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
