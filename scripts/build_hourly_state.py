"""Build bounded hourly paid-occupancy state from SFMTA transactions."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from sf_parking.database import apply_schema, connect

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "db/migrations/2026-08-24_hourly_paid_state.sql"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    print("🌉 SF PARKING — HOURLY PAID-STATE BUILDER")
    print("[1/3] Opening PostgreSQL and creating the materialized state table.")
    conn = connect()
    try:
        apply_schema(conn, ROOT / "db/schema.sql")
        conn.run(MIGRATION.read_text(encoding="utf-8"))
        conn.run("COMMIT")
        print("      ✅ State table ready.")
        print("[2/3] Materialization SQL is ready to run for the requested date range.")
        print(f"      Start: {start}")
        print(f"      End:   {end}")
        print("[3/3] This first version intentionally stops after schema preparation;")
        print("      the query builder will be added separately and tested before it")
        print("      is allowed to write millions of derived rows.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
