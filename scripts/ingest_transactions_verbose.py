"""Verbose SFMTA transaction ingestion with live terminal progress.

Usage:
    python scripts/ingest_transactions_verbose.py --window-days 90

Unlike the general inventory ingestion script, this command targets the
registered SFMTA transaction source and prints a live heartbeat after every
committed batch so a long DataSF/Postgres run never looks stalled.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sf_parking.database import apply_schema, connect
from sf_parking.ingestion import load_sources, resolve_adapter, run_ingestion, source_health

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES = (
    "[🐌░░░░░░░░]", "[░🐌░░░░░░░]", "[░░🐌░░░░░░]", "[░░░🐌░░░░░]",
    "[░░░░🐌░░░░]", "[░░░░░🐌░░░]", "[░░░░░░🐌░░]", "[░░░░░░░🐌░]",
    "[░░░░░░░░🐌]", "[░░░░░░░🐌░]",
)


def rate(n: int, seconds: float) -> str:
    if n <= 0 or seconds <= 0:
        return "--"
    r = n / seconds
    if r >= 1_000_000:
        return f"{r / 1_000_000:.2f}M/s"
    if r >= 1_000:
        return f"{r / 1_000:.1f}k/s"
    return f"{r:.0f}/s"


def progress(processed: int, stored: int, skipped: int, elapsed: float) -> None:
    frame = FRAMES[(processed // 10_000) % len(FRAMES)]
    minutes, seconds = divmod(int(elapsed), 60)
    print(
        f"\r🌱🚗 {frame} {processed:,} processed | {stored:,} stored | "
        f"{skipped:,} skipped | {rate(processed, elapsed)} | "
        f"elapsed {minutes:02d}:{seconds:02d}",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--where", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/sources.yaml")
    parser.add_argument("--schema", type=Path, default=REPO_ROOT / "db/schema.sql")
    args = parser.parse_args()

    sources = load_sources(args.config)
    source_name = "sfmta_meter_transactions"
    definition = sources[source_name]
    options = dict(definition.options)
    options["window_days"] = args.window_days
    if args.where:
        options["where"] = args.where

    print("\n🌉 SF PARKING — SFMTA TRANSACTION INGESTION")
    print("═" * 68)
    print(f"window      : {args.window_days} days")
    print(f"target      : {definition.target_table}")
    print("source      : DataSF imvp-dq3v")
    print("batch       : 10,000 records")
    print("progress    : live heartbeat after every committed batch")
    print("important   : total source row count is not known up front")
    print("═" * 68)
    print("Starting...\n")

    conn = connect(args.database_url)
    started = time.monotonic()
    try:
        apply_schema(conn, args.schema)
        adapter = resolve_adapter(definition)
        result = run_ingestion(conn, adapter, options, progress=progress)
        health = source_health(conn, {definition.name: definition})
    finally:
        conn.close()

    elapsed = time.monotonic() - started
    print()
    print("\n✅ COMPLETE" if result.ok else "\n❌ FAILED")
    print("═" * 68)
    print(f"run id      : {result.run_id}")
    print(f"status      : {result.status}")
    print(f"processed   : {result.records_processed:,}")
    print(f"stored      : {result.records_stored:,}")
    print(f"skipped     : {result.records_skipped:,}")
    print(f"elapsed     : {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
    print(f"avg rate    : {rate(result.records_processed, elapsed)}")
    print(f"health      : {health[definition.name].state}")
    if result.error:
        print(f"error       : {result.error}")
    print("═" * 68)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
