"""Run ingestion for a registered source from config/sources.yaml.

Usage:
    python scripts/run_ingestion.py --source sfmta_meter_transactions
    python scripts/run_ingestion.py --source sfmta_meter_transactions \
        --window-days 3 --where "payment_type = 'PAY BY CELL'"

The run is recorded in ``ingestion_runs``; a failed source exits non-zero
instead of reporting success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sf_parking.database import apply_schema, connect
from sf_parking.ingestion import (
    RegistryError,
    load_sources,
    resolve_adapter,
    run_ingestion,
    source_health,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source name in sources.yaml")
    parser.add_argument("--config", type=Path, default=Path("config/sources.yaml"))
    parser.add_argument("--schema", type=Path, default=REPO_ROOT / "db" / "schema.sql")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--where", default=None)
    args = parser.parse_args()

    sources = load_sources(args.config)
    if args.source not in sources:
        parser.error(f"unknown source {args.source!r}; known: {', '.join(sorted(sources))}")
    definition = sources[args.source]

    options = dict(definition.options)
    if args.window_days is not None:
        options["window_days"] = args.window_days
    if args.where:
        options["where"] = args.where

    conn = connect(args.database_url)
    try:
        apply_schema(conn, args.schema)
        adapter = resolve_adapter(definition)
        result = run_ingestion(conn, adapter, options)
        health = source_health(conn, {definition.name: definition})
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "run": {
                    "run_id": result.run_id,
                    "source": result.source,
                    "status": result.status,
                    "records_processed": result.records_processed,
                    "records_stored": result.records_stored,
                    "records_skipped": result.records_skipped,
                    "source_timestamp": (
                        result.source_timestamp.isoformat() if result.source_timestamp else None
                    ),
                    "error": result.error,
                },
                "health": {
                    "state": health[definition.name].state,
                    "healthy": health[definition.name].healthy,
                },
            },
            indent=2,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
