"""Project ingested SFMTA snapshots into the canonical entity model.

Runs after ``scripts/load_database.py``. Every projection is a derived
source executed through the generic ingestion framework, so each one records
an ``ingestion_runs`` row with provenance and is idempotent on stable
source-id conflict keys.

Usage:
    python scripts/project_canonical.py
"""

from __future__ import annotations

import json

from sf_parking.canonical import project_canonical
from sf_parking.database import connect


def main() -> None:
    conn = connect()
    try:
        results = project_canonical(conn)
    finally:
        conn.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
