"""Load normalized JSONL snapshots into the PostGIS database.

Usage:
    python3 scripts/load_database.py

Reads:
    data/raw/parking_meters.jsonl
    data/raw/meter_policies.jsonl

Requires the database from ``docker compose up -d``. Loading is idempotent;
running it repeatedly never duplicates records.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sf_parking.database import main

if __name__ == "__main__":
    main()
