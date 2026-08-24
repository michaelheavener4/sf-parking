"""One-time migration: retag meter-transaction timestamps to true instants.

History
-------
The DataSF transactions adapter originally parsed Socrata floating timestamps
(``session_start_dt`` / ``session_end_dt``) into *naive* datetimes, which
PostgreSQL stored in ``timestamptz`` columns interpreted under the session
timezone (UTC). Those columns are America/Los_Angeles wall-clock times, so
every stored instant was off by the Pacific UTC offset (-7h PDT / -8h PST):
``2026-08-17T04:31:23`` local was stored as ``04:31:23+00`` instead of
``11:31:23+00``.

The migration reinterprets each stored value — see
``sf_parking.migrations.retag_source_local_timestamps`` for details.

Usage:
    python scripts/migrate_transaction_timestamps.py
"""

from __future__ import annotations

import json

from sf_parking.database import connect
from sf_parking.migrations import retag_source_local_timestamps


def main() -> None:
    conn = connect()
    try:
        result = retag_source_local_timestamps(conn)
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "status": result.status,
                "rows_retagged": result.rows_retagged,
                "total_rows_after": result.total_rows_after,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
