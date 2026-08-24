"""One-time data migrations with recorded provenance."""

from __future__ import annotations

from dataclasses import dataclass

import pg8000.native

from ..database import transaction

#: Provenance source recorded for this migration's ``ingestion_runs`` row.
RETAG_SOURCE_LOCAL_TIMESTAMP_SOURCE = "migration:retag_source_local_timestamps"

TRANSACTIONS_SOURCE = "sfmta_meter_transactions"

# Stored values were America/Los_Angeles wall-clock times misinterpreted as
# UTC when naive datetimes were written to timestamptz columns. Read each
# value back on the UTC clock (recovering the intended wall clock), then
# re-tag that wall clock as source-local time so the column carries the true
# absolute instant. PostgreSQL resolves DST-ambiguous/nonexistent wall clocks
# per its zoneinfo rules.
_UPDATE_SQL = """
UPDATE meter_transactions
SET session_start = (session_start AT TIME ZONE 'UTC')
                    AT TIME ZONE 'America/Los_Angeles',
    session_end = (session_end AT TIME ZONE 'UTC')
                  AT TIME ZONE 'America/Los_Angeles'
"""

_RETAG_RUNS_SQL = """
UPDATE ingestion_runs
SET source_timestamp = (source_timestamp AT TIME ZONE 'UTC')
                       AT TIME ZONE 'America/Los_Angeles'
WHERE source = :source AND status = 'succeeded' AND source_timestamp IS NOT NULL
"""


@dataclass(frozen=True, slots=True)
class MigrationResult:
    status: str  # "applied" or "skipped"
    rows_retagged: int = 0
    total_rows_after: int = 0


def retag_source_local_timestamps(conn: pg8000.native.Connection) -> MigrationResult:
    """Reinterpret meter-transaction timestamps as America/Los_Angeles time.

    Idempotent: records itself as an ``ingestion_runs`` row and exits without
    touching data if that record already exists. Never inserts or deletes
    transaction rows.
    """
    already = conn.run(
        "SELECT 1 FROM ingestion_runs "
        "WHERE source = :source AND status = 'succeeded' LIMIT 1",
        source=RETAG_SOURCE_LOCAL_TIMESTAMP_SOURCE,
    )
    if already:
        return MigrationResult(status="skipped", total_rows_after=_total(conn))

    with transaction(conn):
        retagged = int(
            conn.run(
                "WITH updated AS (" + _UPDATE_SQL + " RETURNING 1) "
                "SELECT count(*) FROM updated"
            )[0][0]
        )
        conn.run(_RETAG_RUNS_SQL, source=TRANSACTIONS_SOURCE)
        total = _total(conn)
        conn.run(
            "INSERT INTO ingestion_runs "
            "(source, started_at, finished_at, status, records_processed, "
            "records_stored, error) "
            "VALUES (:source, now(), now(), 'succeeded', :processed, 0, NULL)",
            source=RETAG_SOURCE_LOCAL_TIMESTAMP_SOURCE,
            processed=retagged,
        )
    return MigrationResult(status="applied", rows_retagged=retagged, total_rows_after=total)


def _total(conn: pg8000.native.Connection) -> int:
    return int(conn.run("SELECT count(*) FROM meter_transactions")[0][0])
