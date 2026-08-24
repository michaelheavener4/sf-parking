"""Integration tests for the generic ingestion framework + provenance.

Requires the database from ``docker compose up -d`` on localhost:5432.

Isolation strategy: every test run creates its own throwaway PostgreSQL
schema (``pytest_sf_parking_<random>``), applies ``db/schema.sql`` inside
it, runs against it via ``search_path``, and drops it at teardown. The
``public`` schema — including real ingested data — is never written to,
deleted, or truncated; an explicit test asserts that stays true.

Uses a deterministic in-repo fake adapter so tests never touch the network;
the real DataSF adapter's normalization is covered in test_datasf_adapter.py
and a live run is performed by scripts/run_ingestion.py.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pg8000
import pytest

from sf_parking.database import apply_schema, connect
from sf_parking.ingestion.framework import (
    IngestionRecord,
    InvalidRecord,
    run_ingestion,
)
from sf_parking.ingestion.health import source_health
from sf_parking.ingestion.registry import SourceDefinition

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"

PUBLIC_TABLES = ("parking_meters", "meter_policies", "meter_transactions", "ingestion_runs")


class FakeAdapter:
    """Minimal adapter over an in-memory record list for framework tests."""

    def __init__(
        self,
        records: Iterable[IngestionRecord | InvalidRecord] = (),
        *,
        error: Exception | None = None,
        name: str = "fake_source",
    ) -> None:
        self.records = list(records)
        self.error = error
        self.name = name
        self.target_table = "meter_transactions"
        self.columns = [
            "transmission_id",
            "post_id",
            "session_start",
            "duration_minutes",
            "gross_paid_amt",
        ]
        self.conflict_columns = ["transmission_id", "post_id"]
        self.options_seen: dict[str, Any] | None = None

    def fetch(self, options):
        self.options_seen = dict(options)
        if self.error is not None:
            raise self.error
        yield from self.records


def _record(n: int) -> IngestionRecord:
    start = datetime.fromisoformat(f"2026-08-20T{8 + n % 3:02d}:15:00")
    return IngestionRecord(
        key=(f"tx-{n}", "100-00001"),
        values={
            "transmission_id": f"tx-{n}",
            "post_id": "100-00001",
            "session_start": start,
            "duration_minutes": 60 * (n + 1),
            "gross_paid_amt": 2.5,
        },
        source_timestamp=start,
    )


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="PostgreSQL/PostGIS not reachable on localhost:5432",
)


def _connect_with_schema(schema: str) -> pg8000.native.Connection:
    conn = connect()
    # SET cannot take bind parameters; schema names are server-generated.
    # public stays last so PostGIS types resolve while our tables win lookups.
    conn.run(f'SET search_path TO "{schema}", public')  # generated name only
    return conn


def _table_counts(conn) -> dict[str, int]:
    return {table: int(conn.run(f"SELECT count(*) FROM {table}")[0][0]) for table in PUBLIC_TABLES}


@pytest.fixture(scope="module")
def db():
    """Dedicated throwaway schema; public/production data is never touched."""
    schema = f"pytest_sf_parking_{uuid.uuid4().hex[:12]}"
    conn = connect()
    conn.run(f'CREATE SCHEMA "{schema}"')
    conn.run("COMMIT")
    isolated = _connect_with_schema(schema)
    apply_schema(isolated, SCHEMA_PATH)

    baseline = _table_counts(conn)
    yield _SchemaScope(connection=isolated, schema=schema, public_baseline=baseline)

    try:
        isolated.close()
    finally:
        conn.run(f'DROP SCHEMA "{schema}" CASCADE')
        conn.run("COMMIT")
        conn.close()


class _SchemaScope:
    def __init__(self, connection, schema: str, public_baseline: dict[str, int]):
        self.connection = connection
        self.schema = schema
        self.public_baseline = public_baseline


@pytest.fixture(autouse=True)
def clean_test_tables(db):
    """Reset ONLY the isolated test schema between tests."""
    yield
    db.connection.run("TRUNCATE meter_transactions, ingestion_runs RESTART IDENTITY CASCADE")
    db.connection.run("COMMIT")  # make the cleanup visible to other connections


@pytest.fixture
def conn(db):
    return db.connection


def _run_row(conn, run_id: int) -> dict:
    row = conn.run(
        "SELECT source, status, records_processed, records_stored, "
        "records_skipped, source_timestamp, finished_at, error "
        "FROM ingestion_runs WHERE run_id = :run_id",
        run_id=run_id,
    )[0]
    return {
        "source": row[0],
        "status": row[1],
        "processed": int(row[2]),
        "stored": int(row[3]),
        "skipped": int(row[4]),
        "source_timestamp": row[5],
        "finished_at": row[6],
        "error": row[7],
    }


def _definition(freshness_hours: float = 24.0) -> SourceDefinition:
    return SourceDefinition(
        name="fake_source",
        provider="test",
        adapter="fake",
        freshness_hours=freshness_hours,
    )


class TestIsolation:
    """Prove the suite neither sees nor mutates production/public data."""

    def test_runs_in_dedicated_non_public_schema(self, conn, db):
        current = conn.run("SELECT current_schema()")[0][0]
        assert current == db.schema
        assert current.startswith("pytest_sf_parking_")
        assert current != "public"

    def test_public_production_tables_untouched(self, db):
        other = connect()
        try:
            counts = _table_counts(other)
        finally:
            other.close()
        assert counts == db.public_baseline


class TestSuccessfulIngestion:
    def test_records_and_run_provenance_are_stored(self, conn):
        result = run_ingestion(conn, FakeAdapter([_record(0), _record(1)]))

        assert result.ok
        assert result.records_processed == 2
        assert result.records_stored == 2
        assert result.source_timestamp == datetime.fromisoformat("2026-08-20T09:15:00")

        stored = conn.run(
            "SELECT transmission_id, post_id, session_start::text, duration_minutes "
            "FROM meter_transactions ORDER BY transmission_id"
        )
        assert len(stored) == 2
        assert stored[0][0] == "tx-0"
        assert stored[0][1] == "100-00001"
        assert stored[0][3] == 60

        # Per-record provenance: traceable to source, run and retrieval time.
        provenance = conn.run(
            "SELECT t.source, t.run_id, t.retrieved_at IS NOT NULL, r.status "
            "FROM meter_transactions t JOIN ingestion_runs r USING (run_id)"
        )
        assert len(provenance) == 2
        assert all(
            p[0] == "fake_source" and p[1] == result.run_id and p[2] and p[3] == "succeeded"
            for p in provenance
        )

        run = _run_row(conn, result.run_id)
        assert run["source"] == "fake_source"
        assert run["status"] == "succeeded"
        assert run["finished_at"] is not None
        assert run["error"] is None


class TestEmptySource:
    def test_empty_source_succeeds_with_zero_counts(self, conn):
        result = run_ingestion(conn, FakeAdapter([]))

        assert result.ok
        assert (result.records_processed, result.records_stored) == (0, 0)
        run = _run_row(conn, result.run_id)
        assert run["status"] == "succeeded"
        assert conn.run("SELECT count(*) FROM meter_transactions")[0][0] == 0


class TestMalformedRecords:
    def test_malformed_rows_are_skipped_and_counted(self, conn):
        records = [_record(0), InvalidRecord(error="bad row"), InvalidRecord(error="worse")]
        result = run_ingestion(conn, FakeAdapter(records))

        assert result.ok
        assert result.records_processed == 1
        assert result.records_skipped == 2
        assert result.records_stored == 1
        assert len(conn.run("SELECT 1 FROM meter_transactions")) == 1
        run = _run_row(conn, result.run_id)
        assert run["status"] == "succeeded"
        assert run["skipped"] == 2


class TestIdempotency:
    def test_repeated_ingestion_does_not_duplicate(self, conn):
        first = run_ingestion(conn, FakeAdapter([_record(0), _record(1)]))
        second = run_ingestion(conn, FakeAdapter([_record(0), _record(1)]))

        assert (first.records_stored, second.records_stored) == (2, 0)
        count, distinct = conn.run(
            "SELECT count(*), count(DISTINCT (transmission_id, post_id)) FROM meter_transactions"
        )[0]
        assert count == distinct == 2
        # Each run is recorded separately with honest counts.
        statuses = {row[0] for row in conn.run("SELECT status FROM ingestion_runs ORDER BY run_id")}
        assert statuses == {"succeeded"}

    def test_updated_record_key_conflict_keeps_original(self, conn):
        run_ingestion(conn, FakeAdapter([_record(0)]))
        changed = IngestionRecord(
            key=_record(0).key,
            values={**_record(0).values, "duration_minutes": 999},
        )
        result = run_ingestion(conn, FakeAdapter([changed]))

        assert result.records_stored == 0
        stored = conn.run(
            "SELECT duration_minutes FROM meter_transactions WHERE transmission_id = 'tx-0'"
        )[0][0]
        assert int(stored) == 60


class TestFailureHandling:
    def test_failing_source_is_marked_failed_with_error(self, db):
        conn = db.connection
        run_ingestion(conn, FakeAdapter([_record(0)]))  # prior good state

        boom = FakeAdapter([_record(1)], error=RuntimeError("DataSF exploded"))
        result = run_ingestion(conn, boom)

        assert not result.ok
        assert result.status == "failed"
        assert "DataSF exploded" in result.error
        run = _run_row(conn, result.run_id)
        assert run["status"] == "failed"
        assert "DataSF exploded" in run["error"]

        # Prior successful data survives; the failure is visible, not silent.
        other = _connect_with_schema(db.schema)  # independent connection
        try:
            ids = {row[0] for row in other.run("SELECT transmission_id FROM meter_transactions")}
            assert ids == {"tx-0"}
            failed_runs = other.run("SELECT count(*) FROM ingestion_runs WHERE status = 'failed'")[
                0
            ][0]
            assert int(failed_runs) == 1
        finally:
            other.close()

    def test_batches_before_failure_are_kept_and_rerun_completes(self, conn):
        class PartialAdapter(FakeAdapter):
            def fetch(self, options):
                yield _record(5)
                yield _record(6)
                raise ConnectionError("network died mid-stream")

        # batch_size=1 so each yielded record is committed before the failure.
        partial = run_ingestion(conn, PartialAdapter(), options={}, batch_size=1)
        assert not partial.ok
        assert partial.records_stored == 2
        assert conn.run("SELECT count(*) FROM meter_transactions")[0][0] == 2

        retry = run_ingestion(
            conn,
            FakeAdapter([_record(5), _record(6), _record(7)]),
            options={},
            batch_size=1,
        )
        assert retry.ok
        assert retry.records_stored == 1  # tx-5/tx-6 already present, tx-7 new
        assert conn.run("SELECT count(*) FROM meter_transactions")[0][0] == 3


class TestFreshnessHealth:
    def test_never_run_source_reports_unknown_state(self, conn):
        health = source_health(conn, {"fake_source": _definition()})["fake_source"]
        assert health.state == "never_run"
        assert not health.healthy

    def test_recent_success_is_fresh(self, conn):
        run_ingestion(conn, FakeAdapter([_record(0)]))
        health = source_health(conn, {"fake_source": _definition(24.0)})["fake_source"]
        assert health.state == "fresh"
        assert health.healthy

    def test_old_success_beyond_sla_is_stale(self, conn):
        result = run_ingestion(conn, FakeAdapter([_record(0)]))
        old = datetime.now(UTC) - timedelta(hours=72)
        conn.run(
            "UPDATE ingestion_runs SET started_at = :t, finished_at = :t WHERE run_id = :r",
            t=old,
            r=result.run_id,
        )
        conn.run("COMMIT")

        health = source_health(conn, {"fake_source": _definition(24.0)})["fake_source"]
        assert health.state == "stale"
        assert not health.healthy
        assert health.age_hours > 24

    def test_recent_failure_is_reported_as_failed_not_healthy(self, conn):
        good = run_ingestion(conn, FakeAdapter([_record(0)]))
        assert good.ok
        bad_result = run_ingestion(conn, FakeAdapter(error=RuntimeError("down")))
        assert bad_result.status == "failed"

        health = source_health(conn, {"fake_source": _definition(168.0)})["fake_source"]
        assert health.state == "failed"
        assert not health.healthy
        assert health.last_success_at is not None
