"""Integration tests for identifier integrity and timestamp migration.

Requires PostgreSQL/PostGIS on localhost:5432 (``docker compose up -d``).
Uses the same throwaway-schema isolation strategy as test_ingestion.py:
public/production tables are never touched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pg8000
import pytest

from sf_parking.database import apply_schema, connect
from sf_parking.ingestion.health import post_id_coverage
from sf_parking.migrations import retag_source_local_timestamps

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"


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


@pytest.fixture()
def conn():
    schema = f"pytest_sf_parking_{uuid.uuid4().hex[:12]}"
    conn = connect()
    conn.run(f'CREATE SCHEMA "{schema}"')
    conn.run("COMMIT")
    conn.run(f'SET search_path TO "{schema}", public')  # generated name only
    apply_schema(conn, SCHEMA_PATH)
    yield conn
    conn.close()
    cleanup = connect()
    try:
        cleanup.run(f'DROP SCHEMA "{schema}" CASCADE')
        cleanup.run("COMMIT")
    finally:
        cleanup.close()


class TestPostIdCoverage:
    @pytest.fixture(autouse=True)
    def _setup(self, conn):
        self.conn = conn

    def _insert_meter(self, post_id: str) -> None:
        self.conn.run(
            "INSERT INTO parking_meters (post_id, latitude, longitude) "
            "VALUES (:p, 37.77, -122.41)",
            p=post_id,
        )

    def _insert_transaction(self, tx: str, post_id: str) -> None:
        self.conn.run(
            "INSERT INTO meter_transactions (transmission_id, post_id, session_start, "
            "source, retrieved_at) VALUES (:t, :p, :s, 'test', now())",
            t=tx,
            p=post_id,
            s=datetime(2026, 8, 20, 9, tzinfo=UTC),
        )

    def test_full_coverage_is_healthy(self):
        for post in ("102-02990", "658-03007"):
            self._insert_meter(post)
            self._insert_transaction(f"tx-{post}", post)

        coverage = post_id_coverage(self.conn)

        assert coverage.transaction_post_ids == 2
        assert coverage.matched_meter_post_ids == 2
        assert coverage.coverage_ratio == 1.0
        assert coverage.healthy
        assert coverage.unmatched_sample == ()

    def test_unmatched_ids_are_reported_not_dropped(self):
        # A transaction whose meter is missing from the inventory snapshot:
        # preserved as an observation, surfaced as incomplete coverage.
        self._insert_meter("102-02990")
        self._insert_transaction("tx-known", "102-02990")
        self._insert_transaction("tx-unknown", "999-99999")

        coverage = post_id_coverage(self.conn)

        assert coverage.transaction_post_ids == 2
        assert coverage.matched_meter_post_ids == 1
        assert not coverage.healthy
        assert coverage.unmatched_sample == ("999-99999",)

    def test_no_transactions_means_full_coverage_by_definition(self):
        self._insert_meter("102-02990")

        coverage = post_id_coverage(self.conn)

        assert coverage.transaction_post_ids == 0
        assert coverage.coverage_ratio == 1.0


class TestRetagMigration:
    @pytest.fixture(autouse=True)
    def _setup(self, conn):
        self.conn = conn

    def _insert_misstored(self, tx: str, stored_naive_utc: datetime) -> None:
        """Insert a row the way the old adapter did: local wall clock as UTC."""
        self.conn.run(
            "INSERT INTO ingestion_runs (source, status) "
            "VALUES ('sfmta_meter_transactions', 'succeeded')"
        )
        run_id = int(self.conn.run("SELECT max(run_id) FROM ingestion_runs")[0][0])
        end = (stored_naive_utc + timedelta(hours=2)).replace(tzinfo=UTC)
        self.conn.run(
            "UPDATE ingestion_runs SET source_timestamp = :ts WHERE run_id = :r",
            ts=stored_naive_utc.replace(tzinfo=UTC),
            r=run_id,
        )
        self.conn.run(
            "INSERT INTO meter_transactions (transmission_id, post_id, session_start, "
            "session_end, duration_minutes, source, run_id, retrieved_at) "
            "VALUES (:t, '665-01003', :s, :e, 120, 'test', :r, now())",
            t=tx,
            s=stored_naive_utc.replace(tzinfo=UTC),
            e=end,
            r=run_id,
        )

    def test_retags_local_wall_clock_as_true_instant(self):
        # 2026-08-17T04:31:23 PDT was mis-stored as 04:31:23+00; the true
        # instant is 11:31:23+00.
        self._insert_misstored("tx-1", datetime(2026, 8, 17, 4, 31, 23))  # noqa: DTZ001

        result = retag_source_local_timestamps(self.conn)

        assert result.status == "applied"
        assert result.rows_retagged == 1
        assert result.total_rows_after == 1
        start, end = self.conn.run(
            "SELECT session_start, session_end FROM meter_transactions"
        )[0]
        assert start.astimezone(UTC) == datetime(2026, 8, 17, 11, 31, 23, tzinfo=UTC)
        assert end.astimezone(UTC) == datetime(2026, 8, 17, 13, 31, 23, tzinfo=UTC)

    def test_is_recorded_and_idempotent(self):
        self._insert_misstored("tx-1", datetime(2026, 8, 17, 4, 31, 23))  # noqa: DTZ001

        first = retag_source_local_timestamps(self.conn)
        second = retag_source_local_timestamps(self.conn)

        assert first.status == "applied"
        assert second.status == "skipped"
        runs = self.conn.run(
            "SELECT count(*) FROM ingestion_runs "
            "WHERE source LIKE 'migration:%' AND status = 'succeeded'"
        )[0][0]
        assert int(runs) == 1
        # Second call must not shift timestamps again.
        start = self.conn.run("SELECT session_start FROM meter_transactions")[0][0]
        assert start.astimezone(UTC) == datetime(2026, 8, 17, 11, 31, 23, tzinfo=UTC)

    def test_preserves_winter_offset_across_dst_boundary(self):
        # PST in November before the fall-back transition: offset -8.
        self._insert_misstored("tx-1", datetime(2022, 11, 18, 7, 0, 0))  # noqa: DTZ001

        retag_source_local_timestamps(self.conn)

        start = self.conn.run("SELECT session_start FROM meter_transactions")[0][0]
        assert start.astimezone(UTC) == datetime(2022, 11, 18, 15, 0, tzinfo=UTC)
