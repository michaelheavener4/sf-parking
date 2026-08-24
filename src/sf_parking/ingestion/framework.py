"""Generic source/adapter ingestion runner with run-level provenance.

The framework knows nothing about providers: an adapter yields normalized
records for a target table, and the runner streams them through a staging
``COPY`` with idempotent conflict handling while recording every run in
``ingestion_runs``. A failed source is always represented as a failed run —
never as a silently successful overall load.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import Any, Protocol

import pg8000.native

from ..database import transaction

BATCH_SIZE = 10_000
ProgressCallback = Callable[[int, int, int, float], None]


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class IngestionRecord:
    """A normalized record destined for ``target_table``.

    ``key`` values must map 1:1 onto the adapter's ``conflict_columns`` so
    re-ingesting the same source row is idempotent.
    """

    key: tuple[Any, ...]
    values: dict[str, Any]
    source_timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class InvalidRecord:
    """A raw source record that could not be normalized."""

    error: str


Record = IngestionRecord | InvalidRecord


class Adapter(Protocol):
    """Contract every source adapter fulfils; provider logic stays isolated."""

    name: str
    target_table: str
    columns: list[str]
    conflict_columns: list[str]

    def fetch(self, options: dict[str, Any]) -> Iterable[Record]: ...


@dataclass(slots=True)
class RunResult:
    source: str
    run_id: int
    status: str
    records_processed: int = 0
    records_stored: int = 0
    records_skipped: int = 0
    source_timestamp: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


def _serialize(value: Any) -> Any:
    return "" if value is None else str(value)


def stage_table_ddl(conn: pg8000.native.Connection, target_table: str, columns: list[str]) -> str:
    """Build a temp staging DDL whose column types mirror the target table.

    Resolves against the connection's ``search_path`` (``current_schema()``)
    so ingestion works in any schema, not just ``public``.
    """
    wanted = set(columns)
    rows = conn.run(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = :table AND table_schema = current_schema()",
        table=target_table,
    )
    types = {name: data_type for name, data_type in rows}
    missing = wanted - set(types)
    if missing:
        raise ValueError(f"unknown columns for {target_table}: {sorted(missing)}")
    defs = ", ".join(f"{col} {types[col]}" for col in columns)
    return f"CREATE TEMP TABLE stage_{target_table} ({defs})"


def _csv_stream(values: Iterable[list[Any]], columns: list[str]) -> Iterator[bytes]:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in values:
        writer.writerow([_serialize(row[col]) for col in columns])
        yield buffer.getvalue().encode("utf-8")
        buffer.seek(0)
        buffer.truncate(0)


def _start_run(conn: pg8000.native.Connection, source: str) -> int:
    with transaction(conn):
        result = conn.run(
            "INSERT INTO ingestion_runs (source, started_at, status) "
            "VALUES (:source, :started_at, 'running') RETURNING run_id",
            source=source,
            started_at=utcnow(),
        )
        run_id = int(result[0][0])
    return run_id


def _finish_run(
    conn: pg8000.native.Connection,
    run_id: int,
    *,
    status: str,
    processed: int,
    stored: int,
    skipped: int,
    source_timestamp: datetime | None,
    error: str | None,
) -> None:
    with transaction(conn):
        conn.run(
            "UPDATE ingestion_runs SET finished_at = :finished_at, status = :status, "
            "records_processed = :processed, records_stored = :stored, "
            "records_skipped = :skipped, source_timestamp = :source_timestamp, "
            "error = :error WHERE run_id = :run_id",
            run_id=run_id,
            finished_at=utcnow(),
            status=status,
            processed=processed,
            stored=stored,
            skipped=skipped,
            source_timestamp=source_timestamp,
            error=error,
        )


def _ingest_batch(
    conn: pg8000.native.Connection,
    adapter: Adapter,
    batch: list[IngestionRecord],
    run_id: int,
    retrieved_at: datetime,
) -> int:
    """Stage one batch and insert it idempotently; returns rows inserted."""
    columns = [*adapter.columns, "source", "run_id", "retrieved_at"]
    updates = ", ".join(adapter.conflict_columns)
    with transaction(conn):
        conn.run(f"DROP TABLE IF EXISTS stage_{adapter.target_table}")
        conn.run(stage_table_ddl(conn, adapter.target_table, adapter.columns))
        conn.run(
            f"COPY stage_{adapter.target_table} ({', '.join(adapter.columns)}) "
            "FROM STDIN WITH (FORMAT csv)",
            stream=_csv_stream((record.values for record in batch), adapter.columns),
        )
        result = conn.run(
            f"WITH ins AS ("
            f"INSERT INTO {adapter.target_table} ({', '.join(columns)}) "
            f"SELECT {', '.join(adapter.columns)}, "
            f":source_name, :run_id, :retrieved_at "
            f"FROM stage_{adapter.target_table} "
            f"ON CONFLICT ({updates}) DO NOTHING "
            f"RETURNING 1) "
            f"SELECT count(*) FROM ins",
            source_name=adapter.name,
            run_id=run_id,
            retrieved_at=retrieved_at,
        )
        inserted = int(result[0][0])
    return inserted


def run_ingestion(
    conn: pg8000.native.Connection,
    adapter: Adapter,
    options: dict[str, Any] | None = None,
    *,
    retrieved_at: datetime | None = None,
    batch_size: int = BATCH_SIZE,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Run one ingestion pass for ``adapter`` and record provenance.

    Batches commit independently so a mid-run failure keeps earlier batches
    (idempotent keys make re-runs safe), but the run itself is marked
    ``failed`` with the error attached.

    ``progress(processed, stored, skipped, elapsed_seconds)`` is called after
    every committed batch (and once at completion when a partial batch exists)
    so CLI callers can provide live feedback without coupling the framework to
    a particular terminal UI.
    """
    options = dict(options or {})
    retrieved_at = retrieved_at or utcnow()
    started_monotonic = __import__("time").monotonic()
    run_id = _start_run(conn, adapter.name)

    processed = stored = skipped = 0
    source_timestamp: datetime | None = None
    status = "succeeded"
    error: str | None = None

    try:
        batch: list[IngestionRecord] = []
        for record in adapter.fetch(options):
            if isinstance(record, InvalidRecord):
                skipped += 1
                continue
            processed += 1
            if record.source_timestamp is not None and (
                source_timestamp is None or record.source_timestamp > source_timestamp
            ):
                source_timestamp = record.source_timestamp
            batch.append(record)
            if len(batch) >= batch_size:
                stored += _ingest_batch(conn, adapter, batch, run_id, retrieved_at)
                batch.clear()
                if progress is not None:
                    progress(processed, stored, skipped, __import__("time").monotonic() - started_monotonic)
        if batch:
            stored += _ingest_batch(conn, adapter, batch, run_id, retrieved_at)
            if progress is not None:
                progress(processed, stored, skipped, __import__("time").monotonic() - started_monotonic)
    except Exception as exc:  # noqa: BLE001 - any failure marks the run failed
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    _finish_run(
        conn,
        run_id,
        status=status,
        processed=processed,
        stored=stored,
        skipped=skipped,
        source_timestamp=source_timestamp,
        error=error,
    )
    return RunResult(
        source=adapter.name,
        run_id=run_id,
        status=status,
        records_processed=processed,
        records_stored=stored,
        records_skipped=skipped,
        source_timestamp=source_timestamp,
        error=error,
    )
