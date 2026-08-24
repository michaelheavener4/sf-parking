"""DataSF adapter for SFMTA parking meter revenue transactions.

Dataset ``imvp-dq3v`` — "SFMTA Parking Meter Detailed Revenue Transactions":
one row per paid transaction at one meter. ``METER_EVENT_TYPE`` is NS (new
session) or AT (additional time on the same session), so sessions can be
reconstructed per meter; start/end times, payment type and amount support
duration, turnover and demand analysis.

All DataSF/Socrata specifics are confined to this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ..datasf import DataSFClient
from ..ingestion.framework import IngestionRecord, InvalidRecord

#: DataSF exposes ``session_start_dt``/``session_end_dt`` as Socrata *floating*
#: timestamps: wall-clock times with no offset, reported in the operating
#: agency's local clock. SFMTA meters operate on America/Los_Angeles time;
#: the observed session distribution (peak 09:00-16:00 with a hard stop at
#: 18:00) only makes sense as Pacific wall-clock time. See docs/ROADMAP.md.
SOURCE_TZ = ZoneInfo("America/Los_Angeles")

DATASET_ID = "imvp-dq3v"
ADAPTER_NAME = "sfmta_meter_transactions"

TARGET_TABLE = "meter_transactions"
COLUMNS = [
    "transmission_id",
    "post_id",
    "street_block",
    "payment_type",
    "meter_event_type",
    "session_start",
    "session_end",
    "duration_minutes",
    "gross_paid_amt",
]
CONFLICT_COLUMNS = ["transmission_id", "post_id"]

DEFAULT_WINDOW_DAYS = 7
DEFAULT_PAGE_SIZE = 50_000


def parse_socrata_timestamp(value: Any) -> datetime | None:
    """Parse a Socrata floating timestamp like ``2022-11-18T07:00:00.000``.

    Floating timestamps carry no offset and represent America/Los_Angeles
    wall-clock time, so the source zone is attached here and the value is
    returned as an aware datetime. PostgreSQL ``timestamptz`` then stores the
    correct absolute instant instead of misreading the wall clock as UTC.

    DST edge cases follow PEP 495 / :mod:`zoneinfo` semantics:

    - Ambiguous local times (fall-back, e.g. ``2025-11-02T01:30``) resolve to
      the *first* occurrence (``fold=0``, still PDT/UTC-7).
    - Nonexistent local times (spring-forward, e.g. ``2026-03-08T02:30``)
      keep the wall clock with the pre-transition offset; converting to UTC
      lands after the transition.
    """
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=SOURCE_TZ)
    return parsed.astimezone(SOURCE_TZ)


def normalize_transaction(row: dict[str, Any]) -> IngestionRecord:
    """Normalize one raw DataSF transaction row.

    Raises ValueError for records missing required identity/timestamp fields;
    the adapter turns those into :class:`InvalidRecord` so a few bad rows do
    not abort an otherwise valid run.
    """
    transmission_id = (row.get("transmission_datetime") or "").strip()
    post_id = (row.get("post_id") or "").strip()
    if not transmission_id or not post_id:
        raise ValueError(f"transaction missing identity fields: {row!r}")

    session_start = parse_socrata_timestamp(row.get("session_start_dt"))
    if session_start is None:
        raise ValueError(f"transaction {transmission_id!r} has no session start")

    session_end = parse_socrata_timestamp(row.get("session_end_dt"))
    duration_minutes = None
    if session_end is not None:
        # Subtract absolute instants: Python ignores utcoffset when both
        # operands share the same tzinfo, which would silently report wall
        # clock elapsed time across DST transitions instead of real time.
        seconds = (
            session_end.astimezone(UTC) - session_start.astimezone(UTC)
        ).total_seconds()
        duration_minutes = int(seconds // 60)

    amount = row.get("gross_paid_amt")
    gross_paid_amt = float(amount) if amount not in (None, "") else None

    return IngestionRecord(
        key=(transmission_id, post_id),
        values={
            "transmission_id": transmission_id,
            "post_id": post_id,
            "street_block": row.get("street_block") or None,
            "payment_type": row.get("payment_type") or None,
            "meter_event_type": row.get("meter_event_type") or None,
            "session_start": session_start,
            "session_end": session_end,
            "duration_minutes": duration_minutes,
            "gross_paid_amt": gross_paid_amt,
        },
        source_timestamp=session_start,
    )


def _window_where(options: dict[str, Any], *, now: datetime | None = None) -> str:
    # DataSF floating timestamps are America/Los_Angeles wall-clock times, so
    # the rolling window boundary must be expressed on that clock. A naive
    # ``now`` is interpreted as UTC (never as source-local time); aware
    # datetimes are used as-is. The boundary is converted onto the source
    # clock and compared against a naive local string, keeping the Socrata
    # filter deterministic regardless of host timezone.
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    local_now = now.astimezone(SOURCE_TZ).replace(tzinfo=None)
    window_days = int(options.get("window_days", DEFAULT_WINDOW_DAYS))
    where = options.get("where")
    clauses = []
    if window_days > 0:
        since = (local_now - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S")
        clauses.append(f"session_start_dt >= '{since}'")
    if where:
        clauses.append(f"({where})")
    if not clauses:
        raise ValueError("refusing unbounded ingestion: set window_days or where")
    return " AND ".join(clauses)


class DataSFMeterTransactionsAdapter:
    """Streams normalized meter transactions from DataSF."""

    name = ADAPTER_NAME
    registry_key = "datasf_meter_transactions"
    target_table = TARGET_TABLE
    columns = COLUMNS
    conflict_columns = CONFLICT_COLUMNS

    def __init__(self, client: DataSFClient | None = None) -> None:
        self._client = client

    def _rows(self, options: dict[str, Any]) -> Iterable[dict[str, Any]]:
        client = self._client or DataSFClient(timeout=120)
        owned = self._client is None
        try:
            yield from client.iter_rows(
                DATASET_ID,
                where=_window_where(options),
                order=":id",
                batch_size=int(options.get("page_size", DEFAULT_PAGE_SIZE)),
            )
        finally:
            if owned:
                client.close()

    def fetch(self, options: dict[str, Any]) -> Iterable[IngestionRecord | InvalidRecord]:
        for raw in self._rows(options):
            try:
                yield normalize_transaction(raw)
            except (ValueError, TypeError, KeyError) as exc:
                yield InvalidRecord(error=f"{type(exc).__name__}: {exc}")
