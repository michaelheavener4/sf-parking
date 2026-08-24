"""Freshness/health checks over recorded ingestion runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pg8000.native

from .registry import SourceDefinition


@dataclass(frozen=True, slots=True)
class SourceHealth:
    source: str
    state: str  # "fresh", "stale", "failed", "never_run"
    last_success_at: datetime | None
    last_run_at: datetime | None
    last_run_status: str | None
    age_hours: float | None
    freshness_hours: float

    @property
    def healthy(self) -> bool:
        return self.state == "fresh"


def _hours_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 3600.0


@dataclass(frozen=True, slots=True)
class PostIdCoverage:
    """How many distinct transaction post_ids resolve to known meters.

    DataSF documents transaction ``post_id`` as the key into the meters
    inventory (dataset ``8vzz-qzz9``), so a coverage drop below ~1.0 signals
    either an inventory refresh lag (brand-new/removed meters) or an
    ingestion regression — not a separate identifier namespace.
    """

    source: str
    transaction_post_ids: int
    matched_meter_post_ids: int
    unmatched_sample: tuple[str, ...]

    @property
    def coverage_ratio(self) -> float:
        if self.transaction_post_ids == 0:
            return 1.0
        return self.matched_meter_post_ids / self.transaction_post_ids

    @property
    def healthy(self) -> bool:
        return self.coverage_ratio >= 0.99


def post_id_coverage(
    conn: pg8000.native.Connection,
    *,
    sample_size: int = 10,
) -> PostIdCoverage:
    """Measure join coverage between meter transactions and parking meters."""
    total_row = conn.run("SELECT count(DISTINCT post_id) FROM meter_transactions")[0]
    total = int(total_row[0])
    matched_row = conn.run(
        "SELECT count(*) FROM ("
        "SELECT DISTINCT post_id FROM meter_transactions"
        ") t JOIN parking_meters m USING (post_id)"
    )[0]
    matched = int(matched_row[0])
    sample_rows = conn.run(
        "SELECT t.post_id FROM ("
        "SELECT DISTINCT post_id FROM meter_transactions"
        ") t LEFT JOIN parking_meters m USING (post_id) "
        "WHERE m.post_id IS NULL LIMIT :limit",
        limit=sample_size,
    )
    return PostIdCoverage(
        source="sfmta_meter_transactions",
        transaction_post_ids=total,
        matched_meter_post_ids=matched,
        unmatched_sample=tuple(row[0] for row in sample_rows),
    )


def source_health(
    conn: pg8000.native.Connection,
    sources: dict[str, SourceDefinition],
    *,
    now: datetime | None = None,
) -> dict[str, SourceHealth]:
    """Report per-source health from provenance history and SLAs.

    A source is healthy ("fresh") only when its latest successful run is
    within its freshness SLA and no run has failed since that success.
    A failed source is always surfaced as ``failed`` — never silently healthy.
    """
    now = now or datetime.now(UTC)

    health: dict[str, SourceHealth] = {}
    for name, definition in sources.items():
        last_row = conn.run(
            "SELECT status, started_at, COALESCE(finished_at, started_at) "
            "FROM ingestion_runs WHERE source = :source "
            "ORDER BY started_at DESC LIMIT 1",
            source=name,
        )
        success_row = conn.run(
            "SELECT max(COALESCE(finished_at, started_at)) FROM ingestion_runs "
            "WHERE source = :source AND status = 'succeeded'",
            source=name,
        )
        last_success_at = success_row[0][0] if success_row else None

        if not last_row:
            row_status, started_at, last_run_at = None, None, None
        else:
            row_status, started_at, last_run_at = last_row[0]

        if row_status is None:
            state = "never_run"
        elif row_status == "failed":
            state = "failed"
        elif last_success_at is None:
            state = "never_run"
        else:
            age_hours = _hours_between(last_success_at, now)
            state = "fresh" if age_hours <= definition.freshness_hours else "stale"

        health[name] = SourceHealth(
            source=name,
            state=state,
            last_success_at=last_success_at,
            last_run_at=last_run_at or started_at,
            last_run_status=row_status,
            age_hours=(
                _hours_between(last_success_at, now)
                if row_status != "failed" and last_success_at is not None
                else None
            ),
            freshness_hours=definition.freshness_hours,
        )
    return health
