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
