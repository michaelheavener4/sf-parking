"""Leakage-safe observation frontiers for historical parking evaluation.

A forecast can only be scored when the outcome window is observable. For a
slot [t, t + horizon), the transaction table must contain observations far
enough beyond t to determine whether a paid session overlaps that slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol


class RowConnection(Protocol):
    def run(self, sql: str, **params: object) -> list[list[object]]: ...


@dataclass(frozen=True, slots=True)
class ObservationFrontier:
    max_session_end: datetime
    horizon: timedelta

    @property
    def safe_until(self) -> datetime:
        """Latest cutoff whose complete outcome horizon is observable."""
        return self.max_session_end - self.horizon

    def clamp(self, requested_until: datetime) -> datetime | None:
        """Clamp a requested cutoff to the observable frontier."""
        requested = requested_until
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=UTC)
        requested = requested.astimezone(UTC)
        safe = self.safe_until.astimezone(UTC)
        if safe <= datetime.min.replace(tzinfo=UTC):
            return None
        return min(requested, safe) if requested <= safe else safe


def observation_frontier(
    conn: RowConnection,
    *,
    horizon_minutes: int = 60,
) -> ObservationFrontier | None:
    """Return the database's latest defensible outcome frontier."""
    rows = conn.run("SELECT max(session_end) FROM meter_transactions")
    value = rows[0][0] if rows else None
    if value is None:
        return None
    end = value
    if not isinstance(end, datetime):
        raise TypeError("database returned a non-datetime max(session_end)")
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return ObservationFrontier(
        max_session_end=end.astimezone(UTC),
        horizon=timedelta(minutes=horizon_minutes),
    )


def safe_until(
    conn: RowConnection,
    requested_until: datetime,
    *,
    horizon_minutes: int = 60,
) -> datetime | None:
    """Convenience wrapper returning a leakage-safe cutoff."""
    frontier = observation_frontier(conn, horizon_minutes=horizon_minutes)
    return None if frontier is None else frontier.clamp(requested_until)
