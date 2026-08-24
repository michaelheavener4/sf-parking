"""Deterministic parking-state features and baseline availability score.

Computes reproducible per-meter features from ingested meter transactions.
Everything here is a *deterministic function of database state plus explicit
parameters* (window, instant): no randomness, no hidden clocks, no fitted
parameters - re-running against the same data yields byte-identical output.

The availability score is NOT a calibrated probability. It is a transparent
occupancy ratio over observed history with documented assumptions; see
docs/PARKING_STATE.md. All time-of-day logic uses America/Los_Angeles wall
clock (the source semantics established for SFMTA timestamps), while elapsed
time uses absolute instants, so DST transitions are handled correctly.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import pg8000.native

SF_TZ = ZoneInfo("America/Los_Angeles")

#: Version tag of the deterministic baseline definition. Bump when the
#: formula changes so stored/reported scores stay interpretable.
BASELINE_METHOD = "deterministic_v0"

SLOT_MINUTES = 60


@dataclass(frozen=True, slots=True)
class MeterFeatures:
    """Aggregate parking activity features for one meter over a window."""

    post_id: str
    window_start: datetime
    window_end: datetime
    session_count: int
    active_days: int
    total_paid_minutes: int
    mean_session_minutes: float
    median_session_minutes: float
    last_session_at: datetime

    @property
    def sessions_per_active_day(self) -> float:
        if self.active_days == 0:
            return 0.0
        return self.session_count / self.active_days


@dataclass(frozen=True, slots=True)
class AvailabilityEstimate:
    """Deterministic baseline availability for one meter at one instant."""

    post_id: str
    at: datetime
    score: float | None  # None = insufficient history; not a probability
    method: str
    evidence_days: int
    evidence_sessions: int
    slot_occupied_minutes: float
    slot_possible_minutes: float

    @property
    def sufficient_history(self) -> bool:
        return self.score is not None


def _window(now: datetime, window_days: int) -> tuple[datetime, datetime]:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    window_end = now.astimezone(UTC)
    return window_end - timedelta(days=window_days), window_end


def meter_features(
    conn: pg8000.native.Connection,
    *,
    window_days: int = 28,
    now: datetime | None = None,
) -> list[MeterFeatures]:
    """Per-meter activity aggregates over completed sessions in the window.

    A session belongs to the window containing its ``session_start``;
    durations always come from absolute instants (DST-safe). Meters without
    completed sessions in the window are omitted entirely.
    """
    window_start, window_end = _window(now or datetime.now(UTC), window_days)
    rows = conn.run(
        "SELECT post_id, count(*), "
        "count(DISTINCT (session_start AT TIME ZONE :tz)::date), "
        "COALESCE(SUM(duration_minutes), 0), AVG(duration_minutes), "
        "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_minutes), "
        "MAX(session_end) "
        "FROM meter_transactions "
        "WHERE session_start >= :ws AND session_start < :we "
        "AND session_end IS NOT NULL "
        "GROUP BY post_id ORDER BY post_id",
        tz="America/Los_Angeles",
        ws=window_start,
        we=window_end,
    )
    return [
        MeterFeatures(
            post_id=row[0],
            window_start=window_start,
            window_end=window_end,
            session_count=int(row[1]),
            active_days=int(row[2]),
            total_paid_minutes=int(row[3] or 0),
            mean_session_minutes=float(row[4] or 0.0),
            median_session_minutes=float(row[5] or 0.0),
            last_session_at=row[6],
        )
        for row in rows
    ]


def _local_hour_slot(at: datetime) -> tuple[datetime, datetime]:
    """Absolute-instant bounds of the America/Los_Angeles clock hour of ``at``.

    Bounds are computed in absolute time (UTC + 1h), never wall-clock
    addition, so slots stay exactly one real hour wide across DST
    transitions. A repeated fall-back hour resolves via ``fold``.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    local_start = at.astimezone(SF_TZ).replace(
        minute=0, second=0, microsecond=0
    )
    lo = local_start.astimezone(UTC)
    return lo, lo + timedelta(hours=1)


def _overlap_seconds(
    session_start: datetime,
    session_end: datetime,
    slot_start: datetime,
    slot_end: datetime,
) -> float:
    """Elapsed seconds between an absolute-instant session and a slot."""
    latest_start = max(session_start, slot_start)
    earliest_end = min(session_end, slot_end)
    return max(0.0, (earliest_end - latest_start).total_seconds())


def _candidate_slot_starts(local_day, hour: int) -> list[datetime]:
    """Distinct absolute instants where ``hour:00`` occurs on ``local_day``.

    Normal days yield one start; fall-back days yield two (the repeated
    local hour); nonexistent spring-forward hours yield none.
    """
    def wall(fold: int) -> datetime:
        return datetime(local_day.year, local_day.month, local_day.day,
                        hour, tzinfo=SF_TZ, fold=fold)

    # Existence test uses the fold-0 mapping: if its UTC offset changes on
    # round-trip through UTC, this wall time does not exist on this date.
    w0 = wall(0)
    if w0.utcoffset() != w0.astimezone(UTC).astimezone(SF_TZ).utcoffset():
        return []

    starts: list[datetime] = []
    for fold in (0, 1):
        lo = wall(fold).astimezone(UTC)
        if all(lo != s for s in starts):
            starts.append(lo)
    return sorted(starts)


def _iter_slot_bounds(
    window_start: datetime,
    window_end: datetime,
    hour: int,
) -> Iterable[tuple[datetime, datetime]]:
    """Yield absolute bounds of the target clock hour for every occurrence
    within the window plus one day of slack on either side."""
    local_day = (
        window_start.astimezone(SF_TZ).date() - timedelta(days=1)
    )
    final_day = window_end.astimezone(SF_TZ).date() + timedelta(days=1)
    while local_day <= final_day:
        for lo in _candidate_slot_starts(local_day, hour):
            yield lo, lo + timedelta(hours=1)
        local_day += timedelta(days=1)


def availability_baseline(
    conn: pg8000.native.Connection,
    post_id: str,
    *,
    at: datetime | None = None,
    window_days: int = 28,
    now: datetime | None = None,
) -> AvailabilityEstimate:
    """Score how likely a spot at this meter was free, historically.

    Definition (method ``deterministic_v0``):

        score = 1 - (sum of session minutes overlapping this clock hour,
                     summed over every local date in the observation span)
                / (evidence_days * 60)

    Assumptions - documented in docs/PARKING_STATE.md, deliberately crude:

    * Paid transactions under-count occupancy (unpaid windows, expired-but-
      parked time are invisible), so the true free probability is lower.
    * The meter is assumed observable from its first session in the window
      onward; ``evidence_days`` counts those days only.
    * The score ignores regulation schedules: during unregulated hours the
      real availability is ~1 regardless of the score.
    * With no completed sessions in the window there is no basis for any
      number: ``score`` is ``None`` rather than a guess.

    Deterministic: a pure function of database state and the parameters.
    """
    if at is None:
        at = now or datetime.now(UTC)
    elif at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    at = at.astimezone(UTC)
    window_start, window_end = _window(at, window_days)

    slot_start, _ = _local_hour_slot(at)
    local_hour = slot_start.astimezone(SF_TZ).hour

    sessions = [
        (row[0], row[1])
        for row in conn.run(
            "SELECT session_start, session_end FROM meter_transactions "
            "WHERE post_id = :post_id AND session_end IS NOT NULL "
            "AND session_start < :we AND session_end >= :ws "
            "ORDER BY session_start",
            post_id=post_id,
            ws=window_start,
            we=window_end,
        )
    ]

    if not sessions:
        return AvailabilityEstimate(
            post_id=post_id,
            at=at,
            score=None,
            method=BASELINE_METHOD,
            evidence_days=0,
            evidence_sessions=0,
            slot_occupied_minutes=0.0,
            slot_possible_minutes=0.0,
        )

    starts = [s for s, _ in sessions]
    occupied_seconds = 0.0
    total = len(sessions)
    for lo, hi in _iter_slot_bounds(window_start, window_end, local_hour):
        # Sessions are sorted by start time, so scanning can stop at the
        # first session starting after the slot; the bisect skip assumes no
        # session spans more than ~48h (SFMTA sessions are bounded by meter
        # time limits far below that).
        i = max(bisect_left(starts, lo - timedelta(days=2)) - 1, 0)
        while i < total:
            session_start, session_end = sessions[i]
            if session_start >= hi:
                break
            occupied_seconds += _overlap_seconds(
                session_start, session_end, lo, hi
            )
            i += 1

    first_local_day = sessions[0][0].astimezone(SF_TZ).date()
    last_local_day = sessions[-1][1].astimezone(SF_TZ).date()
    span_days = (last_local_day - first_local_day).days + 1
    evidence_days = max(span_days, 1)
    slot_possible = evidence_days * SLOT_MINUTES
    occupied_minutes = occupied_seconds / 60.0
    score = round(max(0.0, 1.0 - occupied_minutes / slot_possible), 3)

    return AvailabilityEstimate(
        post_id=post_id,
        at=at,
        score=score,
        method=BASELINE_METHOD,
        evidence_days=evidence_days,
        evidence_sessions=len(sessions),
        slot_occupied_minutes=round(occupied_minutes, 3),
        slot_possible_minutes=float(slot_possible),
    )


def summarize_features(features: Iterable[MeterFeatures]) -> dict[str, Any]:
    """Portfolio-level summary; useful for regression checks."""
    feats = list(features)
    return {
        "meters": len(feats),
        "sessions": sum(f.session_count for f in feats),
        "total_paid_minutes": sum(f.total_paid_minutes for f in feats),
        "mean_session_minutes": (
            round(fmean(f.mean_session_minutes for f in feats), 3) if feats else 0.0
        ),
    }
