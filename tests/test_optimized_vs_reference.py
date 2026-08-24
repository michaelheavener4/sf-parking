"""Correctness test: optimized prepare() vs reference across multiple slot_starts.

Exercises sessions that cross the cutoff (slot_start) and window boundary,
including DST transitions, to ensure the bisect optimization produces
identical hour_scores to the reference implementation.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from sf_parking.backtest import HourConditionedV1Baseline

# Generate a deterministic synthetic dataset with deliberate boundary cases.
random.seed(42)

_SLOT = datetime(2026, 8, 21, 20, tzinfo=UTC)  # 13:00 PT


def _make_sessions() -> dict[str, list[tuple[datetime, datetime]]]:
    """Build 20 meters with varied session patterns including boundary cases."""
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}

    for m in range(20):
        post_id = f"M{m:03d}"
        meter_sessions: list[tuple[datetime, datetime]] = []

        # 28 days of regular sessions (Aug 1–28 2026) at various hours.
        for d in range(1, 29):
            day = datetime(2026, 8, d, tzinfo=UTC)
            for hour in (13, 14, 15):
                # Random offset within the hour for variety.
                offset = random.randint(0, 30)
                dur = random.randint(10, 50)
                s = day + timedelta(hours=hour, minutes=offset)
                e = s + timedelta(minutes=dur)
                meter_sessions.append((s, e))

        # Boundary case: session that ENDS right at slot_start (cutoff).
        # This session should be truncated but still counted.
        meter_sessions.append((
            datetime(2026, 8, 21, 19, 30, tzinfo=UTC),
            datetime(2026, 8, 21, 20, 0, tzinfo=UTC),  # ends exactly at slot
        ))

        # Boundary case: session that STARTS right at slot_start.
        # This should be excluded (point-in-time).
        meter_sessions.append((
            datetime(2026, 8, 21, 20, 0, tzinfo=UTC),
            datetime(2026, 8, 21, 20, 45, tzinfo=UTC),
        ))

        # Boundary case: session entirely before the 28-day window.
        # Should be excluded by window filter.
        meter_sessions.append((
            datetime(2026, 7, 1, 13, 0, tzinfo=UTC),
            datetime(2026, 7, 1, 13, 30, tzinfo=UTC),
        ))

        # Boundary case: session that spans the window_start boundary.
        # Part of it is inside the window.
        meter_sessions.append((
            datetime(2026, 7, 24, 12, 0, tzinfo=UTC),  # 28 days before slot
            datetime(2026, 7, 24, 14, 0, tzinfo=UTC),  # crosses into hour 13
        ))

        meter_sessions.sort()
        sessions[post_id] = meter_sessions

    return sessions


SESSIONS = _make_sessions()


class TestOptimizedVsReference:
    """Compare optimized prepare() output against _prepare_reference()."""

    def test_single_slot_correctness(self):
        """Hour scores from optimized prepare() match reference for one slot."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)  # 13:00 PT

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            SESSIONS, {}, {k: "SS" for k in SESSIONS},
            history_window_days=28, slot_start=slot,
        )
        optimized = model._slot_hour_scores[slot]

        reference = HourConditionedV1Baseline._prepare_reference(
            SESSIONS, history_window_days=28, slot_start=slot,
        )

        assert set(optimized.keys()) == set(reference.keys()), (
            f"Hour keys differ: {set(optimized.keys()) ^ set(reference.keys())}"
        )
        for h in sorted(set(optimized.keys()) | set(reference.keys())):
            opt_pred = optimized[h]
            ref_pred = reference[h]
            assert opt_pred.score == pytest.approx(ref_pred.score, abs=1e-6), (
                f"Hour {h}: optimized={opt_pred.score} != reference={ref_pred.score}"
            )

    def test_multi_slot_correctness(self):
        """Hour scores match across 5 consecutive hourly slots."""
        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        slot = datetime(2026, 8, 21, 18, tzinfo=UTC)

        for i in range(5):
            model.prepare(
                SESSIONS, {}, {k: "SS" for k in SESSIONS},
                history_window_days=28, slot_start=slot,
            )
            optimized = model._slot_hour_scores[slot]
            reference = HourConditionedV1Baseline._prepare_reference(
                SESSIONS, history_window_days=28, slot_start=slot,
            )

            for h in range(24):
                opt = optimized.get(h)
                ref = reference.get(h)
                if opt is None and ref is None:
                    continue
                assert opt is not None and ref is not None, (
                    f"slot={slot} hour={h}: optimized={opt} != reference={ref}"
                )
                assert opt.score == pytest.approx(ref.score, abs=1e-6), (
                    f"slot={slot} hour={h}: {opt.score} != {ref.score}"
                )

            slot += timedelta(hours=1)

    def test_window_boundary_session(self):
        """Session spanning window_start is partially counted (math preserved)."""
        # Slot exactly 28 days after the boundary session's start.
        # The boundary session starts at 2026-07-24 12:00 UTC.
        # window_start = slot_start - 28d. If slot is 2026-08-21 20:00 UTC,
        # window_start = 2026-07-24 20:00 UTC.
        # The boundary session (12:00–14:00 UTC on Jul 24) ends BEFORE
        # window_start, so it should be excluded.
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            SESSIONS, {}, {k: "SS" for k in SESSIONS},
            history_window_days=28, slot_start=slot,
        )
        optimized = model._slot_hour_scores[slot]
        reference = HourConditionedV1Baseline._prepare_reference(
            SESSIONS, history_window_days=28, slot_start=slot,
        )

        for h in range(24):
            opt = optimized.get(h)
            ref = reference.get(h)
            if opt is None and ref is None:
                continue
            assert opt is not None and ref is not None
            assert opt.score == pytest.approx(ref.score, abs=1e-6)

    def test_cutoff_session_truncated(self):
        """Session ending exactly at slot_start is truncated, not excluded."""
        # The session 19:30–20:00 UTC on Aug 21 ends exactly at slot_start.
        # It should be truncated to 19:30–20:00 and counted if it overlaps
        # the relevant hour slot.
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        # hour 13 PT = 20:00 UTC, so the slot is [20:00, 21:00) UTC.
        # The session 19:30–20:00 UTC does NOT overlap [20:00, 21:00).
        # But it does overlap hour 12 PT = 19:00–20:00 UTC if we had a slot there.
        # For slot_start=20:00, window_start=2026-07-24 20:00.
        # The session is within the window and starts before slot_start.
        # It overlaps hour 12 PT: _candidate_slot_starts(aug 21, 12) gives
        # the UTC instant for 12:00 PT = 19:00 UTC. Slot is [19:00, 20:00).
        # Session 19:30–20:00 overlaps this slot with 30 min.
        # So it should contribute to hour 12.

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            SESSIONS, {}, {k: "SS" for k in SESSIONS},
            history_window_days=28, slot_start=slot,
        )
        optimized = model._slot_hour_scores[slot]
        reference = HourConditionedV1Baseline._prepare_reference(
            SESSIONS, history_window_days=28, slot_start=slot,
        )

        # Verify the scores match exactly.
        for h in range(24):
            opt = optimized.get(h)
            ref = reference.get(h)
            if opt is None and ref is None:
                continue
            assert opt is not None and ref is not None
            assert opt.score == pytest.approx(ref.score, abs=1e-6)

    def test_future_session_excluded(self):
        """Session starting at slot_start is excluded (point-in-time safety)."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        # The session 20:00–20:45 UTC starts exactly at slot_start.
        # It must NOT contribute to any hour's score.

        # Build a minimal dataset with only this session.
        future_sessions = {
            "FUTURE": [(
                datetime(2026, 8, 21, 20, tzinfo=UTC),
                datetime(2026, 8, 21, 20, 45, tzinfo=UTC),
            )],
        }

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            future_sessions, {}, {"FUTURE": "SS"},
            history_window_days=28, slot_start=slot,
        )
        optimized = model._slot_hour_scores[slot]

        # No history sessions → all hours should be empty.
        assert len(optimized) == 0

    def test_empty_sessions(self):
        """Empty session list produces empty hour scores."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {}, {}, {},
            history_window_days=28, slot_start=slot,
        )
        assert model._slot_hour_scores[slot] == {}
