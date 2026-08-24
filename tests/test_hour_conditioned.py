"""Pure-function tests for HourConditionedV1Baseline.

These tests require NO database — they exercise prepare() and predict()
directly with hand-computed session data.  Integration tests that need a
database are in test_backtest.py (TestHourConditionedBacktest).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sf_parking.backtest import (
    DeterministicV0Baseline,
    HourConditionedV1Baseline,
    PlacementSpan,
    score_v0,
)

_V0 = DeterministicV0Baseline()


class TestHourConditionedV1:
    """Hand-computed tests for the hour-conditioned model."""

    def test_correct_pooled_hour_baseline(self):
        """Hour baseline pools across ALL meters at a given hour.

        Scenario (August 2026 PDT, slot at 2026-08-21 20:00 UTC = 13:00 PT):
            M1: 3 sessions at hour 13, each 36 min → occupied = 108 min
                evidence_days(m1,h=13) = 3 → possible = 180
            M2: 3 sessions at hour 13, each 24 min → occupied = 72 min
                evidence_days(m2,h=13) = 3 → possible = 180

            hour_occupied = 108 + 72 = 180
            hour_possible = 180 + 180 = 360
            hour_score = 1 - 180/360 = 0.5
        """
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)  # 13:00 PT

        m1_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 36, tzinfo=UTC))
            for d in (18, 19, 20)
        ]
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 24, tzinfo=UTC))
            for d in (18, 19, 20)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        assert model._slot_hour_scores[slot][13].score == pytest.approx(0.5, abs=0.001)

    def test_correct_meter_hour_deviation(self):
        """Meter deviation = meter_hour_score - hour_score.

        Scenario:
            hour_score(13) = 0.5 (from pooled computation)
            M1: occupied 108/180 → meter_hour_score = 0.4
            deviation(M1,13) = 0.4 - 0.5 = -0.1
        """
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        m1_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 36, tzinfo=UTC))
            for d in (18, 19, 20)
        ]
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 24, tzinfo=UTC))
            for d in (18, 19, 20)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        pred = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        assert pred is not None

        # hour_score = 0.5, meter_hour_score(M1) = 0.4
        # deviation = 0.4 - 0.5 = -0.1
        # w = min(3/14, 1) = 3/14 ≈ 0.214
        # final = 0.5 + 0.214 * (-0.1) = 0.5 - 0.0214 ≈ 0.479
        w = 3.0 / 14.0
        expected = round(0.5 + w * (0.4 - 0.5), 3)
        assert pred.score == pytest.approx(expected, abs=0.001)

    def test_evidence_weighting_at_boundary_values(self):
        """Verify weighting at 0, 1, 7, 14, and 28+ days of evidence."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # Need at least 2 meters so hour baseline is non-trivial.
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]

        def make_m1(days):
            return [
                (datetime(2026, 8, d, 20, tzinfo=UTC),
                 datetime(2026, 8, d, 20, 30, tzinfo=UTC))
                for d in days
            ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)

        # Test 0 days evidence → returns hour baseline (no meter history).
        m1_0 = make_m1([])
        model.prepare(
            {"M1": m1_0, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred_0 = model.predict(m1_0, slot, history_window_days=28, post_id="M1")
        assert pred_0 is not None
        hour_score = model._slot_hour_scores[slot][13].score
        assert pred_0.score == pytest.approx(hour_score, abs=0.001)

        # Test 1 day evidence → w = 1/14 ≈ 0.071.
        model._cache.clear()
        model._slot_hour_scores.clear()
        m1_1 = make_m1([20])
        model.prepare(
            {"M1": m1_1, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred_1 = model.predict(m1_1, slot, history_window_days=28, post_id="M1")
        assert pred_1 is not None
        assert pred_1.evidence_days == 1

        # Test 7 days evidence → w = 7/14 = 0.5.
        model._cache.clear()
        model._slot_hour_scores.clear()
        m1_7 = make_m1(range(14, 21))
        model.prepare(
            {"M1": m1_7, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred_7 = model.predict(m1_7, slot, history_window_days=28, post_id="M1")
        assert pred_7 is not None
        assert pred_7.evidence_days == 7

        # Test 14 days evidence → w = 1.0 (full deviation).
        model._cache.clear()
        model._slot_hour_scores.clear()
        m1_14 = make_m1(range(7, 21))
        model.prepare(
            {"M1": m1_14, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred_14 = model.predict(m1_14, slot, history_window_days=28, post_id="M1")
        assert pred_14 is not None
        assert pred_14.evidence_days == 14
        # With w=1.0, final = hour_score + 1.0 * deviation = meter_hour_score
        meter_hour_score = score_v0(30 * 14, 14 * 60)
        assert pred_14.score == pytest.approx(meter_hour_score, abs=0.001)

        # Test 28 days evidence → w = 1.0 (capped).
        model._cache.clear()
        model._slot_hour_scores.clear()
        # 28 days of evidence: July 24 – Aug 20.
        m1_28 = [
            (datetime(2026, 7, 24, 20, tzinfo=UTC) + timedelta(days=i),
             datetime(2026, 7, 24, 20, 30, tzinfo=UTC) + timedelta(days=i))
            for i in range(28)
        ]
        model.prepare(
            {"M1": m1_28, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred_28 = model.predict(m1_28, slot, history_window_days=28, post_id="M1")
        assert pred_28 is not None
        assert pred_28.evidence_days == 28

        # Monotonic: more evidence → score closer to meter_hour_score.
        meter_scores = [
            pred_1.score if pred_1 else None,
            pred_7.score if pred_7 else None,
            pred_14.score if pred_14 else None,
            pred_28.score if pred_28 else None,
        ]
        # All scores should be between hour_score and meter_hour_score.
        for s in meter_scores:
            assert s is not None
            lo = min(hour_score, meter_hour_score)
            hi = max(hour_score, meter_hour_score)
            assert lo - 0.001 <= s <= hi + 0.001

    def test_clamped_to_unit_interval(self):
        """Final score is always in [0, 1]."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # M1 fully occupied every day.
        m1_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 21, tzinfo=UTC))
            for d in range(14, 21)
        ]
        # M2 never occupied.
        m2_sessions = [
            (datetime(2026, 8, d, 16, tzinfo=UTC),
             datetime(2026, 8, d, 16, tzinfo=UTC))
            for d in range(14, 21)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        assert pred is not None
        assert 0.0 <= pred.score <= 1.0

    def test_no_meter_history_uses_hour_baseline(self):
        """Meter with no sessions at the target hour → hour baseline."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        m1_sessions = []  # no sessions
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        assert pred is not None
        hour_score = model._slot_hour_scores[slot][13].score
        assert pred.score == pytest.approx(hour_score, abs=0.001)
        assert pred.evidence_days == 0

    def test_no_hour_history_falls_back_to_v0(self):
        """If no meter has any hour-level history, falls back to V0."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # M1 has sessions at hour 9, not hour 13.
        m1_sessions = [
            (datetime(2026, 8, d, 16, tzinfo=UTC),
             datetime(2026, 8, d, 16, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions},
            {}, {"M1": "SS"},
            history_window_days=28, slot_start=slot,
        )
        # hour_score for hour 13 should not exist (no sessions at that hour).
        assert slot not in model._slot_hour_scores or 13 not in model._slot_hour_scores[slot]
        pred = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        # Should fall back to V0.
        v0_pred = _V0.predict(m1_sessions, slot, history_window_days=28)
        assert pred is not None
        assert pred.score == v0_pred.score

    def test_different_meters_share_hour_baseline(self):
        """Two meters at the same hour share the same hour_score."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        m1_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 45, tzinfo=UTC))
            for d in range(14, 21)
        ]
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 15, tzinfo=UTC))
            for d in range(14, 21)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        hour_score = model._slot_hour_scores[slot][13].score
        pred1 = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        pred2 = model.predict(m2_sessions, slot, history_window_days=28, post_id="M2")
        assert pred1 is not None
        assert pred2 is not None

        # Both use the same hour_score but have different deviations.
        # M1 is less available (45 min occupied) → negative deviation.
        # M2 is more available (15 min occupied) → positive deviation.
        assert pred1.score < hour_score
        assert pred2.score > hour_score

    def test_meters_isolated_from_each_other(self):
        """A meter's deviation comes only from its own history."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # M1: 45 min occupied at hour 13.
        m1_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 45, tzinfo=UTC))
            for d in range(14, 21)
        ]
        # M2: 15 min occupied at hour 13.
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 15, tzinfo=UTC))
            for d in range(14, 21)
        ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            {"M1": m1_sessions, "M2": m2_sessions},
            {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        pred1 = model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        pred2 = model.predict(m2_sessions, slot, history_window_days=28, post_id="M2")
        assert pred1 is not None
        assert pred2 is not None

        # M1's deviation is its own meter_hour_score - hour_score.
        hour_score = model._slot_hour_scores[slot][13].score
        m1_mhs = score_v0(45 * 7, 7 * 60)
        m2_mhs = score_v0(15 * 7, 7 * 60)
        w = min(7.0 / 14.0, 1.0)
        assert pred1.score == pytest.approx(
            round(hour_score + w * (m1_mhs - hour_score), 3), abs=0.001
        )
        assert pred2.score == pytest.approx(
            round(hour_score + w * (m2_mhs - hour_score), 3), abs=0.001
        )

    def test_deterministic_output(self):
        """Identical inputs produce identical outputs across runs."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]
        all_sessions = {"M1": sessions, "M2": sessions}

        def run():
            m = HourConditionedV1Baseline(evidence_halflife=14.0)
            m.prepare(
                all_sessions, {}, {"M1": "SS", "M2": "SS"},
                history_window_days=28, slot_start=slot,
            )
            return m.predict(sessions, slot, history_window_days=28, post_id="M1")

        r1 = run()
        r2 = run()
        assert r1 == r2

    def test_deterministic_v0_unchanged(self):
        """The V0 model is not modified by hour_conditioned_v1."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]
        v0_result = _V0.predict(sessions, slot, history_window_days=28)
        assert v0_result is not None
        assert v0_result.score == pytest.approx(0.5, abs=0.001)

    def test_method_tag(self):
        assert HourConditionedV1Baseline().method == "hour_conditioned_v1"

    def test_hour_baseline_uses_all_meters(self):
        """Hour baseline sums across ALL meters, not just the target meter."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # 10 meters, each with 30 min occupied at hour 13, 7 days each.
        sessions_dict = {}
        for i in range(10):
            sessions_dict[f"M{i}"] = [
                (datetime(2026, 8, d, 20, tzinfo=UTC),
                 datetime(2026, 8, d, 20, 30, tzinfo=UTC))
                for d in range(14, 21)
            ]

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            sessions_dict, {}, {f"M{i}": "SS" for i in range(10)},
            history_window_days=28, slot_start=slot,
        )

        # hour_occupied = 10 × 30 × 7 = 2100 min
        # hour_possible = 10 × 7 × 60 = 4200 min
        # hour_score = 1 - 2100/4200 = 0.5
        hour_score = model._slot_hour_scores[slot][13].score
        assert hour_score == pytest.approx(0.5, abs=0.001)

        # M0's meter_hour_score = 1 - 30*7/(7*60) = 0.5
        # deviation = 0.5 - 0.5 = 0 → final = 0.5
        pred = model.predict(
            sessions_dict["M0"], slot, history_window_days=28, post_id="M0"
        )
        assert pred is not None
        assert pred.score == pytest.approx(0.5, abs=0.001)

    def test_optimized_matches_reference(self):
        """Optimized prepare() produces identical hour_scores to the reference."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # Build a non-trivial dataset with multiple meters, hours, and days.
        all_sessions = {}
        for i in range(20):
            pid = f"M{i}"
            hours = [13, 14, 15] if i % 3 == 0 else [13]
            days = list(range(7, 21)) if i % 2 == 0 else list(range(14, 21))
            all_sessions[pid] = [
                (datetime(2026, 8, d, 20 + h - 13, tzinfo=UTC),
                 datetime(2026, 8, d, 20 + h - 13, 30, tzinfo=UTC))
                for d in days for h in hours
            ]

        ref = HourConditionedV1Baseline._prepare_reference(
            all_sessions, history_window_days=28, slot_start=slot,
        )

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            all_sessions, {}, {f"M{i}": "SS" for i in range(20)},
            history_window_days=28, slot_start=slot,
        )
        opt = model._slot_hour_scores[slot]

        assert set(ref.keys()) == set(opt.keys()), (
            f"Hour keys differ: {set(ref.keys()) ^ set(opt.keys())}"
        )
        for h in ref:
            assert ref[h].score == opt[h].score, (
                f"Hour {h}: ref={ref[h].score} opt={opt[h].score}"
            )
            assert ref[h].evidence_sessions == opt[h].evidence_sessions, (
                f"Hour {h}: ref_sessions={ref[h].evidence_sessions} "
                f"opt_sessions={opt[h].evidence_sessions}"
            )

    def test_optimized_matches_reference_with_truncation(self):
        """Verify correctness when sessions are truncated at slot_start."""
        # Sessions that cross the slot_start boundary.
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)  # 13:00 PT

        m1_sessions = [
            # Aug 19: session fully before slot.
            (datetime(2026, 8, 19, 20, tzinfo=UTC),
             datetime(2026, 8, 19, 20, 45, tzinfo=UTC)),
            # Aug 20: session 19:30-20:30 UTC → overlaps hour 13 (20:00-21:00)
            # but truncated at slot_start (20:00) → 30 min overlap.
            (datetime(2026, 8, 20, 19, 30, tzinfo=UTC),
             datetime(2026, 8, 20, 20, 30, tzinfo=UTC)),
        ]
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]
        all_sessions = {"M1": m1_sessions, "M2": m2_sessions}

        ref = HourConditionedV1Baseline._prepare_reference(
            all_sessions, history_window_days=28, slot_start=slot,
        )

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        model.prepare(
            all_sessions, {}, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )
        opt = model._slot_hour_scores[slot]

        assert set(ref.keys()) == set(opt.keys())
        for h in ref:
            assert ref[h].score == opt[h].score, (
                f"Hour {h}: ref={ref[h].score} opt={opt[h].score}"
            )
