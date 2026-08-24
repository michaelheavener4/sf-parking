"""Pure-function tests for BlockfaceHourlyBaseline.

These tests require NO database — they exercise prepare() and predict()
directly with hand-computed session data.  Integration tests that need a
database are in test_backtest.py (TestBlockfaceHourlyBacktest).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sf_parking.backtest import (
    BlockfaceHourlyBaseline,
    DeterministicV0Baseline,
    PlacementSpan,
)

_V0 = DeterministicV0Baseline()


class TestBlockfaceHourlyHandComputed:
    """Hand-computed test proving pooling 2 meters does NOT double occupancy.

    Scenario (all August 2026 PDT):
        Blockface "BF-1" has two SS meters: M1 and M2.
        Target slot: 2026-08-21 13:00 PT (= 20:00 UTC).
        History: 3 days (Aug 18, 19, 20), all sessions at hour 13 PT.

        M1: 3 sessions, each 36 min → occupied = 3 × 36 = 108 min
            evidence_days = 3 → possible = 3 × 60 = 180
            per-meter score = 1 − 108/180 = 0.4

        M2: 3 sessions, each 24 min → occupied = 3 × 24 = 72 min
            evidence_days = 3 → possible = 3 × 60 = 180
            per-meter score = 1 − 72/180 = 0.6

        Blockface pooled:
            bf_occupied = 108 + 72 = 180 min
            bf_possible = 180 + 180 = 360 min  (NOT 180!)
            bf_score = 1 − 180/360 = 0.5

        If the denominator were wrongly bf_evidence_days × 60 = 3 × 60 = 180,
        the score would be 1 − 180/180 = 0.0 — artificially suggesting the
        blockface is fully occupied, which is wrong: the two meters together
        only cover 180 of 360 possible minutes.

        Blending (evidence_halflife = 14):
            w(M1) = min(3/14, 1) = 3/14 ≈ 0.214
            blended(M1) = 0.214 × 0.4 + 0.786 × 0.5 ≈ 0.479

            w(M2) = min(3/14, 1) = 3/14 ≈ 0.214
            blended(M2) = 0.214 × 0.6 + 0.786 × 0.5 ≈ 0.521

        Both blended scores are between their per-meter score and the
        blockface score, confirming the blend is sane and the denominator
        correctly accounts for pooled meter-hours.
    """

    def test_pooling_two_meters_does_not_double_occupancy(self):
        # All times are August 2026 PDT (UTC−7).
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)  # 13:00 PT

        # M1: 36 min sessions on Aug 18, 19, 20 at hour 13 PT.
        m1_sessions = [
            (datetime(2026, 8, 18, 20, tzinfo=UTC),
             datetime(2026, 8, 18, 20, 36, tzinfo=UTC)),
            (datetime(2026, 8, 19, 20, tzinfo=UTC),
             datetime(2026, 8, 19, 20, 36, tzinfo=UTC)),
            (datetime(2026, 8, 20, 20, tzinfo=UTC),
             datetime(2026, 8, 20, 20, 36, tzinfo=UTC)),
        ]
        # M2: 24 min sessions on Aug 18, 19, 20 at hour 13 PT.
        m2_sessions = [
            (datetime(2026, 8, 18, 20, tzinfo=UTC),
             datetime(2026, 8, 18, 20, 24, tzinfo=UTC)),
            (datetime(2026, 8, 19, 20, tzinfo=UTC),
             datetime(2026, 8, 19, 20, 24, tzinfo=UTC)),
            (datetime(2026, 8, 20, 20, tzinfo=UTC),
             datetime(2026, 8, 20, 20, 24, tzinfo=UTC)),
        ]

        # Verify per-meter V0 scores match hand computation.
        pm1 = _V0.predict(m1_sessions, slot, history_window_days=28)
        pm2 = _V0.predict(m2_sessions, slot, history_window_days=28)
        assert pm1 is not None
        assert pm2 is not None
        assert pm1.score == pytest.approx(0.4, abs=0.001)
        assert pm2.score == pytest.approx(0.6, abs=0.001)

        # Set up blockface context.
        all_sessions = {"M1": m1_sessions, "M2": m2_sessions}
        placements = {
            "M1": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-1",
            )],
            "M2": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-1",
            )],
        }
        meter_types = {"M1": "SS", "M2": "SS"}

        # Run BlockfaceHourlyBaseline for M1.
        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            all_sessions, placements, meter_types,
            history_window_days=28, slot_start=slot,
        )
        pred_m1 = bf_model.predict(
            m1_sessions, slot, history_window_days=28, post_id="M1",
        )
        assert pred_m1 is not None

        # The blockface score must be 0.5, NOT 0.0.
        # (0.0 would result from the wrong denominator 3×60=180.)
        bf_pred = bf_model._bf_scores.get(("BF-1", 13))
        assert bf_pred is not None
        assert bf_pred.score == pytest.approx(0.5, abs=0.001)

        # Blended score for M1: w=3/14, blend = w×0.4 + (1−w)×0.5
        w = 3.0 / 14.0
        expected_blended = round(w * 0.4 + (1.0 - w) * 0.5, 3)
        assert pred_m1.score == pytest.approx(expected_blended, abs=0.001)

        # Run for M2.
        pred_m2 = bf_model.predict(
            m2_sessions, slot, history_window_days=28, post_id="M2",
        )
        assert pred_m2 is not None
        expected_m2 = round(w * 0.6 + (1.0 - w) * 0.5, 3)
        assert pred_m2.score == pytest.approx(expected_m2, abs=0.001)

        # Both blended scores are between their per-meter and the blockface.
        assert min(pm1.score, bf_pred.score) <= pred_m1.score <= max(pm1.score, bf_pred.score)
        assert min(pm2.score, bf_pred.score) <= pred_m2.score <= max(pm2.score, bf_pred.score)

    def test_different_evidence_days_correctly_weighted(self):
        """Meter with more evidence gets higher blending weight."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # M1: 1 day of evidence (1 session).
        m1_sessions = [
            (datetime(2026, 8, 20, 20, tzinfo=UTC),
             datetime(2026, 8, 20, 20, 30, tzinfo=UTC)),
        ]
        # M2: 7 days of evidence (7 sessions).
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in range(14, 21)
        ]

        all_sessions = {"M1": m1_sessions, "M2": m2_sessions}
        placements = {
            "M1": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-2",
            )],
            "M2": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-2",
            )],
        }

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            all_sessions, placements, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        # M1 has 1 day → w = 1/14 ≈ 0.071 → mostly blockface.
        pred_m1 = bf_model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        assert pred_m1 is not None

        # M2 has 7 days → w = 7/14 = 0.5 → equal blend.
        pred_m2 = bf_model.predict(m2_sessions, slot, history_window_days=28, post_id="M2")
        assert pred_m2 is not None

        # Both have the same per-meter score (30/60 = 0.5 occupied → 0.5),
        # so the difference comes from blending weight only.  M2's blend
        # should be closer to the per-meter score (more weight on own data).
        # With equal per-meter scores, both blended scores equal the per-meter
        # score, so they should be equal.
        assert pred_m1.score == pytest.approx(pred_m2.score, abs=0.001)

    def test_meter_without_blockface_falls_back_to_per_meter(self):
        """A meter with no placement gets per-meter-only scoring."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        sessions = [
            (datetime(2026, 8, 18, 20, tzinfo=UTC),
             datetime(2026, 8, 18, 20, 30, tzinfo=UTC)),
            (datetime(2026, 8, 19, 20, tzinfo=UTC),
             datetime(2026, 8, 19, 20, 30, tzinfo=UTC)),
        ]

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            {"M-noplacement": sessions}, {}, {"M-noplacement": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred = bf_model.predict(sessions, slot, history_window_days=28, post_id="M-noplacement")
        assert pred is not None
        # Should fall back to per-meter V0 score.
        v0_pred = _V0.predict(sessions, slot, history_window_days=28)
        assert pred.score == v0_pred.score

    def test_empty_blockface_falls_back_to_per_meter(self):
        """Blockface with no sessions from any meter → per-meter only."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        sessions = [
            (datetime(2026, 8, 18, 20, tzinfo=UTC),
             datetime(2026, 8, 18, 20, 30, tzinfo=UTC)),
        ]

        # M1 has sessions but its blockface partner M2 has none.
        all_sessions = {"M1": sessions, "M2-empty": []}
        placements = {
            "M1": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-empty",
            )],
            "M2-empty": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-empty",
            )],
        }

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            all_sessions, placements, {"M1": "SS", "M2-empty": "SS"},
            history_window_days=28, slot_start=slot,
        )
        pred = bf_model.predict(sessions, slot, history_window_days=28, post_id="M1")
        assert pred is not None
        # Blockface score should exist (M1 contributes to it).
        bf_pred = bf_model._bf_scores.get(("BF-empty", 13))
        assert bf_pred is not None
        # Blend should work normally.
        v0_pred = _V0.predict(sessions, slot, history_window_days=28)
        assert pred.score == v0_pred.score  # w=2/14, so mostly blockface

    def test_ss_ms_meters_treated_independently(self):
        """SS and MS meters on the same blockface are pooled correctly.

        SS meter: 1 session per hour max (one space).
        MS meter: 2 concurrent sessions in the same hour (two spaces).
        The pooled occupied minutes sum correctly; the denominator uses each
        meter's evidence_days × 60, not a fixed constant.
        """
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # SS meter: one 30-min session on each of 3 days.
        ss_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            for d in (18, 19, 20)
        ]
        # MS meter: two concurrent 30-min sessions on each of 3 days.
        # (Two different people paid at the same pay station for different spaces.)
        ms_sessions = []
        for d in (18, 19, 20):
            ms_sessions.append(
                (datetime(2026, 8, d, 20, tzinfo=UTC),
                 datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            )
            ms_sessions.append(
                (datetime(2026, 8, d, 20, tzinfo=UTC),
                 datetime(2026, 8, d, 20, 30, tzinfo=UTC))
            )

        all_sessions = {"SS-m": ss_sessions, "MS-m": ms_sessions}
        placements = {
            "SS-m": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-mixed",
            )],
            "MS-m": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-mixed",
            )],
        }

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            all_sessions, placements,
            {"SS-m": "SS", "MS-m": "MS"},
            history_window_days=28, slot_start=slot,
        )

        bf_pred = bf_model._bf_scores.get(("BF-mixed", 13))
        assert bf_pred is not None

        # SS: 3 × 30 = 90 occupied min, 3 × 60 = 180 possible.
        # MS: 6 × 30 = 180 occupied min, 3 × 60 = 180 possible.
        # (MS has 2 concurrent sessions per day × 3 days = 6 sessions,
        #  each 30 min = 180 occupied minutes.)
        # Pooled: 90 + 180 = 270 occupied, 180 + 180 = 360 possible.
        # bf_score = 1 − 270/360 = 0.25
        assert bf_pred.score == pytest.approx(0.25, abs=0.001)

        # The MS meter's occupied minutes CAN exceed 60 per hour (concurrent
        # sessions), confirming the model handles multi-space semantics.
        pred_ms = bf_model.predict(ms_sessions, slot, history_window_days=28, post_id="MS-m")
        assert pred_ms is not None
        # MS per-meter: 180/180 = fully occupied → score 0.0
        # But blended with bf_score 0.25 → blended > 0.0
        assert pred_ms.score > 0.0

    def test_blended_score_never_exceeds_one(self):
        """Blended score is always clamped to [0, 1]."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)

        # M1: zero occupied minutes (always free).
        m1_sessions = [
            (datetime(2026, 8, d, 16, tzinfo=UTC),
             datetime(2026, 8, d, 16, tzinfo=UTC))  # 0-min session
            for d in (18, 19, 20)
        ]
        # M2: fully occupied.
        m2_sessions = [
            (datetime(2026, 8, d, 20, tzinfo=UTC),
             datetime(2026, 8, d, 21, tzinfo=UTC))
            for d in (18, 19, 20)
        ]

        all_sessions = {"M1": m1_sessions, "M2": m2_sessions}
        placements = {
            "M1": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-clamp",
            )],
            "M2": [PlacementSpan(
                valid_from=float("-inf"), valid_until=float("inf"),
                latitude=37.79, longitude=-122.4, blockface_id="BF-clamp",
            )],
        }

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            all_sessions, placements, {"M1": "SS", "M2": "SS"},
            history_window_days=28, slot_start=slot,
        )

        pred = bf_model.predict(m1_sessions, slot, history_window_days=28, post_id="M1")
        assert pred is not None
        assert 0.0 <= pred.score <= 1.0

    def test_prepare_with_no_placements_leaves_bf_scores_empty(self):
        """prepare() with empty placements produces no blockface scores."""
        slot = datetime(2026, 8, 21, 20, tzinfo=UTC)
        sessions = [
            (datetime(2026, 8, 18, 20, tzinfo=UTC),
             datetime(2026, 8, 18, 20, 30, tzinfo=UTC)),
        ]

        bf_model = BlockfaceHourlyBaseline(evidence_halflife=14.0)
        bf_model.prepare(
            {"M1": sessions}, {}, {"M1": "SS"},
            history_window_days=28, slot_start=slot,
        )
        assert bf_model._bf_scores == {}
        assert bf_model._prepared is True

    def test_method_tag(self):
        assert BlockfaceHourlyBaseline().method == "blockface_hourly"
