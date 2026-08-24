"""Point-in-time-safe historical backtesting for parking-state baselines.

Answers: "If the system had made a parking prediction at time T using only
information actually available at T, how accurate would it have been?"

Protocol
--------
1. Observations are (meter, absolute clock-hour slot) pairs generated for
   every local date a meter was observed within the evaluation window.
2. A prediction for a slot uses ONLY information available at its start:
   * transactions started strictly before the slot,
   * session ends truncated at the slot (an ongoing session's eventual end
     is unknowable at prediction time),
   * meter location/blockface taken from ``meter_placements`` rows whose
     validity range contains the slot (never later inventory state).
3. Outcome: an *observable proxy*, explicitly not ground truth - the paid-
   occupancy minutes overlapped by real transactions (untruncated). The
   dataset observes paid sessions, not occupancy; unpaid parking is
   invisible, so true availability is systematically lower than the proxy
   implies. See docs/BACKTESTING.md for the three-level distinction between
   observed paid sessions, inferred occupancy, and (unavailable) truth.

The V0 baseline formula lives in ``sf_parking.features`` and is not modified
here; this module feeds it honest inputs and measures it. New models can be
added by implementing :class:`BaselineModel` without touching the harness.

Run as a CLI: ``python3 -m sf_parking.backtest --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pg8000.native

from .features import SF_TZ, _candidate_slot_starts, _overlap_seconds

BASELINE_METHOD = "deterministic_v0"
BLOCKFACE_HOURLY_METHOD = "blockface_hourly"
HOUR_CONDITIONED_METHOD = "hour_conditioned_v1"
SLOT_MINUTES = 60
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction:
    score: float
    evidence_days: int
    evidence_sessions: int


class BaselineModel(Protocol):
    """Contract for scorers evaluated by the harness.

    Implementations receive only cutoff-truncated history and must be
    deterministic functions of it plus their own parameters.
    """

    method: str

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
        post_id: str | None = ...,
    ) -> Prediction | None: ...


def score_v0(occupied_minutes: float, possible_minutes: float) -> float:
    """The V0 scoring formula (shared with sf_parking.features), verbatim."""
    if possible_minutes <= 0:
        raise ValueError("possible_minutes must be positive")
    return round(max(0.0, min(1.0, 1.0 - occupied_minutes / possible_minutes)), 3)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class DeterministicV0Baseline:
    """Historical clock-hour occupancy ratio for the same meter.

    score = clamp(1 - paid_occupied_minutes / (evidence_days * 60), 0, 1)

    where occupied minutes come from prior sessions overlapping this local
    clock hour across the meter's evidence span inside the lookback window.
    """

    method = BASELINE_METHOD

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
        post_id: str | None = None,
    ) -> Prediction | None:
        window_start = slot_start - timedelta(days=history_window_days)

        # Point-in-time history: sessions started before the cutoff, ends
        # truncated at the cutoff, restricted to the lookback window.
        hi = bisect_left([s for s, _ in sessions], slot_start)
        history = [
            (s, min(e, slot_start))
            for s, e in sessions[:hi]
            if min(e, slot_start) > window_start
        ]
        if not history:
            return None

        hist_starts = [s for s, _ in history]
        n_hist = len(history)
        local_hour = slot_start.astimezone(SF_TZ).hour
        first_day = history[0][0].astimezone(SF_TZ).date()
        last_day = history[-1][1].astimezone(SF_TZ).date()
        evidence_days = max((last_day - first_day).days + 1, 1)

        occupied = 0.0
        day = first_day
        while day <= last_day:
            for lo in _candidate_slot_starts(day, local_hour):
                if lo < window_start or lo >= slot_start:
                    continue  # never score the slot itself or stale windows
                j = max(bisect_left(hist_starts, lo - timedelta(days=2)) - 1, 0)
                while j < n_hist:
                    s, e = history[j]
                    if s >= lo + timedelta(hours=1):
                        break
                    occupied += _overlap_seconds(s, e, lo, lo + timedelta(hours=1))
                    j += 1
            day += timedelta(days=1)

        return Prediction(
            score=score_v0(occupied / 60.0, evidence_days * SLOT_MINUTES),
            evidence_days=evidence_days,
            evidence_sessions=n_hist,
        )


class BlockfaceHourlyBaseline:
    """Blockface-pooled hourly climatology with per-meter blending.

    Blockface score — the occupancy ratio pooled across all meters on the
    same blockface for the target local clock hour:

        bf_occupied = Σ_meters  overlap_minutes(m, h)      (summed over all
                                                            meters on the
                                                            blockface)
        bf_possible = Σ_meters  evidence_days(m) × 60      (each meter
                                                            contributes its
                                                            own observation
                                                            span)
        bf_score    = clamp(1 − bf_occupied / bf_possible, 0, 1)

    The denominator is the sum of each meter's individual observable
    meter-hours, NOT a shared constant.  A blockface with 10 meters
    observed for 7 days each has bf_possible = 10 × 7 × 60 = 4200
    minutes; one with 3 meters observed for 3 days has 3 × 3 × 60 = 540.

    SS vs MS treatment:
        SS (single-space) meters: post_id = one parking space.  At most
        one concurrent session per post_id.  Occupied minutes per hour ∈
        [0, 60].
        MS (multi-space) meters: post_id = one pay station covering
        potentially many spaces.  Concurrent sessions at the same post_id
        represent different spaces.  Occupied minutes per hour can exceed
        60.  The blockface denominator does NOT attempt to count physical
        spaces for MS meters (the count is unknown without ms_id/space_num
        from the source data); it counts observable meter-hours, which is
        a conservative lower bound on true space-hours.

    Blending — per-meter evidence controls the blend:

        w(m) = min(evidence_days(m) / evidence_halflife, 1.0)
        blended = w × per_meter_score + (1 − w) × bf_score

    With evidence_halflife = 14 (default):
        0 days evidence → w = 0   → pure blockface prior
        7 days evidence → w = 0.5 → equal blend
        14+ days        → w = 1.0 → pure per-meter

    Fallbacks:
        • No blockface assignment  → per-meter score only (deterministic_v0).
        • Blockface has no sessions → per-meter score only.
        • Neither has data → None (insufficient history).

    Method tag: ``blockface_hourly``.
    """

    method = BLOCKFACE_HOURLY_METHOD

    def __init__(self, evidence_halflife: float = 14.0) -> None:
        self.evidence_halflife = evidence_halflife
        self._bf_scores: dict[tuple[str, int], Prediction] = {}
        self._meter_bf: dict[str, str] = {}
        self._meter_type: dict[str, str | None] = {}
        self._bf_meters: dict[str, list[str]] = {}
        self._prepared = False
        self._cache: dict[datetime, None] = {}  # slot_start → computed

    def prepare(
        self,
        all_sessions: dict[str, list[tuple[datetime, datetime]]],
        placements: dict[str, list[PlacementSpan]],
        meter_types: dict[str, str | None],
        *,
        history_window_days: int,
        slot_start: datetime,
    ) -> None:
        """Pre-compute blockface scores from all meters' truncated history.

        Called once per (slot_start) by the harness before the per-meter
        prediction loop.  All sessions are already truncated at slot_start.
        Results are cached by slot_start — repeated calls for the same
        cutoff (different meters at the same hour) are near-free.
        """
        if slot_start in self._cache:
            return
        self._cache[slot_start] = None  # mark as computed
        window_start = slot_start - timedelta(days=history_window_days)

        # 1. Map each meter to its blockface at slot_start.
        bf_meters: dict[str, list[str]] = {}
        for post_id, spans in placements.items():
            placement = _placement_at(spans, slot_start)
            if placement is not None and placement.blockface_id is not None:
                self._meter_bf[post_id] = placement.blockface_id
                bf_meters.setdefault(placement.blockface_id, []).append(post_id)
        self._bf_meters = bf_meters
        self._meter_type = meter_types

        # 2. For each blockface, compute the pooled score for each local
        #    clock hour 0–23.
        for bf_id, bf_post_ids in bf_meters.items():
            local_hour = slot_start.astimezone(SF_TZ).hour

            # Collect all truncated sessions for meters on this blockface,
            # along with each meter's evidence_days.
            bf_data: list[tuple[list[tuple[datetime, datetime]], int]] = []
            for pid in bf_post_ids:
                sessions = all_sessions.get(pid, [])
                hi = bisect_left([s for s, _ in sessions], slot_start)
                history = [
                    (s, min(e, slot_start))
                    for s, e in sessions[:hi]
                    if min(e, slot_start) > window_start
                ]
                if not history:
                    continue
                first_day = history[0][0].astimezone(SF_TZ).date()
                last_day = history[-1][1].astimezone(SF_TZ).date()
                evidence_days = max((last_day - first_day).days + 1, 1)
                bf_data.append((history, evidence_days))

            if not bf_data:
                continue

            # Compute pooled occupied minutes and pooled possible minutes.
            bf_occupied = 0.0
            bf_possible = 0.0
            total_sessions = 0

            for hist, ev_days in bf_data:
                hist_starts = [s for s, _ in hist]
                n_hist = len(hist)
                first_day = hist[0][0].astimezone(SF_TZ).date()
                last_day = hist[-1][1].astimezone(SF_TZ).date()

                meter_occupied = 0.0
                day = first_day
                while day <= last_day:
                    for lo in _candidate_slot_starts(day, local_hour):
                        if lo < window_start or lo >= slot_start:
                            continue
                        j = max(
                            bisect_left(hist_starts, lo - timedelta(days=2)) - 1,
                            0,
                        )
                        while j < n_hist:
                            s, e = hist[j]
                            if s >= lo + timedelta(hours=1):
                                break
                            meter_occupied += _overlap_seconds(
                                s, e, lo, lo + timedelta(hours=1)
                            )
                            j += 1
                    day += timedelta(days=1)

                bf_occupied += meter_occupied
                bf_possible += ev_days * SLOT_MINUTES
                total_sessions += n_hist

            if bf_possible <= 0:
                continue

            self._bf_scores[(bf_id, local_hour)] = Prediction(
                score=score_v0(bf_occupied / 60.0, bf_possible),
                evidence_days=0,  # blockface-level; per-meter tracked separately
                evidence_sessions=total_sessions,
            )

        self._prepared = True

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
        post_id: str | None = None,
    ) -> Prediction | None:
        """Blended per-meter + blockface hourly availability score."""
        # Per-meter score via the V0 formula.
        per_meter = DeterministicV0Baseline().predict(
            sessions, slot_start, history_window_days=history_window_days
        )

        if not self._prepared or post_id is None:
            return per_meter

        # Look up the target meter's blockface at slot_start.
        local_hour = slot_start.astimezone(SF_TZ).hour
        bf_id = self._meter_bf.get(post_id)

        bf_pred = None
        if bf_id is not None:
            bf_pred = self._bf_scores.get((bf_id, local_hour))

        # Blend.
        if per_meter is not None and bf_pred is not None:
            w = min(per_meter.evidence_days / self.evidence_halflife, 1.0)
            blended_score = round(
                w * per_meter.score + (1.0 - w) * bf_pred.score, 3
            )
            return Prediction(
                score=blended_score,
                evidence_days=per_meter.evidence_days,
                evidence_sessions=per_meter.evidence_sessions,
            )

        # Fallback: use whichever estimate is available.
        if per_meter is not None:
            return per_meter
        if bf_pred is not None:
            return bf_pred
        return None


class HourConditionedV1Baseline:
    """Hour-conditioned availability: pooled hour baseline + shrunk meter deviation.

    Decomposes availability into a global hour-of-day component and a
    meter-specific deviation, then shrinks the deviation toward zero
    proportional to evidence depth.

    Formula::

        hour_score(h) = 1 - Σ_meters occupied(m,h) / Σ_meters possible(m,h)

        meter_hour_score(m,h) = 1 - occupied(m,h) / (evidence_days(m,h) × 60)

        deviation(m,h) = meter_hour_score(m,h) - hour_score(h)

        w(m,h) = min(evidence_days(m,h) / evidence_halflife, 1.0)

        final_score = clamp(hour_score(h) + w(m,h) × deviation(m,h), 0, 1)

    where:

    - ``occupied(m,h)`` = total session minutes overlapping local hour *h*
      for meter *m*, summed over all occurrences within the evidence span.
    - ``possible(m,h)`` = ``evidence_days(m,h) × 60``, where
      ``evidence_days(m,h)`` is the number of distinct local calendar dates
      on which meter *m* has at least one session overlapping hour *h*.
    - The hour-level denominator sums possible minutes across ALL meters,
      correctly accounting for each meter's observation coverage.

    Fallbacks:

    - No hour-level history at all → deterministic_v0 per-meter score.
    - No meter-hour history → hour baseline.
    - Neither → None.

    Method tag: ``hour_conditioned_v1``.
    """

    method = HOUR_CONDITIONED_METHOD

    def __init__(
        self,
        evidence_halflife: float = 14.0,
        *,
        session_index: tuple | None = None,
    ) -> None:
        self.evidence_halflife = evidence_halflife
        self._slot_hour_scores: dict[datetime, dict[int, Prediction]] = {}
        self._cache: dict[datetime, None] = {}
        # Precomputed session-hour index: built once per prepare() call family.
        self._session_index_built: bool = False
        # post_id -> sorted list of (session_start, session_end, local_date,
        #   local_hour, slot_lo, slot_hi, overlap_secs)
        self._session_index: dict[str, list[tuple]] = {}
        # post_id -> sorted list of session_start (for predict bisect)
        self._session_starts: dict[str, list[datetime]] = {}
        # (post_id, local_hour) -> list of indices into _session_index[post_id]
        self._meter_hour_index: dict[tuple[str, int], list[int]] = {}
        # (post_id, local_hour) -> entries sorted by lo (slot start) for
        # bisect-based range restriction in prepare().
        self._meter_hour_lo_entries: dict[tuple[str, int], list[tuple]] = {}
        # post_id -> set of local_hours with data
        self._meter_hours: dict[str, set[int]] = {}
        # Optional pre-built index from build_session_index().
        self._prebuilt_index: tuple | None = session_index

    def _build_session_index(
        self, all_sessions: dict[str, list[tuple[datetime, datetime]]],
    ) -> None:
        """One-time preprocessing: for every session, compute the local-hour
        slots it overlaps and the (unclamped) overlap duration."""
        if self._session_index_built:
            return
        self._session_index_built = True

        # Use pre-built index if provided (dataset-level sharing).
        if self._prebuilt_index is not None:
            (
                self._session_index,
                self._session_starts,
                self._meter_hour_index,
                self._meter_hour_lo_entries,
                self._meter_hours,
            ) = self._prebuilt_index
            return

        # Fall back to building from scratch (with memoization).
        (
            self._session_index,
            self._session_starts,
            self._meter_hour_index,
            self._meter_hour_lo_entries,
            self._meter_hours,
        ) = build_session_index(all_sessions)

    # ------------------------------------------------------------------
    # Reference (original) implementation for correctness assertions
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_reference(
        all_sessions: dict[str, list[tuple[datetime, datetime]]],
        *,
        history_window_days: int,
        slot_start: datetime,
    ) -> dict[int, Prediction]:
        """Original prepare() calculation, returned as a dict (not stored)."""
        window_start = slot_start - timedelta(days=history_window_days)
        hour_occupied: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_possible: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_sessions: dict[int, int] = {h: 0 for h in range(24)}

        for post_id, raw_sessions in all_sessions.items():
            hi = bisect_left([s for s, _ in raw_sessions], slot_start)
            history = [
                (s, min(e, slot_start))
                for s, e in raw_sessions[:hi]
                if min(e, slot_start) > window_start
            ]
            if not history:
                continue
            hist_starts = [s for s, _ in history]
            n_hist = len(history)
            for local_hour in range(24):
                meter_days: set[date] = set()
                meter_occupied = 0.0
                first_day = history[0][0].astimezone(SF_TZ).date()
                last_day = history[-1][1].astimezone(SF_TZ).date()
                day = first_day
                while day <= last_day:
                    for lo in _candidate_slot_starts(day, local_hour):
                        if lo < window_start or lo >= slot_start:
                            continue
                        j = max(
                            bisect_left(hist_starts, lo - timedelta(days=2)) - 1,
                            0,
                        )
                        day_has_overlap = False
                        while j < n_hist:
                            s, e = history[j]
                            if s >= lo + timedelta(hours=1):
                                break
                            overlap = _overlap_seconds(
                                s, e, lo, lo + timedelta(hours=1)
                            )
                            if overlap > 0:
                                day_has_overlap = True
                                meter_occupied += overlap
                            j += 1
                        if day_has_overlap:
                            slot_local = lo.astimezone(SF_TZ).date()
                            meter_days.add(slot_local)
                    day += timedelta(days=1)
                ev_days = len(meter_days)
                if ev_days > 0:
                    hour_occupied[local_hour] += meter_occupied
                    hour_possible[local_hour] += ev_days * SLOT_MINUTES
                    hour_sessions[local_hour] += n_hist

        hour_scores: dict[int, Prediction] = {}
        for h in range(24):
            if hour_possible[h] > 0:
                hour_scores[h] = Prediction(
                    score=score_v0(hour_occupied[h] / 60.0, hour_possible[h]),
                    evidence_days=0,
                    evidence_sessions=hour_sessions[h],
                )
        return hour_scores

    # ------------------------------------------------------------------
    # Optimized prepare()
    # ------------------------------------------------------------------

    def prepare(
        self,
        all_sessions: dict[str, list[tuple[datetime, datetime]]],
        placements: dict[str, list[PlacementSpan]],
        meter_types: dict[str, str | None],
        *,
        history_window_days: int,
        slot_start: datetime,
    ) -> None:
        """Pre-compute hour-of-day baseline from all meters' truncated history.

        Optimized: precomputes per-session local-hour overlaps once, then
        aggregates per slot_start using indexed lookup instead of re-walking
        the full meter × 24 × day grid.
        """
        if slot_start in self._cache:
            return
        self._cache[slot_start] = None

        # Ensure the session index is built (once per model lifetime).
        t0 = time.perf_counter()
        self._build_session_index(all_sessions)
        t_index = time.perf_counter() - t0

        window_start = slot_start - timedelta(days=history_window_days)

        hour_occupied: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_possible: dict[int, float] = {h: 0.0 for h in range(24)}
        hour_sessions: dict[int, int] = {h: 0 for h in range(24)}

        n_meters = 0
        n_contributions = 0
        n_candidates = 0
        t_loop = time.perf_counter()
        for post_id in all_sessions:
            raw = all_sessions[post_id]
            n_meters += 1
            # n_hist: count of sessions that started before slot_start.
            n_hist = bisect_left([s for s, _ in raw], slot_start)

            # Iterate only hours this meter actually has sessions at.
            for local_hour in self._meter_hours.get(post_id, set()):
                lo_entries = self._meter_hour_lo_entries.get(
                    (post_id, local_hour), ()
                )

                meter_days: set[date] = set()
                meter_occupied = 0.0

                # Bisect by lo (slot start) to restrict to [window_start, slot_start).
                lo_lo = bisect_left(lo_entries, window_start, key=lambda e: e[4])
                lo_hi = bisect_left(lo_entries, slot_start, key=lambda e: e[4])
                n_candidates += lo_hi - lo_lo

                for entry in lo_entries[lo_lo:lo_hi]:
                    sess_s, sess_e, loc_date, _loc_h, lo, _hi, _ov_full = entry
                    # Point-in-time: session must start before slot_start.
                    if sess_s >= slot_start:
                        continue
                    # Compute overlap with truncation at slot_start.
                    truncated_end = min(sess_e, slot_start)
                    overlap = _overlap_seconds(
                        sess_s, truncated_end, lo, lo + timedelta(hours=1)
                    )
                    if overlap > 0:
                        meter_occupied += overlap
                        meter_days.add(loc_date)
                        n_contributions += 1

                ev_days = len(meter_days)
                if ev_days > 0:
                    hour_occupied[local_hour] += meter_occupied
                    hour_possible[local_hour] += ev_days * SLOT_MINUTES
                    hour_sessions[local_hour] += n_hist

        hour_scores: dict[int, Prediction] = {}
        for h in range(24):
            if hour_possible[h] > 0:
                hour_scores[h] = Prediction(
                    score=score_v0(hour_occupied[h] / 60.0, hour_possible[h]),
                    evidence_days=0,
                    evidence_sessions=hour_sessions[h],
                )

        self._slot_hour_scores[slot_start] = hour_scores
        t_total = time.perf_counter() - t0
        print(
            f"[prepare] slot={slot_start.isoformat()} "
            f"meters={n_meters} candidates={n_candidates} "
            f"contributions={n_contributions} "
            f"index_build={t_index:.3f}s loop={time.perf_counter() - t_loop:.3f}s "
            f"total={t_total:.3f}s"
        )

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
        post_id: str | None = None,
    ) -> Prediction | None:
        """Blended hour baseline + shrunk meter deviation."""
        local_hour = slot_start.astimezone(SF_TZ).hour

        # Hour-level baseline for this specific slot_start.
        slot_hours = self._slot_hour_scores.get(slot_start, {})
        hour_pred = slot_hours.get(local_hour)

        # Per-meter score via V0.
        per_meter = DeterministicV0Baseline().predict(
            sessions, slot_start, history_window_days=history_window_days
        )

        # Fallback: no hour history → use per-meter V0.
        if hour_pred is None:
            return per_meter

        # Compute meter-hour evidence and deviation using precomputed index.
        window_start = slot_start - timedelta(days=history_window_days)

        # Count total history sessions (started before slot_start).
        raw = sessions
        n_hist = bisect_left([s for s, _ in raw], slot_start)

        if n_hist == 0:
            return Prediction(
                score=hour_pred.score,
                evidence_days=0,
                evidence_sessions=hour_pred.evidence_sessions,
            )

        # Use precomputed index if available.
        if post_id is not None and post_id in self._meter_hour_index:
            meter_days: set[date] = set()
            meter_occupied = 0.0
            indices = self._meter_hour_index.get(post_id, [])
            for j in indices:
                sess_s, sess_e, loc_date, loc_h, lo, hi, ov_full = (
                    self._session_index[post_id][j]
                )
                if loc_h != local_hour:
                    continue
                if sess_s >= slot_start:
                    continue
                truncated_end = min(sess_e, slot_start)
                if truncated_end <= window_start:
                    continue
                overlap = _overlap_seconds(
                    sess_s, truncated_end, lo, lo + timedelta(hours=1)
                )
                if overlap > 0:
                    meter_occupied += overlap
                    meter_days.add(loc_date)

            ev_days = len(meter_days)
            if ev_days == 0:
                return Prediction(
                    score=hour_pred.score,
                    evidence_days=0,
                    evidence_sessions=hour_pred.evidence_sessions,
                )
            meter_hour_score = score_v0(
                meter_occupied / 60.0, ev_days * SLOT_MINUTES
            )
            deviation = meter_hour_score - hour_pred.score
            w = min(ev_days / self.evidence_halflife, 1.0)
            final_score = _clamp01(hour_pred.score + w * deviation)
            return Prediction(
                score=round(final_score, 3),
                evidence_days=ev_days,
                evidence_sessions=n_hist,
            )

        # Fallback: no index available (should not happen in normal flow).
        history = [
            (s, min(e, slot_start))
            for s, e in raw[:n_hist]
            if min(e, slot_start) > window_start
        ]
        if not history:
            return Prediction(
                score=hour_pred.score,
                evidence_days=0,
                evidence_sessions=hour_pred.evidence_sessions,
            )
        hist_starts = [s for s, _ in history]
        n_hist_actual = len(history)
        meter_days_fb: set[date] = set()
        meter_occupied_fb = 0.0
        first_day = history[0][0].astimezone(SF_TZ).date()
        last_day = history[-1][1].astimezone(SF_TZ).date()
        day = first_day
        while day <= last_day:
            for lo in _candidate_slot_starts(day, local_hour):
                if lo < window_start or lo >= slot_start:
                    continue
                j = max(
                    bisect_left(hist_starts, lo - timedelta(days=2)) - 1, 0
                )
                day_has_overlap = False
                while j < n_hist_actual:
                    s, e = history[j]
                    if s >= lo + timedelta(hours=1):
                        break
                    overlap = _overlap_seconds(s, e, lo, lo + timedelta(hours=1))
                    if overlap > 0:
                        day_has_overlap = True
                        meter_occupied_fb += overlap
                    j += 1
                if day_has_overlap:
                    slot_local = lo.astimezone(SF_TZ).date()
                    meter_days_fb.add(slot_local)
            day += timedelta(days=1)
        ev_days = len(meter_days_fb)
        if ev_days == 0:
            return Prediction(
                score=hour_pred.score,
                evidence_days=0,
                evidence_sessions=hour_pred.evidence_sessions,
            )
        meter_hour_score = score_v0(
            meter_occupied_fb / 60.0, ev_days * SLOT_MINUTES
        )
        deviation = meter_hour_score - hour_pred.score
        w = min(ev_days / self.evidence_halflife, 1.0)
        final_score = _clamp01(hour_pred.score + w * deviation)
        return Prediction(
            score=round(final_score, 3),
            evidence_days=ev_days,
            evidence_sessions=n_hist_actual,
        )


MODELS: dict[str, BaselineModel] = {
    BASELINE_METHOD: DeterministicV0Baseline(),
    BLOCKFACE_HOURLY_METHOD: BlockfaceHourlyBaseline(),
    HOUR_CONDITIONED_METHOD: HourConditionedV1Baseline(),
}


# ---------------------------------------------------------------------------
# Dataset-level session index: built once, shared across models
# ---------------------------------------------------------------------------

# Cache keyed by id(sessions) so the same dict object is not rebuilt.
# The cache entry holds the5 index structures produced by
# HourConditionedV1Baseline._build_session_index.
_session_index_cache: dict[int, tuple] = {}


def build_session_index(
    all_sessions: dict[str, list[tuple[datetime, datetime]]],
) -> tuple[
    dict[str, list[tuple]],
    dict[str, list[datetime]],
    dict[tuple[str, int], list[int]],
    dict[tuple[str, int], list[tuple]],
    dict[str, set[int]],
]:
    """Build the session-hour index once for a dataset.

    Returns (session_index, session_starts, meter_hour_index,
    meter_hour_lo_entries, meter_hours).  The result is cached by
    ``id(all_sessions)`` so repeated calls with the same dict are free.
    """
    key = id(all_sessions)
    if key in _session_index_cache:
        return _session_index_cache[key]

    _HOUR = timedelta(hours=1)
    session_index: dict[str, list[tuple]] = {}
    session_starts: dict[str, list[datetime]] = {}
    meter_hour_index: dict[tuple[str, int], list[int]] = {}
    meter_hour_lo_entries: dict[tuple[str, int], list[tuple]] = {}
    meter_hours: dict[str, set[int]] = {}
    _slot_cache: dict[tuple[date, int], list[datetime]] = {}

    for post_id, raw_sessions in all_sessions.items():
        entries: list[tuple] = []
        starts_list: list[datetime] = []
        for s, e in raw_sessions:
            starts_list.append(s)
            if e is None:
                continue
            s_local = s.astimezone(SF_TZ)
            e_local = e.astimezone(SF_TZ)
            cur = s_local.replace(minute=0, second=0, microsecond=0)
            end_wall = e_local
            while cur < end_wall:
                local_date = cur.date()
                local_hour = cur.hour
                cache_key = (local_date, local_hour)
                if cache_key not in _slot_cache:
                    _slot_cache[cache_key] = _candidate_slot_starts(
                        local_date, local_hour
                    )
                for lo in _slot_cache[cache_key]:
                    slot_hi = lo + _HOUR
                    ov = _overlap_seconds(s, e, lo, slot_hi)
                    if ov > 0:
                        entries.append(
                            (s, e, local_date, local_hour, lo, slot_hi, ov)
                        )
                cur += _HOUR
        entries.sort(key=lambda x: (x[0], x[4]))
        session_index[post_id] = entries
        session_starts[post_id] = starts_list
        hour_indices: dict[int, list[int]] = {}
        meter_hrs: set[int] = set()
        for idx, entry in enumerate(entries):
            lh = entry[3]
            hour_indices.setdefault(lh, []).append(idx)
            meter_hrs.add(lh)
        for lh, indices in hour_indices.items():
            meter_hour_index[(post_id, lh)] = indices
        meter_hours[post_id] = meter_hrs
        for lh, elist in hour_indices.items():
            lo_sorted = [entries[i] for i in elist]
            lo_sorted.sort(key=lambda x: x[4])
            meter_hour_lo_entries[(post_id, lh)] = lo_sorted

    result = (
        session_index, session_starts, meter_hour_index,
        meter_hour_lo_entries, meter_hours,
    )
    _session_index_cache[key] = result
    return result


def clear_session_index_cache() -> None:
    """Drop all cached session indices (e.g. after dataset mutation)."""
    _session_index_cache.clear()


# ---------------------------------------------------------------------------
# Observation / result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestObservation:
    """One out-of-sample prediction with its observable proxy outcome."""

    post_id: str
    #: The slot is both the prediction cutoff (features may not see past it)
    #: and the target hour being predicted.
    prediction_time: datetime
    cutoff: datetime
    target_hour_start: datetime
    local_date: date
    local_hour: int
    weekday: str
    #: Canonical-spatial context, resolved point-in-time.
    post_blockface_id: str | None
    latitude: float | None
    longitude: float | None
    location_source: str  # "placement_at_t" or "unresolved"
    predicted_score: float
    #: Observable proxy outcome (paid-session overlap), NOT ground truth.
    proxy_occupied_minutes: float
    proxy_availability: int  # 1 iff no paid session overlapped the slot
    #: signed error vs continuous proxy free-fraction
    prediction_error: float
    method_version: str
    evidence_days: int
    evidence_sessions: int

    @property
    def proxy_free_fraction(self) -> float:
        return 1.0 - self.proxy_occupied_minutes / SLOT_MINUTES

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "prediction_time": self.prediction_time.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "target_hour_start": self.target_hour_start.isoformat(),
            "local_date": self.local_date.isoformat(),
            "local_hour": self.local_hour,
            "weekday": self.weekday,
            "blockface_id": self.post_blockface_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "predicted_score": self.predicted_score,
            "proxy_occupied_minutes": self.proxy_occupied_minutes,
            "proxy_availability": self.proxy_availability,
            "prediction_error": self.prediction_error,
            "method_version": self.method_version,
            "evidence_days": self.evidence_days,
            "evidence_sessions": self.evidence_sessions,
        }


#: Historical alias used by earlier revisions/tests.
Observation = BacktestObservation


def compute_metrics(observations: Iterable[BacktestObservation]) -> MetricSet:
    """MAE/RMSE vs the continuous proxy, Brier vs the binary proxy."""
    obs = list(observations)
    n = len(obs)
    if n == 0:
        return MetricSet(0)
    mean_score = sum(o.predicted_score for o in obs) / n
    avail_rate = sum(o.proxy_availability for o in obs) / n
    errors_free = [
        o.predicted_score - o.proxy_free_fraction for o in obs
    ]
    mae = sum(abs(e) for e in errors_free) / n
    rmse = math.sqrt(sum(e * e for e in errors_free) / n)
    brier = sum(
        (o.predicted_score - o.proxy_availability) ** 2 for o in obs
    ) / n
    return MetricSet(
        n=n,
        mean_score=round(mean_score, 4),
        proxy_availability_rate=round(avail_rate, 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        brier=round(brier, 4),
    )


@dataclass(frozen=True, slots=True)
class MetricSet:
    n: int
    mean_score: float | None = None
    proxy_availability_rate: float | None = None
    mae: float | None = None
    rmse: float | None = None
    brier: float | None = None
    suppressed: bool = False

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"n": self.n}
        if self.suppressed:
            d["suppressed"] = True
            return d
        d.update(
            mean_score=self.mean_score,
            proxy_availability_rate=self.proxy_availability_rate,
            mae=self.mae,
            rmse=self.rmse,
            brier=self.brier,
        )
        return d


@dataclass(frozen=True, slots=True)
class BacktestReport:
    method: str
    eval_days: int
    history_window_days: int
    until: datetime
    min_samples: int
    observations_generated: int
    predictions_made: int
    skipped_no_history: int
    overall: MetricSet
    by_hour: dict[int, MetricSet] = field(default_factory=dict)
    by_weekday: dict[str, MetricSet] = field(default_factory=dict)
    by_meter_type: dict[str, MetricSet] = field(default_factory=dict)
    by_blockface: dict[str, MetricSet] = field(default_factory=dict)
    by_evidence_days_bucket: dict[str, MetricSet] = field(default_factory=dict)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    observations: list[BacktestObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method": self.method,
            "eval_days": self.eval_days,
            "history_window_days": self.history_window_days,
            "until": self.until.isoformat(),
            "min_samples": self.min_samples,
            "observations_generated": self.observations_generated,
            "predictions_made": self.predictions_made,
            "skipped_no_history": self.skipped_no_history,
            "overall": self.overall.as_dict(),
            "by_hour": {str(k): v.as_dict() for k, v in self.by_hour.items()},
            "by_weekday": {k: v.as_dict() for k, v in self.by_weekday.items()},
            "by_meter_type": {
                k: v.as_dict() for k, v in self.by_meter_type.items()
            },
            "by_blockface": {
                k: v.as_dict() for k, v in self.by_blockface.items()
            },
            "by_evidence_days_bucket": {
                k: v.as_dict() for k, v in self.by_evidence_days_bucket.items()
            },
            "calibration": self.calibration,
            "target_definition": (
                "paid-session overlap proxy (not ground-truth occupancy)"
            ),
        }
        if self.observations:
            d["observations"] = [obs.as_dict() for obs in self.observations]
        return d



# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _bucket(evidence_days: int) -> str:
    if evidence_days <= 1:
        return "1"
    if evidence_days <= 3:
        return "2-3"
    if evidence_days <= 7:
        return "4-7"
    if evidence_days <= 14:
        return "8-14"
    return "15+"


def _group_by(
    observations: list[BacktestObservation],
    key,
    *,
    min_samples: int,
) -> dict[Any, MetricSet]:
    groups: dict[Any, list[BacktestObservation]] = {}
    for o in observations:
        groups.setdefault(key(o), []).append(o)
    out: dict[Any, MetricSet] = {}
    for k, group in sorted(
        groups.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])
    ):
        if len(group) >= min_samples:
            out[k] = compute_metrics(group)
        else:
            out[k] = MetricSet(n=len(group), suppressed=True)
    return out


def calibration_table(
    observations: list[BacktestObservation],
    *,
    buckets: int = 10,
    min_samples: int = 30,
) -> list[dict[str, Any]]:
    """Score-bucket calibration vs the binary proxy availability outcome."""
    width = 1.0 / buckets
    grouped: dict[int, list[BacktestObservation]] = {}
    for o in observations:
        idx = min(int(o.predicted_score / width), buckets - 1)
        grouped.setdefault(idx, []).append(o)
    table = []
    for idx in range(buckets):
        group = grouped.get(idx, [])
        lo, hi_b = round(idx * width, 2), round((idx + 1) * width, 2)
        entry: dict[str, Any] = {
            "score_bucket": f"{lo:.1f}-{hi_b:.1f}",
            "n": len(group),
        }
        if len(group) >= min_samples:
            entry["mean_score"] = round(
                sum(o.predicted_score for o in group) / len(group), 4
            )
            entry["proxy_availability_rate"] = round(
                sum(o.proxy_availability for o in group) / len(group), 4
            )
        table.append(entry)
    return table


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _load_sessions(
    conn: pg8000.native.Connection,
    *,
    load_since: datetime,
    until: datetime,
    post_ids: list[str] | None,
) -> dict[str, list[tuple[datetime, datetime]]]:
    clause = ""
    params: dict[str, Any] = {"since": load_since, "until": until}
    if post_ids is not None:
        clause = "AND post_id = ANY(:post_ids)"
        params["post_ids"] = post_ids
    rows = conn.run(
        "SELECT post_id, session_start, session_end FROM meter_transactions "
        f"WHERE session_end IS NOT NULL AND session_start < :until "
        f"AND session_end >= :since {clause} "
        "ORDER BY post_id, session_start",
        **params,
    )
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}
    for post_id, start, end in rows:
        sessions.setdefault(post_id, []).append((start, end))
    return sessions


@dataclass(frozen=True, slots=True)
class PlacementSpan:
    valid_from: float  # epoch seconds; -inf allowed
    valid_until: float  # epoch seconds; +inf when open
    latitude: float
    longitude: float
    blockface_id: str | None


def _load_placements(
    conn: pg8000.native.Connection,
    post_ids: set[str],
) -> dict[str, list[PlacementSpan]]:
    """Point-in-time meter geometry from the canonical temporal table."""
    if not post_ids:
        return {}
    rows = conn.run(
        "SELECT m.post_id, "
        "EXTRACT(EPOCH FROM p.valid_from)::float8, "
        "EXTRACT(EPOCH FROM p.valid_until)::float8, "
        "p.latitude, p.longitude, b.source_blockface_id "
        "FROM meter_placements p "
        "JOIN meters m ON m.meter_id = p.meter_id "
        "LEFT JOIN blockfaces b ON b.blockface_id = p.blockface_id "
        "WHERE m.post_id = ANY(:posts)",
        posts=sorted(post_ids),
    )
    spans: dict[str, list[PlacementSpan]] = {}
    for post_id, vf, vu, lat, lon, bf in rows:
        span = PlacementSpan(
            valid_from=vf,
            valid_until=float("inf") if vu is None else vu,
            latitude=lat,
            longitude=lon,
            blockface_id=bf,
        )
        spans.setdefault(post_id, []).append(span)
    for spans_list in spans.values():
        spans_list.sort(key=lambda sp: sp.valid_from)
    return spans


def _placement_at(spans: list[PlacementSpan], instant: datetime) -> PlacementSpan | None:
    epoch = instant.timestamp()
    starts = [sp.valid_from for sp in spans]
    i = bisect_left(starts, epoch)
    if i < len(starts) and starts[i] == epoch:
        return spans[i]
    if i == 0:
        return None
    candidate = spans[i - 1]
    return candidate if candidate.valid_until > epoch else None


def _meter_types(conn: pg8000.native.Connection) -> dict[str, str | None]:
    return {pid: mtype for pid, mtype in conn.run(
        "SELECT post_id, meter_type FROM parking_meters"
    )}


def run_backtest(
    conn: pg8000.native.Connection,
    *,
    until: datetime | None = None,
    eval_days: int = 7,
    history_window_days: int = 28,
    hours: tuple[int, ...] | None = None,
    post_ids: list[str] | None = None,
    max_meters: int | None = None,
    include_observations: bool = False,
    min_samples: int = 30,
    model: BaselineModel | None = None,
) -> BacktestReport:
    """Run the out-of-sample evaluation. Deterministic given DB state."""
    model = model or MODELS[BASELINE_METHOD]
    if until is None:
        until = datetime.now(UTC)
    elif until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    until = until.astimezone(UTC)

    target_hours = set(hours) if hours is not None else set(range(24))

    sessions = _load_sessions(
        conn,
        load_since=until - timedelta(days=eval_days + history_window_days + 2),
        until=until,
        post_ids=post_ids,
    )

    placements = _load_placements(
        conn, set(sessions) if post_ids is None else set(post_ids)
    )
    meter_types = _meter_types(conn)

    eval_start_local = (until - timedelta(days=eval_days)).astimezone(SF_TZ).date()
    # Slots on until's own date are kept; anything at/after ``until`` is
    # skipped per-slot below.
    eval_end_local = until.astimezone(SF_TZ).date()

    observations: list[BacktestObservation] = []
    generated = 0
    skipped_no_history = 0
    meters = sorted(sessions) if max_meters is None else sorted(sessions)[:max_meters]

    for post_id in meters:
        meter_sessions = sessions[post_id]
        meter_starts = [s for s, _ in meter_sessions]
        spans = placements.get(post_id, [])
        first_day = meter_sessions[0][0].astimezone(SF_TZ).date()
        day = max(first_day, eval_start_local)
        while day <= eval_end_local:
            for hour in sorted(target_hours):
                for lo in _candidate_slot_starts(day, hour):
                    if lo >= until:
                        continue
                    generated += 1

                    # Outcome proxy: FULL transactions overlapping the slot.
                    outcome_occupied = 0.0
                    j = max(bisect_left(meter_starts, lo - timedelta(days=2)) - 1, 0)
                    while j < len(meter_sessions):
                        s, e = meter_sessions[j]
                        if s >= lo + timedelta(hours=1):
                            break
                        outcome_occupied += _overlap_seconds(
                            s, e, lo, lo + timedelta(hours=1)
                        )
                        j += 1

                    # Prepare blockface context for models that need it.
                    if hasattr(model, "prepare"):
                        model.prepare(  # type: ignore[union-attr]
                            sessions,
                            placements,
                            meter_types,
                            history_window_days=history_window_days,
                            slot_start=lo,
                        )
                    pred = model.predict(
                        meter_sessions,
                        lo,
                        history_window_days=history_window_days,
                        post_id=post_id,
                    )
                    if pred is None:
                        skipped_no_history += 1
                        continue

                    placement = _placement_at(spans, lo)
                    occupied_minutes = outcome_occupied / 60.0
                    observations.append(
                        BacktestObservation(
                            post_id=post_id,
                            prediction_time=lo,
                            cutoff=lo,
                            target_hour_start=lo,
                            local_date=day,
                            local_hour=lo.astimezone(SF_TZ).hour,
                            weekday=WEEKDAYS[lo.astimezone(SF_TZ).weekday()],
                            post_blockface_id=(
                                placement.blockface_id if placement else None
                            ),
                            latitude=placement.latitude if placement else None,
                            longitude=placement.longitude if placement else None,
                            location_source=(
                                "placement_at_t" if placement else "unresolved"
                            ),
                            predicted_score=pred.score,
                            proxy_occupied_minutes=round(occupied_minutes, 3),
                            proxy_availability=1 if outcome_occupied == 0.0 else 0,
                            prediction_error=round(
                                pred.score - (1.0 - occupied_minutes), 4
                            ),
                            method_version=model.method,
                            evidence_days=pred.evidence_days,
                            evidence_sessions=pred.evidence_sessions,
                        )
                    )
            day += timedelta(days=1)

    return BacktestReport(
        method=model.method,
        eval_days=eval_days,
        history_window_days=history_window_days,
        until=until,
        min_samples=min_samples,
        observations_generated=generated,
        predictions_made=len(observations),
        skipped_no_history=skipped_no_history,
        overall=compute_metrics(observations),
        by_hour=_group_by(observations, lambda o: o.local_hour, min_samples=min_samples),
        by_weekday=_group_by(observations, lambda o: o.weekday, min_samples=min_samples),
        by_meter_type=_group_by(
            observations,
            lambda o: (meter_types.get(o.post_id) or "unknown"),
            min_samples=min_samples,
        ),
        by_blockface=_group_by(
            observations,
            lambda o: o.post_blockface_id or "unresolved",
            min_samples=min_samples,
        ),
        by_evidence_days_bucket=_group_by(
            observations, lambda o: _bucket(o.evidence_days), min_samples=min_samples
        ),
        calibration=calibration_table(observations, min_samples=min_samples),
        observations=observations if include_observations else [],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_summary(report: BacktestReport) -> str:
    lines = [
        (
            f"Backtest — method={report.method} eval_days={report.eval_days} "
            f"history_window={report.history_window_days}d until={report.until.isoformat()}"
        ),
        (
            f"observations generated={report.observations_generated} "
            f"predicted={report.predictions_made} "
            f"skipped(no history)={report.skipped_no_history}"
        ),
    ]
    o = report.overall
    if o.n and not o.suppressed:
        lines.append(
            f"overall: n={o.n} MAE={o.mae} RMSE={o.rmse} Brier={o.brier} "
            f"mean_score={o.mean_score} proxy_avail_rate={o.proxy_availability_rate}"
        )
    else:
        lines.append(f"overall: n={o.n} (insufficient observations for metrics)")
    for name, breakdown in (
        ("by_hour", report.by_hour),
        ("by_weekday", report.by_weekday),
        ("by_meter_type", report.by_meter_type),
        ("by_evidence_days", report.by_evidence_days_bucket),
    ):
        rendered = ", ".join(
            f"{k}: n={m.n}" + ("" if m.suppressed else f" MAE={m.mae}")
            for k, m in breakdown.items()
        )
        if rendered:
            lines.append(f"{name}: {rendered}")
    cal_cells = [
        f"{c['score_bucket']}:n={c['n']}"
        + (f":obs={c['proxy_availability_rate']}" if "proxy_availability_rate" in c else "")
        for c in report.calibration
        if c["n"]
    ]
    if cal_cells:
        lines.append("calibration(vs binary proxy): " + " ".join(cal_cells))
    lines.append(
        "NOTE: outcomes are a paid-session overlap PROXY, not ground-truth "
        "occupancy; scores are not calibrated probabilities."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sf_parking.backtest",
        description="Point-in-time-safe backtesting of parking-state baselines.",
    )
    parser.add_argument("--start", help="eval window start, YYYY-MM-DD (local PT)")
    parser.add_argument("--end", help="eval window end (inclusive), YYYY-MM-DD (local PT)")
    parser.add_argument("--until", help="absolute cutoff instant, ISO-8601")
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--window-days", type=int, default=28,
                        help="baseline lookback window")
    parser.add_argument("--hour", type=int, action="append",
                        help="restrict to local hour(s); repeatable")
    parser.add_argument("--post-id", action="append", dest="post_ids",
                        help="restrict to meter(s); repeatable")
    parser.add_argument("--near", metavar="LAT,LON",
                        help="restrict to meters within RADIUS_METERS of point")
    parser.add_argument("--radius-meters", type=float, default=250.0)
    parser.add_argument("--max-meters", type=int)
    parser.add_argument("--min-samples", type=int, default=30,
                        help="minimum observations per breakdown cell")
    parser.add_argument("--model", choices=sorted(MODELS), default=BASELINE_METHOD)
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of summary")
    args = parser.parse_args(argv)

    until: datetime | None = None
    eval_days = args.eval_days
    if args.start and args.end:
        start_pt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=SF_TZ)
        end_pt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=SF_TZ)
        until = (end_pt + timedelta(days=1)).astimezone(UTC)
        eval_days = max((end_pt.date() - start_pt.date()).days + 1, 1)
    elif args.until:
        until = datetime.fromisoformat(args.until)
    if until is not None and until.tzinfo is None:
        until = until.replace(tzinfo=UTC)

    scoped_post_ids = list(args.post_ids) if args.post_ids else None
    near_clause = ""
    near_params: dict[str, Any] = {}
    if args.near:
        lat_s, lon_s = args.near.split(",")
        near_clause = (
            "AND ST_DWithin(location, "
            "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)"
        )
        near_params = {"lat": float(lat_s), "lon": float(lon_s),
                       "radius": args.radius_meters}

    conn = pg8000.native.Connection(**_conn_kwargs())
    try:
        if scoped_post_ids is None and args.near:
            scoped_post_ids = [
                r[0]
                for r in conn.run(
                    "SELECT post_id FROM parking_meters WHERE true " + near_clause,
                    **near_params,
                )
            ]
            if not scoped_post_ids:
                print(json.dumps({"error": "no meters within radius"}))
                return 1

        report = run_backtest(
            conn,
            until=until,
            eval_days=eval_days,
            history_window_days=args.window_days,
            hours=tuple(args.hour) if args.hour else None,
            post_ids=scoped_post_ids,
            max_meters=args.max_meters,
            include_observations=args.include_observations,
            min_samples=args.min_samples,
            model=MODELS[args.model],
        )
    finally:
        conn.close()

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(format_summary(report))
        print("use --json for machine-readable output")
    return 0


def _conn_kwargs() -> dict[str, Any]:
    import os
    from urllib.parse import urlparse

    from .database import database_url_from_env

    parsed = urlparse(database_url_from_env())
    password = parsed.password or os.environ.get("PGPASSWORD", "postgres")
    return {
        "user": parsed.username or "postgres",
        "password": password,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": (parsed.path or "/").lstrip("/") or "sf_parking",
        "timeout": 60,
    }


if __name__ == "__main__":
    raise SystemExit(main())
