"""A/B benchmark: old scan-all vs new bisect-optimized prepare() inner loop.

Runs both implementations on the same synthetic dataset and reports:
- old prepare ms/slot
- new prepare ms/slot
- candidate entries examined before/after
"""

from __future__ import annotations

import random
import time
from bisect import bisect_left
from datetime import UTC, date, datetime, timedelta

from sf_parking.backtest import (
    HourConditionedV1Baseline,
    SLOT_MINUTES,
    SF_TZ,
    _candidate_slot_starts,
    _overlap_seconds,
    score_v0,
    Prediction,
)

random.seed(42)

_HOUR = timedelta(hours=1)


def _make_sessions(
    n_meters: int, n_days: int, s_per_day: int,
    start_date: datetime = datetime(2026, 6, 1, tzinfo=UTC),
) -> dict[str, list[tuple[datetime, datetime]]]:
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}
    for m in range(n_meters):
        post_id = f"M{m:04d}"
        ss: list[tuple[datetime, datetime]] = []
        for d in range(n_days):
            day = start_date + timedelta(days=d)
            for _ in range(s_per_day):
                h = random.randint(7, 20)
                hutc = (h + 7) % 24
                do = 1 if h + 7 >= 24 else 0
                s = day + timedelta(days=do, hours=hutc)
                e = s + timedelta(minutes=random.randint(10, 55))
                ss.append((s, e))
        ss.sort()
        sessions[post_id] = ss
    return sessions


def prepare_old(
    model: HourConditionedV1Baseline,
    all_sessions: dict[str, list[tuple[datetime, datetime]]],
    *,
    history_window_days: int,
    slot_start: datetime,
) -> tuple[dict[int, Prediction], int, int]:
    """Old prepare: scan ALL entries per (meter, hour), apply 4 inline filters."""
    if slot_start in model._cache:
        return model._slot_hour_scores.get(slot_start, {}), 0, 0
    model._cache[slot_start] = None
    model._build_session_index(all_sessions)

    window_start = slot_start - timedelta(days=history_window_days)
    hour_occupied = {h: 0.0 for h in range(24)}
    hour_possible = {h: 0.0 for h in range(24)}
    hour_sessions = {h: 0 for h in range(24)}

    n_scanned = 0
    n_contributions = 0
    for post_id in all_sessions:
        raw = all_sessions[post_id]
        n_hist = bisect_left([s for s, _ in raw], slot_start)
        for local_hour in model._meter_hours.get(post_id, set()):
            indices = model._meter_hour_index.get((post_id, local_hour), ())
            meter_days: set[date] = set()
            meter_occupied = 0.0
            for j in indices:
                n_scanned += 1
                sess_s, sess_e, loc_date, loc_h, lo, hi, ov_full = (
                    model._session_index[post_id][j]
                )
                if loc_h != local_hour:
                    continue
                if sess_s >= slot_start:
                    continue
                truncated_end = min(sess_e, slot_start)
                if truncated_end <= window_start:
                    continue
                if lo < window_start or lo >= slot_start:
                    continue
                overlap = _overlap_seconds(sess_s, truncated_end, lo, lo + _HOUR)
                if overlap > 0:
                    meter_occupied += overlap
                    meter_days.add(loc_date)
                    n_contributions += 1
            ev_days = len(meter_days)
            if ev_days > 0:
                hour_occupied[local_hour] += meter_occupied
                hour_possible[local_hour] += ev_days * SLOT_MINUTES
                hour_sessions[local_hour] += n_hist

    hour_scores = {}
    for h in range(24):
        if hour_possible[h] > 0:
            hour_scores[h] = Prediction(
                score=score_v0(hour_occupied[h] / 60.0, hour_possible[h]),
                evidence_days=0,
                evidence_sessions=hour_sessions[h],
            )
    model._slot_hour_scores[slot_start] = hour_scores
    return hour_scores, n_scanned, n_contributions


def main() -> None:
    for n_meters, n_days, s_per_day in [(500, 30, 8), (500, 90, 8), (2000, 90, 8)]:
        sessions = _make_sessions(n_meters, n_days, s_per_day)
        total = sum(len(v) for v in sessions.values())
        print(f"\n{'='*60}")
        print(f"Dataset: {n_meters}m × {n_days}d × {s_per_day}s/d = {total} sessions")
        print(f"{'='*60}")

        # --- Old: scan all entries ---
        model_old = HourConditionedV1Baseline(evidence_halflife=14.0)
        slot = datetime(2026, 8, 30, 12, tzinfo=UTC)
        t0 = time.perf_counter()
        _, n_scanned, n_contrib = prepare_old(
            model_old, sessions, history_window_days=28, slot_start=slot,
        )
        t_old = (time.perf_counter() - t0) * 1000
        print(f"\n[old] scanned={n_scanned} contributions={n_contrib} time={t_old:.1f}ms")

        # --- New: bisect-optimized ---
        model_new = HourConditionedV1Baseline(evidence_halflife=14.0)
        t0 = time.perf_counter()
        model_new.prepare(
            sessions, {}, {k: "SS" for k in sessions},
            history_window_days=28, slot_start=slot,
        )
        t_new = (time.perf_counter() - t0) * 1000
        # Verify correctness
        old_scores = model_old._slot_hour_scores[slot]
        new_scores = model_new._slot_hour_scores[slot]
        match = all(
            old_scores.get(h) is not None and new_scores.get(h) is not None
            and old_scores[h].score == new_scores[h].score
            for h in set(old_scores) | set(new_scores)
        )
        print(f"[new] time={t_new:.1f}ms correct={match}")
        print(f"Speedup: {t_old/t_new:.2f}x")


if __name__ == "__main__":
    main()
