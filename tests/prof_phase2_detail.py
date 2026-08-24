"""Drill into Phase 2: separate astimezone, candidate_slot_starts, overlap_seconds."""

from __future__ import annotations

import random
import time
from datetime import UTC, date, datetime, timedelta

from sf_parking.backtest import (
    SF_TZ,
    _candidate_slot_starts,
    _overlap_seconds,
)

random.seed(42)
_HOUR = timedelta(hours=1)


def _make_sessions(n_meters, n_days, s_per_day, start_date):
    sessions = {}
    for m in range(n_meters):
        post_id = f"M{m:04d}"
        ss = []
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


def profile_phase2_detail(sessions):
    """Time each sub-step of Phase 2 independently."""
    n_sessions = sum(len(v) for v in sessions.values())

    # Collect all (s, e) pairs
    all_pairs = [(s, e) for ss in sessions.values() for s, e in ss if e is not None]

    # Sub-step A: astimezone only
    t0 = time.perf_counter()
    for s, e in all_pairs:
        _ = s.astimezone(SF_TZ)
        _ = e.astimezone(SF_TZ)
    t_tz = time.perf_counter() - t0

    # Sub-step B: replace(minute=0...) + .date() + .hour
    t0 = time.perf_counter()
    for s, e in all_pairs:
        s_local = s.astimezone(SF_TZ)
        e_local = e.astimezone(SF_TZ)
        cur = s_local.replace(minute=0, second=0, microsecond=0)
    t_replace = time.perf_counter() - t0

    # Sub-step C: candidate_slot_starts calls only
    t0 = time.perf_counter()
    n_slot_calls = 0
    for s, e in all_pairs:
        s_local = s.astimezone(SF_TZ)
        e_local = e.astimezone(SF_TZ)
        cur = s_local.replace(minute=0, second=0, microsecond=0)
        end_wall = e_local
        while cur < end_wall:
            for lo in _candidate_slot_starts(cur.date(), cur.hour):
                n_slot_calls += 1
            cur += _HOUR
    t_candslot = time.perf_counter() - t0

    # Sub-step D: overlap_seconds calls only (with pre-computed slot starts)
    t0 = time.perf_counter()
    n_overlap = 0
    for s, e in all_pairs:
        s_local = s.astimezone(SF_TZ)
        e_local = e.astimezone(SF_TZ)
        cur = s_local.replace(minute=0, second=0, microsecond=0)
        end_wall = e_local
        while cur < end_wall:
            for lo in _candidate_slot_starts(cur.date(), cur.hour):
                slot_hi = lo + _HOUR
                ov = _overlap_seconds(s, e, lo, slot_hi)
                n_overlap += 1
            cur += _HOUR
    t_full = time.perf_counter() - t0

    # Sub-step E: just the while loop + for + append (no tz, no functions)
    t0 = time.perf_counter()
    n_iters = 0
    for s, e in all_pairs:
        # Simulate: walk hours from start to end using raw UTC
        # (rough approximation of iteration count)
        dur_hours = int((e - s).total_seconds() // 3600) + 1
        for _ in range(dur_hours):
            n_iters += 1
    t_loop = time.perf_counter() - t0

    print(f"\n  Sessions: {n_sessions:,}")
    print(f"  candidate_slot_starts calls: {n_slot_calls:,}")
    print(f"  overlap_seconds calls: {n_overlap:,}")
    print(f"  while-loop iterations: {n_iters:,}")
    print()
    print(f"  A. astimezone (2 per session):       {t_tz*1000:8.1f}ms  ({t_tz/t_full*100:.0f}%)")
    print(f"  B. replace + date + hour:            {t_replace*1000:8.1f}ms  ({t_replace/t_full*100:.0f}%)")
    print(f"  C. candidate_slot_starts calls:      {t_candslot*1000:8.1f}ms  ({t_candslot/t_full*100:.0f}%)")
    print(f"  D. Full (tz + slot + overlap):       {t_full*1000:8.1f}ms  (100%)")
    print(f"  E. Pure loop iterations (no tz):     {t_loop*1000:8.1f}ms  ({t_loop/t_full*100:.0f}%)")
    print(f"  D - C (overlap_seconds calls):       {(t_full-t_candslot)*1000:8.1f}ms")
    print(f"  C - E (candidate_slot_starts - loop):{(t_candslot-t_loop)*1000:8.1f}ms")


def main():
    for n_meters, n_days, s_per_day, label in [
        (500, 90, 8, "500m x 90d"),
        (2000, 90, 8, "2000m x 90d"),
    ]:
        sessions = _make_sessions(n_meters, n_days, s_per_day,
                                  datetime(2026, 6, 1, tzinfo=UTC))
        print(f"\n{'='*55}")
        print(f" {label}")
        print(f"{'='*55}")
        profile_phase2_detail(sessions)


if __name__ == "__main__":
    main()
