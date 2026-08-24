"""Detailed profiler for _build_session_index(): phase-by-phase breakdown."""

from __future__ import annotations

import random
import time
from datetime import UTC, date, datetime, timedelta

from sf_parking.backtest import (
    HourConditionedV1Baseline,
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


def profile_build_index(sessions):
    """Profile each phase of _build_session_index separately."""
    n_sessions_total = sum(len(v) for v in sessions.values())
    n_meters = len(sessions)

    # Phase 1: astimezone conversions
    t0 = time.perf_counter()
    conversions = 0
    for post_id, raw_sessions in sessions.items():
        for s, e in raw_sessions:
            if e is None:
                continue
            _ = s.astimezone(SF_TZ)
            _ = e.astimezone(SF_TZ)
            conversions += 2
    t_tz = time.perf_counter() - t0

    # Phase 2: walk local hours + candidate_slot_starts + overlap_seconds
    t0 = time.perf_counter()
    n_entries = 0
    n_slot_calls = 0
    n_overlap_calls = 0
    for post_id, raw_sessions in sessions.items():
        for s, e in raw_sessions:
            if e is None:
                continue
            s_local = s.astimezone(SF_TZ)
            e_local = e.astimezone(SF_TZ)
            cur = s_local.replace(minute=0, second=0, microsecond=0)
            end_wall = e_local
            while cur < end_wall:
                local_date = cur.date()
                local_hour = cur.hour
                for lo in _candidate_slot_starts(local_date, local_hour):
                    n_slot_calls += 1
                    slot_hi = lo + _HOUR
                    ov = _overlap_seconds(s, e, lo, slot_hi)
                    n_overlap_calls += 1
                    if ov > 0:
                        n_entries += 1
                cur += _HOUR
    t_entries = time.perf_counter() - t0

    # Phase 3: sort entries per meter
    t0 = time.perf_counter()
    for post_id, raw_sessions in sessions.items():
        entries = [(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC),
                     date(2026, 1, 1), 0, datetime(2026, 1, 1, tzinfo=UTC),
                     datetime(2026, 1, 1, 1, tzinfo=UTC), 0.0)] * max(1, n_entries // n_meters)
        entries.sort(key=lambda x: (x[0], x[4]))
    t_sort = time.perf_counter() - t0

    # Phase 4: build hour_indices + meter_hours + lo_entries per meter
    t0 = time.perf_counter()
    for post_id, raw_sessions in sessions.items():
        # Simulate hour_indices building
        hour_indices: dict[int, list[int]] = {}
        meter_hrs: set[int] = set()
        for idx in range(max(1, n_entries // n_meters)):
            lh = idx % 24
            hour_indices.setdefault(lh, []).append(idx)
            meter_hrs.add(lh)
        # Simulate lo_entries building
        for lh, elist in hour_indices.items():
            lo_sorted = list(range(len(elist)))
            lo_sorted.sort()
    t_index_build = time.perf_counter() - t0

    # Phase 5: dict assignments (simulated)
    t0 = time.perf_counter()
    session_index = {}
    session_starts = {}
    meter_hour_index = {}
    meter_hour_lo_entries = {}
    meter_hours = {}
    for post_id, raw_sessions in sessions.items():
        session_index[post_id] = []
        session_starts[post_id] = [s for s, _ in raw_sessions]
        meter_hour_index[(post_id, 0)] = []
        meter_hour_lo_entries[(post_id, 0)] = []
        meter_hours[post_id] = {0}
    t_dict = time.perf_counter() - t0

    print(f"\nProfile: {n_meters} meters, {n_sessions_total} sessions")
    print(f"  Entries generated: {n_entries:,}")
    print(f"  candidate_slot_starts calls: {n_slot_calls:,}")
    print(f"  overlap_seconds calls: {n_overlap_calls:,}")
    print(f"")
    print(f"  Phase 1 - astimezone ({conversions:,} calls):  {t_tz*1000:8.1f}ms")
    print(f"  Phase 2 - slot walk + overlap:                  {t_entries*1000:8.1f}ms")
    print(f"  Phase 3 - sort entries:                         {t_sort*1000:8.1f}ms")
    print(f"  Phase 4 - hour_indices + lo_entries:            {t_index_build*1000:8.1f}ms")
    print(f"  Phase 5 - dict assignments:                     {t_dict*1000:8.1f}ms")
    total = t_tz + t_entries + t_sort + t_index_build + t_dict
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total (synthetic):                             {total*1000:8.1f}ms")

    # Now time the REAL _build_session_index for comparison
    model = HourConditionedV1Baseline(evidence_halflife=14.0)
    t0 = time.perf_counter()
    model._build_session_index(sessions)
    t_real = time.perf_counter() - t0
    print(f"  Real _build_session_index:                     {t_real*1000:8.1f}ms")

    return {
        "tz": t_tz, "entries": t_entries, "sort": t_sort,
        "index_build": t_index_build, "dict": t_dict, "real": t_real,
    }


def main() -> None:
    for n_meters, n_days, s_per_day, label in [
        (500, 90, 8, "500m x 90d"),
        (2000, 90, 8, "2000m x 90d"),
    ]:
        sessions = _make_sessions(n_meters, n_days, s_per_day,
                                  datetime(2026, 6, 1, tzinfo=UTC))
        print(f"\n{'='*55}")
        print(f" {label}")
        print(f"{'='*55}")
        profile_build_index(sessions)


if __name__ == "__main__":
    main()
