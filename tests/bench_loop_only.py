"""Final benchmark: old vs new prepare() inner loop, with entry counts."""

from __future__ import annotations

import random
import time
from bisect import bisect_left
from datetime import UTC, date, datetime, timedelta

from sf_parking.backtest import (
    HourConditionedV1Baseline,
    SLOT_MINUTES,
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


def _bench_loop_old(model, sessions, slot, window_start):
    n_scanned = 0
    n_contrib = 0
    for post_id in sessions:
        raw = sessions[post_id]
        n_hist = bisect_left([s for s, _ in raw], slot)
        for local_hour in model._meter_hours.get(post_id, set()):
            indices = model._meter_hour_index.get((post_id, local_hour), ())
            for j in indices:
                n_scanned += 1
                sess_s, sess_e, loc_date, loc_h, lo, hi, ov_full = (
                    model._session_index[post_id][j]
                )
                if loc_h != local_hour:
                    continue
                if sess_s >= slot:
                    continue
                truncated_end = min(sess_e, slot)
                if truncated_end <= window_start:
                    continue
                if lo < window_start or lo >= slot:
                    continue
                overlap = _overlap_seconds(sess_s, truncated_end, lo, lo + _HOUR)
                if overlap > 0:
                    n_contrib += 1
    return n_scanned, n_contrib


def _bench_loop_new(model, sessions, slot, window_start):
    n_candidates = 0
    n_contrib = 0
    for post_id in sessions:
        raw = sessions[post_id]
        n_hist = bisect_left([s for s, _ in raw], slot)
        for local_hour in model._meter_hours.get(post_id, set()):
            lo_entries = model._meter_hour_lo_entries.get(
                (post_id, local_hour), ()
            )
            lo_lo = bisect_left(lo_entries, window_start, key=lambda e: e[4])
            lo_hi = bisect_left(lo_entries, slot, key=lambda e: e[4])
            n_candidates += lo_hi - lo_lo
            for entry in lo_entries[lo_lo:lo_hi]:
                sess_s, sess_e, loc_date, _loc_h, lo, _hi, _ov_full = entry
                if sess_s >= slot:
                    continue
                truncated_end = min(sess_e, slot)
                overlap = _overlap_seconds(sess_s, truncated_end, lo, lo + _HOUR)
                if overlap > 0:
                    n_contrib += 1
    return n_candidates, n_contrib


def main() -> None:
    configs = [
        (500, 30, 8, "500m x 30d (tight window)"),
        (500, 90, 8, "500m x 90d (wide data)"),
        (2000, 90, 8, "2000m x 90d (production-ish)"),
    ]

    for n_meters, n_days, s_per_day, label in configs:
        sessions = _make_sessions(n_meters, n_days, s_per_day,
                                  datetime(2026, 6, 1, tzinfo=UTC))
        total = sum(len(v) for v in sessions.values())

        model = HourConditionedV1Baseline(evidence_halflife=14.0)
        slot = datetime(2026, 8, 30, 12, tzinfo=UTC)
        window_start = slot - timedelta(days=28)
        model.prepare(sessions, {}, {k: "SS" for k in sessions},
                       history_window_days=28, slot_start=slot)

        # Old loop: 5 iterations, average
        t0 = time.perf_counter()
        for _ in range(5):
            n_scanned, n_c1 = _bench_loop_old(model, sessions, slot, window_start)
        t_old = (time.perf_counter() - t0) / 5 * 1000

        # New loop: 5 iterations, average
        t0 = time.perf_counter()
        for _ in range(5):
            n_cand, n_c2 = _bench_loop_new(model, sessions, slot, window_start)
        t_new = (time.perf_counter() - t0) / 5 * 1000

        print(f"\n{label} ({total} sessions)")
        print(f"  Old: scanned={n_scanned:>8,}  loop={t_old:6.1f}ms")
        print(f"  New: candidates={n_cand:>6,}  loop={t_new:6.1f}ms")
        print(f"  Entries pruned: {n_scanned - n_cand:>7,} ({(1 - n_cand/n_scanned)*100:.0f}%)")
        print(f"  Loop speedup:   {t_old/t_new:.2f}x")


if __name__ == "__main__":
    main()
