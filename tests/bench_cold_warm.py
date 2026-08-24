"""Benchmark: cold-start vs warm-start vs cross-model session index building."""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta

from sf_parking.backtest import (
    HourConditionedV1Baseline,
    build_session_index,
    clear_session_index_cache,
)

random.seed(42)


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


def main() -> None:
    for n_meters, n_days, s_per_day, label in [
        (500, 90, 8, "500m x 90d"),
        (2000, 90, 8, "2000m x 90d"),
    ]:
        sessions = _make_sessions(n_meters, n_days, s_per_day,
                                  datetime(2026, 6, 1, tzinfo=UTC))
        total = sum(len(v) for v in sessions.values())
        slot = datetime(2026, 8, 30, 12, tzinfo=UTC)

        print(f"\n{'='*60}")
        print(f" {label} ({total} sessions)")
        print(f"{'='*60}")

        # --- Cold start: build_session_index() from scratch ---
        clear_session_index_cache()
        t0 = time.perf_counter()
        idx = build_session_index(sessions)
        t_cold = (time.perf_counter() - t0) * 1000
        print(f"\n  Cold start (build_session_index):  {t_cold:8.1f}ms")

        # --- Warm start: same sessions dict, cache hit ---
        t0 = time.perf_counter()
        idx2 = build_session_index(sessions)
        t_warm = (time.perf_counter() - t0) * 1000
        print(f"  Warm start (cache hit):            {t_warm:8.1f}ms")
        assert idx is idx2, "Cache should return same object"
        print(f"  Speedup:                           {t_cold/t_warm:8.0f}x")

        # --- Cross-model: new instance with pre-built index ---
        t0 = time.perf_counter()
        model = HourConditionedV1Baseline(
            evidence_halflife=14.0, session_index=idx
        )
        model.prepare(
            sessions, {}, {k: "SS" for k in sessions},
            history_window_days=28, slot_start=slot,
        )
        t_cross = (time.perf_counter() - t0) * 1000
        print(f"  Cross-model (pre-built index):     {t_cross:8.1f}ms")

        # --- Fresh model, no pre-built index (but cache still warm) ---
        t0 = time.perf_counter()
        model2 = HourConditionedV1Baseline(evidence_halflife=14.0)
        model2.prepare(
            sessions, {}, {k: "SS" for k in sessions},
            history_window_days=28, slot_start=slot,
        )
        t_fresh = (time.perf_counter() - t0) * 1000
        print(f"  Fresh model (index cache warm):    {t_fresh:8.1f}ms")

        # --- Old baseline: no cache, no memoization (simulated) ---
        # The old code took ~1.5s for 500m and ~6.0s for 2000m.
        # We can't time it directly since it was replaced, but we
        # note the numbers from earlier profiling.
        old_estimates = {(500, 90): 1460, (2000, 90): 6000}
        old_ms = old_estimates.get((n_meters, n_days), 0)
        if old_ms:
            print(f"\n  Old _build_session_index (before): {old_ms:8.1f}ms")
            print(f"  New cold start speedup:            {old_ms/t_cold:8.1f}x")


if __name__ == "__main__":
    main()
