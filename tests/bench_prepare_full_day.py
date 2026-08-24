"""Full-day benchmark with realistic data distribution.

Sessions span 90 days (wider than the 28-day window) so the bisect
optimization can actually prune entries outside the window.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta

from sf_parking.backtest import HourConditionedV1Baseline

random.seed(42)


def _make_sessions(
    n_meters: int, n_days: int, s_per_day: int,
    start_date: datetime = datetime(2026, 6, 1, tzinfo=UTC),
) -> dict[str, list[tuple[datetime, datetime]]]:
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}
    for m in range(n_meters):
        post_id = f"M{m:04d}"
        meter_sessions: list[tuple[datetime, datetime]] = []
        for d in range(n_days):
            day = start_date + timedelta(days=d)
            for _ in range(s_per_day):
                hour_pt = random.randint(7, 20)
                hour_utc = (hour_pt + 7) % 24
                day_offset = 1 if hour_pt + 7 >= 24 else 0
                sess_start = day + timedelta(days=day_offset, hours=hour_utc)
                duration_min = random.randint(10, 55)
                sess_end = sess_start + timedelta(minutes=duration_min)
                meter_sessions.append((sess_start, sess_end))
        meter_sessions.sort()
        sessions[post_id] = meter_sessions
    return sessions


def main() -> None:
    # 90 days of data, but 28-day window → ~62 days of entries to prune
    n_meters, n_days, s_per_day = 500, 90, 8
    sessions = _make_sessions(n_meters, n_days, s_per_day)
    total = sum(len(v) for v in sessions.values())
    print(f"Dataset: {n_meters}m × {n_days}d × {s_per_day}s/d = {total} sessions\n")

    model = HourConditionedV1Baseline(evidence_halflife=14.0)

    # 24 consecutive hourly slots
    slot = datetime(2026, 8, 30, 0, tzinfo=UTC)
    t_start = time.perf_counter()
    for i in range(24):
        model.prepare(
            sessions, {}, {k: "SS" for k in sessions},
            history_window_days=28, slot_start=slot,
        )
        slot += timedelta(hours=1)
    t_total = time.perf_counter() - t_start
    print(f"\n24-slot total: {t_total:.3f}s  ({t_total/24*1000:.0f}ms/slot avg)")


if __name__ == "__main__":
    main()
