"""Synthetic benchmark for HourConditionedV1Baseline.prepare().

Generates a dataset sized to expose the hot loop, then runs prepare()
across multiple slot_starts with instrumentation output.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta

from sf_parking.backtest import HourConditionedV1Baseline

random.seed(42)


def _make_sessions(
    n_meters: int,
    n_days: int,
    sessions_per_day: int,
    start_date: datetime = datetime(2026, 7, 1, tzinfo=UTC),
) -> dict[str, list[tuple[datetime, datetime]]]:
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}
    for m in range(n_meters):
        post_id = f"M{m:04d}"
        meter_sessions: list[tuple[datetime, datetime]] = []
        for d in range(n_days):
            day = start_date + timedelta(days=d)
            for _ in range(sessions_per_day):
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
    # Scale: 500 meters × 30 days × 8 sessions/day = 120k sessions
    n_meters, n_days, s_per_day = 500, 30, 8
    sessions = _make_sessions(n_meters, n_days, s_per_day)
    total = sum(len(v) for v in sessions.values())
    print(f"Dataset: {n_meters} meters × {n_days} days × {s_per_day} sess/day = {total} sessions\n")

    model = HourConditionedV1Baseline(evidence_halflife=14.0)

    # Run prepare() across 5 consecutive hourly slots
    slot = datetime(2026, 8, 1, 20, tzinfo=UTC)
    for i in range(5):
        t0 = time.perf_counter()
        model.prepare(
            sessions,
            {},
            {k: "SS" for k in sessions},
            history_window_days=28,
            slot_start=slot,
        )
        elapsed = time.perf_counter() - t0
        print(f"  wall={elapsed:.3f}s")
        slot += timedelta(hours=1)
        print()


if __name__ == "__main__":
    main()
