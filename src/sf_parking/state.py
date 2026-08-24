"""Bounded paid-state inference primitives for hourly parking modeling."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp
from typing import Iterable


@dataclass(frozen=True, slots=True)
class StateEvent:
    post_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class HourlyPaidState:
    slot_start: datetime
    slot_minutes: float
    transaction_count: int
    paid_overlap_minutes: float
    paid_occupancy_probability: float
    paid_availability_probability: float


def event_probability(overlap_minutes: float, *, p90_minutes: float = 120.0) -> float:
    if overlap_minutes <= 0:
        return 0.0
    x = 5.0 * (overlap_minutes / max(1.0, p90_minutes) - 0.2)
    sigmoid = 1.0 / (1.0 + exp(-x))
    return min(0.98, max(0.0, 0.98 * sigmoid))


def infer_hourly_paid_state(
    events: Iterable[StateEvent],
    slot_start: datetime,
    *,
    slot_minutes: int = 60,
    p90_minutes: float = 120.0,
) -> HourlyPaidState:
    """Infer a bounded paid-occupancy probability for one future interval.

    Each overlapping transaction contributes an event probability. Concurrent
    events combine with a noisy-OR, so multi-space meters never create an
    impossible target below zero or above one.
    """
    slot_end = slot_start + timedelta(minutes=slot_minutes)
    not_occupied = 1.0
    total_overlap = 0.0
    count = 0
    for event in events:
        if event.start >= slot_end or event.end <= slot_start:
            continue
        start = max(event.start, slot_start)
        end = min(event.end, slot_end)
        overlap = max(0.0, (end - start).total_seconds() / 60.0)
        if overlap <= 0:
            continue
        count += 1
        total_overlap += overlap
        p = event_probability(overlap, p90_minutes=p90_minutes)
        not_occupied *= 1.0 - p

    probability = min(1.0, max(0.0, 1.0 - not_occupied))
    return HourlyPaidState(
        slot_start=slot_start,
        slot_minutes=float(slot_minutes),
        transaction_count=count,
        paid_overlap_minutes=round(total_overlap, 3),
        paid_occupancy_probability=round(probability, 6),
        paid_availability_probability=round(1.0 - probability, 6),
    )
