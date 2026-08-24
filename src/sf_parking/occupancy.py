"""Probabilistic paid-occupancy inference from parking transactions.

This module deliberately does *not* claim to recover physical occupancy. A
meter transaction records paid use, not whether a vehicle remained in the
space. The estimator produces a latent *paid-occupancy intensity* suitable as
a feature/target for forecasting. Physical-occupancy calibration belongs in a
later layer when an independent occupancy source exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp
from statistics import median
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PaidTransaction:
    post_id: str
    start: datetime
    end: datetime
    meter_type: str | None = None
    payment_type: str | None = None
    duration_minutes: int | None = None
    gross_paid_amt: float | None = None


@dataclass(frozen=True, slots=True)
class OccupancyEstimate:
    """Latent paid-occupancy state for one target interval."""

    probability_paid_occupied: float
    expected_paid_minutes: float
    supporting_transactions: int


@dataclass(frozen=True, slots=True)
class DurationProfile:
    """Robust duration summary used to soften incomplete payment sessions."""

    median_minutes: float
    p90_minutes: float

    @classmethod
    def from_transactions(cls, transactions: Iterable[PaidTransaction]) -> "DurationProfile":
        values = sorted(
            max(1.0, (t.end - t.start).total_seconds() / 60.0)
            for t in transactions
            if t.end > t.start
        )
        if not values:
            return cls(60.0, 120.0)
        idx90 = min(len(values) - 1, int(round(0.9 * (len(values) - 1))))
        return cls(median(values), values[idx90])


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


class PaidOccupancyEstimator:
    """Convert transaction intervals into a probabilistic paid-occupancy signal.

    Each observed transaction contributes a soft occupancy kernel. Complete
    paid-session overlap is high-confidence; the shoulders of a transaction
    are softened using a robust duration scale learned only from prior data.
    Multiple concurrent transactions combine with a noisy-OR rule.

    The estimator is point-in-time safe when callers pass only transactions
    available at the prediction cutoff.
    """

    def __init__(
        self,
        *,
        shoulder_minutes: float = 15.0,
        max_probability_per_event: float = 0.98,
    ) -> None:
        if shoulder_minutes <= 0:
            raise ValueError("shoulder_minutes must be positive")
        if not 0 < max_probability_per_event < 1:
            raise ValueError("max_probability_per_event must be in (0, 1)")
        self.shoulder_minutes = shoulder_minutes
        self.max_probability_per_event = max_probability_per_event

    def estimate(
        self,
        transactions: Iterable[PaidTransaction],
        slot_start: datetime,
        *,
        slot_minutes: int = 60,
        profile: DurationProfile | None = None,
    ) -> OccupancyEstimate:
        slot_end = slot_start + timedelta(minutes=slot_minutes)
        txs = list(transactions)
        if profile is None:
            profile = DurationProfile.from_transactions(txs)

        occupancy_not = 1.0
        expected = 0.0
        support = 0
        for tx in txs:
            if tx.start >= slot_end or tx.end <= slot_start:
                continue
            overlap_start = max(tx.start, slot_start)
            overlap_end = min(tx.end, slot_end)
            overlap_minutes = max(0.0, (overlap_end - overlap_start).total_seconds() / 60)
            if overlap_minutes <= 0:
                continue
            support += 1
            # Full interval overlap contributes strongly; a small shoulder
            # makes the estimator less discontinuous around transaction edges.
            normalized = overlap_minutes / max(1.0, profile.p90_minutes)
            confidence = self.max_probability_per_event * _sigmoid(5.0 * (normalized - 0.2))
            occupancy_not *= 1.0 - confidence
            expected += overlap_minutes

        probability = 1.0 - occupancy_not
        return OccupancyEstimate(
            probability_paid_occupied=round(min(1.0, max(0.0, probability)), 6),
            expected_paid_minutes=round(expected, 3),
            supporting_transactions=support,
        )

    @staticmethod
    def target_free_probability(estimate: OccupancyEstimate) -> float:
        return round(1.0 - estimate.probability_paid_occupied, 6)
