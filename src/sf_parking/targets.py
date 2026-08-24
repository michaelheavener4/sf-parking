"""Leakage-safe future-state target construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .occupancy import OccupancyEstimate, PaidOccupancyEstimator, PaidTransaction


@dataclass(frozen=True, slots=True)
class StateTarget:
    post_id: str
    forecast_time: datetime
    target_time: datetime
    probability_paid_occupied: float
    target_free_probability: float
    support_transactions: int


def make_next_slot_targets(
    transactions: Iterable[PaidTransaction],
    *,
    forecast_times: Iterable[datetime],
    horizon_minutes: int = 60,
    estimator: PaidOccupancyEstimator | None = None,
) -> tuple[StateTarget, ...]:
    """Create targets using the *future* observation window only.

    The caller must build training features from transactions available at the
    corresponding forecast_time. This function derives the label from the
    complete target interval [forecast_time+horizon, +2*horizon), which makes
    it explicit that target information is never fed into features.
    """
    estimator = estimator or PaidOccupancyEstimator()
    txs = list(transactions)
    posts = sorted({t.post_id for t in txs})
    out: list[StateTarget] = []
    horizon = timedelta(minutes=horizon_minutes)
    for forecast_time in sorted(forecast_times):
        target_start = forecast_time + horizon
        target_end = target_start + horizon
        for post_id in posts:
            future = [
                t for t in txs
                if t.post_id == post_id
                and t.start < target_end
                and t.end > target_start
            ]
            est: OccupancyEstimate = estimator.estimate(
                future, target_start, slot_minutes=horizon_minutes
            )
            out.append(
                StateTarget(
                    post_id=post_id,
                    forecast_time=forecast_time,
                    target_time=target_start,
                    probability_paid_occupied=est.probability_paid_occupied,
                    target_free_probability=estimator.target_free_probability(est),
                    support_transactions=est.supporting_transactions,
                )
            )
    return tuple(out)
