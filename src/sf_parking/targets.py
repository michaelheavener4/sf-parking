"""Leakage-safe future-state target construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .occupancy import DurationProfile, OccupancyEstimate, PaidOccupancyEstimator, PaidTransaction


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
    profile: DurationProfile | None = None,
) -> tuple[StateTarget, ...]:
    """Create labels from a future outcome interval only.

    ``profile`` must come from training/history when supplied. If omitted, a
    fixed conservative profile is used; the label is never allowed to derive
    its duration scale from the future target rows themselves.
    """
    estimator = estimator or PaidOccupancyEstimator()
    txs = list(transactions)
    posts = sorted({t.post_id for t in txs})
    out: list[StateTarget] = []
    horizon = timedelta(minutes=horizon_minutes)
    label_profile = profile or DurationProfile(60.0, 120.0)
    for forecast_time in sorted(forecast_times):
        target_start = forecast_time + horizon
        target_end = target_start + horizon
        for post_id in posts:
            future = [
                t
                for t in txs
                if t.post_id == post_id
                and t.start < target_end
                and t.end > target_start
            ]
            est: OccupancyEstimate = estimator.estimate(
                future,
                target_start,
                slot_minutes=horizon_minutes,
                profile=label_profile,
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
