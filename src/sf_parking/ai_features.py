"""Point-in-time feature construction for the urban parking forecaster."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import cos, pi, sin
from typing import Iterable

from .occupancy import OccupancyEstimate, PaidOccupancyEstimator, PaidTransaction
from .spatial import SpatialGraph


@dataclass(frozen=True, slots=True)
class Snapshot:
    timestamp: datetime
    post_id: str
    probability_paid_occupied: float | None
    transaction_count: int
    blockface_id: str | None = None
    meter_type: str | None = None


def _hour_features(ts: datetime) -> tuple[float, float, float, float]:
    local = ts
    h = local.hour + local.minute / 60.0
    angle = 2 * pi * h / 24.0
    weekday_angle = 2 * pi * local.weekday() / 7.0
    return sin(angle), cos(angle), sin(weekday_angle), cos(weekday_angle)


def build_forecast_features(
    snapshots: Iterable[Snapshot],
    *,
    graph: SpatialGraph | None = None,
) -> dict[tuple[str, datetime], tuple[float, ...]]:
    """Build deterministic lag/rolling/spatial features without future access."""
    ordered = sorted(snapshots, key=lambda s: (s.post_id, s.timestamp))
    by_post: dict[str, list[Snapshot]] = defaultdict(list)
    for s in ordered:
        by_post[s.post_id].append(s)

    values_by_time: dict[datetime, dict[str, float]] = defaultdict(dict)
    for s in ordered:
        if s.probability_paid_occupied is not None:
            values_by_time[s.timestamp][s.post_id] = s.probability_paid_occupied

    result: dict[tuple[str, datetime], tuple[float, ...]] = {}
    for post_id, rows in by_post.items():
        history: deque[tuple[datetime, float]] = deque()
        for s in rows:
            while history and (s.timestamp - history[0][0]) > timedelta(hours=24):
                history.popleft()
            vals = [v for _, v in history]
            lag1 = vals[-1] if vals else None
            lag4 = vals[-4] if len(vals) >= 4 else None
            rolling = sum(vals) / len(vals) if vals else None
            hour_sin, hour_cos, wd_sin, wd_cos = _hour_features(s.timestamp)
            spatial = graph.neighbor_mean(values_by_time.get(s.timestamp, {}), post_id) if graph else None
            features = (
                float(lag1) if lag1 is not None else float("nan"),
                float(lag4) if lag4 is not None else float("nan"),
                float(rolling) if rolling is not None else float("nan"),
                float(s.transaction_count),
                hour_sin,
                hour_cos,
                wd_sin,
                wd_cos,
                float(spatial) if spatial is not None else float("nan"),
                1.0 if s.meter_type == "MS" else 0.0,
            )
            result[(post_id, s.timestamp)] = features
            if s.probability_paid_occupied is not None:
                history.append((s.timestamp, s.probability_paid_occupied))
    return result


def _visible_history(
    transactions: list[PaidTransaction],
    slot: datetime,
) -> list[PaidTransaction]:
    """Return only transaction state that was knowable at ``slot``.

    Sessions that started before the cutoff but ended afterward are truncated
    at the cutoff because their eventual departure time was not yet known.
    """
    visible: list[PaidTransaction] = []
    for tx in transactions:
        if tx.start >= slot:
            continue
        end = min(tx.end, slot)
        if end <= tx.start:
            continue
        visible.append(
            PaidTransaction(
                post_id=tx.post_id,
                start=tx.start,
                end=end,
                meter_type=tx.meter_type,
                payment_type=tx.payment_type,
                duration_minutes=tx.duration_minutes,
                gross_paid_amt=tx.gross_paid_amt,
            )
        )
    return visible


def transaction_snapshots(
    transactions: Iterable[PaidTransaction],
    *,
    slots: Iterable[datetime],
    estimator: PaidOccupancyEstimator | None = None,
) -> list[Snapshot]:
    """Convert transaction history into hourly state snapshots.

    Every snapshot is point-in-time safe: future transaction starts are ignored
    and future transaction ends are truncated to the snapshot time.
    """
    estimator = estimator or PaidOccupancyEstimator()
    txs = list(transactions)
    by_post: dict[str, list[PaidTransaction]] = defaultdict(list)
    for tx in txs:
        by_post[tx.post_id].append(tx)
    posts = sorted(by_post)

    output: list[Snapshot] = []
    for slot in sorted(slots):
        for post in posts:
            history = _visible_history(by_post[post], slot)
            estimate: OccupancyEstimate = estimator.estimate(history, slot)
            output.append(
                Snapshot(
                    timestamp=slot,
                    post_id=post,
                    probability_paid_occupied=estimate.probability_paid_occupied,
                    transaction_count=estimate.supporting_transactions,
                )
            )
    return output
