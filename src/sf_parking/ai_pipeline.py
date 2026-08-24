"""End-to-end orchestration primitives for the SF parking intelligence stack.

The pipeline deliberately separates observation, inference, feature building,
and forecasting so every stage can be evaluated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .ai_features import Snapshot, build_forecast_features, transaction_snapshots
from .forecast import ForecastRow
from .occupancy import PaidOccupancyEstimator, PaidTransaction
from .spatial import SpatialGraph
from .targets import make_next_slot_targets


@dataclass(frozen=True, slots=True)
class PipelineArtifacts:
    snapshots: tuple[Snapshot, ...]
    feature_rows: tuple[ForecastRow, ...]


def build_pipeline_artifacts(
    transactions: Iterable[PaidTransaction],
    *,
    slots: Iterable[datetime],
    graph: SpatialGraph | None = None,
    estimator: PaidOccupancyEstimator | None = None,
    horizon_minutes: int = 60,
) -> PipelineArtifacts:
    """Build training rows with an explicit future target and prior-only features.

    A ForecastRow timestamp is the forecast time. Its target is the estimated
    paid-occupancy state in the *next* horizon. Feature construction sees only
    snapshots at or before that forecast time; target construction uses the
    future interval strictly after it.
    """
    txs = tuple(transactions)
    slots_tuple = tuple(sorted(slots))
    estimator = estimator or PaidOccupancyEstimator()

    snapshots = transaction_snapshots(txs, slots=slots_tuple, estimator=estimator)
    feature_map = build_forecast_features(snapshots, graph=graph)
    targets = make_next_slot_targets(
        txs,
        forecast_times=slots_tuple,
        horizon_minutes=horizon_minutes,
        estimator=estimator,
    )
    target_by_key = {(t.post_id, t.forecast_time): t for t in targets}

    rows: list[ForecastRow] = []
    for snapshot in snapshots:
        target = target_by_key.get((snapshot.post_id, snapshot.timestamp))
        if target is None:
            continue
        rows.append(
            ForecastRow(
                timestamp=snapshot.timestamp,
                post_id=snapshot.post_id,
                target=target.probability_paid_occupied,
                features=feature_map[(snapshot.post_id, snapshot.timestamp)],
            )
        )
    return PipelineArtifacts(tuple(snapshots), tuple(rows))
