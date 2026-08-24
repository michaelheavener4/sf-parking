"""End-to-end orchestration primitives for the SF parking intelligence stack.

The pipeline deliberately separates observation, inference, feature building,
and forecasting so every stage can be evaluated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .ai_features import Snapshot, build_forecast_features
from .forecast import ForecastRow
from .occupancy import PaidOccupancyEstimator, PaidTransaction
from .spatial import SpatialGraph


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
) -> PipelineArtifacts:
    """Build leak-safe state/features from caller-supplied historical data.

    No component fetches future data. The caller controls the transaction set
    and slot cutoffs; this is essential for reproducible research.
    """
    from .ai_features import transaction_snapshots

    snapshots = transaction_snapshots(
        transactions,
        slots=slots,
        estimator=estimator,
    )
    feature_map = build_forecast_features(snapshots, graph=graph)
    rows: list[ForecastRow] = []
    for snapshot in snapshots:
        if snapshot.probability_paid_occupied is None:
            continue
        # Forecast the next snapshot's paid occupancy state. Rows whose target
        # is not yet known are created by the caller from future snapshots; the
        # builder itself never peeks beyond the snapshot timestamp.
        features = feature_map[(snapshot.post_id, snapshot.timestamp)]
        rows.append(
            ForecastRow(
                timestamp=snapshot.timestamp,
                post_id=snapshot.post_id,
                target=snapshot.probability_paid_occupied,
                features=features,
            )
        )
    return PipelineArtifacts(tuple(snapshots), tuple(rows))
