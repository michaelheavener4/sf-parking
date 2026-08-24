from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sf_parking.ai_features import Snapshot, build_forecast_features
from sf_parking.forecast import ForecastRow, LogisticFallback, brier_score, temporal_split
from sf_parking.occupancy import PaidOccupancyEstimator, PaidTransaction
from sf_parking.research_frontier import ObservationFrontier
from sf_parking.spatial import Point, build_knn_graph


def dt(hour: int) -> datetime:
    return datetime(2026, 8, 20, hour, tzinfo=UTC)


def test_frontier_clamps_to_complete_outcome_window():
    frontier = ObservationFrontier(
        max_session_end=dt(20), horizon=timedelta(hours=1)
    )
    assert frontier.safe_until == dt(19)
    assert frontier.clamp(dt(22)) == dt(19)
    assert frontier.clamp(dt(18)) == dt(18)


def test_paid_occupancy_is_bounded_and_point_in_time():
    estimator = PaidOccupancyEstimator()
    tx = PaidTransaction("m", dt(9), dt(10))
    estimate = estimator.estimate([tx], dt(9, ))
    assert 0.0 <= estimate.probability_paid_occupied <= 1.0
    assert estimate.supporting_transactions == 1


def test_knn_graph_is_deterministic_and_respects_radius():
    points = [
        Point("a", 37.0, -122.0),
        Point("b", 37.0001, -122.0),
        Point("c", 38.0, -122.0),
    ]
    graph = build_knn_graph(points, k=2, radius_meters=30)
    assert graph.nodes == ("a", "b", "c")
    assert {(e.source, e.target) for e in graph.edges} == {("a", "b"), ("b", "a")}


def test_features_use_only_prior_snapshots():
    snaps = [
        Snapshot(dt(9), "m", 0.2, 1),
        Snapshot(dt(10), "m", 0.8, 1),
        Snapshot(dt(11), "m", 0.4, 1),
    ]
    features = build_forecast_features(snaps)
    first = features[("m", dt(9))]
    second = features[("m", dt(10))]
    assert first[0] != first[0]  # NaN: no prior lag
    assert second[0] == pytest.approx(0.2)


def test_temporal_split_has_strict_time_order():
    rows = tuple(
        ForecastRow(dt(h), "m", 0.5, (float(h),)) for h in range(9, 12)
    )
    split = temporal_split(rows, train_until=dt(10), validation_until=dt(11))
    assert [r.timestamp for r in split.train] == [dt(9)]
    assert [r.timestamp for r in split.validation] == [dt(10)]
    assert [r.timestamp for r in split.test] == [dt(11)]


def test_logistic_fallback_learns_simple_relation():
    rows = tuple(
        ForecastRow(dt(h), "m", float(h >= 11), (float(h),))
        for h in range(9, 15)
    )
    model = LogisticFallback(learning_rate=0.08, epochs=600).fit(rows)
    probs = model.predict_proba(rows)
    assert probs[-1] > probs[0]
    assert 0.0 <= brier_score([r.target for r in rows], probs) <= 1.0
