from __future__ import annotations

from sf_parking.ml_features import FEATURES_SPATIAL


def test_spatial_features_are_explicit_and_stable():
    assert len(FEATURES_SPATIAL) == 31
    assert FEATURES_SPATIAL[:6] == [
        "lag1_availability", "lag2_availability", "lag3_availability",
        "lag6_availability", "lag24_availability", "lag168_availability",
    ]


def test_spatial_features_have_neighbor_signals():
    assert "neighbor_mean_availability" in FEATURES_SPATIAL
    assert "neighbor_distance_weighted_availability" in FEATURES_SPATIAL
    assert "neighbor_mean_delta1" in FEATURES_SPATIAL


def test_spatial_features_have_transition_signals():
    assert "delta1_availability" in FEATURES_SPATIAL
    assert "delta3_availability" in FEATURES_SPATIAL
    assert "acceleration_availability" in FEATURES_SPATIAL
    assert "tx_delta1" in FEATURES_SPATIAL


def test_no_target_derived_feature_is_named_as_observed_target():
    assert "target" not in FEATURES_SPATIAL
    assert "actual_availability" not in FEATURES_SPATIAL
