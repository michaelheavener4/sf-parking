from __future__ import annotations

import numpy as np

from sf_parking.calibration import fit_isotonic
from sf_parking.decision import ParkingCandidate, radius_probability, rank_candidates
from sf_parking.ml_features import FEATURES_SPATIAL


def test_spatial_feature_contract_contains_dynamics_and_neighbors():
    required = {
        "delta1_availability", "delta3_availability", "acceleration_availability",
        "neighbor_mean_availability", "neighbor_std_availability",
        "neighbor_occupied_fraction", "neighbor_distance_weighted_availability",
    }
    assert required <= set(FEATURES_SPATIAL)
    assert len(FEATURES_SPATIAL) >= 30


def test_isotonic_calibration_is_monotone():
    scores = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9])
    events = np.array([0,0,1,0,1,1,1,1,1])
    cal = fit_isotonic(scores, events)
    out = cal.predict(scores)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1))


def test_radius_probability_is_bounded_and_monotone():
    a = [ParkingCandidate("a", .5, 10), ParkingCandidate("b", .4, 20)]
    b = a + [ParkingCandidate("c", .3, 30)]
    pa = radius_probability(a)
    pb = radius_probability(b)
    assert 0 <= pa <= 1
    assert 0 <= pb <= 1
    assert pb >= pa


def test_rank_prefers_probability_then_distance():
    rows = [
        ParkingCandidate("far", .90, 500),
        ParkingCandidate("near", .88, 10),
        ParkingCandidate("best", .95, 100),
    ]
    out = rank_candidates(rows)
    assert out[0].candidate.post_id == "best"
    assert out[-1].candidate.post_id == "near"


def test_rank_is_deterministic_on_ties():
    rows = [ParkingCandidate("b", .8, 100), ParkingCandidate("a", .8, 100)]
    assert [r.candidate.post_id for r in rank_candidates(rows)] == ["a", "b"]
