import math

from sf_parking.dynamics import availability_forecast, two_state_forecast


def test_zero_arrivals_drift_toward_occupied():
    p = two_state_forecast(0.8, 0.0, 2.0, 1.0)
    assert 0.0 < p < 0.8


def test_high_arrivals_drift_toward_occupied_equilibrium():
    p = two_state_forecast(0.0, 2.0, 1.0, 10.0)
    # Stationary occupancy is lambda / (lambda + mu) = 2 / 3 here.
    assert 0.65 < p < 0.68


def test_availability_is_complement_of_occupancy():
    occ = two_state_forecast(0.4, 0.5, 2.0, 1.0)
    avail = availability_forecast(0.6, 0.5, 2.0, 1.0)
    assert math.isclose(avail, 1.0 - occ, rel_tol=1e-12, abs_tol=1e-12)
