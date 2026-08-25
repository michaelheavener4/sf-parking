"""First-principles parking occupancy dynamics.

The model treats each post as a two-state system:
    EMPTY <-> OCCUPIED
with an arrival hazard lambda and departure hazard mu.
"""
from __future__ import annotations

import math


def two_state_forecast(
    occupancy_probability: float,
    arrival_rate_per_hour: float,
    mean_duration_hours: float,
    horizon_hours: float = 1.0,
) -> float:
    """Forecast occupancy under a constant-hazard two-state process.

    lambda is the expected number of arrivals per hour for the post.
    mu is approximated as 1 / mean session duration.
    The stationary occupancy is lambda / (lambda + mu).
    """
    p = min(1.0, max(0.0, float(occupancy_probability)))
    lam = max(0.0, float(arrival_rate_per_hour))
    dur = max(1e-6, float(mean_duration_hours))
    mu = 1.0 / dur
    total = lam + mu
    if total <= 0.0:
        return p
    equilibrium = lam / total
    return min(1.0, max(0.0, equilibrium + (p - equilibrium) * math.exp(-total * max(0.0, horizon_hours))))


def availability_forecast(
    availability_probability: float,
    arrival_rate_per_hour: float,
    mean_duration_hours: float,
    horizon_hours: float = 1.0,
) -> float:
    """Forecast availability by evolving occupancy and taking its complement."""
    p_occ = 1.0 - min(1.0, max(0.0, float(availability_probability)))
    return 1.0 - two_state_forecast(p_occ, arrival_rate_per_hour, mean_duration_hours, horizon_hours)
