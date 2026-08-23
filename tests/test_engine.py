from datetime import datetime, time

from sf_parking.engine import active_policy, haversine_m, nearby
from sf_parking.models import MeterPolicy, ParkingMeter


def test_haversine_zero_distance() -> None:
    assert haversine_m(37.0, -122.0, 37.0, -122.0) == 0


def test_active_policy_matches_post_id_and_two_letter_day() -> None:
    meter = ParkingMeter(123238, "102-02990", 37.0, -122.0, True)
    policy = MeterPolicy(
        parking_space_id=123238,
        post_id="102-02990",
        day_of_week="Su",
        start_time=time(9),
        end_time=time(18),
        hourly_rate=3.0,
        time_limit_minutes=120,
    )
    when = datetime(2026, 8, 23, 12, 0)
    assert active_policy([policy], meter=meter, when=when) == policy


def test_active_policy_returns_none_outside_window() -> None:
    meter = ParkingMeter(None, "102-02990", 37.0, -122.0, True)
    policy = MeterPolicy(
        parking_space_id=1,
        post_id="102-02990",
        day_of_week="Sunday",
        start_time=time(9),
        end_time=time(18),
        hourly_rate=3.0,
        time_limit_minutes=120,
    )
    when = datetime(2026, 8, 23, 19, 0)
    assert active_policy([policy], meter=meter, when=when) is None


def test_nearby_filters_inactive_and_radius() -> None:
    meters = [
        ParkingMeter(1, "A", 37.0, -122.0, True),
        ParkingMeter(2, "B", 37.01, -122.0, True),
        ParkingMeter(3, "C", 37.0, -122.0, False),
    ]
    result = nearby(
        meters,
        [],
        latitude=37.0,
        longitude=-122.0,
        when=datetime(2026, 8, 23, 12),
        radius_m=100,
    )
    assert [candidate.meter.parking_space_id for candidate in result] == [1]
