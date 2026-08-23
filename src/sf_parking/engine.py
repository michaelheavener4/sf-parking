"""Core V0.1 parking eligibility and ranking logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

from .models import MeterPolicy, ParkingMeter

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class ParkingCandidate:
    meter: ParkingMeter
    distance_m: float
    policy: MeterPolicy | None

    @property
    def hourly_rate(self) -> float | None:
        return self.policy.hourly_rate if self.policy else None

    @property
    def time_limit_minutes(self) -> int | None:
        return self.policy.time_limit_minutes if self.policy else None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(a))


def _same_location(policy: MeterPolicy, meter: ParkingMeter) -> bool:
    """Match policy to inventory using the identifiers SFMTA actually exposes."""
    if meter.parking_space_id is not None and policy.parking_space_id is not None:
        if meter.parking_space_id == policy.parking_space_id:
            return True
    return bool(meter.post_id and policy.post_id and meter.post_id == policy.post_id)


def active_policy(
    policies: list[MeterPolicy],
    *,
    meter: ParkingMeter,
    when: datetime,
) -> MeterPolicy | None:
    matches = [
        p
        for p in policies
        if _same_location(p, meter)
        and p.applies_on(when.strftime("%A"))
        and (p.start_date is None or p.start_date <= when.date())
        and (p.end_date is None or when.date() <= p.end_date)
        and p.start_time <= when.time() < p.end_time
    ]
    if not matches:
        return None

    # Prefer explicit FREE/OP schedules over PRE when multiple records overlap;
    # then prefer the lowest effective hourly rate and longest usable duration.
    schedule_priority = {"FREE": 0, "OP": 1, "PRE": 2}
    return min(
        matches,
        key=lambda p: (
            schedule_priority.get((p.schedule_type or "").upper(), 9),
            p.hourly_rate,
            -(p.time_limit_minutes or 0),
        ),
    )


def nearby(
    meters: list[ParkingMeter],
    policies: list[MeterPolicy],
    *,
    latitude: float,
    longitude: float,
    when: datetime,
    radius_m: float = 400,
    limit: int = 20,
) -> list[ParkingCandidate]:
    candidates: list[ParkingCandidate] = []
    for meter in meters:
        if not meter.active:
            continue
        distance = haversine_m(latitude, longitude, meter.latitude, meter.longitude)
        if distance > radius_m:
            continue
        candidates.append(
            ParkingCandidate(
                meter=meter,
                distance_m=distance,
                policy=active_policy(policies, meter=meter, when=when),
            )
        )

    # V0.1: closest first. The ranking layer will later incorporate price,
    # availability probability, walking distance, and restriction confidence.
    candidates.sort(key=lambda candidate: candidate.distance_m)
    return candidates[:limit]
