"""Normalized parking-domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass(frozen=True, slots=True)
class ParkingMeter:
    """A physical parking meter/location from the SFMTA inventory."""

    parking_space_id: int | None
    post_id: str | None
    latitude: float
    longitude: float
    active: bool
    street_name: str | None = None
    street_number: str | None = None
    blockface_id: str | None = None
    meter_type: str | None = None


@dataclass(frozen=True, slots=True)
class MeterPolicy:
    """A time-bounded operating/rate policy for a parking location."""

    parking_space_id: int | None
    post_id: str | None
    day_of_week: str
    start_time: time
    end_time: time
    hourly_rate: float
    time_limit_minutes: int | None
    start_date: date | None = None
    end_date: date | None = None
    schedule_type: str | None = None

    def applies_on(self, day: str) -> bool:
        return self.day_of_week.strip().lower() in {
            day.strip().lower(),
            day.strip()[:2].lower(),
        }
