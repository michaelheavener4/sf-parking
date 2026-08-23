"""Normalized parking-domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass(frozen=True, slots=True)
class ParkingMeter:
    parking_space_id: int
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
    parking_space_id: int
    day_of_week: str
    start_time: time
    end_time: time
    hourly_rate: float
    time_limit_minutes: int | None
    start_date: date | None = None
    end_date: date | None = None
    schedule_type: str | None = None

    def applies_on(self, day: str) -> bool:
        return self.day_of_week.strip().lower() == day.strip().lower()
