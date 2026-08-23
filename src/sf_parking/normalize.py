"""Convert raw DataSF rows into stable domain objects."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from .models import MeterPolicy, ParkingMeter


def _float(row: dict[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"missing numeric field; tried {names}")


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _date(value: Any) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _time(value: Any) -> time:
    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"unsupported time value: {value!r}")


def normalize_meter(row: dict[str, Any]) -> ParkingMeter:
    space_id = row.get("parking_space_id", row.get("parkingspaceid"))
    if space_id in (None, ""):
        raise ValueError("meter row has no ParkingSpaceID")
    active_flag = str(row.get("active_meter_flag", row.get("active_met", ""))).upper()
    return ParkingMeter(
        parking_space_id=int(float(space_id)),
        post_id=row.get("post_id", row.get("postid")),
        latitude=_float(row, "latitude"),
        longitude=_float(row, "longitude"),
        active=active_flag in {"M", "T", "Y", "TRUE", "1"},
        street_name=row.get("street_name"),
        street_number=row.get("street_num"),
        blockface_id=row.get("blockface_id"),
        meter_type=row.get("meter_type"),
    )


def normalize_policy(row: dict[str, Any]) -> MeterPolicy:
    space_id = row.get("parkingspaceid", row.get("parking_space_id"))
    if space_id in (None, ""):
        raise ValueError("policy row has no ParkingSpaceID")
    return MeterPolicy(
        parking_space_id=int(float(space_id)),
        day_of_week=str(row["dayofweek"]),
        start_time=_time(row["starttime"]),
        end_time=_time(row["endtime"]),
        hourly_rate=float(row.get("hourlyrate", 0)),
        time_limit_minutes=_int(row.get("timelimitminutes")),
        start_date=_date(row.get("startdate")),
        end_date=_date(row.get("enddate")),
        schedule_type=row.get("scheduletype"),
    )
