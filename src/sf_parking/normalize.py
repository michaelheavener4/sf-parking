"""Convert raw DataSF rows into stable domain objects."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from .models import MeterPolicy, ParkingMeter

#: SFMTA floating timestamps (e.g. the inventory's ``data_as_of``) are local
#: wall-clock times in the agency's operating zone. See docs/ROADMAP.md.
SFMTA_TZ = ZoneInfo("America/Los_Angeles")


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


def _sfmta_timestamp(value: Any) -> datetime | None:
    """Parse an SFMTA floating timestamp into an aware instant.

    Floating timestamps carry no offset and represent America/Los_Angeles
    wall-clock time (see docs/ROADMAP.md); the source zone is attached so
    ``timestamptz`` storage preserves the true absolute instant.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=SFMTA_TZ)
    return parsed.astimezone(SFMTA_TZ)


def _time(value: Any) -> time:
    """Parse SFMTA schedule times, including the valid sentinel ``24:00``.

    Python's ``datetime`` rejects hour 24, while SFMTA uses ``24:00`` to mean
    the end of the service day (midnight). Represent it as the latest possible
    time on that day so normal ``start <= current < end`` comparisons continue
    to work without inventing a 25th hour.
    """
    text = str(value).strip()
    if text in {"24:00", "24:00:00"}:
        return time.max

    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"unsupported time value: {value!r}")


def normalize_meter(row: dict[str, Any]) -> ParkingMeter:
    """Normalize an inventory row.

    The current SFMTA inventory does not expose ParkingSpaceID in every row;
    the stable cross-dataset identifier available here is PostID. Policy rows
    carry both PostID and ParkingSpaceID, so PostID is retained for the join.
    """
    space_id = row.get("parking_space_id", row.get("parkingspaceid"))
    active_flag = str(row.get("active_meter_flag", row.get("active_met", ""))).upper()
    return ParkingMeter(
        parking_space_id=_int(space_id),
        post_id=row.get("post_id", row.get("postid")),
        latitude=_float(row, "latitude"),
        longitude=_float(row, "longitude"),
        active=active_flag in {"M", "T", "Y", "TRUE", "1"},
        street_name=row.get("street_name"),
        street_number=row.get("street_num"),
        blockface_id=row.get("blockface_id"),
        meter_type=row.get("meter_type"),
        street_id=row.get("street_id"),
        street_centerline_id=row.get("street_seg_ctrln_id"),
        data_as_of=_sfmta_timestamp(row.get("data_as_of")),
    )


def normalize_policy(row: dict[str, Any]) -> MeterPolicy:
    space_id = row.get("parkingspaceid", row.get("parking_space_id"))
    return MeterPolicy(
        parking_space_id=_int(space_id),
        post_id=row.get("postid", row.get("post_id")),
        day_of_week=str(row["dayofweek"]),
        start_time=_time(row["starttime"]),
        end_time=_time(row["endtime"]),
        hourly_rate=float(row.get("hourlyrate") or 0),
        time_limit_minutes=_int(row.get("timelimitminutes")),
        start_date=_date(row.get("startdate")),
        end_date=_date(row.get("enddate")),
        schedule_type=row.get("scheduletype"),
    )
