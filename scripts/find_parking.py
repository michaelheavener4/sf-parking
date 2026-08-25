#!/usr/bin/env python3
"""Spot-level parking search: find predicted-available meters near a location.

Usage:
    python scripts/find_parking.py \\
        --lat 37.7985 --lon -122.4368 \\
        --date 2026-08-24 --hour 18 \\
        --radius 1000 --top 20
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

from sf_parking.database import connect

TZ_LA = ZoneInfo("America/Los_Angeles")
SEARCH_RADIUS_DEFAULT = 1000
TOP_N_DEFAULT = 20


def _parse_local_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find predicted-available parking spots near a location."
    )
    p.add_argument("--lat", required=True, type=float,
                   help="Search center latitude")
    p.add_argument("--lon", required=True, type=float,
                   help="Search center longitude")
    p.add_argument("--date", required=True, type=str,
                   help="Desired arrival date (local LA date, YYYY-MM-DD)")
    p.add_argument("--hour", required=True, type=int,
                   help="Desired arrival hour (local, 0-23)")
    p.add_argument("--radius", type=int, default=SEARCH_RADIUS_DEFAULT,
                   help=f"Search radius in meters (default: {SEARCH_RADIUS_DEFAULT})")
    p.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                   help=f"Number of results to return (default: {TOP_N_DEFAULT})")
    return p.parse_args(args)


def _validate_args(ns: argparse.Namespace) -> list[str]:
    """Return a list of error messages (empty = valid)."""
    errors: list[str] = []
    if not (-90 <= ns.lat <= 90):
        errors.append(f"Invalid latitude: {ns.lat} (must be -90..90)")
    if not (-180 <= ns.lon <= 180):
        errors.append(f"Invalid longitude: {ns.lon} (must be -180..180)")
    if not (0 <= ns.hour <= 23):
        errors.append(f"Invalid hour: {ns.hour} (must be 0..23)")
    if ns.radius <= 0:
        errors.append(f"Invalid radius: {ns.radius} (must be > 0)")
    if ns.top <= 0:
        errors.append(f"Invalid top: {ns.top} (must be > 0)")
    try:
        datetime.strptime(ns.date, "%Y-%m-%d")
    except ValueError:
        errors.append(f"Invalid date: {ns.date} (must be YYYY-MM-DD)")
    return errors


def _local_to_utc(date_str: str, hour: int) -> tuple[datetime, str]:
    """Convert local date+hour to a UTC target_slot.

    Returns (target_utc, local_label) where local_label is a human-readable
    string like ``2026-08-24 18:00 PDT``.
    """
    naive = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour)
    aware = naive.replace(tzinfo=TZ_LA)
    local_label = aware.strftime("%Y-%m-%d %H:%M %Z")
    return aware.astimezone(timezone.utc), local_label


def _resolve_target_slot(conn, target_utc: datetime) -> datetime | None:
    """Return the target_slot if an exact forecast exists, else None.

    The finder must never silently jump to a different hour.  If the
    user asks for 18:00 PDT, we require a forecast at exactly 01:00 UTC.
    """
    result = conn.run(
        "SELECT target_slot FROM parking_state_forecasts "
        "WHERE target_slot = :t LIMIT 1",
        t=target_utc,
    )
    if result:
        return result[0][0]
    return None


def _query_forecasts(
    conn,
    lat: float,
    lon: float,
    radius_m: int,
    target_slot: datetime,
    top_n: int,
) -> list[dict]:
    """Run a single PostGIS query joining forecasts with meters.

    Uses ST_DWithin for the radius filter (uses the GiST index) and
    ST_Distance for exact geographic distance in metres.
    """
    sql = """
        SELECT
            f.post_id,
            m.parking_space_id,
            m.meter_type,
            m.street_number,
            m.street_name,
            ST_Y(m.location::geometry)          AS latitude,
            ST_X(m.location::geometry)          AS longitude,
            ST_Distance(
                m.location,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            )                                    AS distance_m,
            f.predicted_availability,
            f.target_slot,
            f.hours_ahead,
            f.forecast_generated_at,
            f.model_version
        FROM parking_state_forecasts f
        INNER JOIN parking_meters m
            ON m.post_id = f.post_id
        WHERE f.target_slot = :target_slot
          AND m.location IS NOT NULL
          AND ST_DWithin(
              m.location,
              ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
              :radius
          )
        ORDER BY f.predicted_availability DESC, distance_m ASC
        LIMIT :top_n
    """
    rows = conn.run(sql, lon=lon, lat=lat, radius=radius_m,
                    target_slot=target_slot, top_n=top_n)
    results = []
    for r in rows:
        results.append({
            "post_id":                    r[0],
            "parking_space_id":           r[1],
            "meter_type":                 r[2],
            "street_number":              r[3],
            "street_name":                r[4],
            "latitude":                   r[5],
            "longitude":                  r[6],
            "distance_meters":            round(r[7], 1),
            "predicted_availability":     r[8],
            "target_slot":                r[9],
            "hours_ahead":                r[10],
            "forecast_generated_at":      r[11],
            "model_version":              r[12],
        })
    return results


def _combined_score(availability: float, distance_m: float) -> float:
    """Rank by availability, breaking ties by distance.

    Score = availability * 100 - distance_m * 0.01
    Availability dominates; distance is a minor tiebreaker.
    """
    return availability * 100.0 - distance_m * 0.01


def _format_location(row: dict) -> str:
    """Build a human-readable location string."""
    parts: list[str] = []
    if row["street_number"]:
        parts.append(str(row["street_number"]))
    if row["street_name"]:
        parts.append(row["street_name"])
    return " ".join(parts) if parts else row["post_id"]


def _print_results(
    results: list[dict],
    *,
    local_label: str,
    target_utc: datetime,
    radius_m: int,
    total_in_radius: int,
    lat: float,
    lon: float,
    hours_ahead: int,
) -> None:
    """Pretty-print the results to stdout."""
    W = 70
    print()
    print("SF PARKING — PREDICTED PARKING")
    print("═" * W)
    print()
    print(f"Arrival:     {local_label}")
    print(f"Target UTC:  {target_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Search:      {radius_m:,} m")
    print(f"Candidates:  {total_in_radius:,}")
    print(f"Returning:   {len(results)}")
    print(f"Hours ahead: {hours_ahead}")
    print()

    if not results:
        print("No results.")
        return

    # Column headers
    hdr = f" {'#':>3}  {'Probability':>11}  {'Distance':>9}  {'Score':>7}  Location"
    print(hdr)
    print(" " + "─" * (W - 2))

    for i, row in enumerate(results, 1):
        pct = row["predicted_availability"] * 100.0
        dist = row["distance_meters"]
        score = _combined_score(row["predicted_availability"], dist)
        loc = _format_location(row)
        print(f" {i:>3}  {pct:>10.1f}%  {dist:>7.1f} m  {score:>7.2f}  {loc}")

    print()


def find_parking(argv: list[str] | None = None) -> int:
    """Main entry point.  Returns exit code."""
    ns = _parse_local_args(argv)
    errors = _validate_args(ns)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # 1. Convert local time → UTC
    target_utc, local_label = _local_to_utc(ns.date, ns.hour)

    # 2. Connect and validate
    conn = connect()
    try:
        # 3. Check forecast table has data
        r = conn.run("SELECT count(*) FROM parking_state_forecasts")
        if r[0][0] == 0:
            print("ERROR: parking_state_forecasts is empty. "
                  "Run forecast_paid_state.py first.", file=sys.stderr)
            return 1

        # 4. Check forecast range
        r = conn.run(
            "SELECT min(target_slot), max(target_slot) "
            "FROM parking_state_forecasts"
        )
        min_slot, max_slot = r[0][0], r[0][1]
        if target_utc > max_slot:
            print(
                f"ERROR: Requested target {target_utc.isoformat()} is beyond "
                f"the latest forecast ({max_slot.isoformat()}).\n"
                f"Forecast range: {min_slot.isoformat()} .. "
                f"{max_slot.isoformat()}",
                file=sys.stderr,
            )
            return 1

        # 5. Check for exact forecast at requested slot
        resolved = _resolve_target_slot(conn, target_utc)
        if resolved is None:
            # Show what forecasts ARE available.
            r = conn.run(
                "SELECT target_slot, count(*) AS n "
                "FROM parking_state_forecasts "
                "GROUP BY target_slot "
                "ORDER BY target_slot"
            )
            if not r:
                print(
                    f"ERROR: No forecast exists for {local_label} "
                    f"({target_utc.isoformat()}).\n"
                    f"The parking_state_forecasts table is empty.",
                    file=sys.stderr,
                )
            else:
                slots_str = "\n".join(
                    f"    {row[0].astimezone(TZ_LA).strftime('%Y-%m-%d %H:%M %Z'):>24}  "
                    f"({row[1]:,} meters)"
                    for row in r
                )
                print(
                    f"ERROR: No forecast exists for {local_label} "
                    f"({target_utc.isoformat()}).\n"
                    f"Available forecast slots:\n{slots_str}",
                    file=sys.stderr,
                )
            return 1

        # 6. Count total meters with forecasts for this target
        r = conn.run(
            "SELECT count(DISTINCT f.post_id) "
            "FROM parking_state_forecasts f "
            "INNER JOIN parking_meters m ON m.post_id = f.post_id "
            "WHERE f.target_slot = :t AND m.location IS NOT NULL",
            t=resolved,
        )
        total_forecast_meters = r[0][0]

        # 7. Count meters within radius (without top_n limit)
        r = conn.run(
            "SELECT count(*) "
            "FROM parking_state_forecasts f "
            "INNER JOIN parking_meters m ON m.post_id = f.post_id "
            "WHERE f.target_slot = :t "
              "AND m.location IS NOT NULL "
              "AND ST_DWithin("
                  "m.location, "
                  "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, "
                  ":radius"
              ")",
            lon=ns.lon, lat=ns.lat, radius=ns.radius, t=resolved,
        )
        total_in_radius = r[0][0]

        if total_in_radius == 0:
            print(
                f"NOTE: {total_forecast_meters:,} meters have forecasts for "
                f"{local_label}, but none are within {ns.radius:,} m of "
                f"({ns.lat}, {ns.lon}).",
                file=sys.stderr,
            )

        # 8. Query ranked results
        results = _query_forecasts(conn, ns.lat, ns.lon, ns.radius,
                                   resolved, ns.top)

        # 9. Compute hours_ahead from the forecast metadata
        r = conn.run(
            "SELECT hours_ahead FROM parking_state_forecasts "
            "WHERE target_slot = :t LIMIT 1",
            t=resolved,
        )
        h_ahead = r[0][0] if r else 0

        # 10. Print
        _print_results(
            results,
            local_label=local_label,
            target_utc=resolved,
            radius_m=ns.radius,
            total_in_radius=total_in_radius,
            lat=ns.lat,
            lon=ns.lon,
            hours_ahead=h_ahead,
        )

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(find_parking())
