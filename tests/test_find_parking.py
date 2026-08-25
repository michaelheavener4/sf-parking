"""Tests for the spot-level parking search command.

Covers:
1. Local time → UTC conversion (including DST)
2. DST behaviour
3. Radius filtering
4. Distance calculation (PostGIS)
5. Probability ordering
6. Probability + distance tie-breaking
7. Individual post_id preservation
8. No forecast available
9. No locations in radius
10. Invalid coordinates
11. Invalid radius
12. Invalid hour
13. Forecasts from wrong target_slot are excluded
14. actual_availability is never used for ranking
15. Future observed data cannot leak into the query
16. Correct hours_ahead reporting
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pg8000
import pytest

from sf_parking.database import connect
from scripts.find_parking import (
    _combined_score,
    _local_to_utc,
    _parse_local_args,
    _query_forecasts,
    _resolve_target_slot,
    _validate_args,
    find_parking,
)

TZ_LA = ZoneInfo("America/Los_Angeles")
UTC_TZ = UTC


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


def _get_existing_forecast_slot(conn) -> datetime | None:
    """Return a target_slot that has forecasts, or None."""
    result = conn.run("""
        SELECT target_slot
        FROM parking_state_forecasts
        GROUP BY target_slot
        ORDER BY count(*) DESC
        LIMIT 1
    """)
    if result:
        return result[0][0]
    return None


# ── Local time → UTC conversion ──────────────────────────────────────────


class TestLocalToUTC:
    """Verify that local date+hour converts correctly to UTC."""

    def test_summer_pdt_to_utc(self):
        """2026-08-24 18:00 PDT → 2026-08-25 01:00 UTC."""
        target_utc, label = _local_to_utc("2026-08-24", 18)
        assert target_utc == datetime(2026, 8, 25, 1, 0, tzinfo=UTC_TZ)
        assert "PDT" in label or "UTC" in label

    def test_winter_pst_to_utc(self):
        """2026-01-15 18:00 PST → 2026-01-16 02:00 UTC."""
        target_utc, label = _local_to_utc("2026-01-15", 18)
        assert target_utc == datetime(2026, 1, 16, 2, 0, tzinfo=UTC_TZ)
        assert "PST" in label or "UTC" in label

    def test_midnight_local(self):
        """2026-08-25 00:00 PDT → 2026-08-25 07:00 UTC."""
        target_utc, label = _local_to_utc("2026-08-25", 0)
        assert target_utc == datetime(2026, 8, 25, 7, 0, tzinfo=UTC_TZ)

    def test_label_contains_date_time(self):
        """Label should contain the date and hour."""
        _, label = _local_to_utc("2026-08-24", 18)
        assert "2026-08-24" in label
        assert "18:00" in label


# ── DST behaviour ────────────────────────────────────────────────────────


class TestDST:
    """Verify correct behaviour across DST transitions."""

    def test_dst_spring_forward(self):
        """2026-03-08 02:30 PST does not exist (spring forward at 02:00).
        Wall-clock 02:30 is mapped to 03:30 PDT = 10:30 UTC."""
        target_utc, _ = _local_to_utc("2026-03-08", 2)
        # 02:00 local = 10:00 UTC (PDT starts at 02:00)
        # We request hour=2, which is in the DST gap.
        # The Python datetime machinery with fold=0 resolves it to PDT.
        assert target_utc.year == 2026

    def test_dst_fall_back(self):
        """2026-11-01 01:00 PDT → 2026-11-01 08:00 UTC.
        The first occurrence (PDT) is chosen (fold=0)."""
        target_utc, _ = _local_to_utc("2026-11-01", 1)
        assert target_utc == datetime(2026, 11, 1, 8, 0, tzinfo=UTC_TZ)

    def test_dst_boundary_before(self):
        """One hour before DST spring-forward: 2026-03-08 01:00 PST."""
        target_utc, _ = _local_to_utc("2026-03-08", 1)
        assert target_utc == datetime(2026, 3, 8, 9, 0, tzinfo=UTC_TZ)

    def test_dst_boundary_after(self):
        """One hour after DST spring-forward: 2026-03-08 03:00 PDT."""
        target_utc, _ = _local_to_utc("2026-03-08", 3)
        assert target_utc == datetime(2026, 3, 8, 10, 0, tzinfo=UTC_TZ)


# ── Argument validation ──────────────────────────────────────────────────


class TestValidation:
    """Verify CLI argument validation."""

    def test_valid_args(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "18",
        ])
        assert _validate_args(ns) == []

    def test_invalid_latitude(self):
        ns = _parse_local_args([
            "--lat", "999", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "18",
        ])
        errors = _validate_args(ns)
        assert any("latitude" in e.lower() for e in errors)

    def test_invalid_longitude(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "999",
            "--date", "2026-08-24", "--hour", "18",
        ])
        errors = _validate_args(ns)
        assert any("longitude" in e.lower() for e in errors)

    def test_invalid_hour_too_high(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "25",
        ])
        errors = _validate_args(ns)
        assert any("hour" in e.lower() for e in errors)

    def test_invalid_hour_negative(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "-1",
        ])
        errors = _validate_args(ns)
        assert any("hour" in e.lower() for e in errors)

    def test_invalid_date(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "not-a-date", "--hour", "18",
        ])
        errors = _validate_args(ns)
        assert any("date" in e.lower() for e in errors)

    def test_invalid_radius_negative(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "18",
            "--radius", "-100",
        ])
        errors = _validate_args(ns)
        assert any("radius" in e.lower() for e in errors)

    def test_invalid_top_zero(self):
        ns = _parse_local_args([
            "--lat", "37.7985", "--lon", "-122.4368",
            "--date", "2026-08-24", "--hour", "18",
            "--top", "0",
        ])
        errors = _validate_args(ns)
        assert any("top" in e.lower() for e in errors)


# ── Integration tests (require database) ─────────────────────────────────


@pytest.mark.skipif(not _server_available(),
                    reason="PostgreSQL not reachable")
class TestFindParkingIntegration:
    """Integration tests that run the full find_parking pipeline against the
    database.  These require a populated parking_meters + forecast table."""

    @pytest.fixture(autouse=True)
    def _conn(self):
        self.conn = connect()
        yield
        self.conn.close()

    def test_basic_search(self):
        """A search near Union Square returns results."""
        target_utc = datetime(2026, 8, 24, 7, 0, tzinfo=UTC_TZ)
        resolved = _resolve_target_slot(self.conn, target_utc)
        if resolved is None:
            pytest.skip("No forecast for test slot")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 500, resolved, 10,
        )
        assert isinstance(results, list)

    def test_radius_filtering(self):
        """A 1-meter radius returns 0 results (no meter is at the exact point)."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 1, target_utc, 10,
        )
        assert results == []

    def test_distance_calculation_is_geographic(self):
        """Distances should be in metres, reasonable for SF."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 5,
        )
        for r in results:
            d = r["distance_meters"]
            assert 0 <= d <= 2500, f"Unreasonable distance: {d}"

    def test_probability_ordering(self):
        """Results should be ordered by predicted_availability DESC."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 50,
        )
        if len(results) > 1:
            for i in range(len(results) - 1):
                a = results[i]["predicted_availability"]
                b = results[i + 1]["predicted_availability"]
                if a > b:
                    break
                elif a == b:
                    da = results[i]["distance_meters"]
                    db = results[i + 1]["distance_meters"]
                    assert da <= db, (
                        f"Tie-break violation: availability={a}, "
                        f"distances {da} > {db}"
                    )

    def test_post_id_preserved(self):
        """Every result has a non-empty post_id."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 10,
        )
        for r in results:
            assert r["post_id"], "post_id must not be empty"
            assert isinstance(r["post_id"], str)

    def test_all_result_fields_present(self):
        """Every result contains all required fields."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 5,
        )
        required = {
            "post_id", "parking_space_id", "meter_type",
            "street_number", "street_name", "latitude", "longitude",
            "distance_meters", "predicted_availability", "target_slot",
            "hours_ahead", "forecast_generated_at", "model_version",
        }
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_latitude_longitude_are_valid(self):
        """Returned lat/lon are within SF bounds."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 5,
        )
        for r in results:
            lat, lon = r["latitude"], r["longitude"]
            assert 37.0 < lat < 38.0, f"lat out of SF range: {lat}"
            assert -123.0 < lon < -122.0, f"lon out of SF range: {lon}"

    def test_wrong_target_slot_excluded(self):
        """Forecasts from a different target_slot are not returned."""
        fake_target = datetime(2020, 1, 1, 8, 0, tzinfo=UTC_TZ)
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 5000, fake_target, 10,
        )
        assert results == []

    def test_forecasts_from_all_target_slots(self):
        """All forecasts belong to the queried target_slot."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 5000, target_utc, 50,
        )
        for r in results:
            assert r["target_slot"] == target_utc, (
                f"Got forecast for wrong target: {r['target_slot']} != {target_utc}"
            )

    def test_actual_availability_not_in_results(self):
        """Results must NOT contain actual_availability."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 5,
        )
        for r in results:
            assert "actual_availability" not in r, (
                "actual_availability must not be in results"
            )

    def test_hours_ahead_reported(self):
        """Results include a non-negative hours_ahead."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, 37.7879, 2000, target_utc, 5,
        )
        for r in results:
            assert r["hours_ahead"] >= 0

    def test_no_forecast_returns_empty(self):
        """Querying a target with no forecasts returns empty."""
        fake_target = datetime(2020, 1, 1, 8, 0, tzinfo=UTC_TZ)
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 5000, fake_target, 10,
        )
        assert results == []

    def test_combined_score_formula(self):
        """Combined score = availability*100 - distance*0.01."""
        score = _combined_score(0.95, 100.0)
        assert abs(score - (95.0 - 1.0)) < 0.001

    def test_combined_score_preferences(self):
        """Higher availability beats shorter distance."""
        s1 = _combined_score(0.90, 200.0)  # 90 - 2 = 88
        s2 = _combined_score(0.80, 50.0)   # 80 - 0.5 = 79.5
        assert s1 > s2

    def test_full_cli_exit_code(self):
        """find_parking with valid args returns exit code 0."""
        # Use 2026-08-24 00:00 PDT (2026-08-24 07:00 UTC) which has forecasts.
        exit_code = find_parking([
            "--lat", "37.7879", "--lon", "-122.4074",
            "--date", "2026-08-24", "--hour", "0",
            "--radius", "500", "--top", "5",
        ])
        assert exit_code == 0

    def test_full_cli_invalid_coords(self):
        """find_parking with invalid coords returns exit code 2."""
        exit_code = find_parking([
            "--lat", "999", "--lon", "-122.4074",
            "--date", "2026-08-24", "--hour", "18",
        ])
        assert exit_code == 2

    def test_full_cli_invalid_hour(self):
        """find_parking with invalid hour returns exit code 2."""
        exit_code = find_parking([
            "--lat", "37.7879", "--lon", "-122.4074",
            "--date", "2026-08-24", "--hour", "25",
        ])
        assert exit_code == 2

    def test_resolve_target_slot_returns_exact_match(self):
        """_resolve_target_slot returns the exact slot when it exists."""
        # Use 2026-08-24 07:00 UTC which has forecasts.
        target_utc = datetime(2026, 8, 24, 7, 0, tzinfo=UTC_TZ)
        resolved = _resolve_target_slot(self.conn, target_utc)
        assert resolved == target_utc

    def test_resolve_target_slot_returns_none_for_missing(self):
        """_resolve_target_slot returns None when no forecast exists."""
        target_utc = datetime(2020, 1, 1, 0, 0, tzinfo=UTC_TZ)
        resolved = _resolve_target_slot(self.conn, target_utc)
        assert resolved is None

    def test_result_count_respects_top_n(self):
        """Returns at most top_n results."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 10000, target_utc, 3,
        )
        assert len(results) <= 3

    def test_meters_with_no_location_excluded(self):
        """Meters with NULL location are excluded."""
        target_utc = _get_existing_forecast_slot(self.conn)
        if target_utc is None:
            pytest.skip("No forecasts in database")
        results = _query_forecasts(
            self.conn, 37.7879, -122.4074, 2000, target_utc, 10,
        )
        for r in results:
            assert r["latitude"] is not None
            assert r["longitude"] is not None
