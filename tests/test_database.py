"""Integration tests for the PostGIS database layer.

These tests require the database from ``docker compose up -d`` to be running
on localhost:5432.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pg8000
import pytest

from sf_parking.database import (
    apply_schema,
    connect,
    find_meters_near,
    load_meter_policies,
    load_parking_meters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"


def _server_available() -> bool:
    try:
        docker_info = subprocess.run(
            ["docker", "info", "--format", "ok"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if docker_info.returncode != 0:
            return False
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="PostgreSQL/PostGIS not reachable on localhost:5432",
)


@pytest.fixture(scope="module")
def db():
    conn = connect()
    apply_schema(conn, SCHEMA_PATH)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_tables(db):
    yield
    db.run("TRUNCATE parking_meters, meter_policies")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestSchema:
    def test_postgis_extension_installed(self, db):
        assert db.run(
            "SELECT count(*) FROM pg_extension WHERE extname = 'postgis'"
        )[0][0] == 1

    def test_required_tables_exist(self, db):
        names = {
            row[0]
            for row in db.run(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        }
        assert {"parking_meters", "meter_policies"} <= names

    def test_location_is_geography_point(self, db):
        column_type = db.run(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'parking_meters' AND column_name = 'location'"
        )[0][0]
        assert column_type == "geography"

        db.run(
            "INSERT INTO parking_meters (post_id, latitude, longitude) "
            "VALUES ('schema-probe', 37.7880, -122.4075)"
        )
        try:
            geom_type, srid = db.run(
                "SELECT GeometryType(location::geometry), ST_SRID(location::geometry) "
                "FROM parking_meters WHERE post_id = 'schema-probe'"
            )[0]
        finally:
            db.run("DELETE FROM parking_meters WHERE post_id = 'schema-probe'")
        assert (geom_type.upper(), int(srid)) == ("POINT", 4326)

    def test_spatial_and_key_indexes_exist(self, db):
        indexes = {
            row[0]: row[1]
            for row in db.run(
                "SELECT i.indexname, am.amname FROM pg_indexes i "
                "JOIN pg_class c ON c.relname = i.indexname "
                "JOIN pg_am am ON am.oid = c.relam "
                "WHERE i.tablename IN ('parking_meters', 'meter_policies')"
            )
        }
        assert indexes["idx_parking_meters_location_gist"] == "gist"
        assert indexes["uq_meter_policies_row"] == "btree"
        assert indexes["parking_meters_pkey"] == "btree"

    def test_preserves_both_identifier_columns(self, db):
        columns = {
            row[0]
            for row in db.run(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'parking_meters'"
            )
        }
        assert {"post_id", "parking_space_id"} <= columns


class TestIdempotentLoading:
    def test_loading_meters_twice_does_not_duplicate(self, db, tmp_path):
        path = tmp_path / "parking_meters.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "post_id": "612-13010",
                    "parking_space_id": None,
                    "latitude": 37.7888,
                    "longitude": -122.4204,
                    "active": True,
                    "street_name": "POLK ST",
                    "street_number": "1301",
                    "blockface_id": "612131",
                    "meter_type": "SS",
                },
                {
                    "post_id": "500-15310",
                    "parking_space_id": 123238,
                    "latitude": 37.7941,
                    "longitude": -122.4207,
                    "active": False,
                    "street_name": "JACKSON ST",
                    "street_number": "1531",
                    "blockface_id": "500151",
                    "meter_type": "SS",
                },
            ],
        )

        load_parking_meters(db, path)
        load_parking_meters(db, path)

        count, distinct_posts = db.run(
            "SELECT count(*), count(DISTINCT post_id) FROM parking_meters"
        )[0]
        assert count == 2 == distinct_posts

        # A changed row is updated, not duplicated.
        _write_jsonl(
            path,
            [
                {
                    "post_id": "612-13010",
                    "parking_space_id": 999,
                    "latitude": 37.7888,
                    "longitude": -122.4204,
                    "active": False,
                    "street_name": "POLK ST",
                    "street_number": "1301",
                    "blockface_id": "612131",
                    "meter_type": "SS",
                }
            ],
        )
        load_parking_meters(db, path)
        row = db.run(
            "SELECT parking_space_id, active FROM parking_meters "
            "WHERE post_id = '612-13010'"
        )[0]
        assert row[0] == 999 and row[1] is False
        assert db.run("SELECT count(*) FROM parking_meters")[0][0] == 2

    def test_loading_policies_twice_does_not_duplicate(self, db, tmp_path):
        path = tmp_path / "meter_policies.jsonl"
        policy = {
            "parking_space_id": 123238,
            "post_id": "102-02990",
            "day_of_week": "Mo",
            "start_time": "00:00:00",
            "end_time": "04:30:00",
            "hourly_rate": 0.0,
            "time_limit_minutes": None,
            "start_date": "2026-07-13",
            "end_date": "2200-12-31",
            "schedule_type": "FREE",
        }
        _write_jsonl(path, [policy])

        load_meter_policies(db, path)
        load_meter_policies(db, path)

        assert db.run("SELECT count(*) FROM meter_policies")[0][0] == 1
        stored = db.run(
            "SELECT post_id, parking_space_id, day_of_week, start_time::text, "
            "hourly_rate, schedule_type FROM meter_policies"
        )[0]
        assert stored[0] == "102-02990"
        assert stored[1] == 123238
        assert stored[2] == "Mo"
        assert stored[3] == "00:00:00"
        assert float(stored[4]) == 0.0
        assert stored[5] == "FREE"


class TestNearbyQuery:
    @pytest.fixture(autouse=True)
    def seed_meters(self, db, tmp_path):
        rows = [
            # Union Square area: target ~37.7880, -122.4075
            ("100-00001", None, 37.78800, -122.40750),   # ~0 m from query point
            ("100-00002", None, 37.78920, -122.40750),   # ~133 m north
            ("100-00003", None, 37.78800, -122.41100),   # ~388 m west
            ("200-00004", None, 37.76900, -122.48600),   # ~7 km away (outside)
        ]
        path = tmp_path / "meters.jsonl"
        _write_jsonl(
            path,
            (
                {
                    "post_id": post_id,
                    "parking_space_id": space_id,
                    "latitude": lat,
                    "longitude": lon,
                    "active": True,
                    "street_name": "MARKET ST",
                    "street_number": "1",
                    "blockface_id": "100001",
                    "meter_type": "SS",
                }
                for post_id, space_id, lat, lon in rows
            ),
        )
        load_parking_meters(db, path)

    def test_returns_meters_ordered_by_distance(self, db):
        results = find_meters_near(db, latitude=37.78800, longitude=-122.40750, radius_meters=500)
        posts = [m.post_id for m in results]
        assert posts[0] == "100-00001"
        assert set(posts) == {"100-00001", "100-00002", "100-00003"}
        distances = [m.distance_m for m in results]
        assert distances == sorted(distances)
        assert distances[0] < 1.0
        assert 120 < distances[1] < 150

    def test_radius_filters_distant_meters(self, db):
        results = find_meters_near(db, latitude=37.78800, longitude=-122.40750, radius_meters=200)
        posts = {m.post_id for m in results}
        assert posts == {"100-00001", "100-00002"}

    def test_results_carry_identifier_and_distance(self, db):
        result = find_meters_near(db, latitude=37.78800, longitude=-122.40750, radius_meters=50)[0]
        assert result.post_id == "100-00001"
        assert abs(result.latitude - 37.78800) < 1e-9
        assert isinstance(result.distance_m, float)
