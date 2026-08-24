"""Integration tests for the canonical spatial/temporal model.

Requires PostgreSQL/PostGIS on localhost:5432 (``docker compose up -d``).
Uses the throwaway-schema isolation strategy: public/production data is
never touched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pg8000
import pytest

from sf_parking.canonical import project_canonical
from sf_parking.database import apply_schema, connect

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
SF_TZ = ZoneInfo("America/Los_Angeles")


def _server_available() -> bool:
    try:
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


@pytest.fixture()
def conn():
    schema = f"pytest_sf_parking_{uuid.uuid4().hex[:12]}"
    conn = connect()
    conn.run(f'CREATE SCHEMA "{schema}"')
    conn.run("COMMIT")
    conn.run(f'SET search_path TO "{schema}", public')  # generated name only
    apply_schema(conn, SCHEMA_PATH)
    yield conn
    conn.close()
    cleanup = connect()
    try:
        cleanup.run(f'DROP SCHEMA "{schema}" CASCADE')
        cleanup.run("COMMIT")
    finally:
        cleanup.close()


def _meter_row(
    post_id,
    *,
    street_id="363",
    street_name="COLUMBUS AVE",
    blockface_id="363041",
    centerline="4294000",
    lat=37.79895,
    lon=-122.40839,
    active=True,
    data_as_of=None,
):
    return {
        "post_id": post_id,
        "parking_space_id": None,
        "latitude": lat,
        "longitude": lon,
        "active": active,
        "street_name": street_name,
        "street_number": "415",
        "blockface_id": blockface_id,
        "meter_type": "SS",
        "street_id": street_id,
        "street_centerline_id": centerline,
        "data_as_of": data_as_of,
    }


def _seed_inventory(conn, rows):
    for r in rows:
        conn.run(
            "INSERT INTO parking_meters (post_id, parking_space_id, latitude, longitude,"
            " active, street_name, street_number, blockface_id, meter_type, street_id,"
            " street_centerline_id, data_as_of)"
            " VALUES (:post_id, :parking_space_id, :latitude, :longitude, :active,"
            " :street_name, :street_number, :blockface_id, :meter_type, :street_id,"
            " :street_centerline_id, :data_as_of)",
            **r,
        )
    conn.run("COMMIT")


def _seed_policy(conn, space_id, post_id, start=None, end=None):
    conn.run(
        "INSERT INTO meter_policies (parking_space_id, post_id, day_of_week, start_time,"
        " end_time, hourly_rate, start_date, end_date, schedule_type)"
        " VALUES (:s, :p, 'Mo', '08:00', '18:00', 3.5, :start, :end, 'PAID')",
        s=space_id,
        p=post_id,
        start=start,
        end=end,
    )
    conn.run("COMMIT")


OBSERVED_AT = datetime(2026, 8, 22, 2, 11, 50, tzinfo=SF_TZ)


def _standard_seed(conn):
    _seed_inventory(
        conn,
        [
            _meter_row("363-04151", data_as_of=OBSERVED_AT),
            _meter_row(
                "363-04152",
                lat=37.79910,
                lon=-122.40850,
                data_as_of=OBSERVED_AT,
            ),
            # Meter with unknown observation time and no centerline id.
            _meter_row(
                "411-12040",
                street_id="411",
                street_name="FELL ST",
                blockface_id="411122",
                centerline=None,
                lat=37.77409,
                lon=-122.43792,
                data_as_of=None,
            ),
            # Meter with no blockface id at all.
            _meter_row(
                "999-00001",
                street_id="999",
                street_name="NOWHERE ST",
                blockface_id=None,
                centerline=None,
                lat=37.0,
                lon=-122.0,
                data_as_of=OBSERVED_AT,
            ),
        ],
    )
    _seed_policy(conn, 123238, "363-04151", start=date(2026, 7, 13), end=date(2200, 12, 31))
    _seed_policy(conn, 555001, "404-40404")  # space without any meter


@pytest.fixture()
def projected(conn):
    _standard_seed(conn)
    results = project_canonical(conn)
    assert all(r["status"] == "succeeded" for r in results.values()), results
    return conn


class TestCanonicalEntityCreation:
    def test_entities_created_from_inventory_and_policies(self, projected):
        counts = {
            t: int(projected.run(f"SELECT count(*) FROM {t}")[0][0])
            for t in ("streets", "blockfaces", "meters", "meter_placements",
                      "curb_segments", "parking_spaces", "parking_space_meters")
        }
        assert counts == {
            "streets": 3,          # 363, 411, 999
            "blockfaces": 2,       # 363041, 411122 (NULL blockface not invented)
            "meters": 4,
            "meter_placements": 4,
            "curb_segments": 0,    # no source resolves sub-blockface curbs yet
            "parking_spaces": 2,   # one linked to a meter, one without
            "parking_space_meters": 1,
        }

    def test_source_ids_preserved_verbatim(self, projected):
        assert [tuple(r) for r in projected.run(
            "SELECT source_street_id, name FROM streets ORDER BY source_street_id"
        )] == [("363", "COLUMBUS AVE"), ("411", "FELL ST"), ("999", "NOWHERE ST")]
        assert [r[0] for r in projected.run(
            "SELECT source_blockface_id FROM blockfaces ORDER BY 1"
        )] == ["363041", "411122"]
        assert projected.run(
            "SELECT post_id FROM meters WHERE post_id = '363-04151'"
        )[0][0] == "363-04151"
        assert [tuple(r) for r in projected.run(
            "SELECT source_space_id FROM parking_spaces ORDER BY source_space_id"
        )] == [("123238",), ("555001",)]

    def test_hierarchy_relationships_are_authoritative_only(self, projected):
        # Blockface -> street resolved from same-row observation.
        assert [tuple(r) for r in projected.run(
            "SELECT b.source_blockface_id, s.source_street_id FROM blockfaces b "
            "JOIN streets s USING (street_id) ORDER BY 1"
        )] == [("363041", "363"), ("411122", "411")]
        # Placement -> blockface; NULL when the source row had none.
        assert [tuple(r) for r in projected.run(
            "SELECT p.source_post_id, b.source_blockface_id FROM meter_placements p "
            "LEFT JOIN blockfaces b USING (blockface_id) ORDER BY 1"
        )] == [
            ("363-04151", "363041"),
            ("363-04152", "363041"),
            ("411-12040", "411122"),
            ("999-00001", None),
        ]
        # Spaces have unresolved spatial placement - never guessed.
        assert projected.run(
            "SELECT count(*) FROM parking_spaces "
            "WHERE geometry IS NOT NULL OR curb_segment_id IS NOT NULL"
        )[0][0] == 0

    def test_space_without_meter_is_represented(self, projected):
        orphan = projected.run(
            "SELECT count(*) FROM parking_spaces ps "
            "WHERE NOT EXISTS (SELECT 1 FROM parking_space_meters l "
            "WHERE l.parking_space_id = ps.parking_space_id)"
        )[0][0]
        assert int(orphan) >= 1

    def test_curb_segments_empty_but_functional(self, projected):
        # The hierarchy supports curb segments even though no source populates
        # them: insert one manually against an existing blockface.
        projected.run(
            "INSERT INTO curb_segments (blockface_id, source, retrieved_at) "
            "SELECT blockface_id, 'test', now() FROM blockfaces "
            "WHERE source_blockface_id = '363041'"
        )
        projected.run("COMMIT")
        assert int(projected.run("SELECT count(*) FROM curb_segments")[0][0]) == 1
        # A space can now reference it.
        projected.run(
            "UPDATE parking_spaces SET curb_segment_id = ("
            "  SELECT curb_segment_id FROM curb_segments LIMIT 1) "
            "WHERE source_space_id = '123238'"
        )
        projected.run("COMMIT")
        assert projected.run(
            "SELECT curb_segment_id IS NOT NULL FROM parking_spaces "
            "WHERE source_space_id = '123238'"
        )[0][0]


class TestSpatialModel:
    def test_placement_geometry_supports_radius_queries(self, projected):
        # Within ~15 m of the first meter's location.
        hits = projected.run(
            "SELECT p.source_post_id FROM meter_placements p "
            "WHERE ST_DWithin(p.location, "
            "ST_SetSRID(ST_MakePoint(-122.40839, 37.79895), 4326)::geography, 15)"
        )
        assert {r[0] for r in hits} == {"363-04151"}

    def test_spatial_indexes_exist(self, projected):
        indexes = {
            r[0]
            for r in projected.run(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
            )
        }
        assert {
            "idx_meter_placements_location_gist",
            "idx_parking_spaces_geometry_gist",
            "idx_meter_placements_valid_period",
            "idx_parking_space_meters_period",
        } <= indexes


class TestTemporalModel:
    def test_unknown_observation_time_is_infinite_not_fabricated(self, projected):
        rows = dict(
            projected.run(
                "SELECT source_post_id, lower(valid_period)::text "
                "FROM meter_placements"
            )
        )
        assert rows["411-12040"] == "-infinity"
        # Rendered on the UTC session clock: 02:11:50 PDT == 09:11:50Z.
        assert rows["363-04151"].startswith("2026-08-22 09:11:50")

    def test_newer_snapshot_preserves_history(self, conn):
        _seed_inventory(conn, [_meter_row("363-04151", data_as_of=OBSERVED_AT)])
        project_canonical(conn)

        moved_at = datetime(2026, 9, 1, 3, 0, 0, tzinfo=SF_TZ)
        conn.run(
            "UPDATE parking_meters SET latitude = 37.80, longitude = -122.41,"
            " data_as_of = :t WHERE post_id = '363-04151'",
            t=moved_at,
        )
        conn.run("COMMIT")
        project_canonical(conn)

        history = [tuple(r) for r in conn.run(
            "SELECT latitude::text, valid_from = :moved, valid_until IS NULL "
            "FROM meter_placements p JOIN meters m USING (meter_id) "
            "WHERE m.post_id = '363-04151' ORDER BY p.valid_from",
            moved=moved_at,
        )]
        assert history == [
            ("37.79895", False, False),  # old placement closed at move time
            ("37.8", True, True),        # new placement open-ended
        ]

        # "What was true at time T?" is answerable both ways.
        before = conn.run(
            "SELECT latitude::text FROM meter_placements p JOIN meters m USING (meter_id) "
            "WHERE m.post_id = '363-04151' "
            "AND valid_period @> :t::timestamptz",
            t=datetime(2026, 8, 23, tzinfo=UTC),
        )
        after = conn.run(
            "SELECT latitude::text FROM meter_placements p JOIN meters m USING (meter_id) "
            "WHERE m.post_id = '363-04151' AND valid_period @> :t::timestamptz",
            t=datetime(2026, 9, 2, tzinfo=UTC),
        )
        assert before[0][0] == "37.79895"
        assert after[0][0] == "37.8"

    def test_overlapping_placements_rejected_by_constraint(self, projected):
        meter_id = projected.run(
            "SELECT meter_id FROM meters WHERE post_id = '363-04151'"
        )[0][0]
        with pytest.raises(pg8000.Error):
            projected.run(
                "INSERT INTO meter_placements (meter_id, source_post_id, latitude,"
                " longitude, valid_from, source, retrieved_at)"
                " VALUES (:m, '363-04151', 37.79895, -122.40839, :vf, 'x', now())",
                m=meter_id,
                vf=datetime(2026, 8, 22, 12, tzinfo=UTC),
            )
        projected.run("ROLLBACK")


class TestIdempotentProjection:
    def test_rerun_stores_nothing_new(self, projected):
        baseline = {
            t: int(projected.run(f"SELECT count(*) FROM {t}")[0][0])
            for t in ("streets", "blockfaces", "meters", "meter_placements",
                      "parking_spaces", "parking_space_meters")
        }
        results = project_canonical(projected)

        assert all(r["status"] == "succeeded" for r in results.values()), results
        assert all(r["stored"] == 0 for r in results.values()), results
        after = {
            t: int(projected.run(f"SELECT count(*) FROM {t}")[0][0])
            for t in baseline
        }
        assert after == baseline


class TestUnresolvedIdentity:
    def test_transaction_posts_without_canonical_meter_are_visible(self, projected):
        projected.run(
            "INSERT INTO meter_transactions (transmission_id, post_id, session_start,"
            " source, retrieved_at) VALUES ('tx-hist', '000-00000', now(), 'test', now())"
        )
        projected.run(
            "INSERT INTO meter_transactions (transmission_id, post_id, session_start,"
            " source, retrieved_at) VALUES ('tx-live', '363-04151', now(), 'test', now())"
        )
        projected.run("COMMIT")

        unresolved = {
            r[0] for r in projected.run(
                "SELECT post_id FROM v_unresolved_transaction_posts"
            )
        }
        assert unresolved == {"000-00000"}  # known posts resolve; unknown stay honest

    def test_legacy_tables_remain_usable_alongside_canonical(self, projected):
        # Existing geographic query path still works off parking_meters.
        from sf_parking.database import find_meters_near

        nearby = find_meters_near(projected, 37.79895, -122.40839, 20)
        assert nearby[0].post_id == "363-04151"  # nearest first, legacy path intact
        assert {m.post_id for m in nearby} >= {"363-04151"}


class TestProvenanceRecorded:
    def test_each_projection_recorded_as_ingestion_run(self, projected):
        runs = projected.run(
            "SELECT source, status FROM ingestion_runs "
            "WHERE source LIKE 'canonical_projection:%' ORDER BY run_id"
        )
        assert {r[0] for r in runs} == {
            "canonical_projection:streets",
            "canonical_projection:blockfaces",
            "canonical_projection:meters",
            "canonical_projection:meter_placements",
            "canonical_projection:parking_spaces",
            "canonical_projection:parking_space_meters",
        }
        assert all(r[1] == "succeeded" for r in runs)

    def test_rows_trace_to_their_projection_run(self, projected):
        orphan_rows = projected.run(
            "SELECT count(*) FROM streets s LEFT JOIN ingestion_runs r USING (run_id) "
            "WHERE r.run_id IS NULL"
        )[0][0]
        assert int(orphan_rows) == 0
