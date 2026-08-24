"""Canonical spatial/temporal model projections.

Projects the ingested SFMTA snapshots (legacy ``parking_meters`` and
``meter_policies``) into the canonical entity hierarchy:

    street -> blockface -> curb segment -> parking space -> meter

These are *derived* sources: they read local tables, never the network, and
run through the same generic ingestion framework as external adapters so
every projection is recorded in ``ingestion_runs`` with provenance and is
idempotent on stable source-id conflict keys.

Honesty rules encoded here:

* Source identifiers are preserved verbatim in ``source_*_id`` columns.
* Geometry/linkage that no current source establishes stays NULL/unresolved.
* The space<->meter association exists only because policy rows assert both
  identifiers in one source row - never inferred from geography.
* Meter placements are temporal: a newer inventory observation closes the
  previous open-ended validity range instead of overwriting history.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import pg8000.native

from .database import transaction
from .ingestion.framework import IngestionRecord, run_ingestion


class _DbProjectionAdapter:
    """Base for adapters that project locally ingested tables."""

    name: ClassVar[str] = ""
    target_table: ClassVar[str] = ""
    columns: ClassVar[list[str]] = []
    conflict_columns: ClassVar[list[str]] = []

    def __init__(self, conn: pg8000.native.Connection) -> None:
        self._conn = conn

    def fetch(self, options: dict[str, Any]) -> Iterable[IngestionRecord]:
        for row in self._conn.run(self.source_sql()):
            yield IngestionRecord(
                key=tuple(row[self.columns.index(c)] for c in self.conflict_columns),
                values=dict(zip(self.columns, row)),
            )


class StreetsProjection(_DbProjectionAdapter):
    """Canonical streets from observed (street_id, street_name) pairs."""

    name: ClassVar[str] = "canonical_projection:streets"
    target_table: ClassVar[str] = "streets"
    columns: ClassVar[list[str]] = ["source_street_id", "name"]
    conflict_columns: ClassVar[list[str]] = ["source_street_id"]

    def source_sql(self) -> str:
        return (
            "SELECT pm.street_id::text AS source_street_id, MIN(pm.street_name) AS name "
            "FROM parking_meters pm "
            "WHERE COALESCE(pm.street_id, '') <> '' "
            "GROUP BY pm.street_id::text"
        )


class BlockfacesProjection(_DbProjectionAdapter):
    """Canonical blockfaces; street association is a same-row observation."""

    name: ClassVar[str] = "canonical_projection:blockfaces"
    target_table: ClassVar[str] = "blockfaces"
    columns: ClassVar[list[str]] = ["source_blockface_id", "street_id", "street_centerline_source_id"]
    conflict_columns: ClassVar[list[str]] = ["source_blockface_id"]

    def source_sql(self) -> str:
        # MAX() collapses to the single observed value when consistent; a
        # blockface genuinely associated with several streets yields an
        # arbitrary-but-deterministic pick rather than a failure.
        return (
            "SELECT x.source_blockface_id, MAX(x.canonical_street_id) AS street_id, "
            "MAX(x.street_centerline_source_id) AS street_centerline_source_id "
            "FROM ("
            "  SELECT DISTINCT "
            "         pm.blockface_id::text AS source_blockface_id, "
            "         s.street_id AS canonical_street_id, "
            "         pm.street_centerline_id::text AS street_centerline_source_id "
            "  FROM parking_meters pm "
            "  LEFT JOIN streets s ON s.source_street_id = pm.street_id::text "
            "  WHERE COALESCE(pm.blockface_id, '') <> ''"
            ") x GROUP BY x.source_blockface_id"
        )


class MetersProjection(_DbProjectionAdapter):
    """One canonical meter per SFMTA PostID."""

    name: ClassVar[str] = "canonical_projection:meters"
    target_table: ClassVar[str] = "meters"
    columns: ClassVar[list[str]] = ["post_id"]
    conflict_columns: ClassVar[list[str]] = ["post_id"]

    def source_sql(self) -> str:
        return (
            "SELECT DISTINCT post_id FROM parking_meters "
            "WHERE COALESCE(post_id, '') <> ''"
        )


class MeterPlacementsProjection(_DbProjectionAdapter):
    """Point-in-time meter locations with temporal validity.

    Each inventory observation becomes one placement valid from its
    ``data_as_of`` (or ``-infinity`` when the observation time is unknown -
    honest rather than fabricated). Before inserting, previously open-ended
    placements are closed at the new observation time so ranges never overlap
    (enforced by an exclusion constraint).
    """

    name: ClassVar[str] = "canonical_projection:meter_placements"
    target_table: ClassVar[str] = "meter_placements"
    columns: ClassVar[list[str]] = [
        "meter_id",
        "blockface_id",
        "active",
        "source_post_id",
        "latitude",
        "longitude",
        "valid_from",
        "valid_until",
    ]
    conflict_columns: ClassVar[list[str]] = ["source_post_id", "valid_from"]

    def fetch(self, options: dict[str, Any]) -> Iterable[IngestionRecord]:
        self._close_superseded()
        yield from super().fetch(options)

    def _close_superseded(self) -> None:
        with transaction(self._conn):
            self._conn.run(
                "UPDATE meter_placements p SET valid_until = i.observed_at "
                "FROM ("
                "  SELECT post_id, MIN(data_as_of) AS observed_at FROM parking_meters "
                "  WHERE data_as_of IS NOT NULL AND COALESCE(post_id,'') <> '' "
                "  GROUP BY post_id"
                ") i "
                "JOIN meters m ON m.post_id = i.post_id "
                "WHERE p.meter_id = m.meter_id AND p.valid_until IS NULL "
                "AND p.valid_from < i.observed_at"
            )

    def source_sql(self) -> str:
        return (
            "SELECT DISTINCT ON (pm.post_id) "
            "m.meter_id, b.blockface_id, pm.active, pm.post_id AS source_post_id, "
            "pm.latitude, pm.longitude, "
            "COALESCE(pm.data_as_of, '-infinity'::timestamptz) AS valid_from, "
            "NULL::timestamptz AS valid_until "
            "FROM parking_meters pm "
            "JOIN meters m ON m.post_id = pm.post_id "
            "LEFT JOIN blockfaces b ON b.source_blockface_id = pm.blockface_id::text "
            "ORDER BY pm.post_id"
        )


class ParkingSpacesProjection(_DbProjectionAdapter):
    """Spaces identified by SFMTA ParkingSpaceID (policies dataset rows).

    No coordinates exist in any current source, so spaces are created without
    geometry or blockface assignment rather than inventing them.
    """

    name: ClassVar[str] = "canonical_projection:parking_spaces"
    target_table: ClassVar[str] = "parking_spaces"
    columns: ClassVar[list[str]] = ["source_space_id"]
    conflict_columns: ClassVar[list[str]] = ["source_space_id"]

    def source_sql(self) -> str:
        return (
            "SELECT DISTINCT parking_space_id::text AS source_space_id "
            "FROM meter_policies WHERE parking_space_id IS NOT NULL"
        )


class SpaceMeterLinksProjection(_DbProjectionAdapter):
    """Authoritative space<->meter association from policy rows.

    Validity comes from the asserting policy rows' effective dates; rows with
    unknown dates contribute open-ended validity (-infinity).
    """

    name: ClassVar[str] = "canonical_projection:parking_space_meters"
    target_table: ClassVar[str] = "parking_space_meters"
    columns: ClassVar[list[str]] = [
        "parking_space_id",
        "meter_id",
        "valid_from",
        "valid_until",
    ]
    conflict_columns: ClassVar[list[str]] = ["parking_space_id", "meter_id"]

    def source_sql(self) -> str:
        return (
            "SELECT ps.parking_space_id, m.meter_id, "
            "MIN(COALESCE(mp.start_date, '-infinity'::date)) AS valid_from, "
            "MAX(mp.end_date) AS valid_until "
            "FROM meter_policies mp "
            "JOIN parking_spaces ps ON ps.source_space_id = mp.parking_space_id::text "
            "JOIN meters m ON m.post_id = mp.post_id "
            "WHERE mp.parking_space_id IS NOT NULL AND COALESCE(mp.post_id,'') <> '' "
            "GROUP BY ps.parking_space_id, m.meter_id"
        )


#: Projection order respects foreign keys between canonical entities.
PROJECTION_SEQUENCE = [
    StreetsProjection,
    BlockfacesProjection,
    MetersProjection,
    MeterPlacementsProjection,
    ParkingSpacesProjection,
    SpaceMeterLinksProjection,
]


def project_canonical(conn: pg8000.native.Connection) -> dict[str, Any]:
    """Run every canonical projection through the ingestion framework."""
    results = {}
    for adapter_cls in PROJECTION_SEQUENCE:
        result = run_ingestion(conn, adapter_cls(conn))
        results[adapter_cls.target_table] = {
            "run_id": result.run_id,
            "status": result.status,
            "processed": result.records_processed,
            "stored": result.records_stored,
            "skipped": result.records_skipped,
            "error": result.error,
        }
    return results
