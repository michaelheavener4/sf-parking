"""PostgreSQL/PostGIS persistence layer.

Loads normalized JSONL snapshots into PostGIS and answers geographic
queries ("which meters are within N meters of this point?").
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pg8000.native

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/sf_parking"

METERS_BATCH_SIZE = 1_000
POLICIES_BATCH_SIZE = 2_000


def database_url_from_env() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(url: str | None = None) -> pg8000.native.Connection:
    url = url or database_url_from_env()
    parsed = urlparse(url)
    return pg8000.native.Connection(
        user=parsed.username or "postgres",
        password=parsed.password,
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/") or "sf_parking",
        timeout=30,
    )


def apply_schema(conn: pg8000.native.Connection, schema_path: Path) -> None:
    conn.run(schema_path.read_text(encoding="utf-8"))


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _csv_stream(rows: Iterable[dict[str, Any]], columns: list[str]) -> Iterator[bytes]:
    """Encode rows as CSV records suitable for ``COPY ... FROM STDIN``.

    Unquoted empty fields are read back as NULL by PostgreSQL's CSV COPY,
    which matches how missing JSONL values are handled.
    """
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(["" if row.get(col) is None else row[col] for col in columns])
        yield buffer.getvalue().encode("utf-8")
        buffer.seek(0)
        buffer.truncate(0)


METER_COLUMNS = [
    "post_id",
    "parking_space_id",
    "latitude",
    "longitude",
    "active",
    "street_name",
    "street_number",
    "blockface_id",
    "meter_type",
]

POLICY_COLUMNS = [
    "post_id",
    "parking_space_id",
    "day_of_week",
    "start_time",
    "end_time",
    "hourly_rate",
    "time_limit_minutes",
    "start_date",
    "end_date",
    "schedule_type",
]

_METER_STAGE_DDL = (
    "CREATE TEMP TABLE stage_parking_meters ("
    "post_id text, parking_space_id bigint, latitude double precision, "
    "longitude double precision, active boolean, street_name text, "
    "street_number text, blockface_id text, meter_type text)"
)

_POLICY_STAGE_DDL = (
    "CREATE TEMP TABLE stage_meter_policies ("
    "post_id text, parking_space_id bigint, day_of_week text, start_time time, "
    "end_time time, hourly_rate numeric, time_limit_minutes integer, "
    "start_date date, end_date date, schedule_type text)"
)


def load_parking_meters(
    conn: pg8000.native.Connection,
    path: Path,
    batch_size: int = METERS_BATCH_SIZE,
) -> int:
    """Load meters from JSONL via staged COPY. Idempotent on ``post_id``."""
    del batch_size
    conn.run("DROP TABLE IF EXISTS stage_parking_meters")
    conn.run(_METER_STAGE_DDL)
    conn.run(
        f"COPY stage_parking_meters ({', '.join(METER_COLUMNS)}) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=_csv_stream(_rows(path), METER_COLUMNS),
    )
    updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in METER_COLUMNS[1:])
    result = conn.run(
        f"WITH ins AS ("
        f"INSERT INTO parking_meters ({', '.join(METER_COLUMNS)}) "
        f"SELECT {', '.join(METER_COLUMNS)} FROM stage_parking_meters "
        f"ON CONFLICT (post_id) DO UPDATE SET {updates} "
        f"RETURNING 1) "
        f"SELECT count(*) FROM ins"
    )
    processed = int(result[0][0])
    conn.run("DROP TABLE stage_parking_meters")
    conn.run("commit")
    return processed


def load_meter_policies(
    conn: pg8000.native.Connection,
    path: Path,
    batch_size: int = POLICIES_BATCH_SIZE,
) -> int:
    """Load policies from JSONL via staged COPY. Idempotent on the full policy tuple."""
    del batch_size
    conn.run("DROP TABLE IF EXISTS stage_meter_policies")
    conn.run(_POLICY_STAGE_DDL)
    conn.run(
        f"COPY stage_meter_policies ({', '.join(POLICY_COLUMNS)}) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=_csv_stream(_rows(path), POLICY_COLUMNS),
    )
    result = conn.run(
        f"WITH ins AS ("
        f"INSERT INTO meter_policies ({', '.join(POLICY_COLUMNS)}) "
        f"SELECT {', '.join(POLICY_COLUMNS)} FROM stage_meter_policies "
        f"ON CONFLICT DO NOTHING "
        f"RETURNING 1) "
        f"SELECT count(*) FROM ins"
    )
    inserted = int(result[0][0])
    conn.run("DROP TABLE stage_meter_policies")
    conn.run("commit")
    return inserted


@dataclass(frozen=True, slots=True)
class NearbyMeter:
    post_id: str | None
    parking_space_id: int | None
    latitude: float
    longitude: float
    active: bool
    street_name: str | None
    street_number: str | None
    blockface_id: str | None
    meter_type: str | None
    distance_m: float


def find_meters_near(
    conn: pg8000.native.Connection,
    latitude: float,
    longitude: float,
    radius_meters: float,
    limit: int | None = None,
) -> list[NearbyMeter]:
    """Return meters within ``radius_meters`` of a point, nearest first."""
    point = (
        "ST_SetSRID(ST_MakePoint(CAST(:lon AS double precision), "
        "CAST(:lat AS double precision)), 4326)::geography"
    )
    sql = (
        "SELECT post_id, parking_space_id, latitude, longitude, active, "
        "street_name, street_number, blockface_id, meter_type, "
        f"ST_Distance(location, {point}) AS distance_m "
        "FROM parking_meters "
        f"WHERE ST_DWithin(location, {point}, CAST(:radius AS double precision)) "
        "ORDER BY distance_m ASC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.run(sql, lat=latitude, lon=longitude, radius=radius_meters)
    return [
        NearbyMeter(
            post_id=row[0],
            parking_space_id=row[1],
            latitude=row[2],
            longitude=row[3],
            active=row[4],
            street_name=row[5],
            street_number=row[6],
            blockface_id=row[7],
            meter_type=row[8],
            distance_m=row[9],
        )
        for row in rows
    ]


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Load normalized JSONL into PostGIS.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--schema", type=Path, default=Path("db/schema.sql"))
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    conn = connect(args.database_url)
    try:
        apply_schema(conn, args.schema)
        meter_count = load_parking_meters(conn, args.data_dir / "parking_meters.jsonl")
        policy_count = load_meter_policies(conn, args.data_dir / "meter_policies.jsonl")
        meters_total = conn.run("SELECT count(*) FROM parking_meters")[0][0]
        policies_total = conn.run("SELECT count(*) FROM meter_policies")[0][0]
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "processed": {"parking_meters": meter_count, "meter_policies": policy_count},
                "stored": {"parking_meters": meters_total, "meter_policies": policies_total},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
