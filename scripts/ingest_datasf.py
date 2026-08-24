"""Download current SFMTA parking inventory and policies as normalized JSONL.

Usage:
    python scripts/ingest_datasf.py --output-dir data/raw

The raw city datasets are intentionally not committed to Git. This script is
repeatable and records the DataSF refresh timestamp available in each row.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sf_parking.datasf import (
    METER_POLICIES_DATASET,
    PARKING_METERS_DATASET,
    DataSFClient,
)
from sf_parking.normalize import normalize_meter, normalize_policy


def _write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meters_path = args.output_dir / "parking_meters.jsonl"
    policies_path = args.output_dir / "meter_policies.jsonl"
    metadata_path = args.output_dir / "manifest.json"

    with DataSFClient(timeout=60) as client:
        meters = (
            normalize_meter(row)
            for row in client.iter_rows(
                PARKING_METERS_DATASET,
                order="objectid",
            )
        )
        meter_count = _write_jsonl(
            meters_path,
            ({
                "parking_space_id": meter.parking_space_id,
                "post_id": meter.post_id,
                "latitude": meter.latitude,
                "longitude": meter.longitude,
                "active": meter.active,
                "street_name": meter.street_name,
                "street_number": meter.street_number,
                "blockface_id": meter.blockface_id,
                "meter_type": meter.meter_type,
                "street_id": meter.street_id,
                "street_centerline_id": meter.street_centerline_id,
                "data_as_of": (
                    meter.data_as_of.isoformat() if meter.data_as_of else None
                ),
            } for meter in meters),
        )

        policies = (
            normalize_policy(row)
            for row in client.iter_rows(
                METER_POLICIES_DATASET,
                order="pk_mapi",
            )
        )
        policy_count = _write_jsonl(
            policies_path,
            ({
                "parking_space_id": policy.parking_space_id,
                "post_id": policy.post_id,
                "day_of_week": policy.day_of_week,
                "start_time": policy.start_time.isoformat(),
                "end_time": policy.end_time.isoformat(),
                "hourly_rate": policy.hourly_rate,
                "time_limit_minutes": policy.time_limit_minutes,
                "start_date": policy.start_date.isoformat() if policy.start_date else None,
                "end_date": policy.end_date.isoformat() if policy.end_date else None,
                "schedule_type": policy.schedule_type,
            } for policy in policies),
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "parking_meters": PARKING_METERS_DATASET,
            "meter_policies": METER_POLICIES_DATASET,
        },
        "counts": {
            "parking_meters": meter_count,
            "meter_policies": policy_count,
        },
    }
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
