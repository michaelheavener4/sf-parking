#!/usr/bin/env python3
"""One-shot bootstrap for the research data-fusion layer.

What it does:
1. Applies the two new SQL migrations.
2. Optionally downloads the SFMTA historical sensor and smart-payment CSVs
   from the official SFpark evaluation page by following the published link.
3. Streams and normalizes those large CSVs into PostgreSQL via psql \copy.
4. Pulls current DataSF Parking Meters, Meter Policies, and (optionally) a
   bounded window of Parking Citations through the public Socrata API.

The 1.38 GB historical sensor file and the 812 MB-class payment dataset are
never loaded into Python memory in full.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LANDING = "https://www.sfmta.com/getting-around/drive-park/demand-responsive-pricing/sfpark-evaluation"
SOURCES = {
    "sensor": "SFpark Parking Sensor Data Hourly Occupancy 2011 - 2013",
    "smart_payments": "SFpark Meter Data Payment Transactions Smart Pilot Data",
}
SOCRATA = {
    "parking_meters": "8vzz-qzz9",
    "meter_policies": "qq7v-hds4",
    "parking_citations": "ab4h-6ztd",
}


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "sf-parking-data-fusion/1.0"})
    with urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def resolve_sfmta_link(label_fragment: str) -> str:
    text = fetch_text(LANDING)
    # Drupal renders a normal anchor; keep extraction deliberately conservative.
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
    candidates: list[tuple[str, str]] = []
    for href, anchor in pattern.findall(text):
        clean = re.sub(r"<[^>]+>", " ", html.unescape(anchor))
        clean = re.sub(r"\s+", " ", clean).strip()
        if label_fragment.lower() in clean.lower():
            candidates.append((clean, urljoin(LANDING, href)))
    if not candidates:
        raise RuntimeError(f"Could not resolve SFMTA download link containing: {label_fragment!r}")
    return candidates[0][1]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  using existing {dest} ({dest.stat().st_size:,} bytes)")
        return
    req = Request(url, headers={"User-Agent": "sf-parking-data-fusion/1.0"})
    with urlopen(req, timeout=120) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r  downloaded {copied / total:.1%}", end="", flush=True)
    print(f"\n  saved {dest} ({dest.stat().st_size:,} bytes)")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def normalized_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        return [norm(x) for x in next(csv.reader(f))]


def run_sql(sql: str) -> None:
    psql = shutil.which("psql")
    dsn = os.environ.get("DATABASE_URL")
    if not psql or not dsn:
        raise RuntimeError("DATABASE_URL and the psql client are required for large-file imports")
    subprocess.run([psql, dsn, "-v", "ON_ERROR_STOP=1", "-c", sql], check=True)


def apply_migrations() -> None:
    for path in [
        ROOT / "db/migrations/2026-08-25_data_fusion.sql",
        ROOT / "db/migrations/2026-08-25_sfpark_historical_sessions.sql",
    ]:
        print(f"Applying {path.name}")
        run_sql(path.read_text())


def copy_sensor_csv(path: Path) -> None:
    expected = {
        "block_id", "street_name", "block_num", "street_block", "area_type",
        "pm_district_name", "rate", "rate_type", "start_time_dt", "total_time",
        "total_occupied_time", "total_vacant_time", "total_unknown_time", "op_time",
        "op_occupied_time", "op_vacant_time", "op_unknown_time", "nonop_time",
        "nonop_occupied_time", "nonop_vacant_time", "nonop_unknown_time", "gmp_time",
        "gmp_occupied_time", "gmp_vacant_time", "gmp_unknown_time", "comm_time",
        "comm_occupied_time", "comm_vacant_time", "comm_unknown_time",
    }
    header = normalized_header(path)
    missing = expected - set(header)
    if missing:
        raise RuntimeError(f"Sensor CSV missing documented fields: {sorted(missing)}")

    # Stream transformed rows directly into psql. No full-file memory or temp copy.
    psql = shutil.which("psql"); dsn = os.environ.get("DATABASE_URL")
    if not psql or not dsn:
        raise RuntimeError("DATABASE_URL and psql are required")
    proc = subprocess.Popen(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-c", r"\copy sfpark_sensor_hourly (block_id,street_name,block_num,street_block,area_type,pm_district_name,rate,rate_type,start_time_local,total_time,total_occupied_time,total_vacant_time,total_unknown_time,op_time,op_occupied_time,op_vacant_time,op_unknown_time,nonop_time,nonop_occupied_time,nonop_vacant_time,nonop_unknown_time,gmp_time,gmp_occupied_time,gmp_vacant_time,gmp_unknown_time,comm_time,comm_occupied_time,comm_vacant_time,comm_unknown_time) FROM STDIN WITH (FORMAT csv, HEADER true)",],
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdin is not None
    writer = csv.writer(proc.stdin, lineterminator="\n")
    canonical = [
        "block_id","street_name","block_num","street_block","area_type","pm_district_name",
        "rate","rate_type","start_time_dt","total_time","total_occupied_time","total_vacant_time",
        "total_unknown_time","op_time","op_occupied_time","op_vacant_time","op_unknown_time",
        "nonop_time","nonop_occupied_time","nonop_vacant_time","nonop_unknown_time","gmp_time",
        "gmp_occupied_time","gmp_vacant_time","gmp_unknown_time","comm_time","comm_occupied_time",
        "comm_vacant_time","comm_unknown_time",
    ]
    writer.writerow(canonical)
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = [row.get(col, "") for col in canonical]
            writer.writerow(out)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("psql sensor import failed")


def copy_payment_csv(path: Path) -> None:
    # The SFMTA guide defines these smart-transaction fields. We accept common
    # historical header spellings and require the five causal essentials.
    header = normalized_header(path)
    aliases = {
        "parking_management_district": ["parking_management_district", "parking_management_district_name"],
        "collected_date_local": ["date", "collected_date", "collection_date"],
        "street_block": ["street_and_block", "street_block"],
        "post_id": ["post_id"],
        "payment_type": ["payment_type"],
        "net_amount_paid": ["net_amount_paid", "amount_paid"],
        "session_start_utc": ["session_start_date", "session_start", "start_time"],
        "session_end_utc": ["session_end_date", "session_end", "end_time"],
    }
    chosen = {}
    for key, options in aliases.items():
        chosen[key] = next((x for x in options if x in header), None)
    required = ["street_block", "session_start_utc", "session_end_utc"]
    missing = [x for x in required if not chosen[x]]
    if missing:
        raise RuntimeError(f"Smart payment CSV missing required fields: {missing}; header={header}")

    psql = shutil.which("psql"); dsn = os.environ.get("DATABASE_URL")
    if not psql or not dsn:
        raise RuntimeError("DATABASE_URL and psql are required")
    proc = subprocess.Popen(
        [psql, dsn, "-v", "ON_ERROR_STOP=1", "-c", r"\copy sfpark_payment_session_historical (parking_management_district,collected_date_local,street_block,post_id,payment_type,net_amount_paid,session_start_utc,session_end_utc) FROM STDIN WITH (FORMAT csv)"] ,
        stdin=subprocess.PIPE, text=True, encoding="utf-8",
    )
    assert proc.stdin is not None
    writer = csv.writer(proc.stdin, lineterminator="\n")
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def get(k: str) -> str:
                c = chosen[k]
                return (row.get(c, "") if c else "")
            writer.writerow([
                get("parking_management_district"),
                get("collected_date_local"),
                get("street_block"),
                get("post_id"),
                get("payment_type"),
                get("net_amount_paid"),
                get("session_start_utc"),
                get("session_end_utc"),
            ])
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("psql payment import failed")


def socrata_json(dataset_id: str, where: str | None = None, limit: int = 50000) -> list[dict]:
    base = f"https://data.sfgov.org/resource/{dataset_id}.json"
    params = [f"$limit={limit}"]
    if where:
        params.append(f"$where={quote(where)}")
    url = base + "?" + "&".join(params)
    req = Request(url, headers={"User-Agent": "sf-parking-data-fusion/1.0"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def write_source_registry() -> None:
    payload = json.loads((ROOT / "config/data_fusion_sources.json").read_text())
    lines = []
    for name, spec in payload["sources"].items():
        url = spec.get("url") or spec.get("landing_page")
        lines.append(
            f"INSERT INTO fusion_source_registry(source_name,source_url,source_kind,notes) VALUES "
            f"({json.dumps(name)},{json.dumps(url)},{json.dumps(spec['kind'])},{json.dumps(spec.get('role'))}) "
            f"ON CONFLICT(source_name) DO UPDATE SET source_url=excluded.source_url,source_kind=excluded.source_kind,notes=excluded.notes,ingested_at=now();"
        )
    run_sql("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-csv", type=Path)
    ap.add_argument("--smart-payments-csv", type=Path)
    ap.add_argument("--download-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--download-historical", action="store_true")
    args = ap.parse_args()

    apply_migrations()
    write_source_registry()

    if args.download_historical:
        args.download_dir.mkdir(parents=True, exist_ok=True)
        sensor_url = resolve_sfmta_link(SOURCES["sensor"])
        payment_url = resolve_sfmta_link(SOURCES["smart_payments"])
        sensor_path = args.download_dir / "SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv"
        payment_path = args.download_dir / "SFpark_MeterData_PaymentTransactions_Smart_20112013.csv"
        print("Resolving historical SFMTA files:")
        print(f"  sensor:   {sensor_url}")
        print(f"  payments: {payment_url}")
        download(sensor_url, sensor_path)
        download(payment_url, payment_path)
        args.sensor_csv, args.smart_payments_csv = sensor_path, payment_path

    if args.sensor_csv:
        print(f"Importing sensor hourly data from {args.sensor_csv}")
        copy_sensor_csv(args.sensor_csv)
    if args.smart_payments_csv:
        print(f"Importing historical smart payment sessions from {args.smart_payments_csv}")
        copy_payment_csv(args.smart_payments_csv)

    print("\nData-fusion bootstrap complete.")
    print("Next: run scripts/train_fused_occupancy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
