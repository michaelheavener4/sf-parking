# SF Parking Intelligence

Parking intelligence for San Francisco: combine SFMTA parking inventory, meter policies, regulations, and eventually live traffic/availability signals to answer one question: **where should I park right now?**

## V0.1 goal

Given a destination, arrival time, and intended parking duration, return nearby metered parking opportunities with:

- location
- active/inactive meter state
- current operating schedule
- current hourly rate
- time limit
- distance from destination
- source/evidence metadata

V0.1 intentionally does **not** claim live occupancy. The SFMTA meter inventory contains physical meter/location data, while the current Meter Policies dataset supplies daily schedules and rates. These datasets join on `post_id` (verified against live DataSF data; see docs/ROADMAP.md).

## Data sources

- SFMTA Parking Meters — DataSF dataset `8vzz-qzz9`
- SFMTA Meter Policies — DataSF dataset `qq7v-hds4`
- SFMTA Parking Regulations — DataSF dataset `hi6h-neyh` (supplemental; not treated as authoritative)

## Architecture

```text
DataSF / SFMTA
      ↓
  ingestion
      ↓
 normalized parking data
      ↓
 PostgreSQL + PostGIS
      ↓
 canonical spatial/temporal model
 (streets → blockfaces → curb segments → parking spaces → meters;
 see docs/CANONICAL_MODEL.md)
      ↓
 parking rules engine
      ↓
 nearby parking API
      ↓
 map / web client
```

## Setup

Requires Docker and Python 3.11+.

```bash
# 1. Start PostgreSQL + PostGIS on localhost:5432 (database: sf_parking)
docker compose up -d

# 2. Download + normalize current SFMTA data into data/raw/
python3 scripts/ingest_datasf.py

# 3. Create schema and load data/raw/*.jsonl into PostGIS
python3 scripts/load_database.py

# 4. Run the test suite
python3 -m pytest
```

Loading is idempotent: re-running `scripts/load_database.py` never duplicates
records. The connection string defaults to
`postgresql://postgres:postgres@localhost:5432/sf_parking` and can be
overridden with the `DATABASE_URL` environment variable.

## Status

🚧 V0.1: SFMTA inventory + meter policies ingested and loaded into PostGIS.
Nearby-meter geographic queries available via
`sf_parking.database.find_meters_near(latitude, longitude, radius_meters)`.
Live availability is not yet implemented.
