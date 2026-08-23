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

V0.1 intentionally does **not** claim live occupancy. The SFMTA meter inventory contains physical meter/location data, while the current Meter Policies dataset supplies daily schedules and rates. These are separate datasets joined by `ParkingSpaceID`.

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
 parking rules engine
      ↓
 nearby parking API
      ↓
 map / web client
```

## Status

🚧 Initial repository scaffold.
