# SF Parking Intelligence — Roadmap

## V0.1 — Inventory + policies (done)

- 38,711 SFMTA parking meters in PostGIS (`parking_meters`, joined by `post_id`).
- 890,495 meter policy records (`meter_policies`), idempotent loads.
- Geographic queries: `find_meters_near(lat, lon, radius_m)`.
- Repeatable DataSF download/normalize pipeline into `data/raw/*.jsonl`.

## V0.2 — Generic ingestion framework + provenance (current, issue #1)

Turn one-off ingestion scripts into a reusable data platform layer:

- [x] Generic source/adapter interface; adapters resolved from a registry.
- [x] Source registry: `config/sources.yaml`.
- [x] Provenance: `ingestion_runs` records source, start/finish, status,
      processed/stored/skipped counts, source timestamp and errors.
- [x] Per-record provenance: stored observations keep source identifiers,
      the originating `run_id` and the retrieval timestamp.
- [x] Idempotent ingestion for every adapter (stable conflict keys).
- [x] Freshness/health checks against per-source SLAs.
- [x] First adapter: SFMTA meter transactions (`imvp-dq3v`).

## V0.3 — More sources (planned)

Adapters to add without touching core loading code:

1. Parking citations
2. Street sweeping schedules
3. Temporary closures / events
4. Managed off-street facilities
5. Current garage availability (`qahs-fevu`)

Migrate meters/policies JSONL snapshot loads onto the framework.

## V0.4 — API + UI

Nearby-parking API combining inventory, live policy state and observed
demand; map/web client.

## V0.5 — Prediction (ML)

Occupancy/demand estimation from accumulated transaction history.
Explicitly **not** before real observation data exists — no synthetic data.
