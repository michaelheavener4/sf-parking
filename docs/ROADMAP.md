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

## Data-model findings (investigation, Aug 2026)

### Timestamp semantics (`imvp-dq3v` → `meter_transactions`)

`session_start_dt` / `session_end_dt` are Socrata **floating timestamps**:
wall-clock times with **no UTC offset**, reported on SFMTA's operating clock,
**America/Los_Angeles**. Evidence:

1. DataSF/Socrata define floating timestamps as agency-local time; SFMTA
   operates in San Francisco.
2. Observed distribution over 320k real sessions peaks at 09:00–16:00 with a
   hard stop at 18:00 — exactly SF metered hours on a Pacific clock.
   Interpreted as UTC they would fall at ~2–11 AM Pacific.
3. The inventory dataset's `data_as_of` (floating) vs `data_loaded_at`
   (Socrata load time) differ by roughly the Pacific offset plus lag.

Consequence: naive parsing stored these as if UTC, shifting every instant by
−7/−8 h. The adapter now attaches `America/Los_Angeles` before insert
(`sf_parking.adapters.datasf.parse_socrata_timestamp`) so `timestamptz`
columns carry true instants. DST rules (PEP 495): ambiguous fall-back times
resolve to the first occurrence (`fold=0`); nonexistent spring-forward times
keep the pre-transition offset. Session durations subtract absolute UTC
instants, so they stay correct across DST transitions. Existing rows were
retagged in place by the recorded one-time migration
`migration:retag_source_local_timestamps`.

### Identifier semantics: `post_id` is the join key (verified)

The suspected transaction↔inventory post_id mismatch was **not** a namespace
problem. Verified against live DataSF data:

- DataSF documents transaction `post_id` as the key into meters dataset
  `8vzz-qzz9`.
- All 13,776 distinct post_ids in production transactions match the current
  inventory exactly (same `NNN-NNNNN` format; no whitespace/case deltas).
- `post_id` is temporally stable: 99.3% of distinct post_ids from 2017-era
  transactions still exist in the 2026 inventory.
- Meters ↔ policies also join on `post_id` (26,136/26,137 policy post_ids
  match). `parking_space_id` exists in policy rows but the current inventory
  exposes it in **zero** rows, so it cannot serve as the join key today.

Root cause of the observed "zero matches": the production `parking_meters`
table was empty at measurement time (the docker volume was recreated after
the last manual `load_database.py` run), so any join returned nothing.

Model decisions:

- No foreign key on `meter_transactions.post_id`: transactions referencing
  retired meters are legitimate history and must survive inventory refreshes.
- Coverage drift is monitored instead, via
  `sf_parking.ingestion.health.post_id_coverage` (healthy ≥ 99%).

Implications for inference work: sessions can be joined to locations through
`post_id` directly; demand models should treat unmatched post_ids as retired/
new meters rather than data corruption, and all temporal features must be
computed in America/Los_Angeles time.

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
