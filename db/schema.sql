-- SF Parking database schema (PostgreSQL + PostGIS).

CREATE EXTENSION IF NOT EXISTS postgis;
-- Scalar equality inside GiST exclusion constraints (temporal validity).
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Physical parking meters from the SFMTA inventory.
-- The current inventory does not always expose parking_space_id, so post_id
-- (the stable cross-dataset identifier) is the primary key and both columns
-- are retained.
CREATE TABLE IF NOT EXISTS parking_meters (
    post_id          text PRIMARY KEY,
    parking_space_id bigint,
    latitude         double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude        double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    active           boolean NOT NULL DEFAULT true,
    street_name      text,
    street_number    text,
    blockface_id     text,
    meter_type       text,
    location         geography(Point, 4326)
        GENERATED ALWAYS AS (
            ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
        ) STORED
);

-- Fast radius queries: find meters within a distance of a point.
CREATE INDEX IF NOT EXISTS idx_parking_meters_location_gist
    ON parking_meters USING gist (location);

CREATE INDEX IF NOT EXISTS idx_parking_meters_space_id
    ON parking_meters (parking_space_id);

-- Time-bounded operating/rate policies. Rows are joined to meters by either
-- parking_space_id or post_id; both are kept because the datasets differ in
-- which identifier they expose.
CREATE TABLE IF NOT EXISTS meter_policies (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    post_id            text,
    parking_space_id   bigint,
    day_of_week        text NOT NULL,
    start_time         time NOT NULL,
    end_time           time NOT NULL,
    hourly_rate        numeric(10, 4) NOT NULL DEFAULT 0,
    time_limit_minutes integer,
    start_date         date,
    end_date           date,
    schedule_type      text
);

-- Makes repeated loads idempotent even when optional columns are NULL.
CREATE UNIQUE INDEX IF NOT EXISTS uq_meter_policies_row
    ON meter_policies (
        post_id, parking_space_id, day_of_week, start_time, end_time,
        hourly_rate, time_limit_minutes, start_date, end_date, schedule_type
    ) NULLS NOT DISTINCT;

CREATE INDEX IF NOT EXISTS idx_meter_policies_post_id
    ON meter_policies (post_id);

CREATE INDEX IF NOT EXISTS idx_meter_policies_space_id
    ON meter_policies (parking_space_id);

-- ---------------------------------------------------------------------------
-- Generic ingestion provenance (issue #1).
-- One row per ingestion attempt; a failed source is always visible here as
-- status = 'failed' with error details.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source            text NOT NULL,
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    status            text NOT NULL DEFAULT 'running'
                      CHECK (status IN ('running', 'succeeded', 'failed')),
    records_processed bigint NOT NULL DEFAULT 0,
    records_stored    bigint NOT NULL DEFAULT 0,
    records_skipped   bigint NOT NULL DEFAULT 0,
    source_timestamp  timestamptz,
    error             text
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source_started
    ON ingestion_runs (source, started_at DESC);

-- SFMTA parking meter revenue transactions (DataSF imvp-dq3v).
-- One row per paid session/extension event at one meter. Source identifiers
-- are preserved and every record traces to its ingestion run and retrieval
-- time. Idempotent on (transmission_id, post_id).
--
-- Identifier semantics (verified against live DataSF data, Aug 2026):
--   * post_id is the cross-dataset key documented by DataSF ("The identifier
--     of the meter this transaction is related to. See the related meters
--     dataset ... /d/8vzz-qzz9"). Both datasets use the identical
--     NNN-NNNNN namespace with no formatting differences.
--   * All 13,776 distinct post_ids in production transactions match the
--     current inventory snapshot; 99.3% of post_ids from 2017-era
--     transactions are still present in the 2026 inventory, so the namespace
--     is stable across ~9 years. Unmatched rows are meters retired after the
--     transaction occurred (or added after its window) — legitimate history.
--   * For that reason there is deliberately NO foreign key on post_id:
--     transaction observations must survive inventory refreshes, and a
--     missing match is surfaced by the post_id_coverage health check rather
--     than rejected at insert time.
--
-- Timestamp semantics: session_start/session_end arrive as Socrata floating
-- timestamps — America/Los_Angeles wall-clock times with no offset. The
-- adapter attaches the source zone before insert so these timestamptz values
-- represent true absolute instants (DST-safe). Rows ingested before this fix
-- were retagged in place by migration
-- 'migration:retag_source_local_timestamps'.
CREATE TABLE IF NOT EXISTS meter_transactions (
    transmission_id  text NOT NULL,
    post_id          text NOT NULL,
    street_block     text,
    payment_type     text,
    meter_event_type text,
    session_start    timestamptz NOT NULL,
    session_end      timestamptz,
    duration_minutes integer,
    gross_paid_amt   numeric(10, 2),
    source           text NOT NULL,
    run_id           bigint REFERENCES ingestion_runs(run_id),
    retrieved_at     timestamptz NOT NULL,
    PRIMARY KEY (transmission_id, post_id)
);

CREATE INDEX IF NOT EXISTS idx_meter_transactions_post_id
    ON meter_transactions (post_id);

CREATE INDEX IF NOT EXISTS idx_meter_transactions_session_start
    ON meter_transactions (session_start);

CREATE INDEX IF NOT EXISTS idx_meter_transactions_street_block
    ON meter_transactions (street_block);

-- ---------------------------------------------------------------------------
-- Additive observation fields on the legacy inventory table. The canonical
-- model below projects these into streets/blockfaces/meter placements; the
-- legacy table stays untouched otherwise so existing loads keep working.
ALTER TABLE parking_meters ADD COLUMN IF NOT EXISTS street_id         text;
ALTER TABLE parking_meters ADD COLUMN IF NOT EXISTS street_centerline_id text;
ALTER TABLE parking_meters ADD COLUMN IF NOT EXISTS data_as_of        timestamptz;

-- ===========================================================================
-- CANONICAL SPATIAL/TEMPORAL MODEL
--
-- Physical hierarchy (a meter is an installed device, not a place):
--
--   street -> blockface -> curb segment -> parking space -> meter
--
-- Identity rules:
--   * Canonical surrogate ids (bigint identity) are ours; SFMTA source ids
--     are preserved in source_*_id columns and are the ingestion conflict
--     keys, so re-ingestion is idempotent.
--   * Relationships that can change over time (meter placement, space<->meter
--     association) carry validity ranges. Closed ranges are never rewritten,
--     so "what was true at location X at time T" is answerable.
--   * Where current sources cannot establish geometry or linkage (curb
--     segments; space coordinates), columns stay NULL rather than inventing
--     precision.
-- ============================================================================

-- A named street. Source: inventory `street_id` (+ observed name).
CREATE TABLE IF NOT EXISTS streets (
    street_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_street_id   text NOT NULL UNIQUE,
    name               text,
    source             text NOT NULL,
    run_id             bigint REFERENCES ingestion_runs(run_id),
    retrieved_at       timestamptz NOT NULL
);

-- One side of one street between intersections. Source: inventory
-- `blockface_id`. The street association is a direct same-row observation in
-- the inventory, not an inference.
CREATE TABLE IF NOT EXISTS blockfaces (
    blockface_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_blockface_id  text NOT NULL UNIQUE,
    street_id            bigint REFERENCES streets(street_id),
    street_centerline_source_id text,
    source               text NOT NULL,
    run_id               bigint REFERENCES ingestion_runs(run_id),
    retrieved_at         timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blockfaces_street
    ON blockfaces (street_id);

-- A stretch of curb within a blockface (e.g. between driveways). No current
-- SFMTA dataset resolves sub-blockface curb geometry, so this table is
-- intentionally unpopulated until a source provides it; parking spaces link
-- to it only when such a source exists.
CREATE TABLE IF NOT EXISTS curb_segments (
    curb_segment_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    blockface_id     bigint NOT NULL REFERENCES blockfaces(blockface_id),
    geometry         geography(LineString, 4326),
    valid_from       timestamptz NOT NULL DEFAULT '-infinity'::timestamptz,
    valid_until      timestamptz,
    valid_period     tstzrange GENERATED ALWAYS AS (
                         tstzrange(valid_from, valid_until)) STORED,
    source           text NOT NULL,
    run_id           bigint REFERENCES ingestion_runs(run_id),
    retrieved_at     timestamptz NOT NULL,
    EXCLUDE USING gist (blockface_id WITH =, valid_period WITH &&)
);

-- An individual parkable space. Source: `ParkingSpaceID` ("primary key in
-- SFMTA inventory", exposed by the policies dataset; the current inventory
-- snapshot exposes it in zero rows). Spaces have no coordinates in any
-- current source: latitude/longitude/geometry stay NULL (unresolved) instead
-- of guessing from nearby meters.
CREATE TABLE IF NOT EXISTS parking_spaces (
    parking_space_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_space_id   text NOT NULL UNIQUE,
    blockface_id      bigint REFERENCES blockfaces(blockface_id),
    curb_segment_id   bigint REFERENCES curb_segments(curb_segment_id),
    latitude          double precision CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90),
    longitude         double precision CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180),
    geometry          geography(Point, 4326)
                      GENERATED ALWAYS AS (
                          CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL
                          THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
                          END) STORED,
    source            text NOT NULL,
    run_id            bigint REFERENCES ingestion_runs(run_id),
    retrieved_at      timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parking_spaces_geometry_gist
    ON parking_spaces USING gist (geometry);
CREATE INDEX IF NOT EXISTS idx_parking_spaces_blockface
    ON parking_spaces (blockface_id);

-- The canonical meter: a physical payment device identity, identified across
-- datasets by its SFMTA PostID.
CREATE TABLE IF NOT EXISTS meters (
    meter_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    post_id         text NOT NULL UNIQUE,
    source          text NOT NULL,
    run_id          bigint REFERENCES ingestion_runs(run_id),
    retrieved_at    timestamptz NOT NULL
);

-- Where a meter stood, over time. One row per (meter, observation period).
-- A new inventory observation with a newer data_as_of closes the previous
-- open-ended row and inserts a new one, so history is preserved verbatim.
-- Rows whose observation time is unknown get valid_from = -infinity rather
-- than a fabricated timestamp.
CREATE TABLE IF NOT EXISTS meter_placements (
    placement_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    meter_id       bigint NOT NULL REFERENCES meters(meter_id),
    blockface_id   bigint REFERENCES blockfaces(blockface_id),
    active         boolean,
    source_post_id text NOT NULL,
    latitude       double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude      double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    location       geography(Point, 4326)
                   GENERATED ALWAYS AS (
                       ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography) STORED,
    valid_from     timestamptz NOT NULL DEFAULT '-infinity'::timestamptz,
    valid_until    timestamptz,
    valid_period   tstzrange GENERATED ALWAYS AS (
                       tstzrange(valid_from, valid_until)) STORED,
    source         text NOT NULL,
    run_id         bigint REFERENCES ingestion_runs(run_id),
    retrieved_at   timestamptz NOT NULL,
    UNIQUE (source_post_id, valid_from),
    EXCLUDE USING gist (meter_id WITH =, valid_period WITH &&)
);

CREATE INDEX IF NOT EXISTS idx_meter_placements_location_gist
    ON meter_placements USING gist (location);
CREATE INDEX IF NOT EXISTS idx_meter_placements_blockface
    ON meter_placements (blockface_id);
CREATE INDEX IF NOT EXISTS idx_meter_placements_valid_period
    ON meter_placements USING gist (valid_period);

-- Authoritative space<->meter association: policy rows assert both
-- ParkingSpaceID and PostID in the same source row, so the relationship is
-- documented by SFMTA itself (no geographic inference). Validity comes from
-- the policy effective dates of the asserting rows.
CREATE TABLE IF NOT EXISTS parking_space_meters (
    parking_space_id bigint NOT NULL REFERENCES parking_spaces(parking_space_id),
    meter_id         bigint NOT NULL REFERENCES meters(meter_id),
    valid_from       date NOT NULL DEFAULT '-infinity'::date,
    valid_until      date,
    valid_period     daterange GENERATED ALWAYS AS (
                         daterange(valid_from, valid_until)) STORED,
    source           text NOT NULL,
    run_id           bigint REFERENCES ingestion_runs(run_id),
    retrieved_at     timestamptz NOT NULL,
    PRIMARY KEY (parking_space_id, meter_id)
);

CREATE INDEX IF NOT EXISTS idx_parking_space_meters_meter
    ON parking_space_meters (meter_id);
CREATE INDEX IF NOT EXISTS idx_parking_space_meters_period
    ON parking_space_meters USING gist (valid_period);

-- Transaction post_ids with no canonical meter: honest visibility of
-- unresolved historical references (e.g. meters retired before the current
-- inventory snapshot). Never guessed away.
CREATE OR REPLACE VIEW v_unresolved_transaction_posts AS
SELECT DISTINCT t.post_id
FROM meter_transactions t
LEFT JOIN meters m ON m.post_id = t.post_id
WHERE m.meter_id IS NULL;
