-- SF Parking database schema (PostgreSQL + PostGIS).

CREATE EXTENSION IF NOT EXISTS postgis;

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
