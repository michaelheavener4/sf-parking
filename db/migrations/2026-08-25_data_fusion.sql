-- Data-fusion layer for the parking dynamics engine.
-- Sources are deliberately separated from derived model tables.

CREATE TABLE IF NOT EXISTS fusion_source_registry (
    source_name text PRIMARY KEY,
    source_url text NOT NULL,
    source_kind text NOT NULL,
    dataset_as_of timestamptz,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    notes text
);

CREATE TABLE IF NOT EXISTS sfpark_sensor_hourly (
    block_id bigint,
    street_name text,
    block_num text,
    street_block text,
    area_type text,
    pm_district_name text,
    rate numeric,
    rate_type text,
    start_time_local timestamp NOT NULL,
    total_time bigint,
    total_occupied_time bigint,
    total_vacant_time bigint,
    total_unknown_time bigint,
    op_time bigint,
    op_occupied_time bigint,
    op_vacant_time bigint,
    op_unknown_time bigint,
    nonop_time bigint,
    nonop_occupied_time bigint,
    nonop_vacant_time bigint,
    nonop_unknown_time bigint,
    gmp_time bigint,
    gmp_occupied_time bigint,
    gmp_vacant_time bigint,
    gmp_unknown_time bigint,
    comm_time bigint,
    comm_occupied_time bigint,
    comm_vacant_time bigint,
    comm_unknown_time bigint,
    source_name text NOT NULL DEFAULT 'sfpark_sensor_hourly_2011_2013',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (street_block, start_time_local)
);

CREATE INDEX IF NOT EXISTS idx_sfpark_sensor_hourly_time ON sfpark_sensor_hourly(start_time_local);
CREATE INDEX IF NOT EXISTS idx_sfpark_sensor_hourly_block ON sfpark_sensor_hourly(street_block, start_time_local);

CREATE TABLE IF NOT EXISTS fusion_events (
    event_id text PRIMARY KEY,
    start_time timestamptz NOT NULL,
    end_time timestamptz,
    latitude double precision,
    longitude double precision,
    event_type text,
    expected_attendance integer,
    source_name text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fusion_events_time ON fusion_events(start_time, end_time);

CREATE TABLE IF NOT EXISTS fusion_weather (
    observed_at timestamptz PRIMARY KEY,
    temperature_c numeric,
    precipitation_mm numeric,
    wind_speed_mps numeric,
    visibility_m numeric,
    condition text,
    source_name text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fusion_citations_hourly (
    local_hour timestamp NOT NULL,
    citation_count integer NOT NULL,
    source_name text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (local_hour, source_name)
);

CREATE OR REPLACE VIEW v_fusion_sensor_transaction_hourly AS
WITH tx AS (
    SELECT
        date_trunc('hour', t.session_start AT TIME ZONE 'America/Los_Angeles') AS local_hour,
        t.street_block,
        COUNT(DISTINCT t.transmission_id)::int AS transaction_count,
        COALESCE(SUM(t.gross_paid_amt), 0)::numeric AS gross_paid_amt,
        COALESCE(SUM(t.duration_minutes), 0)::numeric AS paid_duration_minutes
    FROM meter_transactions t
    WHERE t.session_start IS NOT NULL
    GROUP BY 1, 2
), sensor AS (
    SELECT
        date_trunc('hour', s.start_time_local) AS local_hour,
        s.street_block,
        CASE WHEN (s.total_occupied_time + s.total_vacant_time) > 0
             THEN s.total_occupied_time::double precision /
                  (s.total_occupied_time + s.total_vacant_time) END AS sensor_total_occupancy,
        CASE WHEN (s.gmp_occupied_time + s.gmp_vacant_time) > 0
             THEN s.gmp_occupied_time::double precision /
                  (s.gmp_occupied_time + s.gmp_vacant_time) END AS sensor_gmp_occupancy,
        s.rate,
        s.rate_type
    FROM sfpark_sensor_hourly s
)
SELECT
    COALESCE(s.local_hour, tx.local_hour) AS local_hour,
    COALESCE(s.street_block, tx.street_block) AS street_block,
    tx.transaction_count,
    tx.gross_paid_amt,
    tx.paid_duration_minutes,
    s.sensor_total_occupancy,
    s.sensor_gmp_occupancy,
    s.rate,
    s.rate_type
FROM sensor s
FULL OUTER JOIN tx
  ON tx.local_hour = s.local_hour AND tx.street_block = s.street_block;

CREATE OR REPLACE VIEW v_fusion_sensor_calibration AS
SELECT * FROM v_fusion_sensor_transaction_hourly
WHERE sensor_total_occupancy IS NOT NULL;
