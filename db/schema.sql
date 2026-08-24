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
