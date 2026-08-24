-- Materialized hourly paid-occupancy state used by the forecasting stack.
-- This is an inferred paid-use state, not physical ground-truth occupancy.
CREATE TABLE IF NOT EXISTS parking_state_hourly (
    post_id                       text NOT NULL,
    slot_start                    timestamptz NOT NULL,
    local_date                    date NOT NULL,
    local_hour                    smallint NOT NULL CHECK (local_hour BETWEEN 0 AND 23),
    meter_type                    text,
    transaction_count             integer NOT NULL DEFAULT 0,
    paid_overlap_minutes          double precision NOT NULL DEFAULT 0 CHECK (paid_overlap_minutes >= 0),
    paid_occupancy_probability    double precision NOT NULL CHECK (paid_occupancy_probability BETWEEN 0 AND 1),
    paid_availability_probability double precision NOT NULL CHECK (paid_availability_probability BETWEEN 0 AND 1),
    source                        text NOT NULL DEFAULT 'derived_from_sfmta_transactions',
    generated_at                  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (post_id, slot_start)
);

CREATE INDEX IF NOT EXISTS idx_parking_state_hourly_slot
    ON parking_state_hourly (slot_start);
CREATE INDEX IF NOT EXISTS idx_parking_state_hourly_post_slot
    ON parking_state_hourly (post_id, slot_start DESC);
CREATE INDEX IF NOT EXISTS idx_parking_state_hourly_local_hour
    ON parking_state_hourly (local_date, local_hour);
