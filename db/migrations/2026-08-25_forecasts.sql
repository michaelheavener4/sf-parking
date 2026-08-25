-- Persisted parking availability forecasts produced by the forecasting pipeline.
-- Each row is a single meter × slot prediction attributed to a specific model
-- version and generation time.
CREATE TABLE IF NOT EXISTS parking_state_forecasts (
    id                     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    post_id                text NOT NULL,
    forecast_generated_at  timestamptz NOT NULL DEFAULT now(),
    target_slot            timestamptz NOT NULL,
    hours_ahead            smallint NOT NULL CHECK (hours_ahead BETWEEN 1 AND 24),
    predicted_availability double precision NOT NULL
                           CHECK (predicted_availability BETWEEN 0 AND 1),
    model_version          text NOT NULL,
    model_path             text NOT NULL,
    feature_data_as_of     timestamptz NOT NULL,
    -- Actual observed value, populated after the target slot has passed.
    actual_availability    double precision
                           CHECK (actual_availability IS NULL OR actual_availability BETWEEN 0 AND 1),
    actual_observed_at     timestamptz,
    -- Prevent duplicate forecasts for the same meter / slot / model.
    UNIQUE (post_id, target_slot, model_version)
);

-- Fast lookup: "what forecasts exist for this slot?"
CREATE INDEX IF NOT EXISTS idx_parking_state_forecasts_target_slot
    ON parking_state_forecasts (target_slot);

-- Fast lookup: "what was predicted for this meter at this time?"
CREATE INDEX IF NOT EXISTS idx_parking_state_forecasts_post_slot
    ON parking_state_forecasts (post_id, target_slot);

-- Fast lookup: "when was this forecast generated?"
CREATE INDEX IF NOT EXISTS idx_parking_state_forecasts_generated_at
    ON parking_state_forecasts (forecast_generated_at);

-- Fast lookup: "compare forecasts for the same slot across models."
CREATE INDEX IF NOT EXISTS idx_parking_state_forecasts_slot_model
    ON parking_state_forecasts (target_slot, model_version);
