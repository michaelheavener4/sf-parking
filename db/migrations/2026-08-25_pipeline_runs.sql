-- Operational history for the hourly forecasting pipeline.
-- One row per pipeline execution attempt.
CREATE TABLE IF NOT EXISTS forecast_pipeline_runs (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at            timestamptz NOT NULL DEFAULT now(),
    completed_at          timestamptz,
    status                text NOT NULL DEFAULT 'running'
                          CHECK (status IN (
                              'running', 'success', 'data_stale',
                              'ingestion_failed', 'forecast_failed',
                              'verification_failed'
                          )),
    latest_observed_slot  timestamptz,
    forecast_target_slot  timestamptz,
    hours_ahead           smallint NOT NULL DEFAULT 1,
    rows_ingested         bigint,
    rows_forecast         bigint,
    forecasts_evaluated   bigint,
    data_age_minutes      double precision,
    model_version         text,
    error_message         text
);

CREATE INDEX IF NOT EXISTS idx_forecast_pipeline_runs_started
    ON forecast_pipeline_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_forecast_pipeline_runs_status
    ON forecast_pipeline_runs (status, started_at DESC);
