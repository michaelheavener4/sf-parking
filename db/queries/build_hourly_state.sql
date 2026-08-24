-- Build bounded paid-occupancy state for one local calendar day.
-- Parameters: :day_start, :day_end, :max_prob, :p90_minutes
--
-- The materialized state represents COMPLETED hourly intervals. A row at
-- slot_start describes paid activity during [slot_start, slot_start + 1h).
-- Downstream forecasting must use only rows whose slot has already completed
-- at the forecast timestamp. Zero-transaction hours are explicit zeros.
WITH params AS (
    SELECT
        CAST(:day_start AS timestamptz) AS day_start,
        CAST(:day_end AS timestamptz) AS day_end
),
hours AS (
    SELECT gs AS slot_start
    FROM params p
    CROSS JOIN LATERAL generate_series(
        date_trunc('hour', p.day_start),
        date_trunc('hour', p.day_end) - INTERVAL '1 hour',
        INTERVAL '1 hour'
    ) AS gs
),
posts AS (
    SELECT post_id
    FROM _state_post_spans s
    CROSS JOIN params p
    WHERE s.first_local_date <= (p.day_end AT TIME ZONE 'America/Los_Angeles')::date
      AND s.last_local_date  >= (p.day_start AT TIME ZONE 'America/Los_Angeles')::date
),
grid AS (
    SELECT p.post_id, h.slot_start
    FROM posts p
    CROSS JOIN hours h
),
raw AS (
    SELECT
        t.post_id,
        t.session_start,
        t.session_end,
        GREATEST(t.session_start, p.day_start) AS clipped_start,
        LEAST(t.session_end, p.day_end) AS clipped_end
    FROM meter_transactions t
    CROSS JOIN params p
    WHERE t.session_end IS NOT NULL
      AND t.session_start < p.day_end
      AND t.session_end > p.day_start
),
slots AS (
    SELECT
        r.post_id,
        gs AS slot_start,
        GREATEST(
            0.0,
            EXTRACT(EPOCH FROM (
                LEAST(r.clipped_end, gs + INTERVAL '1 hour')
                - GREATEST(r.clipped_start, gs)
            )) / 60.0
        ) AS overlap_minutes
    FROM raw r
    CROSS JOIN LATERAL generate_series(
        date_trunc('hour', r.clipped_start),
        date_trunc('hour', r.clipped_end - INTERVAL '1 microsecond'),
        INTERVAL '1 hour'
    ) AS gs
),
scored AS (
    SELECT
        post_id,
        slot_start,
        overlap_minutes,
        CASE
            WHEN overlap_minutes <= 0 THEN 0.0
            ELSE LEAST(
                :max_prob,
                :max_prob / (
                    1.0 + EXP(-5.0 * (
                        overlap_minutes / :p90_minutes - 0.2
                    ))
                )
            )
        END AS event_probability
    FROM slots
),
grouped AS (
    SELECT
        post_id,
        slot_start,
        COUNT(*) FILTER (WHERE overlap_minutes > 0) AS transaction_count,
        SUM(overlap_minutes) AS paid_overlap_minutes,
        1.0 - EXP(
            SUM(LN(GREATEST(1e-12, 1.0 - event_probability)))
        ) AS paid_occupancy_probability
    FROM scored
    WHERE overlap_minutes > 0
    GROUP BY post_id, slot_start
)
SELECT
    g.post_id,
    g.slot_start,
    (g.slot_start AT TIME ZONE 'America/Los_Angeles')::date AS local_date,
    EXTRACT(
        HOUR FROM (g.slot_start AT TIME ZONE 'America/Los_Angeles')
    )::smallint AS local_hour,
    pm.meter_type,
    COALESCE(a.transaction_count, 0)::integer AS transaction_count,
    COALESCE(a.paid_overlap_minutes, 0.0)::double precision AS paid_overlap_minutes,
    COALESCE(
        LEAST(1.0, GREATEST(0.0, a.paid_occupancy_probability)),
        0.0
    )::double precision AS paid_occupancy_probability
FROM grid g
LEFT JOIN grouped a
  ON a.post_id = g.post_id
 AND a.slot_start = g.slot_start
LEFT JOIN parking_meters pm
  ON pm.post_id = g.post_id
ORDER BY g.slot_start, g.post_id;
