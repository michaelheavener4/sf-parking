-- Build bounded paid-occupancy state for one local calendar day.
-- Parameters: :day_start, :day_end, :max_prob, :p90_minutes
WITH raw AS (
    SELECT post_id, session_start, session_end,
           GREATEST(session_start, :day_start) AS clipped_start,
           LEAST(session_end, :day_end) AS clipped_end
    FROM meter_transactions
    WHERE session_end IS NOT NULL
      AND session_start < :day_end
      AND session_end > :day_start
), slots AS (
    SELECT r.post_id, gs AS slot_start,
           GREATEST(0.0, EXTRACT(EPOCH FROM
             (LEAST(r.clipped_end, gs + INTERVAL '1 hour') -
              GREATEST(r.clipped_start, gs))) / 60.0) AS overlap_minutes
    FROM raw r
    CROSS JOIN LATERAL generate_series(
        (date_trunc('hour', r.clipped_start AT TIME ZONE 'America/Los_Angeles')
            AT TIME ZONE 'America/Los_Angeles'),
        (date_trunc('hour', (r.clipped_end - INTERVAL '1 microsecond') AT TIME ZONE 'America/Los_Angeles')
            AT TIME ZONE 'America/Los_Angeles'),
        INTERVAL '1 hour'
    ) gs
), scored AS (
    SELECT post_id, slot_start, overlap_minutes,
           CASE WHEN overlap_minutes <= 0 THEN 0.0 ELSE LEAST(
               :max_prob,
               :max_prob / (1.0 + EXP(-5.0 * (overlap_minutes / :p90_minutes - 0.2)))
           ) END AS event_probability
    FROM slots
), grouped AS (
    SELECT post_id, slot_start,
           COUNT(*) FILTER (WHERE overlap_minutes > 0) AS transaction_count,
           SUM(overlap_minutes) AS paid_overlap_minutes,
           1.0 - EXP(SUM(LN(GREATEST(1e-12, 1.0 - event_probability))))
             AS paid_occupancy_probability
    FROM scored
    WHERE overlap_minutes > 0
    GROUP BY post_id, slot_start
)
SELECT g.post_id, g.slot_start,
       (g.slot_start AT TIME ZONE 'America/Los_Angeles')::date AS local_date,
       EXTRACT(HOUR FROM (g.slot_start AT TIME ZONE 'America/Los_Angeles'))::smallint AS local_hour,
       pm.meter_type,
       g.transaction_count,
       g.paid_overlap_minutes,
       LEAST(1.0, GREATEST(0.0, g.paid_occupancy_probability)) AS paid_occupancy_probability
FROM grouped g
LEFT JOIN parking_meters pm USING (post_id)
ORDER BY g.slot_start, g.post_id;
