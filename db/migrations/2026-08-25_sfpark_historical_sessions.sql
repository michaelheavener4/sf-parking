CREATE TABLE IF NOT EXISTS sfpark_payment_session_historical (
    parking_management_district text,
    collected_date_local date,
    street_block text NOT NULL,
    post_id text,
    payment_type text,
    net_amount_paid numeric,
    session_start_utc timestamp NOT NULL,
    session_end_utc timestamp NOT NULL,
    source_name text NOT NULL DEFAULT 'sfpark_smart_payment_transactions_2011_2013',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    CHECK (session_end_utc >= session_start_utc)
);
CREATE INDEX IF NOT EXISTS idx_sfpark_payment_hist_time ON sfpark_payment_session_historical(session_start_utc);
CREATE INDEX IF NOT EXISTS idx_sfpark_payment_hist_block_time ON sfpark_payment_session_historical(street_block, session_start_utc);

CREATE OR REPLACE VIEW v_fusion_historical_calibration_hourly AS
WITH payments AS (
    SELECT date_trunc('hour', session_start_utc AT TIME ZONE 'America/Los_Angeles') AS local_hour,
           street_block,
           COUNT(*)::int AS payment_session_starts,
           COALESCE(SUM(net_amount_paid),0)::numeric AS net_paid,
           COALESCE(AVG(EXTRACT(EPOCH FROM (session_end_utc-session_start_utc))/60.0),0)::double precision AS mean_paid_minutes
    FROM sfpark_payment_session_historical
    GROUP BY 1,2
), sensor AS (
    SELECT date_trunc('hour',start_time_local) AS local_hour,
           street_block,
           CASE WHEN total_occupied_time+total_vacant_time>0 THEN total_occupied_time::double precision/(total_occupied_time+total_vacant_time) END AS occupancy_total,
           CASE WHEN gmp_occupied_time+gmp_vacant_time>0 THEN gmp_occupied_time::double precision/(gmp_occupied_time+gmp_vacant_time) END AS occupancy_gmp,
           rate, rate_type
    FROM sfpark_sensor_hourly
), base AS (
    SELECT s.local_hour,s.street_block,s.occupancy_total,s.occupancy_gmp,s.rate,s.rate_type,
           COALESCE(p.payment_session_starts,0)::int AS payment_session_starts,
           COALESCE(p.net_paid,0)::numeric AS net_paid,
           COALESCE(p.mean_paid_minutes,0)::double precision AS mean_paid_minutes
    FROM sensor s LEFT JOIN payments p ON p.local_hour=s.local_hour AND p.street_block=s.street_block
)
SELECT b.*,
       LAG(b.occupancy_total) OVER (PARTITION BY b.street_block ORDER BY b.local_hour) AS lag_occupancy_total,
       LAG(b.payment_session_starts) OVER (PARTITION BY b.street_block ORDER BY b.local_hour) AS lag_payment_session_starts,
       AVG(b.payment_session_starts) OVER (PARTITION BY b.street_block ORDER BY b.local_hour ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS prior_3h_mean_payment_starts,
       AVG(b.payment_session_starts) OVER (PARTITION BY b.street_block,EXTRACT(ISODOW FROM b.local_hour),EXTRACT(HOUR FROM b.local_hour) ORDER BY b.local_hour ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS prior_same_slot_payment_mean
FROM base b;
