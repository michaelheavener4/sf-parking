"""Leakage-safe spatial + dynamic ML features for parking forecasting.

The original production model is intentionally kept stable.  This module is
an additive research/production layer that adds neighborhood state and
short-term dynamics without changing the canonical parking-state table.

Spatial features use PostGIS KNN against the existing GiST meter-location
index.  For every target meter, only the N nearest *other* meters are used.
Every state lookup is for a slot strictly before the target slot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import csv

import numpy as np
import pandas as pd

from .database import connect

FEATURES_SPATIAL = [
    "lag1_availability", "lag2_availability", "lag3_availability",
    "lag6_availability", "lag24_availability", "lag168_availability",
    "lag1_transactions", "lag24_transactions",
    "roll3_availability", "roll24_availability",
    "delta1_availability", "delta3_availability",
    "acceleration_availability", "tx_delta1", "tx_ratio_3h",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_ms",
    "neighbor_count", "neighbor_mean_availability",
    "neighbor_median_availability", "neighbor_min_availability",
    "neighbor_max_availability", "neighbor_std_availability",
    "neighbor_occupied_fraction", "neighbor_mean_transactions",
    "neighbor_mean_delta1", "neighbor_mean_delta3",
    "neighbor_distance_weighted_availability",
]


@dataclass(frozen=True)
class SpatialFeatureConfig:
    neighbor_k: int = 24
    max_distance_m: float = 250.0


def _copy_targets(conn, targets: list[tuple]) -> None:
    conn.run("DROP TABLE IF EXISTS _ml_targets")
    conn.run("""
        CREATE TEMP TABLE _ml_targets (
            post_id text,
            slot_start timestamptz,
            target double precision,
            meter_type text,
            local_hour int,
            local_date date
        )
    """)
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(targets)
    conn.run(
        "COPY _ml_targets (post_id, slot_start, target, meter_type, local_hour, local_date) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )


def build_spatial_features(
    conn,
    targets: list[tuple],
    *,
    config: SpatialFeatureConfig = SpatialFeatureConfig(),
) -> pd.DataFrame:
    """Build one leakage-safe row per target.

    ``targets`` rows are ``post_id, slot_start, target, meter_type,
    local_hour, local_date``.  The query intentionally never reads the target
    slot for any feature.  The target column is carried only as the label.
    """
    if not targets:
        return pd.DataFrame(columns=["post_id", "slot_start", "target", *FEATURES_SPATIAL])

    _copy_targets(conn, targets)
    sql = """
    WITH base AS (
        SELECT
            t.post_id, t.slot_start, t.target, t.meter_type,
            t.local_hour, t.local_date, m.location
        FROM _ml_targets t
        INNER JOIN parking_meters m ON m.post_id = t.post_id
        WHERE m.location IS NOT NULL
    ), temporal AS (
        SELECT
            b.*,
            p1.paid_availability_probability AS lag1,
            p2.paid_availability_probability AS lag2,
            p3.paid_availability_probability AS lag3,
            p6.paid_availability_probability AS lag6,
            p24.paid_availability_probability AS lag24,
            p168.paid_availability_probability AS lag168,
            COALESCE(p1.transaction_count, 0) AS tx1,
            COALESCE(p24.transaction_count, 0) AS tx24
        FROM base b
        INNER JOIN parking_state_hourly p1
          ON p1.post_id=b.post_id AND p1.slot_start=b.slot_start-INTERVAL '1 hour'
        INNER JOIN parking_state_hourly p2
          ON p2.post_id=b.post_id AND p2.slot_start=b.slot_start-INTERVAL '2 hours'
        INNER JOIN parking_state_hourly p3
          ON p3.post_id=b.post_id AND p3.slot_start=b.slot_start-INTERVAL '3 hours'
        INNER JOIN parking_state_hourly p6
          ON p6.post_id=b.post_id AND p6.slot_start=b.slot_start-INTERVAL '6 hours'
        INNER JOIN parking_state_hourly p24
          ON p24.post_id=b.post_id AND p24.slot_start=b.slot_start-INTERVAL '24 hours'
        INNER JOIN parking_state_hourly p168
          ON p168.post_id=b.post_id AND p168.slot_start=b.slot_start-INTERVAL '168 hours'
    ),
    neighbors AS (
        SELECT
            t.post_id, t.slot_start, n.post_id AS neighbor_post_id,
            ST_Distance(n.location, t.location) AS distance_m
        FROM temporal t
        CROSS JOIN LATERAL (
            SELECT pm.post_id, pm.location
            FROM parking_meters pm
            WHERE pm.location IS NOT NULL
              AND pm.post_id <> t.post_id
              AND ST_DWithin(pm.location, t.location, :max_distance_m)
            ORDER BY pm.location <-> t.location
            LIMIT :neighbor_k
        ) n
    ),
    neighbor_state AS (
        SELECT
            n.post_id, n.slot_start, n.neighbor_post_id, n.distance_m,
            p1.paid_availability_probability AS a1,
            p2.paid_availability_probability AS a2,
            p3.paid_availability_probability AS a3,
            COALESCE(p1.transaction_count,0) AS tx1
        FROM neighbors n
        LEFT JOIN parking_state_hourly p1
          ON p1.post_id=n.neighbor_post_id AND p1.slot_start=n.slot_start-INTERVAL '1 hour'
        LEFT JOIN parking_state_hourly p2
          ON p2.post_id=n.neighbor_post_id AND p2.slot_start=n.slot_start-INTERVAL '2 hours'
        LEFT JOIN parking_state_hourly p3
          ON p3.post_id=n.neighbor_post_id AND p3.slot_start=n.slot_start-INTERVAL '3 hours'
    ),
    spatial AS (
        SELECT
            post_id, slot_start,
            count(a1) AS neighbor_count,
            avg(a1) AS neighbor_mean_availability,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY a1) AS neighbor_median_availability,
            min(a1) AS neighbor_min_availability,
            max(a1) AS neighbor_max_availability,
            stddev_pop(a1) AS neighbor_std_availability,
            avg(CASE WHEN a1 < 0.5 THEN 1.0 ELSE 0.0 END) AS neighbor_occupied_fraction,
            avg(tx1) AS neighbor_mean_transactions,
            avg(a1-a2) AS neighbor_mean_delta1,
            avg(a1-a3) AS neighbor_mean_delta3,
            sum(a1 / GREATEST(distance_m, 5.0)) /
              NULLIF(sum(1.0 / GREATEST(distance_m, 5.0)),0) AS neighbor_distance_weighted_availability
        FROM neighbor_state
        WHERE a1 IS NOT NULL
        GROUP BY post_id, slot_start
    )
    SELECT
        t.post_id, t.slot_start, t.target,
        t.lag1, t.lag2, t.lag3, t.lag6, t.lag24, t.lag168,
        t.tx1, t.tx24,
        (t.lag1+t.lag2+t.lag3)/3.0 AS roll3,
        (t.lag1+t.lag2+t.lag3+t.lag6+t.lag24)/5.0 AS roll24,
        t.lag1-t.lag2 AS delta1,
        t.lag1-t.lag3 AS delta3,
        (t.lag1-t.lag2)-(t.lag2-t.lag3) AS acceleration,
        t.tx1-t.tx24 AS tx_delta1,
        t.tx1/GREATEST(t.tx24,1.0) AS tx_ratio_3h,
        sin(2*pi()*t.local_hour/24.0) AS hour_sin,
        cos(2*pi()*t.local_hour/24.0) AS hour_cos,
        sin(2*pi()*((extract(isodow from t.local_date)::int)-1)/7.0) AS weekday_sin,
        cos(2*pi()*((extract(isodow from t.local_date)::int)-1)/7.0) AS weekday_cos,
        CASE WHEN t.meter_type='MS' THEN 1.0 ELSE 0.0 END AS is_ms,
        COALESCE(s.neighbor_count,0),
        COALESCE(s.neighbor_mean_availability,t.lag1),
        COALESCE(s.neighbor_median_availability,t.lag1),
        COALESCE(s.neighbor_min_availability,t.lag1),
        COALESCE(s.neighbor_max_availability,t.lag1),
        COALESCE(s.neighbor_std_availability,0),
        COALESCE(s.neighbor_occupied_fraction,CASE WHEN t.lag1<0.5 THEN 1.0 ELSE 0.0 END),
        COALESCE(s.neighbor_mean_transactions,0),
        COALESCE(s.neighbor_mean_delta1,0),
        COALESCE(s.neighbor_mean_delta3,0),
        COALESCE(s.neighbor_distance_weighted_availability,t.lag1)
    FROM temporal t
    LEFT JOIN spatial s USING (post_id, slot_start)
    """
    rows = conn.run(sql, max_distance_m=config.max_distance_m, neighbor_k=config.neighbor_k)
    conn.run("DROP TABLE IF EXISTS _ml_targets")

    cols = ["post_id", "slot_start", "target", *FEATURES_SPATIAL]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df[FEATURES_SPATIAL] = df[FEATURES_SPATIAL].replace([np.inf, -np.inf], np.nan)
    df[FEATURES_SPATIAL] = df[FEATURES_SPATIAL].fillna(0.0)
    return df


def sample_targets(conn, start_slot: datetime, end_slot: datetime, limit_rows: int, seed: int) -> list[tuple]:
    """Deterministically sample target rows in a chronological window."""
    sql = """
    SELECT post_id, slot_start, paid_availability_probability, meter_type,
           local_hour, local_date
    FROM parking_state_hourly
    WHERE slot_start >= :start_slot AND slot_start <= :end_slot
      AND mod(abs(hashtext(post_id || '|' || slot_start::text || :seed::text)), 1000) < 50
    ORDER BY slot_start, post_id
    LIMIT :limit_rows
    """
    return conn.run(sql, start_slot=start_slot, end_slot=end_slot, seed=str(seed), limit_rows=limit_rows)
