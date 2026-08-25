"""Production forecasting pipeline for parking availability.

Core components:
- Feature construction with optional forecast overrides (for recursive T+2+)
- Forecast persistence (INSERT INTO parking_state_forecasts)
- Forecast evaluation (compare stored forecasts against observed state)

Architecture for recursive multi-step forecasting:

The model uses 6 availability lags: lag1, lag2, lag3, lag6, lag24, lag168.
For T+N where N > 1, some lags reference slots that have not been observed
yet.  Those lags are supplied by previously-stored forecasts.

Example for T+3 (hours_ahead=3, target = latest + 3h):
  lag1  → T+2 forecast  (hours_ahead=2, not yet observed)
  lag2  → T+1 forecast  (hours_ahead=1, not yet observed)
  lag3  → observed       (latest state)
  lag6  → observed
  lag24 → observed
  l168  → observed

Transaction features (lag1_transactions, lag24_transactions) are ALWAYS
sourced from observed state.  The model was trained with observed
transaction counts and has no mechanism to predict future transactions.
For horizons where the observed slot is unavailable the transaction lag
falls back to 0 (the column default in parking_state_hourly).  This is
documented and honest: we do not fabricate transaction data.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np

from sf_parking.database import connect, transaction

TZ = "America/Los_Angeles"

FEATURES = [
    "lag1_availability", "lag2_availability", "lag3_availability",
    "lag6_availability", "lag24_availability", "lag168_availability",
    "lag1_transactions", "lag24_transactions", "roll3_availability",
    "roll24_availability", "hour_sin", "hour_cos", "weekday_sin",
    "weekday_cos", "is_ms",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "paid_state_lgbm.txt"
DEFAULT_META_PATH = Path(__file__).resolve().parents[2] / "models" / "paid_state_lgbm.meta.json"


# ── lag offset definitions ────────────────────────────────────────────────
_LAG_OFFSETS = {
    "lag1":  1,
    "lag2":  2,
    "lag3":  3,
    "lag6":  6,
    "lag24": 24,
    "lag168": 168,
}


def load_model(model_path: Path | None = None):
    """Load LightGBM model and metadata. Raises on failure."""
    import lightgbm as lgb
    model_path = model_path or DEFAULT_MODEL_PATH
    meta_path = model_path.with_suffix("").with_name(model_path.stem + ".meta.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    model = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def latest_observed_slot(conn) -> datetime:
    """Return the latest *completed* slot_start from parking_state_hourly.

    ``build_hourly_state.py`` materializes every hour of the local day,
    including slots whose local hour has not yet occurred.  Those future
    slots carry zero-transaction rows (availability 1.0) and must not
    be treated as observed data.
    """
    result = conn.run(
        "SELECT max(slot_start) FROM parking_state_hourly "
        "WHERE slot_start <= NOW()"
    )
    if not result or result[0][0] is None:
        raise RuntimeError("parking_state_hourly has no completed slots")
    return result[0][0]


def _ensure_forecast_overrides_temp(conn, overrides: list[dict]) -> None:
    """Create a temp table holding forecast values for recursive lags."""
    conn.run("DROP TABLE IF EXISTS _forecast_overrides")
    conn.run(
        "CREATE TEMP TABLE _forecast_overrides "
        "(post_id text, lag_offset int, predicted_value double precision)"
    )
    if not overrides:
        return
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for o in overrides:
        writer.writerow([o["post_id"], o["lag_offset"], o["predicted_value"]])
    conn.run(
        "COPY _forecast_overrides (post_id, lag_offset, predicted_value) "
        "FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )


def _discover_meters(
    conn,
    slot_utc: datetime,
    forecast_overrides: list[dict] | None = None,
) -> list[tuple]:
    """Find meters with complete lag history for *slot_utc*.

    For T+1 (hours_ahead=1) all lags come from observed state.
    For T+N (N>1) some lags are supplied by *forecast_overrides*.
    """
    _ensure_forecast_overrides_temp(conn, forecast_overrides or [])
    has_overrides = bool(forecast_overrides)

    # For discovery we need ALL six lag slots to be either observed or forecast.
    # T+1: all observed → standard discovery.
    # T+2: lag1 from forecast, rest observed → check observed lags 2,3,6,24,168
    #      plus forecast lag1 exists.
    # General: for each lag, check if offset >= hours_ahead (observed) or
    #          if a forecast override exists.

    if not has_overrides:
        # Pure T+1: all lags observed.  Same as existing predict_paid_state.
        sql = """
        SELECT p1.post_id, p1.meter_type
        FROM parking_state_hourly p1
        INNER JOIN parking_state_hourly p2
          ON p2.post_id = p1.post_id
          AND p2.slot_start = CAST(:slot AS timestamptz) - INTERVAL '2 hours'
        INNER JOIN parking_state_hourly p3
          ON p3.post_id = p1.post_id
          AND p3.slot_start = CAST(:slot AS timestamptz) - INTERVAL '3 hours'
        INNER JOIN parking_state_hourly p6
          ON p6.post_id = p1.post_id
          AND p6.slot_start = CAST(:slot AS timestamptz) - INTERVAL '6 hours'
        INNER JOIN parking_state_hourly p24
          ON p24.post_id = p1.post_id
          AND p24.slot_start = CAST(:slot AS timestamptz) - INTERVAL '24 hours'
        INNER JOIN parking_state_hourly p168
          ON p168.post_id = p1.post_id
          AND p168.slot_start = CAST(:slot AS timestamptz) - INTERVAL '168 hours'
        WHERE p1.slot_start = CAST(:slot AS timestamptz) - INTERVAL '1 hour'
        """
        return conn.run(sql, slot=slot_utc)

    # Multi-step: discover from observed lags + forecast overrides.
    # A meter is eligible if, for every required lag, either:
    #   (a) the lag slot is in parking_state_hourly, OR
    #   (b) a forecast override exists for that lag offset.
    #
    # When the lag-1 slot itself is forecast-only (not in parking_state_hourly),
    # derive the base meter set from the forecast overrides for lag-1.
    sql = """
    WITH required_lags AS (
        SELECT post_id, meter_type
        FROM parking_state_hourly
        WHERE slot_start = CAST(:slot AS timestamptz) - INTERVAL '1 hour'
        UNION
        SELECT fo.post_id,
               COALESCE(
                   (SELECT psh.meter_type FROM parking_state_hourly psh
                    WHERE psh.post_id = fo.post_id
                    ORDER BY psh.slot_start DESC LIMIT 1),
                   'MS'
               ) AS meter_type
        FROM _forecast_overrides fo
        WHERE fo.lag_offset = 1
          AND NOT EXISTS (
              SELECT 1 FROM parking_state_hourly psh
              WHERE psh.slot_start = CAST(:slot AS timestamptz) - INTERVAL '1 hour'
          )
    )
    SELECT rl.post_id, rl.meter_type
    FROM required_lags rl
    -- Check lag-1: observed or forecast
    LEFT JOIN parking_state_hourly obs1
      ON obs1.post_id = rl.post_id
      AND obs1.slot_start = CAST(:slot AS timestamptz) - INTERVAL '1 hour'
    LEFT JOIN _forecast_overrides fo1
      ON fo1.post_id = rl.post_id AND fo1.lag_offset = 1
    -- Check lag-2
    LEFT JOIN parking_state_hourly obs2
      ON obs2.post_id = rl.post_id
      AND obs2.slot_start = CAST(:slot AS timestamptz) - INTERVAL '2 hours'
    LEFT JOIN _forecast_overrides fo2
      ON fo2.post_id = rl.post_id AND fo2.lag_offset = 2
    -- Check lag-3
    LEFT JOIN parking_state_hourly obs3
      ON obs3.post_id = rl.post_id
      AND obs3.slot_start = CAST(:slot AS timestamptz) - INTERVAL '3 hours'
    LEFT JOIN _forecast_overrides fo3
      ON fo3.post_id = rl.post_id AND fo3.lag_offset = 3
    -- Check lag-6
    LEFT JOIN parking_state_hourly obs6
      ON obs6.post_id = rl.post_id
      AND obs6.slot_start = CAST(:slot AS timestamptz) - INTERVAL '6 hours'
    LEFT JOIN _forecast_overrides fo6
      ON fo6.post_id = rl.post_id AND fo6.lag_offset = 6
    -- Check lag-24
    LEFT JOIN parking_state_hourly obs24
      ON obs24.post_id = rl.post_id
      AND obs24.slot_start = CAST(:slot AS timestamptz) - INTERVAL '24 hours'
    LEFT JOIN _forecast_overrides fo24
      ON fo24.post_id = rl.post_id AND fo24.lag_offset = 24
    -- Check lag-168
    LEFT JOIN parking_state_hourly obs168
      ON obs168.post_id = rl.post_id
      AND obs168.slot_start = CAST(:slot AS timestamptz) - INTERVAL '168 hours'
    LEFT JOIN _forecast_overrides fo168
      ON fo168.post_id = rl.post_id AND fo168.lag_offset = 168
    WHERE
      (obs1.post_id IS NOT NULL OR fo1.post_id IS NOT NULL)
      AND (obs2.post_id IS NOT NULL OR fo2.post_id IS NOT NULL)
      AND (obs3.post_id IS NOT NULL OR fo3.post_id IS NOT NULL)
      AND (obs6.post_id IS NOT NULL OR fo6.post_id IS NOT NULL)
      AND (obs24.post_id IS NOT NULL OR fo24.post_id IS NOT NULL)
      AND (obs168.post_id IS NOT NULL OR fo168.post_id IS NOT NULL)
    """
    return conn.run(sql, slot=slot_utc)


def _build_features(
    conn,
    slot_utc: datetime,
    forecast_overrides: list[dict] | None = None,
) -> list[dict]:
    """Construct leakage-safe features for every eligible meter at *slot_utc*.

    When *forecast_overrides* is provided, specified lag values are sourced
    from previously-stored forecasts instead of observed state, enabling
    recursive multi-step forecasting.
    """
    targets = _discover_meters(conn, slot_utc, forecast_overrides)
    if not targets:
        return []

    conn.run("DROP TABLE IF EXISTS _predict_targets")
    conn.run(
        "CREATE TEMP TABLE _predict_targets "
        "(post_id text, meter_type text)"
    )
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(targets)
    conn.run(
        "COPY _predict_targets (post_id, meter_type) FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )

    _ensure_forecast_overrides_temp(conn, forecast_overrides or [])

    # Build lag availability using COALESCE(observed, forecast).
    # Forecast overrides are pivoted into a wide table so each lag offset
    # becomes a column, avoiding correlated subqueries.
    # Transaction lags always use observed values; COALESCE falls back to 0
    # when the observed slot does not yet exist (the column default).
    sql = """
    WITH target AS (
        SELECT CAST(:slot AS timestamptz) AS slot_start
    ),
    overrides_wide AS (
        SELECT
            post_id,
            MAX(CASE WHEN lag_offset = 1   THEN predicted_value END) AS ov1,
            MAX(CASE WHEN lag_offset = 2   THEN predicted_value END) AS ov2,
            MAX(CASE WHEN lag_offset = 3   THEN predicted_value END) AS ov3,
            MAX(CASE WHEN lag_offset = 6   THEN predicted_value END) AS ov6,
            MAX(CASE WHEN lag_offset = 24  THEN predicted_value END) AS ov24,
            MAX(CASE WHEN lag_offset = 168 THEN predicted_value END) AS ov168
        FROM _forecast_overrides
        GROUP BY post_id
    ),
    lags AS (
        SELECT
            t.post_id,
            t.meter_type,
            COALESCE(ov.ov1,  p1.paid_availability_probability) AS lag1,
            COALESCE(ov.ov2,  p2.paid_availability_probability) AS lag2,
            COALESCE(ov.ov3,  p3.paid_availability_probability) AS lag3,
            COALESCE(ov.ov6,  p6.paid_availability_probability) AS lag6,
            COALESCE(ov.ov24, p24.paid_availability_probability) AS lag24,
            COALESCE(ov.ov168, p168.paid_availability_probability) AS lag168,
            COALESCE(p1.transaction_count, 0)   AS tx1,
            COALESCE(p24.transaction_count, 0)  AS tx24
        FROM _predict_targets t
        INNER JOIN target tg ON TRUE
        LEFT JOIN overrides_wide ov ON ov.post_id = t.post_id
        LEFT JOIN parking_state_hourly p1
          ON p1.post_id = t.post_id AND p1.slot_start = tg.slot_start - INTERVAL '1 hour'
        LEFT JOIN parking_state_hourly p2
          ON p2.post_id = t.post_id AND p2.slot_start = tg.slot_start - INTERVAL '2 hours'
        LEFT JOIN parking_state_hourly p3
          ON p3.post_id = t.post_id AND p3.slot_start = tg.slot_start - INTERVAL '3 hours'
        LEFT JOIN parking_state_hourly p6
          ON p6.post_id = t.post_id AND p6.slot_start = tg.slot_start - INTERVAL '6 hours'
        LEFT JOIN parking_state_hourly p24
          ON p24.post_id = t.post_id AND p24.slot_start = tg.slot_start - INTERVAL '24 hours'
        LEFT JOIN parking_state_hourly p168
          ON p168.post_id = t.post_id AND p168.slot_start = tg.slot_start - INTERVAL '168 hours'
    )
    SELECT
        lags.post_id,
        lags.meter_type,
        lags.lag1, lags.lag2, lags.lag3, lags.lag6, lags.lag24, lags.lag168,
        lags.tx1, lags.tx24,
        EXTRACT(DOW FROM tg.slot_start)::int AS dow,
        EXTRACT(HOUR FROM tg.slot_start AT TIME ZONE 'America/Los_Angeles')::int AS local_hour
    FROM lags, target tg
    """
    rows = conn.run(sql, slot=slot_utc)
    conn.run("DROP TABLE IF EXISTS _predict_targets")
    conn.run("DROP TABLE IF EXISTS _forecast_overrides")

    dow_to_iso = {0: 7, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    features = []
    for row in rows:
        hour = int(row[11])
        dow = dow_to_iso.get(int(row[10]), 1)
        lag1 = float(row[2])
        lag2 = float(row[3])
        lag3 = float(row[4])
        lag6 = float(row[5])
        lag24 = float(row[6])
        lag168 = float(row[7])
        is_ms = 1.0 if row[1] == "MS" else 0.0
        features.append({
            "post_id": row[0],
            "meter_type": row[1],
            "lag1_availability": lag1,
            "lag2_availability": lag2,
            "lag3_availability": lag3,
            "lag6_availability": lag6,
            "lag24_availability": lag24,
            "lag168_availability": lag168,
            "lag1_transactions": float(row[8]),
            "lag24_transactions": float(row[9]),
            "roll3_availability": (lag1 + lag2 + lag3) / 3.0,
            "roll24_availability": (lag1 + lag2 + lag3 + lag6 + lag24) / 5.0,
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "weekday_sin": math.sin(2 * math.pi * (dow - 1) / 7.0),
            "weekday_cos": math.cos(2 * math.pi * (dow - 1) / 7.0),
            "is_ms": is_ms,
        })
    return features


def enrich_meters(conn, post_ids: list[str]) -> dict[str, dict]:
    """Fetch parking_meters metadata for the given post_ids."""
    if not post_ids:
        return {}
    conn.run("DROP TABLE IF EXISTS _predict_meters")
    conn.run("CREATE TEMP TABLE _predict_meters (post_id text)")
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for pid in post_ids:
        writer.writerow([pid])
    conn.run(
        "COPY _predict_meters (post_id) FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )
    rows = conn.run("""
        SELECT pm.post_id, pm.parking_space_id, pm.latitude, pm.longitude,
               pm.meter_type, pm.street_name, pm.street_number
        FROM parking_meters pm
        INNER JOIN _predict_meters m ON m.post_id = pm.post_id
    """)
    conn.run("DROP TABLE IF EXISTS _predict_meters")
    return {
        row[0]: {
            "parking_space_id": row[1],
            "latitude": float(row[2]) if row[2] is not None else None,
            "longitude": float(row[3]) if row[3] is not None else None,
            "meter_type": row[4],
            "street_name": row[5],
            "street_number": row[6],
        }
        for row in rows
    }


def store_forecasts(
    conn,
    *,
    target_slot: datetime,
    hours_ahead: int,
    model_version: str,
    model_path: str,
    feature_data_as_of: datetime,
    rows: list[dict],
) -> int:
    """Bulk-insert forecast rows into parking_state_forecasts.

    Uses a staging table + INSERT...SELECT to avoid per-row round trips.
    Each dict in *rows* must have: post_id, predicted_availability.
    Returns the number of rows inserted.
    """
    if not rows:
        return 0
    with transaction(conn):
        conn.run("DROP TABLE IF EXISTS _stage_forecasts")
        conn.run(
            "CREATE TEMP TABLE _stage_forecasts "
            "(post_id text, predicted_availability double precision)"
        )
        buf = StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in rows:
            writer.writerow([row["post_id"], row["predicted_availability"]])
        conn.run(
            "COPY _stage_forecasts (post_id, predicted_availability) "
            "FROM STDIN WITH (FORMAT csv)",
            stream=[buf.getvalue().encode("utf-8")],
        )
        conn.run(
            "INSERT INTO parking_state_forecasts "
            "    (post_id, target_slot, hours_ahead, predicted_availability,"
            "     model_version, model_path, feature_data_as_of) "
            "SELECT s.post_id, CAST(:slot AS timestamptz), :ha,"
            "       s.predicted_availability, :mv, :mp, :fdao "
            "FROM _stage_forecasts s "
            "ON CONFLICT (post_id, target_slot, model_version) DO UPDATE "
            "    SET predicted_availability = EXCLUDED.predicted_availability,"
            "        forecast_generated_at = now(),"
            "        feature_data_as_of = EXCLUDED.feature_data_as_of",
            slot=target_slot,
            ha=hours_ahead,
            mv=model_version,
            mp=model_path,
            fdao=feature_data_as_of,
        )
        conn.run("DROP TABLE IF EXISTS _stage_forecasts")
    return len(rows)


def fetch_unverified_forecasts(conn) -> list[dict]:
    """Return forecasts whose actual value has not yet been recorded."""
    rows = conn.run("""
        SELECT id, post_id, target_slot, hours_ahead,
               predicted_availability, model_version
        FROM parking_state_forecasts
        WHERE actual_availability IS NULL
        ORDER BY target_slot, post_id
    """)
    return [
        {
            "id": r[0],
            "post_id": r[1],
            "target_slot": r[2],
            "hours_ahead": r[3],
            "predicted_availability": r[4],
            "model_version": r[5],
        }
        for r in rows
    ]


def verify_forecasts(conn) -> int:
    """Match unverified forecasts against observed state.  Returns count updated."""
    with transaction(conn):
        result = conn.run("""
            UPDATE parking_state_forecasts f
            SET actual_availability = p.paid_availability_probability,
                actual_observed_at = now()
            FROM parking_state_hourly p
            WHERE f.actual_availability IS NULL
              AND f.post_id = p.post_id
              AND f.target_slot = p.slot_start
        """)
        return result[0][0] if result else 0


def evaluate_verified_forecasts(
    conn,
    *,
    hours_ahead: int | None = None,
    model_version: str | None = None,
) -> dict:
    """Compute metrics for forecasts that have been verified against observed state.

    Returns a dict with overall metrics and per-group breakdowns.
    """
    where_clauses = ["f.actual_availability IS NOT NULL"]
    params: dict[str, object] = {}
    if hours_ahead is not None:
        where_clauses.append("f.hours_ahead = :ha")
        params["ha"] = hours_ahead
    if model_version is not None:
        where_clauses.append("f.model_version = :mv")
        params["mv"] = model_version

    where_sql = " AND ".join(where_clauses)

    sql = f"""
    SELECT
        f.post_id,
        f.target_slot,
        f.hours_ahead,
        f.predicted_availability,
        f.actual_availability,
        f.model_version
    FROM parking_state_forecasts f
    WHERE {where_sql}
    ORDER BY f.target_slot, f.post_id
    """
    rows = conn.run(sql, **params)
    if not rows:
        return {"overall": _empty_metrics(), "by_hour_bucket": {}, "by_day_type": {}, "by_meter_type": {}}

    post_ids = list({r[0] for r in rows})
    meter_meta = enrich_meters(conn, post_ids)

    predictions = np.array([r[3] for r in rows])
    actuals = np.array([r[4] for r in rows])
    target_slots = [r[1] for r in rows]
    local_hours = np.array([
        r[1].astimezone(timezone.utc).hour  # simplified; actual tz needs care
        for r in rows
    ])

    overall = _compute_metrics(actuals, predictions)

    # Group by hour bucket
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    tz_la = ZoneInfo(TZ)
    hour_buckets: dict[str, list[int]] = {b: [] for b in _HOUR_BUCKETS}
    for i, slot in enumerate(target_slots):
        local_h = slot.astimezone(tz_la).hour
        for name, rng in _HOUR_BUCKETS.items():
            if local_h in rng:
                hour_buckets[name].append(i)
                break

    by_hour = {}
    for name, indices in hour_buckets.items():
        idx = np.array(indices, dtype=int)
        if len(idx) > 0:
            by_hour[name] = _compute_metrics(actuals[idx], predictions[idx])

    # Group by meter type
    type_groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        mt = meter_meta.get(r[0], {}).get("meter_type", "SS") or "SS"
        type_groups.setdefault(mt, []).append(i)

    by_type = {}
    for mt, indices in type_groups.items():
        idx = np.array(indices, dtype=int)
        if len(idx) > 0:
            by_type[mt] = _compute_metrics(actuals[idx], predictions[idx])

    # Group by weekday/weekend
    weekday_idx = []
    weekend_idx = []
    for i, slot in enumerate(target_slots):
        local_date = slot.astimezone(tz_la).date()
        if local_date.isocalendar().weekday >= 6:
            weekend_idx.append(i)
        else:
            weekday_idx.append(i)

    by_day_type = {}
    for label, indices in [("weekday", weekday_idx), ("weekend", weekend_idx)]:
        idx = np.array(indices, dtype=int)
        if len(idx) > 0:
            by_day_type[label] = _compute_metrics(actuals[idx], predictions[idx])

    return {
        "overall": overall,
        "by_hour_bucket": by_hour,
        "by_day_type": by_day_type,
        "by_meter_type": by_type,
    }


# ── helpers ──────────────────────────────────────────────────────────────

_HOUR_BUCKETS: dict[str, range] = {
    "overnight":  range(0, 6),
    "morning":    range(6, 12),
    "afternoon":  range(12, 17),
    "evening":    range(17, 22),
    "late_night": range(22, 24),
}


def _empty_metrics() -> dict:
    return {
        "rows": 0,
        "model_mae": float("nan"),
        "model_rmse": float("nan"),
        "coverage": 0.0,
    }


def _compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    n = len(actuals)
    if n == 0:
        return _empty_metrics()
    err = predictions - actuals
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {
        "rows": n,
        "model_mae": mae,
        "model_rmse": rmse,
        "coverage": 1.0,
    }
