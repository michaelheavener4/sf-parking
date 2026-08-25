"""Predict parking availability for all meters at a given local date/hour.

Loads a persisted LightGBM model and constructs identical features to those
used during training, ensuring no timestamp leakage: every lag feature
references only rows with ``slot_start`` strictly before the prediction target.

The target slot may be in the future (up to one hour after the latest
available observation).  Meter eligibility is discovered from the lag
tables, so no existing target-row is required.

Usage::

    python scripts/predict_paid_state.py --date 2026-08-20 --hour 14
    python scripts/predict_paid_state.py --date 2026-08-20 --hour 14 --top 20
    python scripts/predict_paid_state.py --date 2026-08-20 --hour 14 --lat 37.78 --lon -122.41 --radius 500
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect

TZ = "America/Los_Angeles"
FEATURES = [
    "lag1_availability", "lag2_availability", "lag3_availability",
    "lag6_availability", "lag24_availability", "lag168_availability",
    "lag1_transactions", "lag24_transactions", "roll3_availability",
    "roll24_availability", "hour_sin", "hour_cos", "weekday_sin",
    "weekday_cos", "is_ms",
]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.txt"
DEFAULT_META_PATH = Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.meta.json"


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _target_slot_utc(target_date: datetime, tz_name: str) -> datetime:
    """Convert a naive local datetime to an aware UTC slot_start."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    local = target_date.replace(tzinfo=ZoneInfo(tz_name))
    return local.astimezone(timezone.utc)


def _load_model(model_path: Path):
    """Load LightGBM model and metadata.  Exit with a clear message on failure."""
    import lightgbm as lgb
    meta_path = model_path.with_suffix("").with_name(model_path.stem + ".meta.json")
    if not model_path.exists():
        print(
            f"ERROR: Model not found at {model_path}\n"
            "       Train the model first with:\n"
            "         python scripts/benchmark_paid_state_lgbm_chunked.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not meta_path.exists():
        print(
            f"ERROR: Metadata not found at {meta_path}\n"
            "       Train the model first with:\n"
            "         python scripts/benchmark_paid_state_lgbm_chunked.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    model = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def _predict_features(conn, slot_utc: datetime) -> list[dict]:
    """Construct leakage-safe features for every meter at *slot_utc*.

    All lag and transaction joins reference rows strictly BEFORE *slot_utc*.
    Only meters with complete lag history (all 6 lags present) are returned.

    The meter list is discovered from the lag tables (not from a target-row
    lookup), so the target slot may be in the future and need not exist in
    ``parking_state_hourly``.
    """
    # Discover meters that have complete lag history at slot_utc.
    # All six lag timestamps must exist for a post_id to be eligible.
    sql_discover = """
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
    targets = conn.run(sql_discover, slot=slot_utc)
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

    # All JOINs use slot_start = tg.slot_start - INTERVAL — strictly historical.
    # The target CTE supplies the timestamptz for interval arithmetic; no row
    # at tg.slot_start is read from parking_state_hourly.
    sql_features = """
    WITH target AS (
        SELECT CAST(:slot AS timestamptz) AS slot_start
    ),
    lags AS (
        SELECT
            t.post_id,
            t.meter_type,
            p1.paid_availability_probability  AS lag1,
            p2.paid_availability_probability  AS lag2,
            p3.paid_availability_probability  AS lag3,
            p6.paid_availability_probability  AS lag6,
            p24.paid_availability_probability AS lag24,
            p168.paid_availability_probability AS lag168,
            p1.transaction_count  AS tx1,
            p24.transaction_count AS tx24
        FROM _predict_targets t
        INNER JOIN target tg ON TRUE
        INNER JOIN parking_state_hourly p1
          ON p1.post_id = t.post_id AND p1.slot_start = tg.slot_start - INTERVAL '1 hour'
        INNER JOIN parking_state_hourly p2
          ON p2.post_id = t.post_id AND p2.slot_start = tg.slot_start - INTERVAL '2 hours'
        INNER JOIN parking_state_hourly p3
          ON p3.post_id = t.post_id AND p3.slot_start = tg.slot_start - INTERVAL '3 hours'
        INNER JOIN parking_state_hourly p6
          ON p6.post_id = t.post_id AND p6.slot_start = tg.slot_start - INTERVAL '6 hours'
        INNER JOIN parking_state_hourly p24
          ON p24.post_id = t.post_id AND p24.slot_start = tg.slot_start - INTERVAL '24 hours'
        INNER JOIN parking_state_hourly p168
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
    rows = conn.run(sql_features, slot=slot_utc)
    conn.run("DROP TABLE IF EXISTS _predict_targets")

    # Determine weekday from DOW (0=Sun..6=Sat → ISO 1=Mon..7=Sun)
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


def _enrich_metadata(conn, post_ids: list[str]) -> dict[str, dict]:
    """Fetch latitude, longitude, street_name, street_number from parking_meters."""
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
        SELECT pm.post_id, pm.latitude, pm.longitude,
               pm.street_name, pm.street_number
        FROM parking_meters pm
        INNER JOIN _predict_meters m ON m.post_id = pm.post_id
    """)
    conn.run("DROP TABLE IF EXISTS _predict_meters")
    return {
        row[0]: {
            "latitude": float(row[1]) if row[1] is not None else None,
            "longitude": float(row[2]) if row[2] is not None else None,
            "street_name": row[3],
            "street_number": row[4],
        }
        for row in rows
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Predict parking availability for a given local date/hour."
    )
    p.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format (LA local time)")
    p.add_argument("--hour", required=True, type=int, help="Target hour in 0-23 (LA local time)")
    p.add_argument("--lat", type=float, default=None, help="Filter meters within --radius of this latitude")
    p.add_argument("--lon", type=float, default=None, help="Filter meters within --radius of this longitude")
    p.add_argument("--radius", type=float, default=1000.0, help="Radius in metres for geographic filter")
    p.add_argument("--top", type=int, default=50, help="Number of top predictions to display")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to saved LightGBM model")
    args = p.parse_args()

    # Parse target local time
    try:
        target_naive = datetime.strptime(f"{args.date} {args.hour:02d}:00", "%Y-%m-%d %H:%M")
    except ValueError:
        print("ERROR: Invalid date or hour.  Use --date YYYY-MM-DD --hour HH", file=sys.stderr)
        raise SystemExit(1)

    slot_utc = _target_slot_utc(target_naive, TZ)

    print(f"Target local time: {args.date} {args.hour:02d}:00 {TZ}")
    print(f"Target UTC slot:   {slot_utc.isoformat()}")

    model, meta = _load_model(args.model)
    print(f"Model version: {meta.get('model_version', 'unknown')}")
    print(f"Model features: {len(meta.get('features', []))}")

    conn = connect()
    try:
        # Show data range and validate prediction feasibility.
        bounds = conn.run(
            "SELECT min(slot_start), max(slot_start) FROM parking_state_hourly"
        )
        if bounds and bounds[0][0] is not None:
            db_first, db_last = bounds[0]
            print(f"Database range: {db_first} .. {db_last}")
            lag1_slot = slot_utc - timedelta(hours=1)
            lag168_slot = slot_utc - timedelta(hours=168)
            if lag1_slot > db_last:
                print(
                    f"ERROR: Target slot {slot_utc.isoformat()} is too far "
                    f"beyond the latest data ({db_last}). The required lag-1 "
                    f"slot ({lag1_slot.isoformat()}) does not exist. "
                    "Direct single-step prediction is not possible; "
                    "recursive / multi-step forecasting is required.",
                    file=sys.stderr,
                )
                conn.close()
                return 1
            if lag168_slot < db_first:
                print(
                    f"WARNING: lag-168 slot {lag168_slot.isoformat()} is "
                    f"before the earliest data ({db_first}). Some meters "
                    "may lack the full 7-day history.",
                    file=sys.stderr,
                )
        else:
            print("WARNING: parking_state_hourly is empty.", file=sys.stderr)

        print("\nConstructing features (leakage-safe: only prior-state data)...")
        features = _predict_features(conn, slot_utc)
        if not features:
            print("No meters found with sufficient prior-state history for this slot.")
            return 0

        print(f"Meters with complete history: {len(features):,}")

        # Enrich with metadata from parking_meters
        post_ids = [f["post_id"] for f in features]
        meta_map = _enrich_metadata(conn, post_ids)

        # Run predictions
        df = pd.DataFrame(features)
        feature_cols = [c for c in FEATURES if c in df.columns]
        preds = np.clip(model.predict(df[feature_cols]), 0.0, 1.0)
        df["predicted_availability"] = preds

        # Merge metadata
        for col in ("latitude", "longitude", "street_name", "street_number"):
            df[col] = df["post_id"].map(lambda pid, c=col: meta_map.get(pid, {}).get(c))

        # Geographic filter
        if args.lat is not None and args.lon is not None:
            df["distance_m"] = df.apply(
                lambda row: (
                    _haversine_m(args.lat, args.lon, row["latitude"], row["longitude"])
                    if row["latitude"] is not None and row["longitude"] is not None
                    else float("inf")
                ),
                axis=1,
            )
            df = df[df["distance_m"] <= args.radius].copy()
            print(f"Meters within {args.radius:.0f}m of ({args.lat:.4f}, {args.lon:.4f}): {len(df):,}")
        else:
            df["distance_m"] = None

        if df.empty:
            print("No meters match the specified filters.")
            return 0

        # Sort by predicted availability (highest first)
        df = df.sort_values("predicted_availability", ascending=False).head(args.top)

        # Display results
        print(f"\n{'='*90}")
        print(f"TOP {len(df)} PREDICTED AVAILABILITY — {args.date} {args.hour:02d}:00 {TZ}")
        print(f"{'='*90}")
        print(
            f"{'Rank':>4}  {'post_id':<16} {'avail%':>6}  {'lat':>9} {'lon':>10}  "
            f"{'street':<30} {'type':<6}"
        )
        print("-" * 90)
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            lat_str = f"{row['latitude']:.5f}" if row["latitude"] is not None else "N/A"
            lon_str = f"{row['longitude']:.5f}" if row["longitude"] is not None else "N/A"
            street = f"{row['street_number'] or ''} {row['street_name'] or ''}".strip() or "N/A"
            mtype = row["meter_type"] or "N/A"
            print(
                f"{rank:4d}  {row['post_id']:<16} {row['predicted_availability']*100:5.1f}%  "
                f"{lat_str:>9} {lon_str:>10}  {street:<30} {mtype:<6}"
            )

        print(f"\n{'='*90}")
        print(f"Total meters predicted: {len(df):,}")
        avg_avail = df["predicted_availability"].mean()
        print(f"Average predicted availability: {avg_avail*100:.1f}%")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
