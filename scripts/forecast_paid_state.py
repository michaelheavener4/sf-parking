"""Generate parking availability forecasts and store them in PostgreSQL.

Determines the latest observed state, constructs leakage-safe features,
runs the persisted LightGBM model, and writes predictions to the
``parking_state_forecasts`` table.

For T+1 (hours_ahead=1) all lag features come from observed state.
For T+N (N>1) previously-stored T+1…T+(N-1) forecasts supply the
appropriate lags, enabling recursive multi-step forecasting.

Usage::

    python scripts/forecast_paid_state.py --hours-ahead 1
    python scripts/forecast_paid_state.py --hours-ahead 2
    python scripts/forecast_paid_state.py --hours-ahead 24 --top 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect
from sf_parking.forecasting import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    FEATURES,
    TZ,
    _build_features,
    enrich_meters,
    latest_observed_slot,
    load_model,
    store_forecasts,
)


def _collect_forecast_overrides(
    conn,
    target_slot: datetime,
    hours_ahead: int,
    *,
    db_latest: datetime | None = None,
) -> list[dict]:
    """Gather forecast overrides for recursive multi-step prediction.

    For a target at T+N, we need lags at T+(N-1), T+(N-2), …, T+1.
    Each of those may itself be a forecast (not yet observed).

    Accepts an optional *db_latest* to avoid redundant DB queries for
    the latest_observed_slot across multiple horizon iterations.
    """
    if hours_ahead <= 1:
        return []

    if db_latest is None:
        db_latest = latest_observed_slot(conn)

    # The lag offsets the model needs: 1, 2, 3, 6, 24, 168.
    required_offsets = [1, 2, 3, 6, 24, 168]

    # Determine which lag slots are in the future (need overrides).
    future_lag_slots = []
    for offset in required_offsets:
        lag_slot = target_slot - timedelta(hours=offset)
        hours_until_lag = (lag_slot - db_latest).total_seconds() / 3600
        if hours_until_lag > 0:
            future_lag_slots.append(lag_slot)

    if not future_lag_slots:
        return []

    # Batch-fetch all overrides for future lag slots in one query.
    placeholders = ", ".join(f":ls{i}" for i in range(len(future_lag_slots)))
    params = {f"ls{i}": slot for i, slot in enumerate(future_lag_slots)}
    result = conn.run(f"""
        SELECT post_id, target_slot, predicted_availability
        FROM parking_state_forecasts
        WHERE target_slot IN ({placeholders})
    """, **params)

    # Map lag_slot → offset and build override list.
    slot_to_offset = {
        target_slot - timedelta(hours=offset): offset
        for offset in required_offsets
    }
    overrides: list[dict] = []
    for row in result:
        offset = slot_to_offset.get(row[1])
        if offset is not None:
            overrides.append({
                "post_id": row[0],
                "lag_offset": offset,
                "predicted_value": float(row[2]),
            })

    return overrides


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate parking availability forecasts and store them.",
    )
    p.add_argument(
        "--hours-ahead", type=int, default=1, choices=range(1, 25),
        metavar="[1-24]",
        help="How many hours ahead to forecast (default: 1)",
    )
    p.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path to saved LightGBM model",
    )
    p.add_argument(
        "--top", type=int, default=0,
        help="Show top N predictions after storing (0 = none)",
    )
    args = p.parse_args()
    started = time.monotonic()

    # ── load model ──────────────────────────────────────────────────────
    try:
        model, meta = load_model(args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model_version = meta.get("model_version", "unknown")
    model_path_str = str(args.model.resolve())

    # ── connect and determine target ────────────────────────────────────
    conn = connect()
    try:
        db_latest = latest_observed_slot(conn)
        target_slot = db_latest + timedelta(hours=args.hours_ahead)
        feature_data_as_of = db_latest

        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        tz_la = ZoneInfo(TZ)
        target_local = target_slot.astimezone(tz_la)
        latest_local = db_latest.astimezone(tz_la)

        print("SF PARKING — FORECAST GENERATION")
        print("══════════════════════════════════════════════════════════════")
        print(f"  Latest observed : {db_latest.isoformat()} ({latest_local.strftime('%Y-%m-%d %H:%M %Z')})")
        print(f"  Forecast target : {target_slot.isoformat()} ({target_local.strftime('%Y-%m-%d %H:%M %Z')})")
        print(f"  Hours ahead     : {args.hours_ahead}")
        print(f"  Model version   : {model_version}")
        print(f"  Model features  : {len(meta.get('features', []))}")
        print()

        # ── validate feasibility ────────────────────────────────────────
        lag1_slot = target_slot - timedelta(hours=1)
        if args.hours_ahead == 1 and lag1_slot > db_latest:
            print(
                f"ERROR: Target {target_slot.isoformat()} is beyond the "
                f"latest data ({db_latest.isoformat()}).  "
                f"The required lag-1 slot ({lag1_slot.isoformat()}) does not exist.",
                file=sys.stderr,
            )
            return 1

        # ── gather forecast overrides for recursive steps ───────────────
        overrides = _collect_forecast_overrides(conn, target_slot, args.hours_ahead)
        if args.hours_ahead > 1 and not overrides:
            # Check if we have the required forecasts stored.
            required_offsets = [1, 2, 3, 6, 24, 168]
            missing = []
            for offset in required_offsets:
                lag_slot = target_slot - timedelta(hours=offset)
                hours_until_lag = (lag_slot - db_latest).total_seconds() / 3600
                if hours_until_lag > 0:
                    count = conn.run("""
                        SELECT count(*) FROM parking_state_forecasts
                        WHERE target_slot = :lag_slot
                    """, lag_slot=lag_slot)
                    if count[0][0] == 0:
                        missing.append(f"lag-{offset} ({lag_slot.isoformat()})")
            if missing:
                print(
                    f"ERROR: Recursive T+{args.hours_ahead} requires stored "
                    f"forecasts for: {', '.join(missing)}.\n"
                    "       Generate shorter-horizon forecasts first.",
                    file=sys.stderr,
                )
                return 1

        # ── build features and predict ──────────────────────────────────
        print("Constructing features (leakage-safe: only prior-state data)...")
        features = _build_features(conn, target_slot, overrides)
        if not features:
            print("No meters found with sufficient prior-state history for this slot.")
            return 0

        print(f"Meters with complete history: {len(features):,}")

        # Run model predictions
        df = pd.DataFrame(features)
        feature_cols = [c for c in FEATURES if c in df.columns]
        preds = np.clip(model.predict(df[feature_cols]), 0.0, 1.0)
        df["predicted_availability"] = preds

        # Enrich with metadata
        post_ids = df["post_id"].tolist()
        meta_map = enrich_meters(conn, post_ids)

        df["parking_space_id"] = df["post_id"].map(
            lambda pid: meta_map.get(pid, {}).get("parking_space_id")
        )
        df["latitude"] = df["post_id"].map(
            lambda pid: meta_map.get(pid, {}).get("latitude")
        )
        df["longitude"] = df["post_id"].map(
            lambda pid: meta_map.get(pid, {}).get("longitude")
        )

        # ── store forecasts ─────────────────────────────────────────────
        store_rows = [
            {
                "post_id": row["post_id"],
                "predicted_availability": float(row["predicted_availability"]),
            }
            for _, row in df.iterrows()
        ]
        stored = store_forecasts(
            conn,
            target_slot=target_slot,
            hours_ahead=args.hours_ahead,
            model_version=model_version,
            model_path=model_path_str,
            feature_data_as_of=feature_data_as_of,
            rows=store_rows,
        )
        print(f"\nStored {stored:,} forecasts → parking_state_forecasts")
        print(f"  target_slot  = {target_slot.isoformat()}")
        print(f"  hours_ahead  = {args.hours_ahead}")
        print(f"  model_version = {model_version}")

        # ── summary stats ───────────────────────────────────────────────
        avg = df["predicted_availability"].mean()
        high = (df["predicted_availability"] >= 0.8).sum()
        low = (df["predicted_availability"] <= 0.2).sum()
        print(f"\n  Average predicted availability: {avg*100:.1f}%")
        print(f"  High availability (>=80%): {high:,} meters")
        print(f"  Low  availability (<=20%): {low:,} meters")

        # ── optional top-N display ──────────────────────────────────────
        if args.top > 0:
            df_sorted = df.sort_values("predicted_availability", ascending=False).head(args.top)
            print(f"\n{'='*90}")
            print(f"TOP {len(df_sorted)} PREDICTED AVAILABILITY — {target_local.strftime('%Y-%m-%d %H:%M %Z')}")
            print(f"{'='*90}")
            print(
                f"{'Rank':>4}  {'post_id':<16} {'avail%':>6}  {'lat':>9} {'lon':>10}  {'type':<6}"
            )
            print("-" * 90)
            for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
                lat_str = f"{row['latitude']:.5f}" if row["latitude"] is not None else "N/A"
                lon_str = f"{row['longitude']:.5f}" if row["longitude"] is not None else "N/A"
                mtype = row["meter_type"] or "N/A"
                print(
                    f"{rank:4d}  {row['post_id']:<16} "
                    f"{row['predicted_availability']*100:5.1f}%  "
                    f"{lat_str:>9} {lon_str:>10}  {mtype:<6}"
                )

        elapsed = int(time.monotonic() - started)
        print(f"\nForecast complete — elapsed {elapsed}s")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
