"""Leakage-safe held-out evaluation of the persisted LightGBM model, broken
down by operational parking periods.

Reuses the benchmark's feature construction (same temp-table pipeline, same
INNER JOINs, same leakage guarantees) without modifying or re-training the
model.  The test set is the same held-out final week used by the benchmark.

Usage::

    python scripts/evaluate_paid_state_by_period.py
    python scripts/evaluate_paid_state_by_period.py --test-days 7 --sample-per-day 14500
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from datetime import date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect

# ── import shared utilities from benchmark script ────────────────────────
_BENCH = Path(__file__).resolve().parent / "benchmark_paid_state_lgbm_chunked.py"
_spec = importlib.util.spec_from_file_location("_bench", _BENCH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FEATURES = _mod.FEATURES
TZ = _mod.TZ
build_features = _mod.build_features
metric_arrays = _mod.metric_arrays
run_query = _mod.run_query
sample_day_targets = _mod.sample_day_targets

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.txt"
DEFAULT_META_PATH = Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.meta.json"

# ── hour bucket definitions ──────────────────────────────────────────────
_HOUR_BUCKETS: dict[str, range] = {
    "overnight":  range(0, 6),    # 00-05
    "morning":    range(6, 12),   # 06-11
    "afternoon":  range(12, 17),  # 12-16
    "evening":    range(17, 22),  # 17-21
    "late_night": range(22, 24),  # 22-23
}


def hour_bucket(local_hour: int) -> str:
    """Map a local hour (0-23) to an operational period name."""
    for name, rng in _HOUR_BUCKETS.items():
        if local_hour in rng:
            return name
    return "unknown"


def is_weekend(local_date: date) -> bool:
    """True for Saturday (6) and Sunday (7) in ISO weekday convention."""
    return local_date.isocalendar().weekday >= 6


def group_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Compute model MAE, RMSE, persistence MAE/RMSE, gain, and relative %."""
    persist = np.roll(y, 1)  # not used; caller provides persistence directly
    model_mae, model_rmse = metric_arrays(y, pred)
    return {"model_mae": model_mae, "model_rmse": model_rmse}


def group_report(
    y: np.ndarray,
    pred: np.ndarray,
    persistence: np.ndarray,
) -> dict:
    """Full metrics dict for a group of rows."""
    n = len(y)
    if n == 0:
        return {
            "rows": 0,
            "model_mae": float("nan"),
            "model_rmse": float("nan"),
            "persist_mae": float("nan"),
            "persist_rmse": float("nan"),
            "gain_mae": float("nan"),
            "rel_improvement_pct": float("nan"),
        }
    model_mae, model_rmse = metric_arrays(y, pred)
    persist_mae, persist_rmse = metric_arrays(y, persistence)
    gain_mae = persist_mae - model_mae
    rel_pct = (gain_mae / persist_mae * 100.0) if persist_mae > 0 else 0.0
    return {
        "rows": n,
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "persist_mae": persist_mae,
        "persist_rmse": persist_rmse,
        "gain_mae": gain_mae,
        "rel_improvement_pct": rel_pct,
    }


def _print_section(title: str, rows: list[dict], col_width: int = 14) -> None:
    """Pretty-print a table of group metrics."""
    hdr = (
        f"{'Group':<16} {'Rows':>8} {'Model MAE':>{col_width}} {'Model RMSE':>{col_width}} "
        f"{'Persist MAE':>{col_width}} {'Persist RMSE':>{col_width}} "
        f"{'Gain MAE':>{col_width}} {'Rel %':>{8}}"
    )
    print(f"\n{'─' * len(hdr)}")
    print(f"  {title}")
    print(f"{'─' * len(hdr)}")
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        print(
            f"  {r['label']:<14} {r['rows']:>8,} "
            f"{r['model_mae']:>{col_width}.6f} {r['model_rmse']:>{col_width}.6f} "
            f"{r['persist_mae']:>{col_width}.6f} {r['persist_rmse']:>{col_width}.6f} "
            f"{r['gain_mae']:>{col_width}.6f} {r['rel_improvement_pct']:>7.2f}%"
        )
    print("─" * len(hdr))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate held-out LightGBM predictions by operational period.",
    )
    p.add_argument("--test-days", type=int, default=7,
                   help="Number of held-out test days (default: 7)")
    p.add_argument("--sample-per-day", type=int, default=14_500,
                   help="Max rows sampled per test day")
    p.add_argument("--feature-chunk-size", type=int, default=10_000,
                   help="Rows per feature-extraction chunk")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Path to saved LightGBM model")
    p.add_argument("--seed", type=int, default=44,
                   help="Hash seed for row sampling")
    args = p.parse_args()
    started = time.monotonic()

    import lightgbm as lgb
    meta_path = args.model.with_suffix("").with_name(args.model.stem + ".meta.json")
    if not args.model.exists():
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        return 1
    if not meta_path.exists():
        print(f"ERROR: Metadata not found at {meta_path}", file=sys.stderr)
        return 1
    model = lgb.Booster(model_file=str(args.model))

    conn = connect()
    try:
        first, latest = conn.run(
            "SELECT min(slot_start), max(slot_start) FROM parking_state_hourly"
        )[0]
        latest_day = latest.astimezone(timezone.utc).date()
        test_start_day = (latest - timedelta(days=args.test_days - 1)).astimezone(timezone.utc).date()

        print("SF PARKING — HELD-OUT EVALUATION BY OPERATIONAL PERIOD")
        print("══════════════════════════════════════════════════════════════════════")
        print(f"  Database range : {first} .. {latest}")
        print(f"  Test period    : {test_start_day} .. {latest_day} ({args.test_days} days)")
        print(f"  Model features : {len(FEATURES)}")
        print()

        # ── collect test rows day-by-day ────────────────────────────────
        per_day = min(args.sample_per_day, 50_000)
        all_local_hour: list[np.ndarray] = []
        all_local_date: list[np.ndarray] = []
        all_meter_type: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_persist: list[np.ndarray] = []
        total_rows = 0

        day = test_start_day
        while day <= latest_day:
            raw_targets = [
                r for r in sample_day_targets(conn, day, min(50_000, per_day * 2), args.seed)
                if r[1] <= latest
            ]
            # Build a lookup from (post_id, slot_start) → meter_type
            # raw_targets columns: post_id(0), slot_start(1), target(2),
            #                      meter_type(3), local_hour(4), local_date(5)
            mt_lookup = {(r[0], r[1]): r[3] for r in raw_targets}

            features = build_features(
                conn, raw_targets,
                f"test {day}",
                args.feature_chunk_size,
            )
            if not features.empty:
                pred = np.clip(model.predict(features[FEATURES]), 0.0, 1.0)
                y = features["target"].to_numpy(float)
                persistence = features["lag1_availability"].to_numpy(float)

                slot_hours = pd.to_datetime(features["slot_start"], utc=True).dt.tz_convert(TZ).dt.hour
                slot_dates = pd.to_datetime(features["slot_start"], utc=True).dt.tz_convert(TZ).dt.date

                # Recover meter_type from the raw-target lookup.
                mtypes = np.array([
                    mt_lookup.get((pid, ss), "SS")
                    for pid, ss in zip(features["post_id"], features["slot_start"])
                ])

                all_local_hour.append(slot_hours.to_numpy())
                all_local_date.append(slot_dates.to_numpy())
                all_meter_type.append(mtypes)
                all_y.append(y)
                all_pred.append(pred)
                all_persist.append(persistence)
                total_rows += len(y)
                print(f"  test {day}: {len(y):,} rows", flush=True)
            day += timedelta(days=1)

    finally:
        conn.close()

    if total_rows == 0:
        print("No test rows collected.")
        return 1

    local_hours = np.concatenate(all_local_hour)
    local_dates = np.concatenate(all_local_date)
    meter_types = np.concatenate(all_meter_type)
    y_all = np.concatenate(all_y)
    pred_all = np.concatenate(all_pred)
    persist_all = np.concatenate(all_persist)

    # ── overall ─────────────────────────────────────────────────────────
    overall = group_report(y_all, pred_all, persist_all)

    # ── hour buckets ────────────────────────────────────────────────────
    bucket_labels = list(_HOUR_BUCKETS.keys())
    bucket_groups: dict[str, list[int]] = {b: [] for b in bucket_labels}
    for i, h in enumerate(local_hours):
        b = hour_bucket(int(h))
        if b in bucket_groups:
            bucket_groups[b].append(i)

    hour_rows = []
    for b in bucket_labels:
        idx = np.array(bucket_groups[b], dtype=int)
        if len(idx) == 0:
            hour_rows.append({"label": b, **group_report(np.array([]), np.array([]), np.array([]))})
        else:
            hour_rows.append({"label": b, **group_report(y_all[idx], pred_all[idx], persist_all[idx])})

    # ── weekday vs weekend ──────────────────────────────────────────────
    weekday_idx = []
    weekend_idx = []
    for i, d in enumerate(local_dates):
        if is_weekend(d):
            weekend_idx.append(i)
        else:
            weekday_idx.append(i)
    weekday_idx = np.array(weekday_idx, dtype=int)
    weekend_idx = np.array(weekend_idx, dtype=int)

    day_rows = []
    if len(weekday_idx) > 0:
        day_rows.append({"label": "weekday", **group_report(y_all[weekday_idx], pred_all[weekday_idx], persist_all[weekday_idx])})
    else:
        day_rows.append({"label": "weekday", **group_report(np.array([]), np.array([]), np.array([]))})
    if len(weekend_idx) > 0:
        day_rows.append({"label": "weekend", **group_report(y_all[weekend_idx], pred_all[weekend_idx], persist_all[weekend_idx])})
    else:
        day_rows.append({"label": "weekend", **group_report(np.array([]), np.array([]), np.array([]))})

    # ── meter type ──────────────────────────────────────────────────────
    type_set = sorted(set(meter_types))
    type_groups: dict[str, list[int]] = {t: [] for t in type_set}
    for i, t in enumerate(meter_types):
        type_groups[t].append(i)

    type_rows = []
    for t in type_set:
        idx = np.array(type_groups[t], dtype=int)
        type_rows.append({"label": t or "NULL", **group_report(y_all[idx], pred_all[idx], persist_all[idx])})

    # ── print report ────────────────────────────────────────────────────
    print(f"\nTotal held-out rows: {total_rows:,}")
    print(f"Overall — Model MAE: {overall['model_mae']:.6f}  RMSE: {overall['model_rmse']:.6f}")
    print(f"         Persist MAE: {overall['persist_mae']:.6f}  RMSE: {overall['persist_rmse']:.6f}")
    print(f"         Gain vs persistence (MAE): {overall['gain_mae']:.6f}  ({overall['rel_improvement_pct']:.2f}%)")

    _print_section("BY HOUR BUCKET (local time)", hour_rows)
    _print_section("BY DAY TYPE", day_rows)
    _print_section("BY METER TYPE", type_rows)

    # ── highlight the key question ──────────────────────────────────────
    evening = next((r for r in hour_rows if r["label"] == "evening"), None)
    if evening and evening["rows"] > 0:
        print(f"\n{'═' * 80}")
        print(f"  KEY QUESTION: Model value during parking-demand hours (17:00–21:00)")
        print(f"{'═' * 80}")
        print(f"  Evening rows       : {evening['rows']:,}")
        print(f"  Model MAE          : {evening['model_mae']:.6f}")
        print(f"  Persistence MAE    : {evening['persist_mae']:.6f}")
        print(f"  Gain over persist  : {evening['gain_mae']:.6f}")
        print(f"  Relative improvement: {evening['rel_improvement_pct']:.2f}%")
        print(f"{'═' * 80}")

    elapsed = int(time.monotonic() - started)
    print(f"\nEvaluation complete — elapsed {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
