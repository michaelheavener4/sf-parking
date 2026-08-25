"""Comprehensive forecast evaluation: accuracy, calibration, and parking-finder utility.

Parts:
  1. Forecast maturation — match stored forecasts against observed state
  2. Accuracy by horizon — MAE, RMSE, bias, persistence comparison
  3. Calibration — probability bins, Brier score, calibration error
  4. Regression accuracy vs probability quality distinction
  5. Breakdown by hour/day-type/meter-type
  6. User-relevant metric — P(at least one space within R meters)

Usage::

    python scripts/evaluate_parking_probability.py
    python scripts/evaluate_parking_probability.py --model-version 20260825T002130Z
    python scripts/evaluate_parking_probability.py --max-horizon 6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np

from sf_parking.database import connect, transaction

TZ = "America/Los_Angeles"

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ── hour bucket definitions ─────────────────────────────────────────────
HOUR_BUCKETS = {
    "overnight":  range(0, 6),
    "morning":    range(6, 12),
    "afternoon":  range(12, 17),
    "evening":    range(17, 22),
    "late_night": range(22, 24),
}

# ── calibration bins ────────────────────────────────────────────────────
CALIBRATION_BINS = [(i / 10, (i + 1) / 10) for i in range(10)]
CALIBRATION_LABELS = [f"{i*10}–{(i+1)*10}%" for i in range(10)]

# ── evaluation radii ────────────────────────────────────────────────────
EVAL_RADII = [50, 100, 250, 500, 1000]


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — FORECAST MATURATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_forecasts(conn, model_version: str | None = None) -> int:
    """Match unverified forecasts against observed state. Returns count updated."""
    where = "f.actual_availability IS NULL"
    params: dict[str, object] = {}
    if model_version:
        where += " AND f.model_version = :mv"
        params["mv"] = model_version
    with transaction(conn):
        result = conn.run(f"""
            UPDATE parking_state_forecasts f
            SET actual_availability = p.paid_availability_probability,
                actual_observed_at = now()
            FROM parking_state_hourly p
            WHERE {where}
              AND f.post_id = p.post_id
              AND f.target_slot = p.slot_start
        """, **params)
        return result[0][0] if result else 0


def fetch_matured_forecasts(
    conn,
    *,
    model_version: str | None = None,
    max_horizon: int | None = None,
) -> list[dict]:
    """Return all forecasts that have been matched against observed state.

    Each row includes predicted, actual, error terms, and provenance fields.
    """
    where_clauses = ["f.actual_availability IS NOT NULL"]
    params: dict[str, object] = {}
    if model_version:
        where_clauses.append("f.model_version = :mv")
        params["mv"] = model_version
    if max_horizon:
        where_clauses.append("f.hours_ahead <= :mh")
        params["mh"] = max_horizon

    where_sql = " AND ".join(where_clauses)
    rows = conn.run(f"""
        SELECT
            f.post_id,
            f.target_slot,
            f.hours_ahead,
            f.predicted_availability,
            f.actual_availability,
            f.model_version,
            f.forecast_generated_at,
            f.feature_data_as_of,
            m.meter_type
        FROM parking_state_forecasts f
        LEFT JOIN parking_meters m ON m.post_id = f.post_id
        WHERE {where_sql}
        ORDER BY f.target_slot, f.post_id
    """, **params)

    tz_la = ZoneInfo(TZ)
    results = []
    for r in rows:
        target_slot = r[1]
        local_dt = target_slot.astimezone(tz_la)
        pred = float(r[3])
        actual = float(r[4])
        results.append({
            "post_id": r[0],
            "target_slot": target_slot,
            "hours_ahead": int(r[2]),
            "predicted_availability": pred,
            "actual_availability": actual,
            "absolute_error": abs(pred - actual),
            "squared_error": (pred - actual) ** 2,
            "signed_error": pred - actual,
            "model_version": r[5],
            "forecast_generated_at": r[6],
            "feature_data_as_of": r[7],
            "meter_type": r[8] or "SS",
            "local_hour": local_dt.hour,
            "local_date": local_dt.date(),
            "is_weekend": local_dt.date().isocalendar().weekday >= 6,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — ACCURACY BY HORIZON
# ═══════════════════════════════════════════════════════════════════════════

def horizon_metrics(matured: list[dict]) -> dict[int, dict]:
    """Compute per-horizon MAE, RMSE, bias, persistence MAE, and improvement."""
    from collections import defaultdict
    by_horizon: dict[int, list[dict]] = defaultdict(list)
    for r in matured:
        by_horizon[r["hours_ahead"]].append(r)

    metrics = {}
    for ha in sorted(by_horizon):
        rows = by_horizon[ha]
        n = len(rows)
        pred = np.array([r["predicted_availability"] for r in rows])
        actual = np.array([r["actual_availability"] for r in rows])
        err = pred - actual
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        # Persistence: predict the most recent observed value (lag-1 availability).
        # For a fair comparison, persistence = the observed value at the slot
        # immediately before the target (i.e., the last known state).
        # We approximate this using lag1 availability from the features.
        # When we don't have lag1 stored, persistence is approximated as
        # the mean actual availability across all meters (uninformative baseline).
        # A better approach: use the average actual as persistence for T+1
        # and shift for deeper horizons.
        # Actually, for a proper persistence baseline, we use the previous
        # hour's actual value. We can get this from the forecast data:
        # persistence_MAE = mean(|actual[t] - actual[t-1]|)
        # But we don't have actual[t-1] directly. We'll compute persistence
        # as the lag-1 actual for each meter.
        # For now, use: persistence prediction = same as predicting the
        # current observed value (i.e., the forecast for T+1 uses the
        # latest observed state, so persistence for T+N is the latest
        # observed value, which for deep horizons is just the mean).
        # We'll compute persistence per-horizon from the data.
        persist = np.array([r.get("persistence_availability", actual.mean()) for r in rows])
        persist_err = persist - actual
        persist_mae = float(np.mean(np.abs(persist_err)))
        persist_rmse = float(np.sqrt(np.mean(persist_err ** 2)))

        gain_mae = persist_mae - mae
        rel_pct = (gain_mae / persist_mae * 100.0) if persist_mae > 0 else 0.0

        metrics[ha] = {
            "hours_ahead": ha,
            "rows": n,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "persist_mae": persist_mae,
            "persist_rmse": persist_rmse,
            "gain_mae": gain_mae,
            "rel_improvement_pct": rel_pct,
        }
    return metrics


def grouped_horizon_metrics(horizon_m: dict[int, dict]) -> list[dict]:
    """Group horizons into buckets: 1–6, 7–12, 13–18, 19–24."""
    groups = [
        ("T+1–T+6", range(1, 7)),
        ("T+7–T+12", range(7, 13)),
        ("T+13–T+18", range(13, 19)),
        ("T+19–T+24", range(19, 25)),
    ]
    results = []
    for label, hr_range in groups:
        rows_total = 0
        sum_abs_err = 0.0
        sum_sq_err = 0.0
        sum_signed = 0.0
        sum_persist_abs = 0.0
        for ha in hr_range:
            if ha in horizon_m:
                m = horizon_m[ha]
                rows_total += m["rows"]
                sum_abs_err += m["mae"] * m["rows"]
                sum_sq_err += m["rmse"] ** 2 * m["rows"]
                sum_signed += m["bias"] * m["rows"]
                sum_persist_abs += m["persist_mae"] * m["rows"]
        if rows_total > 0:
            mae = sum_abs_err / rows_total
            rmse = math.sqrt(sum_sq_err / rows_total)
            bias = sum_signed / rows_total
            persist_mae = sum_persist_abs / rows_total
            gain = persist_mae - mae
            rel = (gain / persist_mae * 100.0) if persist_mae > 0 else 0.0
        else:
            mae = rmse = bias = persist_mae = gain = rel = float("nan")
        results.append({
            "label": label,
            "rows": rows_total,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "persist_mae": persist_mae,
            "gain_mae": gain,
            "rel_improvement_pct": rel,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calibration_analysis(matured: list[dict]) -> dict:
    """Bin predictions and compute calibration metrics."""
    if not matured:
        return {
            "bins": [{"label": l, "lo": lo, "hi": hi, "n": 0,
                       "mean_predicted": float("nan"), "actual_rate": float("nan"),
                       "calibration_error": float("nan")}
                      for (lo, hi), l in zip(CALIBRATION_BINS, CALIBRATION_LABELS)],
            "n": 0,
            "overall_mean_predicted": 0.0,
            "overall_actual_rate": 0.0,
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
            "max_calibration_error": 0.0,
        }

    pred = np.array([r["predicted_availability"] for r in matured])
    actual = np.array([r["actual_availability"] for r in matured])

    bins = []
    for (lo, hi), label in zip(CALIBRATION_BINS, CALIBRATION_LABELS):
        mask = (pred >= lo) & (pred < hi)
        # Include the upper edge in the last bin
        if hi == 1.0:
            mask = (pred >= lo) & (pred <= hi)
        n = int(mask.sum())
        if n > 0:
            mean_pred = float(pred[mask].mean())
            mean_actual = float(actual[mask].mean())
            cal_error = abs(mean_pred - mean_actual)
        else:
            mean_pred = mean_actual = cal_error = float("nan")
        bins.append({
            "label": label,
            "lo": lo,
            "hi": hi,
            "n": n,
            "mean_predicted": mean_pred,
            "actual_rate": mean_actual,
            "calibration_error": cal_error,
        })

    # Overall metrics
    n = len(pred)
    if n == 0:
        overall_mean_pred = 0.0
        overall_actual_rate = 0.0
        brier = 0.0
        ece = 0.0
        mce = 0.0
    else:
        overall_mean_pred = float(pred.mean())
        overall_actual_rate = float(actual.mean())
        brier = float(np.mean((pred - actual) ** 2))

        # Expected Calibration Error (ECE) — weighted by bin count
        valid_bins = [b for b in bins if b["n"] > 0]
        if valid_bins:
            ece = sum(b["n"] / n * b["calibration_error"] for b in valid_bins)
        else:
            ece = float("nan")

        # Maximum Calibration Error (MCE)
        if valid_bins:
            mce = max(b["calibration_error"] for b in valid_bins)
        else:
            mce = float("nan")

    return {
        "bins": bins,
        "n": n,
        "overall_mean_predicted": overall_mean_pred,
        "overall_actual_rate": overall_actual_rate,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
    }


def calibration_by_horizon(matured: list[dict]) -> dict[int, dict]:
    """Calibration analysis broken down by horizon."""
    from collections import defaultdict
    by_horizon: dict[int, list[dict]] = defaultdict(list)
    for r in matured:
        by_horizon[r["hours_ahead"]].append(r)
    return {ha: calibration_analysis(rows) for ha, rows in sorted(by_horizon.items())}


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — BREAKDOWN BY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

def _hour_bucket_label(local_hour: int) -> str:
    for name, rng in HOUR_BUCKETS.items():
        if local_hour in rng:
            return name
    return "unknown"


def breakdown_by_hour(matured: list[dict]) -> dict[str, dict]:
    """Metrics grouped by local hour bucket."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in matured:
        bucket = _hour_bucket_label(r["local_hour"])
        groups[bucket].append(r)
    return {k: _group_stats(v) for k, v in groups.items()}


def breakdown_by_day_type(matured: list[dict]) -> dict[str, dict]:
    """Metrics grouped by weekday vs weekend."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in matured:
        key = "weekend" if r["is_weekend"] else "weekday"
        groups[key].append(r)
    return {k: _group_stats(v) for k, v in groups.items()}


def breakdown_by_meter_type(matured: list[dict]) -> dict[str, dict]:
    """Metrics grouped by meter type."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in matured:
        groups[r["meter_type"]].append(r)
    return {k: _group_stats(v) for k, v in groups.items()}


def breakdown_by_hour_and_horizon(matured: list[dict]) -> dict[str, dict[int, dict]]:
    """Metrics grouped by hour bucket AND horizon — for the key evening question."""
    from collections import defaultdict
    groups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in matured:
        bucket = _hour_bucket_label(r["local_hour"])
        groups[bucket][r["hours_ahead"]].append(r)
    result = {}
    for bucket, by_ha in sorted(groups.items()):
        result[bucket] = {}
        for ha in sorted(by_ha):
            result[bucket][ha] = _group_stats(by_ha[ha])
    return result


def _group_stats(rows: list[dict]) -> dict:
    """Compute MAE, RMSE, bias for a group of matured forecasts."""
    n = len(rows)
    if n == 0:
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "mean_pred": float("nan"),
                "mean_actual": float("nan")}
    pred = np.array([r["predicted_availability"] for r in rows])
    actual = np.array([r["actual_availability"] for r in rows])
    err = pred - actual
    return {
        "rows": n,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mean_pred": float(pred.mean()),
        "mean_actual": float(actual.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — PARKING FINDER PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points (Haversine)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_forecasts_with_geometry(
    conn,
    *,
    model_version: str | None = None,
    max_horizon: int | None = None,
) -> list[dict]:
    """Fetch matured forecasts with meter geometry for radius analysis."""
    where_clauses = ["f.actual_availability IS NOT NULL"]
    params: dict[str, object] = {}
    if model_version:
        where_clauses.append("f.model_version = :mv")
        params["mv"] = model_version
    if max_horizon:
        where_clauses.append("f.hours_ahead <= :mh")
        params["mh"] = max_horizon

    where_sql = " AND ".join(where_clauses)
    rows = conn.run(f"""
        SELECT
            f.post_id,
            f.target_slot,
            f.hours_ahead,
            f.predicted_availability,
            f.actual_availability,
            m.latitude,
            m.longitude
        FROM parking_state_forecasts f
        INNER JOIN parking_meters m ON m.post_id = f.post_id
        WHERE {where_sql}
          AND m.latitude IS NOT NULL
          AND m.longitude IS NOT NULL
        ORDER BY f.target_slot, f.post_id
    """, **params)

    tz_la = ZoneInfo(TZ)
    results = []
    for r in rows:
        ts = r[1]
        local_dt = ts.astimezone(tz_la)
        results.append({
            "post_id": r[0],
            "target_slot": ts,
            "hours_ahead": int(r[2]),
            "predicted_availability": float(r[3]),
            "actual_availability": float(r[4]),
            "latitude": float(r[5]),
            "longitude": float(r[6]),
            "local_hour": local_dt.hour,
            "is_weekend": local_dt.date().isocalendar().weekday >= 6,
        })
    return results


def evaluate_radius(
    forecast_rows: list[dict],
    *,
    radius_m: int,
    sample_events: int = 2000,
    seed: int = 42,
) -> dict:
    """Evaluate P(at least one space) at a given radius.

    For each unique (target_slot, search_center) event:
    1. Find all meters within radius_m.
    2. Compute naive independence: P = 1 - prod(1 - p_i).
    3. Compare with actual: was at least one space actually available?

    We use each forecasted meter as a potential search center to maximize
    the use of available data.
    """
    rng = np.random.default_rng(seed)

    # Group by target_slot
    from collections import defaultdict
    by_slot: dict[datetime, list[dict]] = defaultdict(list)
    for r in forecast_rows:
        by_slot[r["target_slot"]].append(r)

    total_events = 0
    naive_correct = 0
    naive_pred_sum = 0.0
    naive_actual_sum = 0.0
    naive_squared_err_sum = 0.0
    naive_brier_sum = 0.0

    # For each slot, sample some meters as "search centers"
    slot_list = sorted(by_slot.keys())
    events_per_slot = max(1, sample_events // max(1, len(slot_list)))

    for slot in slot_list:
        meters = by_slot[slot]
        if len(meters) < 2:
            continue
        # Sample search centers
        n_centers = min(events_per_slot, len(meters))
        center_indices = rng.choice(len(meters), size=n_centers, replace=False)

        for ci in center_indices:
            center = meters[ci]
            clat, clon = center["latitude"], center["longitude"]

            # Find meters within radius
            nearby = []
            for m in meters:
                d = _haversine_m(clat, clon, m["latitude"], m["longitude"])
                if d <= radius_m:
                    nearby.append(m)

            if len(nearby) < 1:
                continue

            # Naive independence: P(at least one) = 1 - prod(1 - p_i)
            pred_probs = np.array([m["predicted_availability"] for m in nearby])
            actual_vals = np.array([m["actual_availability"] for m in nearby])

            p_none_pred = np.prod(1.0 - pred_probs)
            p_at_least_one_pred = 1.0 - p_none_pred
            p_at_least_one_actual = 1.0 if actual_vals.max() > 0.5 else 0.0

            total_events += 1
            naive_pred_sum += p_at_least_one_pred
            naive_actual_sum += p_at_least_one_actual
            naive_squared_err_sum += (p_at_least_one_pred - p_at_least_one_actual) ** 2
            naive_brier_sum += (p_at_least_one_pred - p_at_least_one_actual) ** 2
            if (p_at_least_one_pred >= 0.5) == (p_at_least_one_actual == 1.0):
                naive_correct += 1

    if total_events == 0:
        return {
            "radius_m": radius_m,
            "events": 0,
            "mean_predicted_prob": float("nan"),
            "actual_success_rate": float("nan"),
            "brier_score": float("nan"),
            "rmse": float("nan"),
            "mean_nearby_meters": float("nan"),
            "classification_accuracy": float("nan"),
        }

    mean_pred = naive_pred_sum / total_events
    actual_rate = naive_actual_sum / total_events
    brier = naive_brier_sum / total_events
    rmse = math.sqrt(naive_squared_err_sum / total_events)
    accuracy = naive_correct / total_events

    # Mean number of nearby meters per event
    avg_nearby = 0.0
    count = 0
    for slot in slot_list:
        meters = by_slot[slot]
        if len(meters) < 2:
            continue
        n_centers = min(events_per_slot, len(meters))
        center_indices = rng.choice(len(meters), size=n_centers, replace=False)
        for ci in center_indices:
            center = meters[ci]
            nearby_count = sum(
                1 for m in meters
                if _haversine_m(center["latitude"], center["longitude"],
                               m["latitude"], m["longitude"]) <= radius_m
            )
            avg_nearby += nearby_count
            count += 1
    avg_nearby = avg_nearby / count if count > 0 else 0

    return {
        "radius_m": radius_m,
        "events": total_events,
        "mean_predicted_prob": mean_pred,
        "actual_success_rate": actual_rate,
        "brier_score": brier,
        "rmse": rmse,
        "mean_nearby_meters": avg_nearby,
        "classification_accuracy": accuracy,
    }


def evaluate_radius_by_hour(
    forecast_rows: list[dict],
    *,
    radius_m: int,
    sample_events: int = 2000,
    seed: int = 42,
) -> dict[str, dict]:
    """Radius evaluation broken down by hour bucket."""
    # Filter to specific hour buckets and run evaluation
    results = {}
    for bucket_name, bucket_range in HOUR_BUCKETS.items():
        filtered = [r for r in forecast_rows if r["local_hour"] in bucket_range]
        if filtered:
            results[bucket_name] = evaluate_radius(
                filtered, radius_m=radius_m,
                sample_events=sample_events // 5, seed=seed,
            )
        else:
            results[bucket_name] = {"radius_m": radius_m, "events": 0}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
# CLI REPORT
# ═══════════════════════════════════════════════════════════════════════════

def _print_horizon_table(horizon_m: dict[int, dict]) -> None:
    print()
    print("HORIZON PERFORMANCE")
    print("─" * 80)
    print(f"  {'Horizon':<10} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9} "
          f"{'Persist':>9} {'Improv':>9}")
    print("─" * 80)
    for ha in sorted(horizon_m):
        m = horizon_m[ha]
        sign = "+" if m["bias"] >= 0 else ""
        print(f"  T+{ha:<7} {m['rows']:>8,} {m['mae']:>9.5f} {m['rmse']:>9.5f} "
              f"{sign}{m['bias']:>8.5f} {m['persist_mae']:>9.5f} "
              f"{m['rel_improvement_pct']:>8.2f}%")
    print("─" * 80)

    # Grouped
    grouped = grouped_horizon_metrics(horizon_m)
    print()
    print("GROUPED HORIZONS")
    print("─" * 80)
    print(f"  {'Group':<12} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9} "
          f"{'Persist':>9} {'Improv':>9}")
    print("─" * 80)
    for g in grouped:
        sign = "+" if g["bias"] >= 0 else ""
        print(f"  {g['label']:<12} {g['rows']:>8,} {g['mae']:>9.5f} {g['rmse']:>9.5f} "
              f"{sign}{g['bias']:>8.5f} {g['persist_mae']:>9.5f} "
              f"{g['rel_improvement_pct']:>8.2f}%")
    print("─" * 80)


def _print_calibration_table(cal: dict) -> None:
    print()
    print("CALIBRATION")
    print("─" * 75)
    print(f"  {'Bin':<10} {'Predictions':>12} {'Mean Pred':>10} {'Actual':>10} {'Cal Err':>10}")
    print("─" * 75)
    for b in cal["bins"]:
        if b["n"] > 0:
            print(f"  {b['label']:<10} {b['n']:>12,} {b['mean_predicted']:>9.1%} "
                  f"{b['actual_rate']:>9.1%} {b['calibration_error']:>9.1%}")
        else:
            print(f"  {b['label']:<10} {'0':>12} {'—':>10} {'—':>10} {'—':>10}")
    print("─" * 75)
    print(f"  Brier score          : {cal['brier_score']:.6f}")
    print(f"  Expected cal error   : {cal['expected_calibration_error']:.6f}")
    print(f"  Max cal error        : {cal['max_calibration_error']:.6f}")
    print(f"  Mean predicted       : {cal['overall_mean_predicted']:.4f}")
    print(f"  Actual rate          : {cal['overall_actual_rate']:.4f}")


def _print_radius_table(radius_results: list[dict]) -> None:
    print()
    print("PARKING FIND PROBABILITY")
    print("─" * 80)
    print(f"  {'Radius':>8} {'Events':>8} {'Pred Prob':>10} {'Actual':>10} "
          f"{'Brier':>10} {'Accuracy':>10} {'Avg Meters':>10}")
    print("─" * 80)
    for r in radius_results:
        if r["events"] > 0:
            print(f"  {r['radius_m']:>6} m {r['events']:>8,} "
                  f"{r['mean_predicted_prob']:>9.1%} {r['actual_success_rate']:>9.1%} "
                  f"{r['brier_score']:>10.4f} {r['classification_accuracy']:>9.1%} "
                  f"{r['mean_nearby_meters']:>10.1f}")
        else:
            print(f"  {r['radius_m']:>6} m {'0':>8} {'—':>10} {'—':>10} "
                  f"{'—':>10} {'—':>10} {'—':>10}")
    print("─" * 80)


def _print_hour_breakdown(hour_groups: dict[str, dict]) -> None:
    print()
    print("BY LOCAL HOUR")
    print("─" * 70)
    print(f"  {'Period':<12} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9} "
          f"{'Mean Pred':>10} {'Mean Actual':>12}")
    print("─" * 70)
    for name in HOUR_BUCKETS:
        if name in hour_groups:
            g = hour_groups[name]
            if g["rows"] > 0:
                sign = "+" if g["bias"] >= 0 else ""
                print(f"  {name:<12} {g['rows']:>8,} {g['mae']:>9.5f} "
                      f"{g['rmse']:>9.5f} {sign}{g['bias']:>8.5f} "
                      f"{g['mean_pred']:>9.1%} {g['mean_actual']:>11.1%}")
    print("─" * 70)


def _print_day_type_breakdown(day_groups: dict[str, dict]) -> None:
    print()
    print("BY DAY TYPE")
    print("─" * 70)
    print(f"  {'Type':<12} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9}")
    print("─" * 70)
    for label in ["weekday", "weekend"]:
        if label in day_groups:
            g = day_groups[label]
            if g["rows"] > 0:
                sign = "+" if g["bias"] >= 0 else ""
                print(f"  {label:<12} {g['rows']:>8,} {g['mae']:>9.5f} "
                      f"{g['rmse']:>9.5f} {sign}{g['bias']:>8.5f}")
    print("─" * 70)


def _print_meter_type_breakdown(type_groups: dict[str, dict]) -> None:
    print()
    print("BY METER TYPE")
    print("─" * 70)
    print(f"  {'Type':<12} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9}")
    print("─" * 70)
    for label in sorted(type_groups):
        g = type_groups[label]
        if g["rows"] > 0:
            sign = "+" if g["bias"] >= 0 else ""
            print(f"  {label:<12} {g['rows']:>8,} {g['mae']:>9.5f} "
                  f"{g['rmse']:>9.5f} {sign}{g['bias']:>8.5f}")
    print("─" * 70)


def _print_evening_deep_dive(hour_ha: dict[str, dict[int, dict]]) -> None:
    """Key question: how does evening performance degrade with horizon?"""
    if "evening" not in hour_ha:
        return
    evening = hour_ha["evening"]
    print()
    print("═" * 80)
    print("  KEY QUESTION: Evening (17:00–21:00) performance by horizon")
    print("═" * 80)
    print(f"  {'Horizon':<10} {'Rows':>8} {'MAE':>9} {'RMSE':>9} {'Bias':>9} "
          f"{'Mean Pred':>10} {'Mean Actual':>12}")
    print("─" * 80)
    for ha in sorted(evening):
        g = evening[ha]
        if g["rows"] > 0:
            sign = "+" if g["bias"] >= 0 else ""
            print(f"  T+{ha:<7} {g['rows']:>8,} {g['mae']:>9.5f} "
                  f"{g['rmse']:>9.5f} {sign}{g['bias']:>8.5f} "
                  f"{g['mean_pred']:>9.1%} {g['mean_actual']:>11.1%}")
    print("═" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCE BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def enrich_with_persistence(conn, matured: list[dict]) -> list[dict]:
    """Add persistence_availability to each matured forecast.

    Persistence = the observed availability at the same meter one slot earlier.
    For T+1 this is the latest observed value. For T+N it's the observed value
    at (target_slot - 1 hour), which may itself be a forecast for deep horizons.
    We use the actual observed lag-1 value from parking_state_hourly.
    """
    if not matured:
        return matured

    # Collect unique (post_id, target_slot) pairs and compute lag-1 slots
    lag1_slots = {}
    for r in matured:
        lag1 = r["target_slot"] - timedelta(hours=1)
        key = (r["post_id"], lag1)
        lag1_slots[key] = None

    # Batch-fetch lag-1 actual values
    if lag1_slots:
        post_ids = list({r["post_id"] for r in matured})
        lag1_slot_set = list({k[1] for k in lag1_slots})

        # Use a temp table for efficient batch lookup
        with transaction(conn):
            conn.run("DROP TABLE IF EXISTS _eval_lag1")
            conn.run("CREATE TEMP TABLE _eval_lag1 (post_id text, slot_start timestamptz)")
            buf = StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            for pid, slot in lag1_slots:
                writer.writerow([pid, slot.isoformat()])
            conn.run(
                "COPY _eval_lag1 (post_id, slot_start) FROM STDIN WITH (FORMAT csv)",
                stream=[buf.getvalue().encode("utf-8")],
            )
            rows = conn.run("""
                SELECT l.post_id, l.slot_start, p.paid_availability_probability
                FROM _eval_lag1 l
                LEFT JOIN parking_state_hourly p
                  ON p.post_id = l.post_id AND p.slot_start = l.slot_start
            """)
            conn.run("DROP TABLE IF EXISTS _eval_lag1")
            for r in rows:
                lag1_slots[(r[0], r[1])] = float(r[2]) if r[2] is not None else None

    # Enrich matured records
    for r in matured:
        lag1 = r["target_slot"] - timedelta(hours=1)
        val = lag1_slots.get((r["post_id"], lag1))
        if val is not None:
            r["persistence_availability"] = val
        else:
            # Fallback: use mean actual (uninformative)
            r["persistence_availability"] = np.mean(
                [m["actual_availability"] for m in matured]
            )

    return matured


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(
        description="Comprehensive forecast evaluation: accuracy, calibration, "
                    "and parking-finder utility.",
    )
    p.add_argument("--model-version", type=str, default=None,
                   help="Filter to a specific model version")
    p.add_argument("--max-horizon", type=int, default=None,
                   help="Only evaluate up to this horizon")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be evaluated without modifying data")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip forecast maturation (verify) step")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for radius sampling")
    args = p.parse_args()
    started = time.monotonic()

    conn = connect()
    try:
        print("SF PARKING — FORECAST VALIDATION")
        print("═" * 70)

        # ── Part 1: maturation ──────────────────────────────────────────
        print("\n[Part 1] Forecast maturation...")
        if args.dry_run:
            unverified = conn.run(
                "SELECT count(*) FROM parking_state_forecasts "
                "WHERE actual_availability IS NULL"
            )
            print(f"  Unverified forecasts: {unverified[0][0]:,}")
            print("  Dry run — not updating.")
        elif not args.no_verify:
            updated = verify_forecasts(conn, args.model_version)
            print(f"  Newly verified: {updated:,}")

        # ── fetch matured ───────────────────────────────────────────────
        matured = fetch_matured_forecasts(
            conn, model_version=args.model_version, max_horizon=args.max_horizon,
        )
        if not matured:
            print("\nNo matured forecasts found. Forecasts need time to mature.")
            print("Run the pipeline and wait for target slots to pass.")
            return 0

        print(f"  Total matured forecasts: {len(matured):,}")

        # ── enrichment ──────────────────────────────────────────────────
        print("  Enriching with persistence baseline...")
        matured = enrich_with_persistence(conn, matured)

        # ── Part 2: horizon metrics ─────────────────────────────────────
        print("\n[Part 2] Accuracy by horizon...")
        h_metrics = horizon_metrics(matured)
        _print_horizon_table(h_metrics)

        # ── Part 3: calibration ─────────────────────────────────────────
        print("\n[Part 3] Calibration analysis...")
        cal = calibration_analysis(matured)
        _print_calibration_table(cal)

        cal_by_ha = calibration_by_horizon(matured)
        print()
        print("  Calibration by horizon (Brier score):")
        for ha in sorted(cal_by_ha):
            c = cal_by_ha[ha]
            if c["n"] > 0:
                print(f"    T+{ha:<3} n={c['n']:>7,}  Brier={c['brier_score']:.5f}  "
                      f"ECE={c['expected_calibration_error']:.5f}  "
                      f"mean_pred={c['overall_mean_predicted']:.4f}  "
                      f"actual={c['overall_actual_rate']:.4f}")

        # ── Part 5: breakdowns ──────────────────────────────────────────
        print("\n[Part 5] Breakdown by conditions...")
        hour_groups = breakdown_by_hour(matured)
        _print_hour_breakdown(hour_groups)

        day_groups = breakdown_by_day_type(matured)
        _print_day_type_breakdown(day_groups)

        type_groups = breakdown_by_meter_type(matured)
        _print_meter_type_breakdown(type_groups)

        hour_ha = breakdown_by_hour_and_horizon(matured)
        _print_evening_deep_dive(hour_ha)

        # ── Part 6: parking finder probability ──────────────────────────
        print("\n[Part 6] Parking-finder radius analysis...")
        print("  Fetching forecasts with geometry...")
        geo_rows = fetch_forecasts_with_geometry(
            conn, model_version=args.model_version, max_horizon=args.max_horizon,
        )
        print(f"  Forecasts with geometry: {len(geo_rows):,}")

        radius_results = []
        for radius in EVAL_RADII:
            print(f"  Evaluating radius {radius}m...", end="", flush=True)
            r = evaluate_radius(geo_rows, radius_m=radius, seed=args.seed)
            radius_results.append(r)
            if r["events"] > 0:
                print(f" {r['events']:,} events, "
                      f"pred={r['mean_predicted_prob']:.1%}, "
                      f"actual={r['actual_success_rate']:.1%}")
            else:
                print(" no events")

        _print_radius_table(radius_results)

        # Evening radius breakdown
        print("\n  Evening (17:00–21:00) radius analysis...")
        evening_radius = []
        for radius in [100, 250, 500]:
            r = evaluate_radius(
                [row for row in geo_rows if 17 <= row["local_hour"] < 22],
                radius_m=radius, seed=args.seed,
            )
            evening_radius.append(r)
        if evening_radius:
            print(f"  {'Radius':>8} {'Events':>8} {'Pred':>10} {'Actual':>10} "
                  f"{'Brier':>10}")
            for r in evening_radius:
                if r["events"] > 0:
                    print(f"  {r['radius_m']:>6} m {r['events']:>8,} "
                          f"{r['mean_predicted_prob']:>9.1%} "
                          f"{r['actual_success_rate']:>9.1%} "
                          f"{r['brier_score']:>10.4f}")

        # ── write CSVs ──────────────────────────────────────────────────
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        horizon_rows = []
        for ha in sorted(h_metrics):
            m = h_metrics[ha]
            horizon_rows.append({
                "hours_ahead": ha,
                "rows": m["rows"],
                "mae": round(m["mae"], 6),
                "rmse": round(m["rmse"], 6),
                "bias": round(m["bias"], 6),
                "persist_mae": round(m["persist_mae"], 6),
                "gain_mae": round(m["gain_mae"], 6),
                "rel_improvement_pct": round(m["rel_improvement_pct"], 2),
            })
        _write_csv(RESULTS_DIR / "horizon_metrics.csv", horizon_rows)

        cal_rows = []
        for b in cal["bins"]:
            cal_rows.append({
                "bin": b["label"],
                "n": b["n"],
                "mean_predicted": round(b["mean_predicted"], 6) if not math.isnan(b["mean_predicted"]) else "",
                "actual_rate": round(b["actual_rate"], 6) if not math.isnan(b["actual_rate"]) else "",
                "calibration_error": round(b["calibration_error"], 6) if not math.isnan(b["calibration_error"]) else "",
            })
        cal_rows.append({"bin": "OVERALL", "n": cal["n"],
                         "mean_predicted": round(cal["overall_mean_predicted"], 6),
                         "actual_rate": round(cal["overall_actual_rate"], 6),
                         "calibration_error": ""})
        cal_rows.append({"bin": "BRIER", "n": "", "mean_predicted": "",
                         "actual_rate": "", "calibration_error": round(cal["brier_score"], 6)})
        cal_rows.append({"bin": "ECE", "n": "", "mean_predicted": "",
                         "actual_rate": "", "calibration_error": round(cal["expected_calibration_error"], 6)})
        _write_csv(RESULTS_DIR / "calibration_metrics.csv", cal_rows)

        radius_rows = [r for r in radius_results if r["events"] > 0]
        if radius_rows:
            _write_csv(RESULTS_DIR / "radius_metrics.csv", [
                {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
                for r in radius_rows
            ])

        print(f"\n  CSVs written to {RESULTS_DIR}/")

        # ── CONCLUSION ──────────────────────────────────────────────────
        print()
        print("CONCLUSION")
        print("═" * 70)

        # Best useful horizon
        best_ha = None
        for ha in sorted(h_metrics):
            m = h_metrics[ha]
            if m["gain_mae"] > 0:
                best_ha = ha
        if best_ha:
            print(f"  Best useful horizon    : T+{best_ha} "
                  f"(MAE={h_metrics[best_ha]['mae']:.5f}, "
                  f"beats persistence by {h_metrics[best_ha]['rel_improvement_pct']:.1f}%)")
        else:
            print("  Best useful horizon    : Model does not beat persistence at any horizon")

        # Where does model stop beating persistence?
        last_beating = None
        for ha in sorted(h_metrics):
            if h_metrics[ha]["gain_mae"] > 0:
                last_beating = ha
        if last_beating:
            next_ha = last_beating + 1
            if next_ha in h_metrics and h_metrics[next_ha]["gain_mae"] <= 0:
                print(f"  Stops beating persist  : After T+{last_beating} "
                      f"(T+{next_ha} gain={h_metrics[next_ha]['gain_mae']:.5f})")
            else:
                print(f"  Stops beating persist  : Still beating at T+{last_beating}")
        else:
            print("  Stops beating persist  : N/A (never beats)")

        # Calibration quality
        if cal["n"] > 0:
            if cal["expected_calibration_error"] < 0.03:
                cal_quality = "well calibrated"
            elif cal["expected_calibration_error"] < 0.06:
                cal_quality = "moderately calibrated"
            else:
                cal_quality = "poorly calibrated"
            print(f"  Calibration           : {cal_quality} "
                  f"(ECE={cal['expected_calibration_error']:.4f}, "
                  f"Brier={cal['brier_score']:.4f})")
            print(f"  If model says 80%     : ~{cal['overall_mean_predicted']:.0%} predicted "
                  f"vs {cal['overall_actual_rate']:.0%} actual")

            # Find the 80% bin
            bin_80 = next((b for b in cal["bins"] if b["lo"] == 0.7), None)
            if bin_80 and bin_80["n"] > 0:
                print(f"  Bin 70–80% actual     : {bin_80['actual_rate']:.1%} "
                      f"(n={bin_80['n']:,})")

        # Best radius
        valid_radii = [r for r in radius_results if r["events"] > 0]
        if valid_radii:
            best_r = max(valid_radii, key=lambda r: r["classification_accuracy"])
            print(f"  Best radius           : {best_r['radius_m']}m "
                  f"(accuracy={best_r['classification_accuracy']:.1%}, "
                  f"events={best_r['events']:,})")

        # Evening performance
        evening_ha = hour_ha.get("evening", {})
        if evening_ha:
            # Find best horizon in evening
            best_ev_ha = max(evening_ha.items(),
                            key=lambda x: x[1].get("rows", 0) and -x[1]["mae"])
            if best_ev_ha[1]["rows"] > 0:
                print(f"  Evening performance   : T+{best_ev_ha[0]} MAE={best_ev_ha[1]['mae']:.5f} "
                      f"(n={best_ev_ha[1]['rows']:,})")

        # Overall
        overall_rows = len(matured)
        overall_mae = np.mean([r["absolute_error"] for r in matured])
        print(f"  Overall               : {overall_rows:,} matured forecasts, "
              f"MAE={overall_mae:.5f}")

        elapsed = int(time.monotonic() - started)
        print(f"\nEvaluation complete — elapsed {elapsed}s")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
