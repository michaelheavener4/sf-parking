"""Benchmark a first-principles two-state parking dynamics model.

This is intentionally not a machine-learning tournament. It tests whether a
simple arrival/departure hazard model can improve on one-hour persistence.
All fitted rates are learned only from each fold's training window and all
prediction features come from T-1 or earlier.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sf_parking.database import connect
from sf_parking.dynamics import availability_forecast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "first_principles_dynamics_v1.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def local_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, TZ)


def local_window(start_day: date, end_day: date) -> tuple[datetime, datetime]:
    return local_midnight(start_day).astimezone(UTC), local_midnight(end_day + timedelta(days=1)).astimezone(UTC)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def transition_metrics(y: np.ndarray, lag: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    d = y - lag
    mask = np.abs(d) >= threshold - 1e-12
    if not mask.any():
        return {"n": 0, "mae": None, "direction_accuracy": None, "mean_abs_delta": None}
    pd = pred - lag
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(pred[mask] - y[mask]))),
        "direction_accuracy": float(np.mean((d[mask] > 0) == (pd[mask] > 0))),
        "mean_abs_delta": float(np.mean(np.abs(d[mask]))),
    }


def make_folds(first: datetime, latest: datetime, train_days: int, validation_days: int, test_days: int, max_folds: int):
    first_day = first.astimezone(TZ).date()
    last_day = latest.astimezone(TZ).date()
    end = last_day
    result = []
    while end >= first_day and len(result) < max_folds:
        test_start = end - timedelta(days=test_days - 1)
        val_end = test_start - timedelta(days=1)
        val_start = val_end - timedelta(days=validation_days - 1)
        train_end = val_start - timedelta(days=1)
        train_start = train_end - timedelta(days=train_days - 1)
        if train_start < first_day:
            break
        result.append({
            "train": local_window(train_start, train_end),
            "validation": local_window(val_start, val_end),
            "test": local_window(test_start, end),
            "local_days": {
                "train": [str(train_start), str(train_end)],
                "validation": [str(val_start), str(val_end)],
                "test": [str(test_start), str(end)],
            },
        })
        end -= timedelta(days=test_days)
    result = list(reversed(result))
    if result:
        result[-1]["test"] = (result[-1]["test"][0], min(result[-1]["test"][1], latest + timedelta(microseconds=1)))
    return result


def learn_rates(conn, train_start: datetime, train_end: datetime) -> dict[str, object]:
    duration_rows = conn.run("""
        SELECT post_id,
               AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0) AS mean_duration_h,
               COUNT(*)::double precision AS sessions,
               COUNT(*)::double precision /
                 GREATEST(
                     EXTRACT(EPOCH FROM (
                         CAST(:end AS timestamptz) - CAST(:start AS timestamptz)
                     )) / 3600.0,
                     1.0
                 ) AS baseline_arrival_h
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_end IS NOT NULL
          AND session_end <= CAST(:end AS timestamptz)
          AND session_end > session_start
        GROUP BY post_id
    """, start=train_start, end=train_end)
    durations = {
        str(r[0]): {"mean_duration_h": float(r[1]), "sessions": float(r[2]), "baseline_arrival_h": float(r[3])}
        for r in duration_rows
        if r[1] is not None and float(r[1]) > 0
    }
    hour_rows = conn.run("""
        SELECT EXTRACT(HOUR FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS h,
               EXTRACT(ISODOW FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int AS dow,
               COUNT(*)::double precision AS arrivals
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_start < CAST(:end AS timestamptz)
        GROUP BY 1, 2
    """, start=train_start, end=train_end)
    days = max(1.0, (train_end - train_start).total_seconds() / 86400.0)
    total_arrivals = sum(float(r[2]) for r in hour_rows)
    baseline = max(1e-9, total_arrivals / (days * 168.0))
    seasonality = {
        (int(r[0]), int(r[1])): max(0.05, float(r[2]) / max(days * baseline, 1e-9))
        for r in hour_rows
    }
    global_duration = conn.run("""
        SELECT AVG(EXTRACT(EPOCH FROM (session_end - session_start)) / 3600.0)
        FROM meter_transactions
        WHERE session_start >= CAST(:start AS timestamptz)
          AND session_end IS NOT NULL
          AND session_end <= CAST(:end AS timestamptz)
          AND session_end > session_start
    """, start=train_start, end=train_end)[0][0]
    global_duration_h = float(global_duration) if global_duration is not None and float(global_duration) > 0 else 1.5
    return {"post": durations, "seasonality": seasonality, "global_duration_h": global_duration_h, "baseline_hourly_rate": baseline}
