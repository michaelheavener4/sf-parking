"""Leakage-safe tournament for persistence, climatology, baseline LGBM, and spatial LGBM."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.calibration import fit_isotonic, save_calibrator
from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_paid_state_lgbm import FEATURES as FEATURES_BASE
from scripts.benchmark_paid_state_lgbm import sample_day_targets

MODELS = ROOT / "models"
TZ = "America/Los_Angeles"
MODEL_VERSION = "spatial_dynamic_v1"


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e)))}


def mask_for(df: pd.DataFrame, name: str) -> np.ndarray:
    y = df.target.to_numpy(float)
    lag = df.lag1_availability.to_numpy(float)
    h = pd.to_datetime(df.slot_start, utc=True).dt.tz_convert(TZ).dt.hour.to_numpy()
    return {
        "0_30": y < .30,
        "30_50": (y >= .30) & (y < .50),
        "50_70": (y >= .50) & (y < .70),
        "70_90": (y >= .70) & (y < .90),
        "90_100": y >= .90,
        "transition": np.abs(y - lag) >= .15,
        "daytime": (h >= 7) & (h < 19),
        "peak_evening": (h >= 16) & (h < 21),
    }[name]
