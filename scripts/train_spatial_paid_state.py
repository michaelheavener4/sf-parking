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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sf_parking.calibration import fit_isotonic, save_calibrator
from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features
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


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df.target.to_numpy(float)
    out = {"n": int(len(df)), "overall": scores(y, pred)}
    for name in ("0_30", "30_50", "50_70", "70_90", "90_100", "transition", "daytime", "peak_evening"):
        m = mask_for(df, name)
        out[name] = {"n": int(m.sum()), **scores(y[m], pred[m])} if m.any() else {"n": 0, "mae": float("nan"), "rmse": float("nan")}
    return out


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, features: list[str]):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=1000, learning_rate=.03,
        num_leaves=31, min_child_samples=100, subsample=.9,
        colsample_bytree=.9, reg_alpha=.05, reg_lambda=.2,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(train[features], train.target,
              eval_set=[(val[features], val.target)],
              callbacks=[lgb.early_stopping(75, verbose=False)])
    return model


def collect(conn, start, end, per_day, seed):
    rows = []
    d = start
    while d <= end:
        part = sample_day_targets(conn, d, per_day, seed)
        rows.extend(part)
        print(f"      {d}: +{len(part):,} targets; total={len(rows):,}", flush=True)
        d += timedelta(days=1)
    return rows


def features_in_batches(conn, targets, batch_size, config):
    frames = []
    for start in range(0, len(targets), batch_size):
        chunk = targets[start:start + batch_size]
        frame = build_spatial_features(conn, chunk, config=config)
        if not frame.empty:
            frames.append(frame)
        print(f"      features {min(start + batch_size, len(targets)):,}/{len(targets):,}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def climatology(train, test):
    hours = pd.to_datetime(train.slot_start, utc=True).dt.tz_convert(TZ).dt.hour
    means = train.groupby(hours).target.mean()
    fallback = float(train.target.mean())
    th = pd.to_datetime(test.slot_start, utc=True).dt.tz_convert(TZ).dt.hour
    return th.map(means).fillna(fallback).to_numpy(float)


def save_model(model, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def choose_windows(first, latest, requested_train, requested_val, requested_test):
    first_day, latest_day = first.date(), latest.date()
    available_days = (latest_day - first_day).days + 1
    if available_days < 4:
        raise RuntimeError(
            f"Only {available_days} calendar days of data exist ({first_day}..{latest_day}); "
            "at least 4 are required for a chronological tournament."
        )
    test_days = min(requested_test, max(1, available_days // 4))
    val_days = min(requested_val, max(1, available_days // 4))
    train_days = min(requested_train, available_days - test_days - val_days)
    if train_days < 1:
        raise RuntimeError("Not enough history remains for a non-empty training window")
    test_start = latest_day - timedelta(days=test_days - 1)
    val_start = test_start - timedelta(days=val_days)
    train_start = max(first_day, val_start - timedelta(days=train_days))
    train_end = val_start - timedelta(days=1)
    return train_start, train_end, val_start, test_start, latest_day, train_days, val_days, test_days


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--validation-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--max-train-rows", type=int, default=250000)
    ap.add_argument("--max-validation-rows", type=int, default=100000)
    ap.add_argument("--max-test-rows", type=int, default=150000)
    ap.add_argument("--neighbor-k", type=int, default=24)
    ap.add_argument("--neighbor-radius-m", type=float, default=250)
    ap.add_argument("--feature-batch-size", type=int, default=10000)
    args = ap.parse_args()
    t0 = time.monotonic()
    cfg = SpatialFeatureConfig(args.neighbor_k, args.neighbor_radius_m)
    print("🅿️ SF PARKING — LEAKAGE-SAFE MODEL TOURNAMENT")
    conn = connect()
    try:
        first, latest = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly WHERE slot_start <= NOW()")[0]
        if first is None or latest is None:
            raise RuntimeError("No completed hourly state exists")
        windows = choose_windows(first, latest, args.train_days, args.validation_days, args.test_days)
        train_start, train_end, val_start, test_start, test_end, train_days, val_days, test_days = windows
        requested = (args.train_days, args.validation_days, args.test_days)
        actual = (train_days, val_days, test_days)
        if actual != requested:
            print(f"⚠️ Requested {requested[0]}/{requested[1]}/{requested[2]} days, but only {actual[0]}/{actual[1]}/{actual[2]} fit in available history.")
        print(f"Train      {train_start} → {train_end}")
        print(f"Validation {val_start} → {test_start - timedelta(days=1)}")
        print(f"Test       {test_start} → {test_end}")

        train_targets = collect(conn, train_start, train_end, math.ceil(args.max_train_rows / train_days), 42)[:args.max_train_rows]
        val_targets = collect(conn, val_start, test_start - timedelta(days=1), math.ceil(args.max_validation_rows / val_days), 43)[:args.max_validation_rows]
        test_targets = collect(conn, test_start, test_end, math.ceil(args.max_test_rows / test_days), 44)[:args.max_test_rows]
        print(f"Targets train={len(train_targets):,} val={len(val_targets):,} test={len(test_targets):,}")
        if min(len(train_targets), len(val_targets), len(test_targets)) == 0:
            raise RuntimeError("A chronological split contains no target rows")

        print("\n[1/4] Building leakage-safe temporal + spatial feature matrices")
        train_s = features_in_batches(conn, train_targets, args.feature_batch_size, cfg)
        val_s = features_in_batches(conn, val_targets, args.feature_batch_size, cfg)
        test_s = features_in_batches(conn, test_targets, args.feature_batch_size, cfg)
        if min(len(train_s), len(val_s), len(test_s)) == 0:
            raise RuntimeError("Feature construction returned an empty chronological split")

        train_b = train_s[["post_id", "slot_start", "target", *FEATURES_BASE]].copy()
        val_b = val_s[["post_id", "slot_start", "target", *FEATURES_BASE]].copy()
        test_b = test_s[["post_id", "slot_start", "target", *FEATURES_BASE]].copy()

        print("\n[2/4] Training baseline and spatial models")
        base_model = train_lgbm(train_b, val_b, FEATURES_BASE)
        spatial_model = train_lgbm(train_s, val_s, FEATURES_SPATIAL)

        print("\n[3/4] Scoring tournament")
        base_pred = np.clip(base_model.predict(test_b[FEATURES_BASE]), 0, 1)
        spatial_pred = np.clip(spatial_model.predict(test_s[FEATURES_SPATIAL]), 0, 1)
        persistence = test_b.lag1_availability.to_numpy(float)
        clim = climatology(train_b, test_b)
        results = {
            "persistence": evaluate(test_b, persistence),
            "hour_climatology": evaluate(test_b, clim),
            "current_lgbm": evaluate(test_b, base_pred),
            "spatial_dynamic_lgbm": evaluate(test_s, spatial_pred),
        }
        for name, r in results.items():
            print(f"\n{name}: MAE={r['overall']['mae']:.6f} RMSE={r['overall']['rmse']:.6f} n={r['n']:,}")
            for regime in ("0_30", "30_50", "50_70", "70_90", "transition", "peak_evening"):
                q = r[regime]
                if q["n"]:
                    print(f"  {regime:16s} n={q['n']:,} MAE={q['mae']:.6f}")

        s, b, p = results["spatial_dynamic_lgbm"], results["current_lgbm"], results["persistence"]
        overall_gate = s["overall"]["mae"] < b["overall"]["mae"] and s["overall"]["mae"] < p["overall"]["mae"]
        transition_gate = s["transition"]["n"] >= 100 and s["transition"]["mae"] < b["transition"]["mae"] and s["transition"]["mae"] < p["transition"]["mae"]
        promoted = overall_gate and transition_gate

        print("\n[4/4] Artifacts and promotion")
        MODELS.mkdir(parents=True, exist_ok=True)
        candidate = MODELS / "paid_state_spatial_candidate_lgbm.txt"
        meta = {
            "model_version": MODEL_VERSION,
            "features": FEATURES_SPATIAL,
            "neighbor_k": args.neighbor_k,
            "neighbor_radius_m": args.neighbor_radius_m,
            "train_rows": len(train_s),
            "validation_rows": len(val_s),
            "test_rows": len(test_s),
            "train_window": [str(train_start), str(train_end)],
            "validation_window": [str(val_start), str(test_start - timedelta(days=1))],
            "test_window": [str(test_start), str(test_end)],
            "metrics": s,
            "promotion": {"overall": overall_gate, "transition": transition_gate, "promoted": promoted},
            "lag168_note": "When historical coverage is shorter than 168 hours, lag168 falls back to lag24; this is explicit bootstrap behavior, not future leakage.",
        }
        save_model(spatial_model, candidate, meta)
        if promoted:
            shutil.copy2(candidate, MODELS / "paid_state_spatial_lgbm.txt")
            shutil.copy2(candidate.with_suffix(".meta.json"), MODELS / "paid_state_spatial_lgbm.meta.json")
            print("🏆 PROMOTED spatial/dynamic model")
        else:
            print("🛑 Candidate retained; production spatial model not changed")

        val_pred = np.clip(spatial_model.predict(val_s[FEATURES_SPATIAL]), 0, 1)
        events = (val_s.target.to_numpy(float) >= .50).astype(float)
        save_calibrator(
            fit_isotonic(val_pred, events),
            MODELS / "paid_state_spatial_probability_calibrator.json",
            event="actual_availability >= 0.500",
            training_window=(str(val_start), str(test_start - timedelta(days=1))),
        )
        importance = sorted(
            zip(FEATURES_SPATIAL, spatial_model.booster_.feature_importance("gain")),
            key=lambda x: x[1], reverse=True,
        )
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "rows": {"train": len(train_s), "validation": len(val_s), "test": len(test_s)},
            "windows": {
                "train": [str(train_start), str(train_end)],
                "validation": [str(val_start), str(test_start - timedelta(days=1))],
                "test": [str(test_start), str(test_end)],
            },
            "requested_days": {"train": args.train_days, "validation": args.validation_days, "test": args.test_days},
            "actual_days": {"train": train_days, "validation": val_days, "test": test_days},
            "results": results,
            "promotion": {"overall": overall_gate, "transition": transition_gate, "promoted": promoted},
            "feature_importance_gain": [{"feature": k, "gain": float(v)} for k, v in importance],
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }
        (MODELS / "paid_state_spatial_tournament.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("Report: models/paid_state_spatial_tournament.json")
        print(f"Runtime: {time.monotonic() - t0:.1f}s")
        return 0 if promoted else 3
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
