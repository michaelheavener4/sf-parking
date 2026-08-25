"""Train and evaluate the next-generation spatial/dynamic parking model.

This is deliberately a research-grade, chronological tournament:

    persistence -> hour climatology -> current feature set -> spatial/dynamic LightGBM

Only the final candidate is written to models/. The script never promotes a
model automatically; promotion is based on the machine-readable scorecard
and difficult-regime gates.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features, sample_targets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "paid_state_spatial_lgbm.txt"
DEFAULT_META = ROOT / "models" / "paid_state_spatial_lgbm.meta.json"


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p-y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e*e))),
        "bias": float(np.mean(e)),
    }


def difficult_metrics(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    # Difficult means the observed paid-state target is not trivially near 1.
    mask = y < 0.9
    if not np.any(mask):
        return {"rows": 0, "mae": float("nan"), "baseline_mae": float("nan"), "improvement_pct": float("nan")}
    m = metrics(y[mask], p[mask])
    b = metrics(y[mask], baseline[mask])
    improvement = (b["mae"]-m["mae"])/b["mae"]*100 if b["mae"] else 0.0
    return {"rows": int(mask.sum()), "mae": m["mae"], "baseline_mae": b["mae"], "improvement_pct": improvement}


def by_hour(y: np.ndarray, p: np.ndarray, baseline: np.ndarray, hours: np.ndarray) -> dict[str, dict[str, float | int]]:
    out = {}
    for h in range(24):
        mask = hours == h
        if not np.any(mask):
            continue
        m = metrics(y[mask], p[mask]); b = metrics(y[mask], baseline[mask])
        out[str(h)] = {
            "rows": int(mask.sum()), "mae": m["mae"], "baseline_mae": b["mae"],
            "improvement_pct": ((b["mae"]-m["mae"])/b["mae"]*100 if b["mae"] else 0.0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--max-train-rows", type=int, default=250_000)
    ap.add_argument("--max-validation-rows", type=int, default=100_000)
    ap.add_argument("--max-test-rows", type=int, default=150_000)
    ap.add_argument("--neighbor-k", type=int, default=24)
    ap.add_argument("--neighbor-radius-m", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--promote", action="store_true", help="Write model only if promotion gates pass")
    args = ap.parse_args()

    import lightgbm as lgb

    conn = connect()
    try:
        first, latest = conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly WHERE slot_start <= NOW()")[0]
        test_start = latest - timedelta(days=args.test_days)
        validation_start = test_start - timedelta(days=args.validation_days)
        train_start = validation_start - timedelta(days=args.train_days)

        print("SF PARKING — SPATIAL/DYNAMIC MODEL TOURNAMENT")
        print("="*72)
        print(f"Train      : {train_start} .. {validation_start - timedelta(hours=1)}")
        print(f"Validation : {validation_start} .. {test_start - timedelta(hours=1)}")
        print(f"Test       : {test_start} .. {latest}")
        print(f"Neighbors  : k={args.neighbor_k}, radius={args.neighbor_radius_m:.0f}m")

        cfg = SpatialFeatureConfig(args.neighbor_k, args.neighbor_radius_m)
        train_targets = sample_targets(conn, train_start, validation_start-timedelta(hours=1), args.max_train_rows, args.seed)
        val_targets = sample_targets(conn, validation_start, test_start-timedelta(hours=1), args.max_validation_rows, args.seed+1)
        test_targets = sample_targets(conn, test_start, latest, args.max_test_rows, args.seed+2)
        print(f"Targets: train={len(train_targets):,} val={len(val_targets):,} test={len(test_targets):,}")

        print("[1/4] Building leakage-safe spatial/dynamic features...")
        train = build_spatial_features(conn, train_targets, config=cfg)
        val = build_spatial_features(conn, val_targets, config=cfg)
        test = build_spatial_features(conn, test_targets, config=cfg)
        print(f"Features: train={len(train):,} val={len(val):,} test={len(test):,}")
    finally:
        conn.close()

    if train.empty or val.empty or test.empty:
        raise SystemExit("Insufficient complete feature rows for model training")

    print("[2/4] Training spatial/dynamic LightGBM...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=args.seed,
        verbosity=-1,
    )
    model.fit(
        train[FEATURES_SPATIAL], train.target,
        eval_set=[(val[FEATURES_SPATIAL], val.target)],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    y = test.target.to_numpy(float)
    pred = np.clip(model.predict(test[FEATURES_SPATIAL]), 0.0, 1.0)
    persistence = test.lag1_availability.to_numpy(float)
    # Hour climatology is learned only from train.
    train_hours = train.local_date.astype(str).map(lambda x: 0) if False else None
    # The target query already includes local_hour; derive it from slot_start for safety.
    import pandas as pd
    train_hours = pd.to_datetime(train.slot_start, utc=True).dt.tz_convert("America/Los_Angeles").dt.hour
    test_hours = pd.to_datetime(test.slot_start, utc=True).dt.tz_convert("America/Los_Angeles").dt.hour.to_numpy()
    hour_mean = train.groupby(train_hours).target.mean()
    clim = np.array([float(hour_mean.get(int(h), train.target.mean())) for h in test_hours])

    mm = metrics(y, pred); pm = metrics(y, persistence); cm = metrics(y, clim)
    difficult = difficult_metrics(y, pred, persistence)
    hours = by_hour(y, pred, persistence, test_hours)
    print("[3/4] Scorecard")
    print(f"Model       MAE={mm['mae']:.6f} RMSE={mm['rmse']:.6f}")
    print(f"Persistence MAE={pm['mae']:.6f} RMSE={pm['rmse']:.6f}")
    print(f"Climatology MAE={cm['mae']:.6f} RMSE={cm['rmse']:.6f}")
    print(f"Difficult   n={difficult['rows']:,} MAE={difficult['mae']:.6f} vs persistence={difficult['baseline_mae']:.6f} improvement={difficult['improvement_pct']:.2f}%")

    # Conservative promotion gate: improve persistence on difficult rows and do
    # not regress aggregate MAE by >2%. The final production model can still be
    # selected manually after calibration review.
    aggregate_gate = mm["mae"] <= pm["mae"] * 1.02
    difficult_gate = difficult["rows"] >= 100 and difficult["improvement_pct"] > 0
    promotion_pass = bool(aggregate_gate and difficult_gate)

    meta = {
        "model": "paid_state_spatial_lgbm",
        "feature_version": "spatial_dynamic_v1",
        "features": FEATURES_SPATIAL,
        "neighbor_k": args.neighbor_k,
        "neighbor_radius_m": args.neighbor_radius_m,
        "train_window": [str(train_start), str(validation_start-timedelta(hours=1))],
        "validation_window": [str(validation_start), str(test_start-timedelta(hours=1))],
        "test_window": [str(test_start), str(latest)],
        "metrics": {"model": mm, "persistence": pm, "climatology": cm, "difficult": difficult, "by_hour": hours},
        "promotion": {"aggregate_gate": aggregate_gate, "difficult_gate": difficult_gate, "passed": promotion_pass},
    }

    print("[4/4] Artifact handling")
    if args.promote and not promotion_pass:
        print("❌ Promotion blocked: candidate did not beat the required gates.")
        print(json.dumps(meta["promotion"], indent=2))
        return 2

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_path))
    meta_path = args.model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Model written: {args.model_path}")
    print(f"Metadata:      {meta_path}")
    print(f"Promotion:     {'PASS' if promotion_pass else 'RESEARCH ONLY'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
