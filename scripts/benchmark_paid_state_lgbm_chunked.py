"""Chunked representative, leakage-safe LightGBM benchmark for paid-state forecasting."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from datetime import date, datetime, timedelta, timezone
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
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def heartbeat(message: str, stop: threading.Event) -> None:
    icons = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌")
    i = 0
    started = time.monotonic()
    while not stop.wait(5):
        elapsed = int(time.monotonic() - started)
        print(f"{icons[i % len(icons)]} Still working after {elapsed // 60:02d}:{elapsed % 60:02d}. {message}", flush=True)
        i += 1


def run_query(conn, sql: str, params: dict[str, object], message: str):
    stop = threading.Event()
    t = threading.Thread(target=heartbeat, args=(message, stop), daemon=True)
    t.start()
    try:
        return conn.run(sql, **params)
    finally:
        stop.set()
        t.join(timeout=0.2)


def metric_arrays(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    err = pred - y
    return float(np.mean(np.abs(err))), float(math.sqrt(np.mean(err * err)))


def sample_day_targets(conn, day: date, limit_rows: int, seed: int) -> list[tuple[object, ...]]:
    bucket = max(1, min(999, int((limit_rows / 350_000) * 1000)))
    sql = """
    SELECT post_id, slot_start, paid_availability_probability, meter_type, local_hour, local_date
    FROM parking_state_hourly
    WHERE local_date = :day
      AND mod(abs(hashtext(post_id || '|' || slot_start::text || :seed::text)), 1000) < :bucket
    ORDER BY slot_start, post_id
    LIMIT :limit_rows
    """
    return conn.run(sql, day=day, seed=str(seed), bucket=bucket, limit_rows=limit_rows)


def feature_chunk(conn, targets: list[tuple[object, ...]], label: str) -> pd.DataFrame:
    conn.run("DROP TABLE IF EXISTS _benchmark_targets")
    conn.run("""
        CREATE TEMP TABLE _benchmark_targets (
            post_id text, slot_start timestamptz, target double precision,
            meter_type text, local_hour int, local_date date
        )
    """)
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(targets)
    conn.run(
        "COPY _benchmark_targets (post_id, slot_start, target, meter_type, local_hour, local_date) FROM STDIN WITH (FORMAT csv)",
        stream=[buf.getvalue().encode("utf-8")],
    )
    sql = """
    SELECT
        t.post_id, t.slot_start, t.target,
        CASE WHEN t.meter_type = 'MS' THEN 1.0 ELSE 0.0 END AS is_ms,
        t.local_hour, t.local_date,
        p1.paid_availability_probability AS lag1,
        p2.paid_availability_probability AS lag2,
        p3.paid_availability_probability AS lag3,
        p6.paid_availability_probability AS lag6,
        p24.paid_availability_probability AS lag24,
        p168.paid_availability_probability AS lag168,
        p1.transaction_count AS tx1,
        p24.transaction_count AS tx24
    FROM _benchmark_targets t
    INNER JOIN parking_state_hourly p1
      ON p1.post_id=t.post_id AND p1.slot_start=t.slot_start-INTERVAL '1 hour'
    INNER JOIN parking_state_hourly p2
      ON p2.post_id=t.post_id AND p2.slot_start=t.slot_start-INTERVAL '2 hours'
    INNER JOIN parking_state_hourly p3
      ON p3.post_id=t.post_id AND p3.slot_start=t.slot_start-INTERVAL '3 hours'
    INNER JOIN parking_state_hourly p6
      ON p6.post_id=t.post_id AND p6.slot_start=t.slot_start-INTERVAL '6 hours'
    INNER JOIN parking_state_hourly p24
      ON p24.post_id=t.post_id AND p24.slot_start=t.slot_start-INTERVAL '24 hours'
    INNER JOIN parking_state_hourly p168
      ON p168.post_id=t.post_id AND p168.slot_start=t.slot_start-INTERVAL '168 hours'
    """
    rows = run_query(conn, sql, {}, f"PostgreSQL is fetching prior-state features for {label} chunk ({len(targets):,} targets).")
    conn.run("DROP TABLE IF EXISTS _benchmark_targets")
    out=[]
    for row in rows:
        hour=int(row[4]); dow=int(row[5].isocalendar().weekday)
        lag1,lag2,lag3,lag6,lag24,lag168=map(float,row[6:12])
        out.append({
            "post_id":row[0],"slot_start":row[1],"target":float(row[2]),
            "lag1_availability":lag1,"lag2_availability":lag2,"lag3_availability":lag3,
            "lag6_availability":lag6,"lag24_availability":lag24,"lag168_availability":lag168,
            "lag1_transactions":float(row[12]),"lag24_transactions":float(row[13]),
            "roll3_availability":(lag1+lag2+lag3)/3.0,
            "roll24_availability":(lag1+lag2+lag3+lag6+lag24)/5.0,
            "hour_sin":math.sin(2*math.pi*hour/24.0),"hour_cos":math.cos(2*math.pi*hour/24.0),
            "weekday_sin":math.sin(2*math.pi*(dow-1)/7.0),"weekday_cos":math.cos(2*math.pi*(dow-1)/7.0),
            "is_ms":float(row[3]),
        })
    return pd.DataFrame(out)


def build_features(conn, targets: list[tuple[object, ...]], label: str, chunk_size: int) -> pd.DataFrame:
    frames=[]; total=len(targets)
    for start in range(0,total,chunk_size):
        chunk=targets[start:start+chunk_size]
        frame=feature_chunk(conn,chunk,f"{label} {start+1:,}-{min(start+len(chunk),total):,}/{total:,}")
        frames.append(frame)
        print(f"      ✅ {label} feature chunk {min(start+len(chunk),total):,}/{total:,}",flush=True)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["target",*FEATURES])


def train_model(train: pd.DataFrame, validation: pd.DataFrame):
    import lightgbm as lgb
    model=lgb.LGBMRegressor(objective="regression",n_estimators=600,learning_rate=0.035,num_leaves=31,min_child_samples=100,subsample=0.9,colsample_bytree=0.9,random_state=42,verbosity=-1)
    model.fit(train[FEATURES],train["target"],eval_X=validation[FEATURES],eval_y=validation["target"],callbacks=[lgb.early_stopping(50,verbose=False)])
    return model


def save_model(model, model_dir: Path, mm: float, mr: float, pm: float, pr: float,
               train_rows: int, val_rows: int, test_rows: int) -> Path:
    """Persist the fitted model and metadata to *model_dir*."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "paid_state_lgbm.txt"
    meta_path = model_dir / "paid_state_lgbm.meta.json"
    model.booster_.save_model(str(model_path))
    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "test_rows": test_rows,
        "model_mae": round(mm, 6),
        "model_rmse": round(mr, 6),
        "persistence_mae": round(pm, 6),
        "persistence_rmse": round(pr, 6),
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\n💾 Model saved to {model_path}")
    print(f"💾 Metadata saved to {meta_path}")
    return model_path


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--test-days",type=int,default=7); p.add_argument("--validation-days",type=int,default=14)
    p.add_argument("--max-train-rows",type=int,default=1_000_000); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--feature-chunk-size",type=int,default=10_000); p.add_argument("--sample-per-day",type=int,default=14_500)
    p.add_argument("--model-dir",type=Path,default=MODEL_DIR,help="Directory for saved model and metadata")
    args=p.parse_args(); started=time.monotonic()
    print("🌉 SF PARKING — CHUNKED REPRESENTATIVE 7-DAY LIGHTGBM")
    print("════════════════════════════════════════════════════════════════════")
    conn=connect()
    try:
        first,latest=conn.run("SELECT min(slot_start), max(slot_start) FROM parking_state_hourly")[0]
        latest_day=latest.astimezone(timezone.utc).date()
        test_start_day=(latest-timedelta(days=args.test_days-1)).astimezone(timezone.utc).date()
        validation_start_day=test_start_day-timedelta(days=args.validation_days); train_end_day=validation_start_day-timedelta(days=1)
        print(f"[1/5] state first={first} latest={latest}")
        print(f"      train_end={train_end_day} validation={validation_start_day}..{test_start_day-timedelta(days=1)} test={test_start_day}..{latest_day}")
        per_day=min(args.sample_per_day,50_000)
        train_targets=[]; day=first.astimezone(timezone.utc).date()
        while day<=train_end_day and len(train_targets)<args.max_train_rows:
            rows=sample_day_targets(conn,day,per_day,args.seed); train_targets.extend(rows)
            print(f"      🚗 train {day}: +{len(rows):,}; total={len(train_targets):,}",flush=True); day+=timedelta(days=1)
        train_targets=train_targets[:args.max_train_rows]
        val_targets=[]; day=validation_start_day
        while day<test_start_day:
            rows=sample_day_targets(conn,day,per_day,args.seed+1); val_targets.extend(rows)
            print(f"      🧪 validation {day}: +{len(rows):,}; total={len(val_targets):,}",flush=True); day+=timedelta(days=1)
        print("\n[2/5] Chunking prior-only feature extraction.")
        train=build_features(conn,train_targets,"train",args.feature_chunk_size)
        validation=build_features(conn,val_targets,"validation",args.feature_chunk_size)
        print(f"      ✅ train features={len(train):,}"); print(f"      ✅ validation features={len(validation):,}")
        print("\n[3/5] Training LightGBM."); model=train_model(train,validation)
        local_hours=pd.to_datetime(train["slot_start"],utc=True).dt.tz_convert(TZ).dt.hour; hour_mean=train.groupby(local_hours)["target"].mean(); global_mean=float(train["target"].mean())
        print("\n[4/5] Evaluating held-out week day-by-day.")
        all_y=[];all_pred=[];all_persist=[];all_clim=[];test_rows=0;day=test_start_day
        while day<=latest_day:
            targets=[r for r in sample_day_targets(conn,day,min(50_000,per_day*2),args.seed+2) if r[1]<=latest]
            features=build_features(conn,targets,f"test {day}",args.feature_chunk_size)
            if not features.empty:
                pred=np.clip(model.predict(features[FEATURES]),0,1); y=features["target"].to_numpy(float); persistence=features["lag1_availability"].to_numpy(float)
                hours=pd.to_datetime(features["slot_start"],utc=True).dt.tz_convert(TZ).dt.hour; clim=hours.map(hour_mean).fillna(global_mean).to_numpy(float)
                m,r=metric_arrays(y,pred); mp,rp=metric_arrays(y,persistence); mc,rc=metric_arrays(y,clim)
                print(f"      🚕 test {day}: rows={len(y):,} model_mae={m:.4f} persist_mae={mp:.4f} climatology_mae={mc:.4f}",flush=True)
                all_y.append(y);all_pred.append(pred);all_persist.append(persistence);all_clim.append(clim);test_rows+=len(y)
            day+=timedelta(days=1)
    finally:
        conn.close()
    y=np.concatenate(all_y) if all_y else np.array([]); pred=np.concatenate(all_pred) if all_pred else np.array([]); persist=np.concatenate(all_persist) if all_persist else np.array([]); clim=np.concatenate(all_clim) if all_clim else np.array([])
    mm,mr=metric_arrays(y,pred);pm,pr=metric_arrays(y,persist);cm,cr=metric_arrays(y,clim)
    print("\n[5/5] Final benchmark."); print(f"      train_rows={len(train):,}"); print(f"      validation_rows={len(validation):,}"); print(f"      test_rows={test_rows:,}"); print(f"      model:        MAE={mm:.6f} RMSE={mr:.6f}"); print(f"      persistence:  MAE={pm:.6f} RMSE={pr:.6f}"); print(f"      climatology:  MAE={cm:.6f} RMSE={cr:.6f}"); print(f"      gain_vs_persistence_mae={pm-mm:.6f}"); print(f"      gain_vs_climatology_mae={cm-mm:.6f}"); print(f"      best_iteration={getattr(model,'best_iteration_',None)}"); print(f"\n✅ COMPLETE — elapsed {int(time.monotonic()-started)}s")
    save_model(model, args.model_dir, mm, mr, pm, pr, len(train), len(validation), test_rows)
    text=f"model_mae={mm:.6f}\nmodel_rmse={mr:.6f}\npersistence_mae={pm:.6f}\npersistence_rmse={pr:.6f}\nclimatology_mae={cm:.6f}\nclimatology_rmse={cr:.6f}\ngain_vs_persistence_mae={pm-mm:.6f}\ngain_vs_climatology_mae={cm-mm:.6f}\ntrain_rows={len(train)}\nvalidation_rows={len(validation)}\ntest_rows={test_rows}\n"
    try: subprocess.run(["pbcopy"],input=text,text=True,check=True); print("📋 Results copied to macOS clipboard.")
    except (OSError,subprocess.CalledProcessError): print("⚠️ Clipboard copy failed.")
    return 0


if __name__=="__main__": raise SystemExit(main())
