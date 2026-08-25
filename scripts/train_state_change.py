"""Train and evaluate a leakage-safe model of hourly availability change.

The model predicts delta = availability(T) - availability(T-1). The reconstructed
state is lag1 + predicted_delta. This isolates the actual transition signal while
retaining the existing spatial/dynamic feature architecture.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features
from scripts.benchmark_paid_state_lgbm import sample_day_targets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "paid_state_change_tournament.json"
TZ = "America/Los_Angeles"


def collect(conn, start, end, per_day, seed):
    rows=[]; d=start
    while d<=end:
        part=sample_day_targets(conn,d,per_day,seed); rows.extend(part)
        print(f"  {d}: +{len(part):,} targets; total={len(rows):,}", flush=True); d += timedelta(days=1)
    return rows


def features(conn, targets, cfg, batch):
    frames=[]
    for i in range(0,len(targets),batch):
        f=build_spatial_features(conn,targets[i:i+batch],config=cfg)
        if not f.empty: frames.append(f)
        print(f"  features {min(i+batch,len(targets)):,}/{len(targets):,}",flush=True)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def fit(train,val):
    import lightgbm as lgb
    m=lgb.LGBMRegressor(objective="regression",n_estimators=1000,learning_rate=.03,num_leaves=31,min_child_samples=100,subsample=.9,colsample_bytree=.9,reg_alpha=.05,reg_lambda=.2,random_state=42,n_jobs=-1,verbosity=-1)
    m.fit(train[FEATURES_SPATIAL],train.delta,eval_X=val[FEATURES_SPATIAL],eval_y=val.delta,callbacks=[lgb.early_stopping(75,verbose=False)])
    return m


def metric(y,p):
    e=p-y
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(np.mean(e))}


def transition(y,lag,p,threshold):
    d=y-lag; m=np.abs(d)>=threshold
    if not m.any(): return {"n":0,"mae":None,"direction_accuracy":None,"mean_abs_delta":None}
    pd=p-lag
    return {"n":int(m.sum()),"mae":float(np.mean(np.abs(p[m]-y[m]))),"direction_accuracy":float(np.mean((d[m]>0)==(pd[m]>0))),"mean_abs_delta":float(np.mean(np.abs(d[m])))}


def choose_windows(first,latest,train_days,val_days,test_days):
    first,last=first.date(),latest.date(); avail=(last-first).days+1
    td=min(test_days,max(1,avail//4)); vd=min(val_days,max(1,avail//4)); tr=min(train_days,avail-td-vd)
    test_start=last-timedelta(days=td-1); val_start=test_start-timedelta(days=vd); train_start=max(first,val_start-timedelta(days=tr)); train_end=val_start-timedelta(days=1)
    return train_start,train_end,val_start,test_start,last


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train-days",type=int,default=90); ap.add_argument("--validation-days",type=int,default=7); ap.add_argument("--test-days",type=int,default=7); ap.add_argument("--max-train-rows",type=int,default=250000); ap.add_argument("--max-validation-rows",type=int,default=100000); ap.add_argument("--max-test-rows",type=int,default=150000); ap.add_argument("--neighbor-k",type=int,default=24); ap.add_argument("--neighbor-radius-m",type=float,default=250); ap.add_argument("--feature-batch-size",type=int,default=10000); ap.add_argument("--transition-threshold",type=float,default=.10); args=ap.parse_args()
    print("🔬 SF PARKING — STATE-CHANGE TOURNAMENT")
    conn=connect()
    try:
        first,latest=conn.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0]
        tr0,tr1,v0,t0,t1=choose_windows(first,latest,args.train_days,args.validation_days,args.test_days)
        print(f"Train {tr0} → {tr1}; Validation {v0} → {t0-timedelta(days=1)}; Test {t0} → {t1}")
        train=collect(conn,tr0,tr1,math.ceil(args.max_train_rows/max(1,(tr1-tr0).days+1)),51)[:args.max_train_rows]
        val=collect(conn,v0,t0-timedelta(days=1),math.ceil(args.max_validation_rows/max(1,(t0-v0).days)),52)[:args.max_validation_rows]
        test=collect(conn,t0,t1,math.ceil(args.max_test_rows/max(1,(t1-t0).days+1)),53)[:args.max_test_rows]
        cfg=SpatialFeatureConfig(args.neighbor_k,args.neighbor_radius_m)
        print("[1/3] Features")
        tr=features(conn,train,cfg,args.feature_batch_size); va=features(conn,val,cfg,args.feature_batch_size); te=features(conn,test,cfg,args.feature_batch_size)
    finally: conn.close()
    for df in (tr,va,te): df["delta"]=df.target-df.lag1_availability
    print("[2/3] Training delta model")
    model=fit(tr,va)
    pred_delta=np.asarray(model.predict(te[FEATURES_SPATIAL]),float)
    lag=te.lag1_availability.to_numpy(float); y=te.target.to_numpy(float)
    state_pred=np.clip(lag+pred_delta,0,1)
    persistence=lag
    results={"persistence":metric(y,persistence),"state_change_reconstructed":metric(y,state_pred)}
    thresholds={str(t):{"persistence":transition(y,lag,persistence,t),"state_change":transition(y,lag,state_pred,t)} for t in (.05,.10,.15,.25)}
    print("[3/3] Results")
    for k,v in results.items(): print(f"{k}: MAE={v['mae']:.6f} RMSE={v['rmse']:.6f} bias={v['bias']:.6f}")
    for t,v in thresholds.items(): print(f"|Δ|>={t}: n={v['state_change']['n']} MAE={v['state_change']['mae']} direction={v['state_change']['direction_accuracy']}")
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({"version":1,"windows":{"train":[str(tr0),str(tr1)],"validation":[str(v0),str(t0-timedelta(days=1))],"test":[str(t0),str(t1)]},"results":results,"transitions":thresholds,"feature_count":len(FEATURES_SPATIAL),"model":"spatial_dynamic_delta_v1"},indent=2,default=str),encoding="utf-8")
    print(f"Report: {OUT}")

if __name__=="__main__": main()
