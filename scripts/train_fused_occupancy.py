#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sf_parking.database import connect

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"models"/"fused_sensor_calibration_v1.json"

def metric(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(np.mean(e))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--max-rows",type=int,default=1000000); ap.add_argument("--test-fraction",type=float,default=.2); a=ap.parse_args()
    c=connect()
    try:
        rows=c.run("""
          SELECT local_hour,street_block,occupancy_total,lag_occupancy_total,
                 lag_payment_session_starts,prior_3h_mean_payment_starts,
                 prior_same_slot_payment_mean,rate,
                 EXTRACT(HOUR FROM local_hour)::int AS hour_of_day,
                 EXTRACT(ISODOW FROM local_hour)::int AS dow
          FROM v_fusion_historical_calibration_hourly
          WHERE occupancy_total IS NOT NULL AND lag_occupancy_total IS NOT NULL
          ORDER BY local_hour,street_block LIMIT :k""",k=a.max_rows)
    finally: c.close()
    if len(rows)<1000: raise RuntimeError(f"Only {len(rows):,} rows available; import both historical sensor and smart-payment files first.")
    x=np.column_stack([
        np.nan_to_num(np.asarray([r[3] for r in rows],float),nan=.5),
        np.nan_to_num(np.asarray([r[4] for r in rows],float),nan=0),
        np.nan_to_num(np.asarray([r[5] for r in rows],float),nan=0),
        np.nan_to_num(np.asarray([r[6] for r in rows],float),nan=0),
        np.nan_to_num(np.asarray([r[7] for r in rows],float),nan=0),
        np.asarray([r[8] for r in rows],int),
        np.asarray([r[9] for r in rows],int),
    ])
    y=np.asarray([r[2] for r in rows],float); cut=max(int(len(rows)*(1-a.test_fraction)),1)
    xt,yt=x[cut:],y[cut:]; persistence=x[cut:,0]
    try:
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(n_estimators=500,learning_rate=.03,num_leaves=31,subsample=.9,colsample_bytree=.9,random_state=42,objective="regression_l1",verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=31,random_state=42,loss="absolute_error")
    model.fit(x[:cut],y[:cut]); pred=np.clip(model.predict(xt),0,1)
    pm,mm=metric(yt,persistence),metric(yt,pred)
    result={"version":1,"target":"SFpark sensor-measured hourly total occupancy","features":["lag_occupancy_total","lag_payment_session_starts","prior_3h_mean_payment_starts","prior_same_slot_payment_mean","rate","hour_of_day","dow"],"rows":{"total":len(rows),"train":cut,"test":len(yt)},"persistence":pm,"fused_model":mm,"improvement_over_persistence":(pm["mae"]-mm["mae"])/pm["mae"] if pm["mae"] else None,"promotion":"candidate" if mm["mae"]<pm["mae"] else "retained_only"}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print(f"Report: {OUT}")
if __name__=="__main__": main()
