"""Generate a production-style T+1 forecast with spatial/dynamic features."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta, datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np

from sf_parking.database import connect
from sf_parking.forecasting import latest_observed_slot
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_inference_features

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MODEL=ROOT/"models"/"paid_state_spatial_lgbm.txt"
MODEL_VERSION="spatial_dynamic_v1"


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    ap.add_argument("--hours-ahead",type=int,default=1)
    ap.add_argument("--neighbor-k",type=int,default=24)
    ap.add_argument("--radius-m",type=float,default=250)
    args=ap.parse_args()
    if args.hours_ahead!=1:
        raise SystemExit("spatial model currently supports T+1 only; use the established recursive model for T+2..T+24")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run train_spatial_paid_state.py first.")
    import lightgbm as lgb
    model=lgb.Booster(model_file=str(args.model))
    conn=connect()
    try:
        latest=latest_observed_slot(conn); target=latest+timedelta(hours=1)
        df=build_spatial_inference_features(conn,target,config=SpatialFeatureConfig(args.neighbor_k,args.radius_m))
        if df.empty: raise SystemExit("No eligible meters with complete lag history")
        pred=np.clip(model.predict(df[FEATURES_SPATIAL]),0,1)
        now=datetime.now(timezone.utc)
        buf=StringIO(); writer=csv.writer(buf,lineterminator="\n")
        for pid,p in zip(df.post_id,pred): writer.writerow([pid,now,target,1,float(p),MODEL_VERSION,str(args.model),latest])
        conn.run("CREATE TEMP TABLE _spatial_forecasts(post_id text,forecast_generated_at timestamptz,target_slot timestamptz,hours_ahead int,predicted_availability double precision,model_version text,model_path text,feature_data_as_of timestamptz)")
        conn.run("COPY _spatial_forecasts FROM STDIN WITH (FORMAT csv)",stream=[buf.getvalue().encode()])
        conn.run("""
          INSERT INTO parking_state_forecasts(post_id,forecast_generated_at,target_slot,hours_ahead,predicted_availability,model_version,model_path,feature_data_as_of)
          SELECT post_id,forecast_generated_at,target_slot,hours_ahead,predicted_availability,model_version,model_path,feature_data_as_of FROM _spatial_forecasts
          ON CONFLICT(post_id,target_slot,model_version) DO UPDATE SET predicted_availability=EXCLUDED.predicted_availability,
            forecast_generated_at=EXCLUDED.forecast_generated_at,model_path=EXCLUDED.model_path,feature_data_as_of=EXCLUDED.feature_data_as_of
        """)
        conn.run("DROP TABLE _spatial_forecasts")
        print("SF PARKING — SPATIAL/DYNAMIC T+1")
        print(f"Latest observed : {latest.isoformat()}")
        print(f"Target slot     : {target.isoformat()}")
        print(f"Meters forecast : {len(df):,}")
        print(f"Mean prediction : {float(np.mean(pred))*100:.2f}%")
        print(f"Model version   : {MODEL_VERSION}")
    finally: conn.close()
    return 0

if __name__=="__main__": raise SystemExit(main())
