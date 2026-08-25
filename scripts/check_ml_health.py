"""Operational ML health and drift checks."""
from __future__ import annotations

import argparse
import json
from datetime import timedelta

from sf_parking.database import connect


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours",type=int,default=48)
    ap.add_argument("--max-missing-rate",type=float,default=.10)
    ap.add_argument("--max-mean-abs-error",type=float,default=.15)
    args=ap.parse_args()
    conn=connect()
    try:
        rows=conn.run("""
          SELECT model_version,count(*) n,count(actual_availability) matured,
                 avg(abs(predicted_availability-actual_availability)) mae,
                 avg(predicted_availability) pred,avg(actual_availability) actual
          FROM parking_state_forecasts WHERE forecast_generated_at>=NOW()-(:hours||' hours')::interval
          GROUP BY model_version ORDER BY model_version
        """,hours=args.hours)
    finally: conn.close()
    bad=False
    print("SF PARKING — ML HEALTH")
    print("="*72)
    for version,n,matured,mae,pred,actual in rows:
        maturity=float(matured or 0)/max(int(n),1); missing=1-maturity
        print(f"{version}: rows={n:,} matured={matured:,} maturity={maturity:.1%} mae={float(mae or 0):.4f}")
        if matured and float(mae)>args.max_mean_abs_error: bad=True
        if missing>args.max_missing_rate and matured: bad=True
    print("STATUS:","FAIL" if bad else "PASS")
    return 2 if bad else 0

if __name__=="__main__": raise SystemExit(main())
