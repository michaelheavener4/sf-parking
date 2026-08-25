"""Causal blockface transaction-occupancy benchmark V3.

This version closes the key real-time leakage path: predictors never use a
session_end timestamp for a session that would still be open at the forecast
as-of time. session_end is used only in the training window to learn duration
survival and in the held-out target construction.

Target: transaction-implied occupancy = active paid sessions / mapped
blockface capacity.
Forecast origin: target slot minus one hour.
Predictors at forecast origin: session starts observed by the origin, training
only duration survival, training-only arrival intensity, mapped capacity.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "blockface_transaction_dynamics_v3.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc
LOOKBACK_H = 72.0


def lm(d: date) -> datetime:
    return datetime.combine(d, time.min, TZ)


def win(a: date, b: date) -> tuple[datetime, datetime]:
    return lm(a).astimezone(UTC), lm(b + timedelta(days=1)).astimezone(UTC)


def folds(first, latest, train_days, test_days, max_folds):
    fd = first.astimezone(TZ).date(); end = latest.astimezone(TZ).date(); out = []
    while end >= fd and len(out) < max_folds:
        ts = end - timedelta(days=test_days - 1); te = ts - timedelta(days=1); trs = te - timedelta(days=train_days - 1)
        if trs < fd: break
        out.append({"train": win(trs, te), "test": win(ts, end), "local_days": {"train": [str(trs), str(te)], "test": [str(ts), str(end)]}})
        end -= timedelta(days=test_days)
    return list(reversed(out))


def metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def mapping_sql():
    return "SELECT DISTINCT post_id, blockface_id FROM parking_meters WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL"


def targets(conn, start, end, k, seed):
    return conn.run(f"""
        WITH mapping AS ({mapping_sql()}), capacity AS (
            SELECT blockface_id, COUNT(DISTINCT parking_space_id)::int capacity
            FROM parking_spaces WHERE blockface_id IS NOT NULL GROUP BY blockface_id
        ), slots AS (
            SELECT DISTINCT m.blockface_id,s.slot_start,
                (s.slot_start AT TIME ZONE 'America/Los_Angeles')::date local_date,
                EXTRACT(HOUR FROM(s.slot_start AT TIME ZONE 'America/Los_Angeles'))::int local_hour
            FROM mapping m JOIN parking_state_hourly s ON s.post_id=m.post_id
            WHERE s.slot_start>=:start AND s.slot_start<:end
        )
        SELECT s.blockface_id,s.slot_start,c.capacity,s.local_date,s.local_hour
        FROM slots s JOIN capacity c USING(blockface_id) WHERE c.capacity>0
        ORDER BY hashtext(s.blockface_id::text||'|'||s.slot_start::text||:seed::text),s.blockface_id,s.slot_start
        LIMIT :k""", start=start, end=end, seed=str(seed), k=k)


def labels(conn, ts):
    conn.run("DROP TABLE IF EXISTS _bfv3_labels")
    conn.run("CREATE TEMP TABLE _bfv3_labels(blockface_id text,slot_start timestamptz,capacity int,local_date date,local_hour int)")
    import csv
    from io import StringIO
    b=StringIO(); csv.writer(b,lineterminator='\n').writerows(ts)
    conn.run("COPY _bfv3_labels(blockface_id,slot_start,capacity,local_date,local_hour) FROM STDIN WITH(FORMAT csv)",stream=[b.getvalue().encode()])
    rows=conn.run(f"""
        WITH mapping AS ({mapping_sql()}), sessions AS (
            SELECT DISTINCT t.transmission_id,t.post_id,t.session_start,t.session_end
            FROM meter_transactions t WHERE t.session_end IS NOT NULL AND t.session_end>t.session_start
        ), mapped AS (
            SELECT s.transmission_id,m.blockface_id,s.session_start,s.session_end FROM sessions s JOIN mapping m ON m.post_id=s.post_id
        )
        SELECT z.blockface_id,z.slot_start,z.capacity,
          COUNT(DISTINCT m.transmission_id) FILTER(WHERE m.session_start<=z.slot_start AND m.session_end>z.slot_start)::int active,
          COUNT(DISTINCT m.transmission_id) FILTER(WHERE m.session_start<=z.slot_start-INTERVAL '1 hour' AND m.session_end>z.slot_start-INTERVAL '1 hour')::int active_prev
        FROM _bfv3_labels z LEFT JOIN mapped m ON m.blockface_id=z.blockface_id
        GROUP BY z.blockface_id,z.slot_start,z.capacity""")
    conn.run("DROP TABLE IF EXISTS _bfv3_labels")
    return rows


def learn(conn, start, end):
    hours=max((end-start).total_seconds()/3600.0,1.0)
    hist=conn.run(f"""
      WITH mapping AS ({mapping_sql()})
      SELECT m.blockface_id,
        EXTRACT(ISODOW FROM(t.session_start AT TIME ZONE 'America/Los_Angeles'))::int dow,
        EXTRACT(HOUR FROM(t.session_start AT TIME ZONE 'America/Los_Angeles'))::int hour,
        COUNT(DISTINCT t.transmission_id)::double precision n
      FROM meter_transactions t JOIN mapping m ON m.post_id=t.post_id
      WHERE t.session_start>=:start AND t.session_start<:end GROUP BY 1,2,3""",start=start,end=end)
    rates={(str(r[0]),int(r[1]),int(r[2])):float(r[3])/max(hours/168.0,1/24.0) for r in hist}
    global_rate=float(sum(float(r[3]) for r in hist)/hours) if hist else 0.0
    drows=conn.run(f"""
      WITH mapping AS ({mapping_sql()})
      SELECT m.blockface_id,EXTRACT(EPOCH FROM(t.session_end-t.session_start))/3600.0 duration_h
      FROM meter_transactions t JOIN mapping m ON m.post_id=t.post_id
      WHERE t.session_start>=:start AND t.session_start<:end
        AND t.session_end IS NOT NULL AND t.session_end>t.session_start""",start=start,end=end)
    by_block={}; all_d=[]
    for r in drows:
        d=max(float(r[1]),1e-6);by_block.setdefault(str(r[0]),[]).append(d);all_d.append(d)
    return rates,global_rate,by_block,np.asarray(all_d,float)


def survival_prob(durs, age, extra):
    if len(durs)==0:return math.exp(-extra/1.5)
    denom=max(int(np.sum(durs>age)),1)
    return float(np.sum(durs>age+extra)/denom)


def forecast(conn, ts, rates, global_rate, by_block, global_durs):
    """Causal forecast: session_start is observable; session_end is not."""
    conn.run("DROP TABLE IF EXISTS _bfv3_pred")
    conn.run("CREATE TEMP TABLE _bfv3_pred(blockface_id text,target_slot timestamptz,capacity int,local_date date,local_hour int)")
    import csv
    from io import StringIO
    b=StringIO();csv.writer(b,lineterminator='\n').writerows(ts)
    conn.run("COPY _bfv3_pred(blockface_id,target_slot,capacity,local_date,local_hour) FROM STDIN WITH(FORMAT csv)",stream=[b.getvalue().encode()])
    starts=conn.run(f"""
      WITH mapping AS ({mapping_sql()})
      SELECT p.blockface_id,p.target_slot,
        EXTRACT(EPOCH FROM((p.target_slot-INTERVAL '1 hour')-t.session_start))/3600.0 age_h
      FROM _bfv3_pred p JOIN mapping m ON m.blockface_id=p.blockface_id
      JOIN meter_transactions t ON t.post_id=m.post_id
      WHERE t.session_start<=p.target_slot-INTERVAL '1 hour'
        AND t.session_start>p.target_slot-INTERVAL '73 hours'
        AND t.session_start>=CAST(:floor AS timestamptz)
    """,floor='1970-01-01T00:00:00Z')
    by_target={}
    for r in starts:by_target.setdefault((str(r[0]),r[1]),[]).append(float(r[2]))
    out=[]
    for bf,slot,cap,local_date,local_hour in ts:
        durs=np.asarray(by_block.get(str(bf),global_durs),float); ages=by_target.get((str(bf),slot),[])
        # Probability each session that began before T-1h remains active at T.
        existing=sum(survival_prob(durs,float(age),1.0) for age in ages)
        lam=float(rates.get((str(bf),int(local_date.isoweekday()),int(local_hour)),global_rate))
        new=0.0
        for j in range(12):
            remaining=1.0-(j+0.5)/12.0
            surv=float(np.mean(durs>remaining)) if len(durs) else math.exp(-remaining/1.5)
            new += (lam/12.0)*surv
        expected=max(0.0,existing+new)
        out.append(1.0-min(1.0,expected/max(int(cap),1)))
    conn.run("DROP TABLE IF EXISTS _bfv3_pred")
    return np.asarray(out,float)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-days',type=int,default=6);ap.add_argument('--test-days',type=int,default=1);ap.add_argument('--max-folds',type=int,default=2);ap.add_argument('--max-test-rows',type=int,default=50000);a=ap.parse_args()
    print('🚦 SF PARKING — CAUSAL BLOCKFACE TRANSACTION DYNAMICS V3');c=connect()
    try:
        first,latest=c.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0];fs=folds(first,latest,a.train_days,a.test_days,a.max_folds);reps=[]
        for i,f in enumerate(fs,1):
            print(f"\n[Fold {i}/{len(fs)}] {f['local_days']}");ts=targets(c,*f['test'],a.max_test_rows,31000+i);lab=labels(c,ts);obs={(str(r[0]),r[1]):(int(r[2]),int(r[3]),int(r[4])) for r in lab};rates,gr,bb,gd=learn(c,*f['train']);p=forecast(c,ts,rates,gr,bb,gd);y=[];lag=[]
            for bf,slot,cap,*_ in ts:
                cur=obs[(str(bf),slot)];y.append(1.0-min(1.0,cur[0]/max(cap,1)));lag.append(1.0-min(1.0,cur[1]/max(cap,1)))
            y=np.asarray(y,float);lag=np.asarray(lag,float);dm,pm=metric(y,p),metric(y,lag);print(f"    rows={len(y):,} causal_dynamics_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}");reps.append({'fold':i,'local_days':f['local_days'],'rows':len(y),'causal_dynamics':dm,'persistence':pm})
    finally:c.close()
    w=np.asarray([r['rows'] for r in reps],float);dm=float(np.average([r['causal_dynamics']['mae'] for r in reps],weights=w));pm=float(np.average([r['persistence']['mae'] for r in reps],weights=w));res={'version':3,'model':'causal_blockface_transaction_session_dynamics','ground_truth':'active paid sessions / mapped blockface capacity','causality':'session_start only at forecast origin; session_end used only in training survival and held-out labels','aggregate':{'test_rows':int(w.sum()),'persistence_mae':pm,'causal_dynamics_mae':dm,'improvement_over_persistence':(pm-dm)/pm if pm else None,'promotion':'candidate' if dm<pm else 'retained_only'},'folds':reps};OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(res,indent=2));print('\nFINAL:');print(json.dumps(res['aggregate'],indent=2));print(f'Report: {OUT}')
if __name__=='__main__':main()
