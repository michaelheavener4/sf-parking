"""Blockface transaction-implied occupancy dynamics V2.

Research formulation:
- Unit: blockface rather than meter/post.
- Label: active paid sessions / mapped parking-space capacity.
- As-of state: active paid sessions exactly one hour before target.
- Forecast: existing-session residual-life survival + expected arrivals during
  the one-hour horizon, using training-only non-homogeneous arrival rates.
- Baseline: exact one-hour persistence of the same transaction-implied state.

The meter->blockface mapping is explicitly deduplicated so historical mapping
rows cannot multiply sessions.
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
OUT = ROOT / "models" / "blockface_transaction_dynamics_v2.json"
TZ = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def lm(d: date) -> datetime:
    return datetime.combine(d, time.min, TZ)


def win(a: date, b: date) -> tuple[datetime, datetime]:
    return lm(a).astimezone(UTC), lm(b + timedelta(days=1)).astimezone(UTC)


def make_folds(first, latest, train_days, test_days, max_folds):
    fd = first.astimezone(TZ).date()
    end = latest.astimezone(TZ).date()
    out = []
    while end >= fd and len(out) < max_folds:
        ts = end - timedelta(days=test_days - 1)
        te = ts - timedelta(days=1)
        trs = te - timedelta(days=train_days - 1)
        if trs < fd:
            break
        out.append({
            "train": win(trs, te),
            "test": win(ts, end),
            "local_days": {"train": [str(trs), str(te)], "test": [str(ts), str(end)]},
        })
        end -= timedelta(days=test_days)
    return list(reversed(out))


def metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def mapping_sql():
    return """
        SELECT DISTINCT post_id, blockface_id
        FROM parking_meters
        WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL
    """


def build_targets(conn, start, end, k, seed):
    return conn.run(
        f"""
        WITH mapping AS ({mapping_sql()}),
        capacity AS (
            SELECT blockface_id, COUNT(DISTINCT parking_space_id)::int AS capacity
            FROM parking_spaces
            WHERE blockface_id IS NOT NULL
            GROUP BY blockface_id
        ),
        slots AS (
            SELECT DISTINCT m.blockface_id, s.slot_start,
                   (s.slot_start AT TIME ZONE 'America/Los_Angeles')::date local_date,
                   EXTRACT(HOUR FROM (s.slot_start AT TIME ZONE 'America/Los_Angeles'))::int local_hour
            FROM mapping m
            JOIN parking_state_hourly s ON s.post_id=m.post_id
            WHERE s.slot_start>=:start AND s.slot_start<:end
        )
        SELECT s.blockface_id,s.slot_start,c.capacity,s.local_date,s.local_hour
        FROM slots s JOIN capacity c USING(blockface_id)
        WHERE c.capacity>0
        ORDER BY hashtext(s.blockface_id::text||'|'||s.slot_start::text||:seed::text),s.blockface_id,s.slot_start
        LIMIT :k
        """,
        start=start, end=end, seed=str(seed), k=k,
    )


def observed(conn, targets):
    conn.run("DROP TABLE IF EXISTS _bfv2_targets")
    conn.run("""CREATE TEMP TABLE _bfv2_targets(blockface_id text,slot_start timestamptz,capacity int,local_date date,local_hour int)""")
    import csv
    from io import StringIO
    b = StringIO(); csv.writer(b, lineterminator="\n").writerows(targets)
    conn.run("COPY _bfv2_targets(blockface_id,slot_start,capacity,local_date,local_hour) FROM STDIN WITH(FORMAT csv)", stream=[b.getvalue().encode()])
    rows = conn.run(
        f"""
        WITH mapping AS ({mapping_sql()}),
        sessions AS (
            SELECT DISTINCT t.transmission_id,t.post_id,t.session_start,t.session_end
            FROM meter_transactions t
            WHERE t.session_end IS NOT NULL AND t.session_end>t.session_start
        ),
        mapped AS (
            SELECT s.transmission_id,m.blockface_id,s.session_start,s.session_end
            FROM sessions s JOIN mapping m ON m.post_id=s.post_id
        )
        SELECT z.blockface_id,z.slot_start,z.capacity,
               COUNT(DISTINCT m.transmission_id) FILTER(WHERE m.session_start<=z.slot_start AND m.session_end>z.slot_start) active_now,
               COUNT(DISTINCT m.transmission_id) FILTER(WHERE m.session_start<=z.slot_start-INTERVAL '1 hour' AND m.session_end>z.slot_start-INTERVAL '1 hour') active_prev
        FROM _bfv2_targets z
        LEFT JOIN mapped m ON m.blockface_id=z.blockface_id
        GROUP BY z.blockface_id,z.slot_start,z.capacity
        """
    )
    conn.run("DROP TABLE IF EXISTS _bfv2_targets")
    return rows


def learn(conn, start, end):
    hours = max((end-start).total_seconds()/3600.0, 1.0)
    hist = conn.run(
        f"""
        WITH mapping AS ({mapping_sql()})
        SELECT m.blockface_id,
               EXTRACT(ISODOW FROM(t.session_start AT TIME ZONE 'America/Los_Angeles'))::int dow,
               EXTRACT(HOUR FROM(t.session_start AT TIME ZONE 'America/Los_Angeles'))::int hour,
               COUNT(DISTINCT t.transmission_id)::double precision n
        FROM meter_transactions t JOIN mapping m ON m.post_id=t.post_id
        WHERE t.session_start>=:start AND t.session_start<:end
        GROUP BY 1,2,3
        """, start=start, end=end)
    rates = {(str(r[0]),int(r[1]),int(r[2])): float(r[3])/max(hours/168.0,1.0/24.0) for r in hist}
    global_rate = float(sum(float(r[3]) for r in hist)/hours) if hist else 0.0

    drows = conn.run(
        f"""
        WITH mapping AS ({mapping_sql()})
        SELECT m.blockface_id, EXTRACT(EPOCH FROM(t.session_end-t.session_start))/3600.0 duration_h
        FROM meter_transactions t JOIN mapping m ON m.post_id=t.post_id
        WHERE t.session_start>=:start AND t.session_start<:end
          AND t.session_end IS NOT NULL AND t.session_end>t.session_start
        """, start=start, end=end)
    by_block = {}
    all_d = []
    for r in drows:
        d=max(float(r[1]),1e-6); by_block.setdefault(str(r[0]),[]).append(d); all_d.append(d)
    return rates,global_rate,by_block,np.asarray(all_d,float)


def cond_survival(durs, ages, extra):
    if len(durs)==0:
        return np.full(len(ages),math.exp(-extra/1.5))
    durs=np.asarray(durs,float); out=[]
    for age in ages:
        denom=max(int(np.sum(durs>age)),1)
        out.append(float(np.sum(durs>age+extra))/denom)
    return np.asarray(out,float)


def predict(conn, targets, rates, global_rate, by_block, global_durs):
    conn.run("DROP TABLE IF EXISTS _bfv2_pred")
    conn.run("""CREATE TEMP TABLE _bfv2_pred(blockface_id text,target_slot timestamptz,capacity int,local_date date,local_hour int)""")
    import csv
    from io import StringIO
    b=StringIO();csv.writer(b,lineterminator="\n").writerows(targets)
    conn.run("COPY _bfv2_pred(blockface_id,target_slot,capacity,local_date,local_hour) FROM STDIN WITH(FORMAT csv)",stream=[b.getvalue().encode()])
    active=conn.run(
        f"""
        WITH mapping AS ({mapping_sql()})
        SELECT p.blockface_id,p.target_slot,
               EXTRACT(EPOCH FROM((p.target_slot-INTERVAL '1 hour')-t.session_start))/3600.0 age_h
        FROM _bfv2_pred p
        JOIN mapping m ON m.blockface_id=p.blockface_id
        JOIN meter_transactions t ON t.post_id=m.post_id
        WHERE t.session_start<=p.target_slot-INTERVAL '1 hour'
          AND t.session_end>p.target_slot-INTERVAL '1 hour'
          AND t.session_end IS NOT NULL
          AND t.session_start>=p.target_slot-INTERVAL '49 hours'
        """
    )
    by_target={}
    for r in active: by_target.setdefault((str(r[0]),r[1]),[]).append(float(r[2]))
    out=[]
    for blockface_id,slot,cap,local_date,local_hour in targets:
        key=(str(blockface_id),slot); ages=np.asarray(by_target.get(key,[]),float); durs=np.asarray(by_block.get(str(blockface_id),global_durs),float)
        existing=float(np.sum(cond_survival(durs,ages,1.0))) if len(ages) else 0.0
        dow=int(local_date.isoweekday()); hr=int(local_hour); lam=float(rates.get((str(blockface_id),dow,hr),global_rate))
        arrivals=0.0
        for j in range(6):
            remaining=1.0-(j+0.5)/6.0
            surv=float(np.mean(durs>remaining)) if len(durs) else math.exp(-remaining/1.5)
            arrivals += (lam/6.0)*surv
        expected=existing+arrivals
        availability=1.0-min(1.0,max(0.0,expected/max(int(cap),1)))
        out.append(availability)
    conn.run("DROP TABLE IF EXISTS _bfv2_pred")
    return np.asarray(out,float)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-days',type=int,default=6);ap.add_argument('--test-days',type=int,default=1);ap.add_argument('--max-folds',type=int,default=2);ap.add_argument('--max-test-rows',type=int,default=50000);a=ap.parse_args()
    print('🚦 SF PARKING — BLOCKFACE TRANSACTION DYNAMICS V2');c=connect()
    try:
        first,latest=c.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0];fs=make_folds(first,latest,a.train_days,a.test_days,a.max_folds);reps=[]
        for i,f in enumerate(fs,1):
            print(f"\n[Fold {i}/{len(fs)}] {f['local_days']}");targets=build_targets(c,*f['test'],a.max_test_rows,21000+i);obs=observed(c,targets);obs_map={(str(r[0]),r[1]):(int(r[2]),int(r[3]),int(r[4])) for r in obs};rates,gr,bb,gd=learn(c,*f['train']);pred=predict(c,targets,rates,gr,bb,gd);y=[];lag=[]
            for bf,slot,cap,*_ in targets:
                now,prev,_=obs_map[(str(bf),slot)];y.append(1.0-min(1.0,now/max(cap,1)));lag.append(1.0-min(1.0,prev/max(cap,1)))
            y=np.asarray(y,float);lag=np.asarray(lag,float);dm,pm=metric(y,pred),metric(y,lag);print(f"    rows={len(y):,} dynamics_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}");reps.append({'fold':i,'local_days':f['local_days'],'rows':len(y),'dynamics':dm,'persistence':pm})
    finally:c.close()
    w=np.asarray([r['rows'] for r in reps],float);dm=float(np.average([r['dynamics']['mae'] for r in reps],weights=w));pm=float(np.average([r['persistence']['mae'] for r in reps],weights=w));res={'version':2,'model':'blockface_transaction_session_dynamics','ground_truth':'raw transaction-implied occupancy = active paid sessions / mapped blockface capacity','aggregate':{'test_rows':int(w.sum()),'persistence_mae':pm,'dynamics_mae':dm,'improvement_over_persistence':(pm-dm)/pm if pm else None,'promotion':'candidate' if dm<pm else 'retained_only'},'folds':reps};OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(res,indent=2));print('\nFINAL:');print(json.dumps(res['aggregate'],indent=2));print(f'Report: {OUT}')
if __name__=='__main__':main()
