"""M/G/infinity-style parking dynamics benchmark.

A post can carry multiple overlapping paid sessions, so model occupancy as a
Poisson number of active sessions rather than a single binary Markov state.
The forecast uses only completed training sessions plus T-1 state/transaction
history.  Availability is exp(-expected_active_sessions).
"""
from __future__ import annotations

import argparse, json, math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from sf_parking.database import connect

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"models"/"parking_dynamics_mginfinity_v1.json"
TZ=ZoneInfo("America/Los_Angeles"); UTC=timezone.utc


def local_midnight(d:date)->datetime:return datetime.combine(d,time.min,TZ)
def local_window(a:date,b:date):return local_midnight(a).astimezone(UTC),local_midnight(b+timedelta(days=1)).astimezone(UTC)

def folds(first,latest,train_days,test_days,max_folds):
    fd=first.astimezone(TZ).date(); end=latest.astimezone(TZ).date(); out=[]
    while end>=fd and len(out)<max_folds:
        ts=end-timedelta(days=test_days-1); te=ts-timedelta(days=1); trs=te-timedelta(days=train_days-1)
        if trs<fd:break
        out.append({'train':local_window(trs,te),'test':local_window(ts,end),'local_days':{'train':[str(trs),str(te)],'test':[str(ts),str(end)]}})
        end-=timedelta(days=test_days)
    return list(reversed(out))

def metric(y,p):
    e=np.asarray(p)-np.asarray(y); return {'mae':float(np.mean(np.abs(e))), 'rmse':float(np.sqrt(np.mean(e*e))), 'bias':float(np.mean(e))}

def learn_training_stats(conn,start,end):
    rows=conn.run("""
      SELECT post_id,
             COUNT(*)::double precision sessions,
             AVG(EXTRACT(EPOCH FROM (session_end-session_start))/3600.0) mean_duration_h
      FROM meter_transactions
      WHERE session_start>=CAST(:start AS timestamptz)
        AND session_start<CAST(:end AS timestamptz)
        AND session_end IS NOT NULL AND session_end>session_start
      GROUP BY post_id
    """,start=start,end=end)
    hours=max((end-start).total_seconds()/3600,1.0)
    post={str(r[0]):{'lambda_h':float(r[1])/hours,'mean_duration_h':float(r[2])} for r in rows if r[2] and float(r[2])>0}
    hist=conn.run("""
      SELECT EXTRACT(ISODOW FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int dow,
             EXTRACT(HOUR FROM (session_start AT TIME ZONE 'America/Los_Angeles'))::int hr,
             COUNT(*)::double precision n
      FROM meter_transactions
      WHERE session_start>=CAST(:start AS timestamptz) AND session_start<CAST(:end AS timestamptz)
      GROUP BY 1,2
    """,start=start,end=end)
    baseline=sum(float(r[2]) for r in hist)/hours
    season={(int(r[0]),int(r[1])):max(.05,float(r[2])/max(baseline,1e-9)) for r in hist}
    dur=conn.run("""SELECT session_end-session_start FROM meter_transactions WHERE session_start>=CAST(:start AS timestamptz) AND session_start<CAST(:end AS timestamptz) AND session_end IS NOT NULL AND session_end>session_start""",start=start,end=end)
    durations=np.array([max(1e-6,float(r[0].total_seconds()/3600)) for r in dur],float)
    return post,season,baseline,durations

def sample_targets(conn,start,end,k,seed):
    return conn.run("""SELECT t.post_id,t.slot_start,t.paid_availability_probability,p.paid_availability_probability,p.transaction_count,t.local_date,t.local_hour FROM parking_state_hourly t JOIN parking_state_hourly p ON p.post_id=t.post_id AND p.slot_start=t.slot_start-INTERVAL '1 hour' WHERE t.slot_start>=:start AND t.slot_start<:end ORDER BY hashtext(t.post_id||'|'||t.slot_start::text||:seed::text),t.post_id,t.slot_start LIMIT :k""",start=start,end=end,seed=str(seed),k=k)

def predict(conn,targets,train_start,train_end,post,season,baseline,durations):
    # Build recent-start counts by post/target over the previous 24h.
    conn.run('DROP TABLE IF EXISTS _mginf_targets')
    conn.run('CREATE TEMP TABLE _mginf_targets(post_id text,slot_start timestamptz,target double precision,prev double precision,prev_tx double precision,local_date date,local_hour int)')
    import csv
    from io import StringIO
    b=StringIO();csv.writer(b,lineterminator='\n').writerows(targets)
    conn.run("COPY _mginf_targets(post_id,slot_start,target,prev,prev_tx,local_date,local_hour) FROM STDIN WITH(FORMAT csv)",stream=[b.getvalue().encode()])
    rows=conn.run("""
      SELECT t.post_id,t.slot_start,t.target,t.prev,t.prev_tx,t.local_date,t.local_hour,
             EXTRACT(EPOCH FROM (t.slot_start-x.session_start))/3600.0 age_h
      FROM _mginf_targets t
      JOIN meter_transactions x ON x.post_id=t.post_id
       AND x.session_start<t.slot_start
       AND x.session_start>=t.slot_start-INTERVAL '24 hours'
       AND x.session_start>=CAST(:train_start AS timestamptz)
       AND x.session_start<CAST(:train_end AS timestamptz)
      WHERE x.session_end IS NOT NULL AND x.session_end>x.session_start
    """,train_start=train_start,train_end=train_end)
    by_target={}
    for r in rows: by_target.setdefault((str(r[0]),r[1]),[]).append(float(r[7]))
    out=[]
    for r in targets:
        key=(str(r[0]),r[1]); ages=by_target.get(key,[]); dow=int(r[5].isoweekday()); hr=int(r[6]); params=post.get(key[0]); lam=(params['lambda_h'] if params else baseline)*season.get((dow,hr),1.0); mean_d=(params['mean_duration_h'] if params else float(np.mean(durations) if len(durations) else 1.5))
        # Empirical survival estimate; shrink to exponential if sample is sparse.
        if len(durations):
            surv_recent=[float(np.mean(durations>=a+1.0)) for a in ages] if ages else []
            active=sum(surv_recent)
        else: active=0.0
        future_active=lam*mean_d
        # Preserve observed T-1 load as a stabilizing prior when raw sessions are sparse.
        prior=-math.log(max(float(r[3]),1e-9))
        active=.75*active+.25*prior+future_active
        pred=math.exp(-max(0.0,active))
        out.append((float(r[2]),float(r[3]),pred,float(lam),len(ages),mean_d))
    conn.run('DROP TABLE IF EXISTS _mginf_targets')
    return np.asarray(out,float)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-days',type=int,default=6);ap.add_argument('--test-days',type=int,default=1);ap.add_argument('--max-folds',type=int,default=2);ap.add_argument('--max-test-rows',type=int,default=75000);a=ap.parse_args();print('🚦 SF PARKING — M/G/∞ OCCUPANCY DYNAMICS V1');c=connect()
    try:
        first,latest=c.run('SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()')[0];fs=folds(first,latest,a.train_days,a.test_days,a.max_folds);print(f'Local data: {first.astimezone(TZ).date()} → {latest.astimezone(TZ).date()}; folds={len(fs)}');reps=[]
        for i,f in enumerate(fs,1):
            print(f"\n[Fold {i}/{len(fs)}] {f['local_days']}");post,season,baseline,durations=learn_training_stats(c,*f['train']);ts=sample_targets(c,*f['test'],a.max_test_rows,12000+i);arr=predict(c,ts,*f['train'],post,season,baseline,durations);y,lag,p=arr[:,0],arr[:,1],arr[:,2];dm,pm=metric(y,p),metric(y,lag);print(f"    rows={len(y):,} dynamics_mae={dm['mae']:.6f} persistence_mae={pm['mae']:.6f}");reps.append({'fold':i,'local_days':f['local_days'],'rows':len(y),'dynamics':dm,'persistence':pm,'mean_lambda_h':float(np.mean(arr[:,3])),'mean_recent_sessions':float(np.mean(arr[:,4])),'mean_duration_h':float(np.mean(arr[:,5]))})
    finally:c.close()
    w=np.asarray([r['rows'] for r in reps],float);dm=float(np.average([r['dynamics']['mae'] for r in reps],weights=w));pm=float(np.average([r['persistence']['mae'] for r in reps],weights=w));res={'version':1,'model':'mginfinity_active_session_dynamics','causality':'training session history + exact T-1 availability only','aggregate':{'test_rows':int(w.sum()),'persistence_mae':pm,'dynamics_mae':dm,'improvement_over_persistence':(pm-dm)/pm if pm else None,'promotion':'candidate' if dm<pm else 'retained_only'},'folds':reps};OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(res,indent=2));print('\nFINAL:');print(json.dumps(res['aggregate'],indent=2));print(f'Report: {OUT}')
if __name__=='__main__':main()
