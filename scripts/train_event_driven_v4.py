"""V4 authoritative transition tournament: exact timestamp windows, unbiased samples, rolling folds."""
from __future__ import annotations
import argparse, json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"models"/"paid_state_event_tournament_v4.json"; TZ=ZoneInfo("America/Los_Angeles"); UTC=timezone.utc

def metric(y,p):
 e=np.asarray(p)-np.asarray(y); return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(np.mean(e))}

def rank_auc(y,s):
 y,s=np.asarray(y,int),np.asarray(s,float); pos=int(y.sum()); neg=int((1-y).sum())
 if not pos or not neg:return None
 o=np.argsort(s,kind="mergesort"); ss=s[o]; r=np.empty(len(s),float); i=0
 while i<len(s):
  j=i+1
  while j<len(s) and ss[j]==ss[i]:j+=1
  r[o[i:j]]=(i+j+1)/2;i=j
 return float((r[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def ap(y,s):
 y,s=np.asarray(y,int),np.asarray(s,float); pos=int(y.sum())
 if not pos:return None
 o=np.argsort(-s,kind="mergesort"); ys=y[o]; tp=np.cumsum(ys); fp=np.cumsum(1-ys); prec=tp/np.maximum(tp+fp,1); rec=tp/pos
 return float(np.sum((rec[1:]-rec[:-1])*prec[1:])+prec[0]*rec[0])

def binary(y,p):
 y,p=np.asarray(y,int),np.asarray(p,float); z=p>=.5; tp=int(np.sum(z&(y==1))); fp=int(np.sum(z&(y==0))); fn=int(np.sum(~z&(y==1))); tn=int(np.sum(~z&(y==0))); pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.
 return {"n":len(y),"positives":int(y.sum()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":pr,"recall":rc,"f1":2*pr*rc/(pr+rc) if pr+rc else 0.,"brier":float(np.mean((p-y)**2)),"roc_auc":rank_auc(y,p),"pr_auc":ap(y,p)}

def correct_prior(p,sp,np_):
 p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
 if not(0<sp<1 and 0<np_<1):return p
 o=p/(1-p); so=sp/(1-sp); no=np_/(1-np_); q=o*(no/so); return np.clip(q/(1+q),1e-9,1-1e-9)

def local_midnight(d):return datetime.combine(d,time.min,TZ)
def local_window(a,b):return local_midnight(a).astimezone(UTC),local_midnight(b+timedelta(days=1)).astimezone(UTC)
def folds(first,latest,tr,va,te,n):
 fd,ll=first.astimezone(TZ).date(),latest.astimezone(TZ).date(); end=ll; out=[]
 while end>=fd:
  ts=end-timedelta(days=te-1); ve=ts-timedelta(days=1); vs=ve-timedelta(days=va-1); tre=vs-timedelta(days=1); trs=tre-timedelta(days=tr-1)
  if trs<fd:break
  t0,t1=local_window(ts,end); v0,v1=local_window(vs,ve); r0,r1=local_window(trs,tre); t1=min(t1,latest+timedelta(microseconds=1))
  out.append({"train":[r0,r1],"validation":[v0,v1],"test":[t0,t1],"local_days":{"train":[str(trs),str(tre)],"validation":[str(vs),str(ve)],"test":[str(ts),str(end)]}}); end-=timedelta(days=te)
 return list(reversed(out[-n:]))

def population(conn,a,b,thr):
 r=conn.run("""SELECT count(*),count(p.paid_availability_probability),count(*) FILTER(WHERE p.paid_availability_probability IS NOT NULL AND abs(s.paid_availability_probability-p.paid_availability_probability)>=:thr),avg(abs(s.paid_availability_probability-p.paid_availability_probability)) FILTER(WHERE p.paid_availability_probability IS NOT NULL),avg((s.paid_availability_probability=p.paid_availability_probability)::int) FILTER(WHERE p.paid_availability_probability IS NOT NULL) FROM parking_state_hourly s LEFT JOIN parking_state_hourly p ON p.post_id=s.post_id AND p.slot_start=s.slot_start-INTERVAL '1 hour' WHERE s.slot_start>=:a AND s.slot_start<:b""",a=a,b=b,thr=thr)[0]
 n,e,ev,m,u=r; return {"rows":int(n),"rows_with_exact_prev":int(e),"events":int(ev),"event_rate":float(ev/e) if e else None,"mean_abs_delta":float(m) if m is not None else None,"unchanged_rate":float(u) if u is not None else None}

def sample(conn,a,b,k,seed):
 return conn.run("""SELECT post_id,slot_start,paid_availability_probability,meter_type,local_hour,local_date FROM parking_state_hourly WHERE slot_start>=:a AND slot_start<:b ORDER BY hashtext(post_id||'|'||slot_start::text||:seed::text),post_id,slot_start LIMIT :k""",a=a,b=b,seed=str(seed),k=k)

def event_sample(conn,a,b,k,seed,thr):
 return conn.run("""SELECT s.post_id,s.slot_start,s.paid_availability_probability,s.meter_type,s.local_hour,s.local_date FROM parking_state_hourly s JOIN parking_state_hourly p ON p.post_id=s.post_id AND p.slot_start=s.slot_start-INTERVAL '1 hour' WHERE s.slot_start>=:a AND s.slot_start<:b AND abs(s.paid_availability_probability-p.paid_availability_probability)>=:thr ORDER BY hashtext(s.post_id||'|'||s.slot_start::text||:seed::text),s.post_id,s.slot_start LIMIT :k""",a=a,b=b,k=k,seed=str(seed),thr=thr)

def audit_targets(conn,ts,thr):
 if not ts:return {"rows":0,"rows_with_exact_prev":0,"events":0,"event_rate":None}
 conn.run("DROP TABLE IF EXISTS _v4_targets"); conn.run("CREATE TEMP TABLE _v4_targets(post_id text,slot_start timestamptz,target double precision,meter_type text,local_hour int,local_date date)")
 import csv; from io import StringIO; b=StringIO(); csv.writer(b,lineterminator="\n").writerows(ts); conn.run("COPY _v4_targets(post_id,slot_start,target,meter_type,local_hour,local_date) FROM STDIN WITH(FORMAT csv)",stream=[b.getvalue().encode()])
 r=conn.run("""SELECT count(*),count(p.paid_availability_probability),count(*) FILTER(WHERE p.paid_availability_probability IS NOT NULL AND abs(t.target-p.paid_availability_probability)>=:thr) FROM _v4_targets t LEFT JOIN parking_state_hourly p ON p.post_id=t.post_id AND p.slot_start=t.slot_start-INTERVAL '1 hour'""",thr=thr)[0]; conn.run("DROP TABLE IF EXISTS _v4_targets"); n,e,ev=r; return {"rows":int(n),"rows_with_exact_prev":int(e),"events":int(ev),"event_rate":float(ev/e) if e else None}

def feats(conn,ts,cfg,batch):
 fs=[]
 for i in range(0,len(ts),batch):
  f=build_spatial_features(conn,ts[i:i+batch],config=cfg)
  if not f.empty:fs.append(f)
  print(f"      features {min(i+batch,len(ts)):,}/{len(ts):,}",flush=True)
 import pandas as pd
 return pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()

def fit(tr,va,thr):
 import lightgbm as lgb
 c=dict(n_estimators=700,learning_rate=.04,num_leaves=31,min_child_samples=100,subsample=.9,colsample_bytree=.9,reg_alpha=.05,reg_lambda=.2,random_state=42,n_jobs=-1,verbosity=-1); yt=(np.abs(tr.delta)>=thr).astype(int); yv=(np.abs(va.delta)>=thr).astype(int)
 if int(yt.sum())<20 or int(yv.sum())<5:raise RuntimeError(f"too few transition examples: train={int(yt.sum())}, validation={int(yv.sum())}")
 ev=lgb.LGBMClassifier(objective="binary",**c); ev.fit(tr[FEATURES_SPATIAL],yt,eval_X=va[FEATURES_SPATIAL],eval_y=yv,callbacks=[lgb.early_stopping(60,verbose=False)]); te,ve=tr[yt==1],va[yv==1]
 dr=lgb.LGBMClassifier(objective="binary",**c); dr.fit(te[FEATURES_SPATIAL],(te.delta>0).astype(int),eval_X=ve[FEATURES_SPATIAL],eval_y=(ve.delta>0).astype(int),callbacks=[lgb.early_stopping(60,verbose=False)]); mag=lgb.LGBMRegressor(objective="regression",**c); mag.fit(te[FEATURES_SPATIAL],np.abs(te.delta),eval_X=ve[FEATURES_SPATIAL],eval_y=np.abs(ve.delta),callbacks=[lgb.early_stopping(60,verbose=False)]); return ev,dr,mag,int(yt.sum()),int(yv.sum())

def run_fold(conn,f,args,idx):
 r0,r1=f["train"];v0,v1=f["validation"];t0,t1=f["test"]; pop=population(conn,t0,t1,args.threshold); tests=sample(conn,t0,t1,args.max_test_rows,4000+idx); sa=audit_targets(conn,tests,args.threshold); ratio=sa["event_rate"]/pop["event_rate"] if pop["event_rate"] and sa["event_rate"] else None
 if ratio is not None and not(.5<=ratio<=2):raise RuntimeError(f"fold {idx}: sample/population event-rate ratio {ratio:.3f}x")
 trbg=sample(conn,r0,r1,args.max_train_rows,1000+idx); evts=event_sample(conn,r0,r1,args.event_enrichment,2000+idx,args.threshold); seen={(x[0],x[1]) for x in trbg}; evts=[x for x in evts if (x[0],x[1]) not in seen]; trts=trbg+evts; vats=sample(conn,v0,v1,args.max_validation_rows,3000+idx); cfg=SpatialFeatureConfig(args.neighbor_k,args.neighbor_radius_m)
 print(f"    fold {idx}: pop={pop['rows']:,} pop_events={pop['events']:,} sample={len(tests):,} sample_events={sa['events']:,}")
 tr,va,te=feats(conn,trts,cfg,args.batch),feats(conn,vats,cfg,args.batch),feats(conn,tests,cfg,args.batch)
 for d in (tr,va,te):d["delta"]=d.target-d.lag1_availability
 fa=np.abs(te.delta.to_numpy(float)); fev=int(np.sum(fa>=args.threshold-1e-12)); f_rate=fev/len(te) if len(te) else 0.; retention=f_rate/sa["event_rate"] if sa["event_rate"] else None
 if retention is not None and retention<.9:raise RuntimeError(f"fold {idx}: feature retention {retention:.3f}x")
 ev,dr,mag,trn,van=fit(tr,va,args.threshold); x=te[FEATURES_SPATIAL];lag=te.lag1_availability.to_numpy(float);y=te.target.to_numpy(float);d=y-lag;ey=(np.abs(d)>=args.threshold).astype(int);raw=ev.predict_proba(x)[:,1];sp=float(np.mean(np.abs(tr.delta)>=args.threshold));np_=float(np.mean(np.abs(va.delta)>=args.threshold));pe=correct_prior(raw,sp,np_);up=dr.predict_proba(x)[:,1];mg=np.maximum(0,mag.predict(x));state=np.clip(lag+pe*(2*up-1)*mg,0,1)
 return {"windows":{"local_days":f["local_days"],"utc":{k:[v.isoformat() for v in f[k]] for k in ("train","validation","test")}},"population_audit":pop,"sample_audit":{**sa,"rate_ratio":ratio},"feature_audit":{"rows":len(te),"events":fev,"event_rate":f_rate,"retention_vs_sample":retention},"rows":{"train":len(tr),"validation":len(va),"test":len(te),"enriched_training_events":len(evts)},"transition_examples":{"train":trn,"validation":van},"event_detector":{**binary(ey,pe),"raw_training_event_rate":sp,"validation_event_rate":np_},"persistence":metric(y,lag),"event_driven":metric(y,state)}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--train-days",type=int,default=6);p.add_argument("--validation-days",type=int,default=1);p.add_argument("--test-days",type=int,default=1);p.add_argument("--max-folds",type=int,default=10);p.add_argument("--max-train-rows",type=int,default=250000);p.add_argument("--max-validation-rows",type=int,default=75000);p.add_argument("--max-test-rows",type=int,default=75000);p.add_argument("--event-enrichment",type=int,default=50000);p.add_argument("--neighbor-k",type=int,default=24);p.add_argument("--neighbor-radius-m",type=float,default=250);p.add_argument("--batch",type=int,default=10000);p.add_argument("--threshold",type=float,default=.10);a=p.parse_args(); print("🚦 SF PARKING — EVENT-DRIVEN STATE TRANSITION TOURNAMENT V4")
 c=connect()
 try:
  first,latest=c.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0]; fs=folds(first,latest,a.train_days,a.validation_days,a.test_days,a.max_folds); print(f"Local data: {first.astimezone(TZ).date()} → {latest.astimezone(TZ).date()}; folds={len(fs)}")
  if not fs:raise RuntimeError("not enough history for rolling-origin folds")
  reps=[]
  for i,f in enumerate(fs,1):print(f"\n[Fold {i}/{len(fs)}]");reps.append(run_fold(c,f,a,i))
 finally:c.close()
 weights=[r["rows"]["test"] for r in reps]; pm=float(np.average([r["persistence"]["mae"] for r in reps],weights=weights)); em=float(np.average([r["event_driven"]["mae"] for r in reps],weights=weights)); tp=sum(r["event_detector"]["tp"] for r in reps);fp=sum(r["event_detector"]["fp"] for r in reps);fn=sum(r["event_detector"]["fn"] for r in reps);pos=sum(r["event_detector"]["positives"] for r in reps);res={"version":4,"ground_truth":"exact same-post T-1 hour; timestamp windows derived from Pacific local calendar days","fold_count":len(reps),"aggregate":{"test_rows":sum(weights),"actual_events":pos,"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else 0.,"recall":tp/(tp+fn) if tp+fn else 0.,"persistence_mae":pm,"event_driven_mae":em,"improvement_over_persistence":(pm-em)/pm if pm else None,"promotion":"candidate" if em<pm else "retained_only"},"folds":reps}; OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(res,indent=2,default=str));print("\nFINAL:");print(json.dumps(res["aggregate"],indent=2));print(f"Report: {OUT}")
if __name__=="__main__":raise SystemExit(main())
