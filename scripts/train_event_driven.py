"""Event-driven parking state transition tournament.

Three-stage model: detect a meaningful change, predict direction, then predict
conditional magnitude. Training may enrich rare events; validation/test remain
chronological and unbiased. All model features are strictly pre-target.
"""
from __future__ import annotations
import argparse, json, math
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sf_parking.database import connect
from sf_parking.ml_features import FEATURES_SPATIAL, SpatialFeatureConfig, build_spatial_features

try:
    from scripts.benchmark_paid_state_lgbm import sample_day_targets
except ModuleNotFoundError as exc:
    if exc.name != "scripts": raise
    from benchmark_paid_state_lgbm import sample_day_targets

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"models"/"paid_state_event_tournament.json"

def metric(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(np.mean(e))}

def binary_metrics(y,p,threshold=.5):
    y,p=np.asarray(y,int),np.asarray(p,float); pred=p>=threshold
    tp,fp=int(np.sum(pred&(y==1))),int(np.sum(pred&(y==0))); fn,tn=int(np.sum(~pred&(y==1))),int(np.sum(~pred&(y==0)))
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {"n":len(y),"positives":int(y.sum()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.,"brier":float(np.mean((p-y)**2))}

def rank_auc(y,score):
    y,score=np.asarray(y,int),np.asarray(score,float); order=np.argsort(score,kind="mergesort"); s=score[order]; ranks=np.empty(len(score),float); i=0
    while i<len(score):
        j=i+1
        while j<len(score) and s[j]==s[i]: j+=1
        ranks[order[i:j]]=(i+j+1)/2; i=j
    npos,nneg=int(y.sum()),int((1-y).sum())
    if not npos or not nneg: return None
    return float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def average_precision(y,score):
    y,score=np.asarray(y,int),np.asarray(score,float); positives=int(y.sum())
    if not positives: return None
    order=np.argsort(-score,kind="mergesort"); ys=y[order]; tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    precision=tp/np.maximum(tp+fp,1); recall=tp/positives
    return float(np.sum((recall[1:]-recall[:-1])*precision[1:])+precision[0]*recall[0])

def prior_correct_probability(p,sample_prior,natural_prior):
    """Undo the prior shift caused by deliberately oversampling rare events."""
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    if sample_prior<=0 or sample_prior>=1 or natural_prior<=0 or natural_prior>=1: return p
    odds=p/(1-p); sample_odds=sample_prior/(1-sample_prior); natural_odds=natural_prior/(1-natural_prior)
    return np.clip((odds*(natural_odds/sample_odds))/(1+odds*(natural_odds/sample_odds)),1e-9,1-1e-9)

def transition_metrics(y,lag,pred,threshold):
    d=y-lag; mask=np.abs(d)>=threshold-1e-12
    if not mask.any(): return {"n":0,"mae":None,"direction_accuracy":None,"mean_abs_delta":None}
    pd=pred-lag
    return {"n":int(mask.sum()),"mae":float(np.mean(np.abs(pred[mask]-y[mask]))),"direction_accuracy":float(np.mean((d[mask]>0)==(pd[mask]>0))),"mean_abs_delta":float(np.mean(np.abs(d[mask])))}

def choose_windows(first,latest,train_days,val_days,test_days):
    first_day,last_day=first.date(),latest.date(); available=(last_day-first_day).days+1
    td,vd=min(test_days,max(1,available//4)),min(val_days,max(1,available//4)); tr=min(train_days,max(1,available-td-vd))
    test_start=last_day-timedelta(days=td-1); val_start=test_start-timedelta(days=vd); train_start=max(first_day,val_start-timedelta(days=tr))
    return train_start,val_start,test_start,last_day

def collect_background(conn,start:date,end:date,per_day:int,seed:int,cap:int):
    rows=[]; day=start
    while day<=end and len(rows)<cap:
        part=sample_day_targets(conn,day,per_day,seed); rows.extend(part); print(f"  background {day}: +{len(part):,}; total={len(rows):,}",flush=True); day+=timedelta(days=1)
    return rows[:cap]

def collect_events(conn,start:date,end:date,per_day:int,seed:int,threshold:float,cap:int):
    rows=[]; day=start
    sql="""SELECT s.post_id,s.slot_start,s.paid_availability_probability,s.meter_type,s.local_hour,s.local_date
      FROM parking_state_hourly s JOIN parking_state_hourly p ON p.post_id=s.post_id AND p.slot_start=s.slot_start-INTERVAL '1 hour'
      WHERE s.local_date=:day AND abs(s.paid_availability_probability-p.paid_availability_probability)>=:threshold
      ORDER BY hashtext(s.post_id||'|'||s.slot_start::text||:seed::text) LIMIT :limit_rows"""
    while day<=end and len(rows)<cap:
        part=conn.run(sql,day=day,threshold=threshold,seed=str(seed),limit_rows=per_day); rows.extend(part); print(f"  enriched   {day}: +{len(part):,}; total={len(rows):,}",flush=True); day+=timedelta(days=1)
    return rows[:cap]

def features(conn,targets,cfg,batch):
    frames=[]
    for i in range(0,len(targets),batch):
        f=build_spatial_features(conn,targets[i:i+batch],config=cfg)
        if not f.empty: frames.append(f)
        print(f"  features {min(i+batch,len(targets)):,}/{len(targets):,}",flush=True)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def fit_models(train,val,threshold):
    import lightgbm as lgb
    common=dict(n_estimators=900,learning_rate=.035,num_leaves=31,min_child_samples=100,subsample=.9,colsample_bytree=.9,reg_alpha=.05,reg_lambda=.2,random_state=42,n_jobs=-1,verbosity=-1)
    yt=(np.abs(train.delta)>=threshold).astype(int); yv=(np.abs(val.delta)>=threshold).astype(int)
    event=lgb.LGBMClassifier(objective="binary",**common); event.fit(train[FEATURES_SPATIAL],yt,eval_X=val[FEATURES_SPATIAL],eval_y=yv,callbacks=[lgb.early_stopping(75,verbose=False)])
    te,ve=train[yt==1],val[yv==1]
    if len(te)==0 or len(ve)==0: raise RuntimeError("No transition examples available for conditional models")
    direction=lgb.LGBMClassifier(objective="binary",**common); direction.fit(te[FEATURES_SPATIAL],(te.delta>0).astype(int),eval_X=ve[FEATURES_SPATIAL],eval_y=(ve.delta>0).astype(int),callbacks=[lgb.early_stopping(75,verbose=False)])
    magnitude=lgb.LGBMRegressor(objective="regression",**common); magnitude.fit(te[FEATURES_SPATIAL],np.abs(te.delta),eval_X=ve[FEATURES_SPATIAL],eval_y=np.abs(ve.delta),callbacks=[lgb.early_stopping(75,verbose=False)])
    return event,direction,magnitude

def main():
    ap=argparse.ArgumentParser()
    for name,default in (("train-days",90),("validation-days",7),("test-days",7),("max-train-rows",250000),("max-validation-rows",100000),("max-test-rows",150000),("neighbor-k",24),("feature-batch-size",10000)): ap.add_argument("--"+name,type=int,default=default)
    ap.add_argument("--neighbor-radius-m",type=float,default=250); ap.add_argument("--transition-threshold",type=float,default=.10); ap.add_argument("--event-enrichment",type=int,default=50000); args=ap.parse_args()
    print("🚦 SF PARKING — EVENT-DRIVEN STATE TRANSITION TOURNAMENT"); conn=connect()
    try:
        first,latest=conn.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0]
        tr0,v0,t0,t1=choose_windows(first,latest,args.train_days,args.validation_days,args.test_days); print(f"Train {tr0} → {v0-timedelta(days=1)}; Validation {v0} → {t0-timedelta(days=1)}; Test {t0} → {t1}")
        tr_days,va_days,te_days=max(1,(v0-tr0).days),max(1,(t0-v0).days),max(1,(t1-t0).days+1); bg=min(50000,max(1000,math.ceil(args.max_train_rows/tr_days))); ev=max(500,math.ceil(args.event_enrichment/tr_days))
        print("[1/4] Chronological samples")
        background=collect_background(conn,tr0,v0-timedelta(days=1),bg,61,args.max_train_rows); enriched=collect_events(conn,tr0,v0-timedelta(days=1),ev,62,args.transition_threshold,args.event_enrichment)
        seen={(r[0],r[1]) for r in background}; enriched=[r for r in enriched if (r[0],r[1]) not in seen]; train_targets=background+enriched
        val_targets=collect_background(conn,v0,t0-timedelta(days=1),max(1000,math.ceil(args.max_validation_rows/va_days)),63,args.max_validation_rows); test_targets=collect_background(conn,t0,t1,max(1000,math.ceil(args.max_test_rows/te_days)),64,args.max_test_rows)
        print(f"Training={len(train_targets):,} (enriched events={len(enriched):,}); validation={len(val_targets):,}; test={len(test_targets):,}")
        cfg=SpatialFeatureConfig(args.neighbor_k,args.neighbor_radius_m); print("[2/4] Leakage-safe features")
        train=features(conn,train_targets,cfg,args.feature_batch_size); val=features(conn,val_targets,cfg,args.feature_batch_size); test=features(conn,test_targets,cfg,args.feature_batch_size)
    finally: conn.close()
    for df in (train,val,test): df["delta"]=df.target-df.lag1_availability
    print("[3/4] Training event detector + direction + magnitude")
    event,direction,magnitude=fit_models(train,val,args.transition_threshold)
    x=test[FEATURES_SPATIAL]; lag=test.lag1_availability.to_numpy(float); y=test.target.to_numpy(float); actual=y-lag; event_y=(np.abs(actual)>=args.transition_threshold).astype(int)
    raw_event=event.predict_proba(x)[:,1]
    sample_prior=float(np.mean(np.abs(train.delta)>=args.transition_threshold)); natural_prior=float(np.mean(np.abs(val.delta)>=args.transition_threshold))
    p_event=prior_correct_probability(raw_event,sample_prior,natural_prior)
    p_up=direction.predict_proba(x)[:,1]; mag=np.maximum(0,magnitude.predict(x)); expected_delta=p_event*(2*p_up-1)*mag; state=np.clip(lag+expected_delta,0,1)
    print("[4/4] Results")
    er=binary_metrics(event_y,p_event); er["roc_auc"]=rank_auc(event_y,p_event); er["pr_auc"]=average_precision(event_y,p_event); er["raw_training_event_rate"]=sample_prior; er["calibration_target_event_rate"]=natural_prior
    print("event_detector:",json.dumps(er,sort_keys=True)); em=event_y==1; da=float(np.mean((actual[em]>0)==(p_up[em]>=.5))) if em.any() else None
    print(f"direction_on_actual_events: {da}"); print("persistence:",json.dumps(metric(y,lag))); print("event_driven_expected_state:",json.dumps(metric(y,state)))
    transitions={str(t):transition_metrics(y,lag,state,t) for t in (.05,.10,.15,.25)}
    for t,r in transitions.items(): print(f"|Δ|>={t}: {r}")
    sm,pm=metric(y,state),metric(y,lag)
    report={"version":1,"model":"event_direction_magnitude_v1","transition_threshold":args.transition_threshold,"windows":{"train":[str(tr0),str(v0-timedelta(days=1))],"validation":[str(v0),str(t0-timedelta(days=1))],"test":[str(t0),str(t1)]},"rows":{"train":len(train),"validation":len(val),"test":len(test),"enriched_training_events":len(enriched)},"event_detector":er,"direction_accuracy_on_actual_events":da,"persistence":pm,"event_driven_expected_state":sm,"transitions":transitions,"feature_count":len(FEATURES_SPATIAL),"promotion":"candidate" if sm["mae"]<pm["mae"] else "retained_only"}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,default=str)); print(f"Report: {OUT}"); return 0

if __name__=="__main__": raise SystemExit(main())
