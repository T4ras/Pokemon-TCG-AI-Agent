"""Vectorized scoring and validation-only inference, with whole-match metrics."""
from .common import *
from .distributions import moments
from .models import point_weights
from .fit import predict,load_model,allowed_items,fit_key
from .preprocess import load_features
import numpy as np,resource

METRICS=('uncertain_brier','all_brier','hand_MAE','deck_MAE','prize_MAE','deck_log_loss','deck_Brier','deck_accuracy','deck_true_probability','count_NLL','objective_NLL','support_mass_violation','hand_total_MAE','deck_total_MAE','prize_total_MAE','card_total_violation_MAE','card_total_violation_fraction')

def decision_scores(f,t,d,p):
 if not np.isfinite(p).all() or not np.isfinite(d).all() or np.any(p<-1e-8) or not np.allclose(p.sum(-1),1,atol=1e-5) or not np.allclose(d.sum(-1),1,atol=1e-5):raise InvariantFailure('Invalid model probabilities')
 N,Q,Z,S=p.shape;k=np.arange(S);presence=1-p[:,:,0,0];A=1-f['A'][:,:,0,0];y=t['counts'][:,:,0]>0;uncertain=f.get('uncertain',(A>=.1)&(A<=.9));mu=p@k;truth=t['counts'];bs=(presence-y)**2;den=uncertain.sum(1);u=np.divide((bs*uncertain).sum(1),den,out=np.full(N,np.nan),where=den>0)
 targetprob=np.take_along_axis(p,truth[...,None],axis=-1)[...,0];NLL=-np.log(np.maximum(targetprob,1e-30)).mean((1,2));yd=t['deck'];deckloss=-np.log(np.maximum(d[np.arange(N),yd],1e-30));mask=(k<f['lower'][...,None])|(k>f['upper'][...,None]);bad=(p*mask).sum((1,2,3))/(Q*3);totals=mu.sum(1);totals[:,0]+=f['known'].sum(1);totalerr=np.abs(totals-f['zone_sizes']);cardmean=mu.sum(2)+f['known'];cardbad=np.maximum(0,cardmean-f.get('card_total_upper',np.full_like(cardmean,S-1)))+np.maximum(0,f.get('card_total_lower',np.zeros_like(cardmean))-cardmean)
 out=np.column_stack((u,bs.mean(1),np.abs(mu-truth).mean(1),deckloss,(d*d).sum(1)-2*d[np.arange(N),yd]+1,(d.argmax(1)==yd).astype(float),d[np.arange(N),yd],NLL,deckloss+NLL,bad,totalerr,cardbad.mean(1),(cardbad>1e-6).mean(1)))
 # Column construction above puts the three zone count MAEs together.
 if out.shape[1]!=len(METRICS):raise InvariantFailure('Metric schema mismatch')
 bins=[]
 for j in range(10):
  b=np.minimum((np.maximum(presence,0)*10).astype(int),9)==j;bins.extend([b.mean(1),(presence*b).mean(1),(y*b).mean(1)])
 return out,np.column_stack(bins)

def aggregate(values,games):
 w=point_weights(games);result=[]
 for col in values.T:
  valid=np.isfinite(col);ww=w*valid;result.append(float(np.nansum(col*w)/ww.sum()) if ww.sum() else None)
 return result

def evaluate(job,check=lambda:None):
 family=job['family'];key='A' if family=='A' else fit_key(job);startup=time.process_time();model=None if family=='A' else load_model(job);startup=time.process_time()-startup;records=[];started=time.process_time();wall=time.monotonic();lat=[];parameters=None
 if family in ('MLP','GRU'):parameters=sum(p.numel() for p in model.parameters())
 for item in allowed_items('validation'):
  check();f,t=load_features(item['id']);tick=time.process_time();d,p=predict(model,f,family);cost=time.process_time()-tick;lat.append(cost);values,bins=decision_scores(f,t,d,p);public=read(OUT/'feature-results'/(item['id']+'.json'))
  if sha(R/public['detail_path'])!=public['detail_sha256']:raise InvariantFailure('Point metadata checksum mismatch')
  info=read(R/public['detail_path']);bygame={};bystage={}
  for group in np.unique(f['games']):
   at=f['games']==group;bygame[str(group)]=dict(zip(METRICS,aggregate(values[at],f['games'][at])))
  for group in ('setup','early','middle','late'):
   at=np.array([x['stage']==group for x in info['points']])
   if at.any():bystage[group]=dict(zip(METRICS,aggregate(values[at],f['games'][at])))
  metrics=dict(zip(METRICS,aggregate(values,f['games'])));binvalues=aggregate(bins,f['games']);metrics.update({f'bin{j}_{field}':binvalues[j*3+i] for j in range(10) for i,field in enumerate(('count','pred','label'))})
  path=PRIVATE/'predictions'/key/(item['id']+'.npz');npwrite(path,deck=d.astype(np.float32),counts=p.astype(np.float32),selected=f['selected']);records.append(dict(id=item['id'],opponent_list=item['opponent_list'],chooser=item['chooser'],scored_points=len(d),expected_points=len(info['points']),metrics=metrics,by_game=bygame,by_stage=bystage,inference_CPU_seconds=cost,common_feature_CPU_seconds=info['CPU_seconds'],all_callbacks=len(f['sequence']),full_GRU_sequence_processed=family=='GRU',prediction_path=str(path.relative_to(R)),prediction_sha256=sha(path)))
  write(OUT/'evaluation-active.json',dict(key=key,completed=len(records),expected=592,at_utc=iso()))
 if len(records)!=592:raise InvariantFailure('Fixed validation allocation incomplete')
 from .timing import evaluate as online_timing
 full_timing=online_timing(model,family,predict,check)
 result=dict(model_load_CPU_seconds=startup,full_online_timing=full_timing,at_utc=iso(),key=key,job=job,state='COMPLETE',expected_BO3=592,completed_BO3=len(records),records=records,parameters=parameters,CPU_seconds=time.process_time()-started,incremental_inference_CPU_seconds=sum(lat),sampled_validation_workload_CPU_seconds_per_BO3=sum(x['common_feature_CPU_seconds']+x['inference_CPU_seconds'] for x in records)/592,readout_CPU_p50_p95=np.quantile(lat,[.5,.95]).tolist(),peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,locked_test_scored=False,elapsed_seconds=time.monotonic()-wall)
 write(OUT/'validation-results'/(key+'.json'),result);return result
