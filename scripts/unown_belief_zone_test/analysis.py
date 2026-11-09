from .common import *
from .evaluate import METRICS
import numpy as np

def analyze(records):
 records=[x for x in records if x['state']=='COMPLETE'];byid={r['id']:r for r in records};models=['A']+[x['key'] for x in read(OUT/'models.json')];entry=read(OUT/'test-manifest.json');keys=models.copy()
 for size in (8,16,'full'):
  for family in ('Trees','MLP','GRU'):
   a,b=[f'{family}-n{size}-s{s}' for s in (11,29)];name=f'{family}-n{size}-seedmean';keys.append(name)
   for r in records:r['metrics'][name]={k:(r['metrics'][a][k]+r['metrics'][b][k])/2 if r['metrics'][a][k] is not None and r['metrics'][b][k] is not None else None for k in METRICS}
 shape=(len(keys),len(METRICS));rng=np.random.default_rng(SEED);B=10000;point={p:np.zeros(shape) for p in ('eligible','full_field')};den={p:np.zeros(shape) for p in point};boot={p:np.zeros((B,*shape)) for p in point};perdeck=[]
 for deck,desc in sorted(prior().items()):
  for chooser in (0,1):
   ids=[x['id'] for x in entry if x['opponent_list']==deck and x['chooser']==chooser];assert len(ids)==8
   valid_ids=[i for i in ids if i in byid];perdeck.append(dict(deck=deck,chooser=chooser,expected=8,completed=len(valid_ids)))
   # Never silently turn partial allocation into a complete confirmatory estimate.
   if len(valid_ids)!=8:continue
   values=np.array([[[byid[i]['metrics'][m][k] if byid[i]['metrics'][m][k] is not None else np.nan for k in METRICS] for m in keys] for i in ids]);valid=np.isfinite(values).all(0);means=np.nan_to_num(values).mean(0)*valid;counts=rng.multinomial(8,np.full(8,1/8),size=B)/8;sample=(counts@np.nan_to_num(values).reshape(8,-1)).reshape(B,*shape)*valid
   for pop in point:
    w=desc['entries']/(100 if pop=='full_field' else 82)/2 if pop=='full_field' or desc['group']=='passing' else 0
    point[pop]+=w*means;den[pop]+=w*valid;boot[pop]+=w*sample
 result=dict(at_utc=iso(),expected_BO3=592,completed_BO3=len(records),failures=[r for r in read(OUT/'test-manifest.json') if r['id'] not in byid],populations={},strata=perdeck,bootstrap_replicates=B,seedmeans='Mean scores withinBO3, not probability ensembles',test_set_opened=True)
 for pop in point:
  result['populations'][pop]={}
  for j,key in enumerate(keys):
   metrics={}
   for n,name in enumerate(METRICS):
    weight=den[pop][j,n];estimate=float(point[pop][j,n]/weight) if weight else None;ci=(np.quantile(boot[pop][:,j,n],[.025,.975])/weight).tolist() if weight>=1-1e-8 else None
    metrics[name]=dict(value=estimate,CI95=ci,stratum_weight_coverage=float(weight))
   result['populations'][pop][key]=metrics
 write(OUT/'analysis.json',result);return result
