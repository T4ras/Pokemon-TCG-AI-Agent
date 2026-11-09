"""Paired whole-BO3 validation learning curves; never opens locked test."""
from .common import *
from .scoring import METRICS
from .fit import fit_key
import numpy as np

def summarize(results,replicates=20000):
 results={r['key']:r for r in results if r['state']=='COMPLETE' and r['completed_BO3']==592};methods={key:{x['id']:x for x in value['records']} for key,value in results.items()}
 # Seeds are averaged inside each independent validation BO3, not resampled as matches.
 for n in SIZES:
  for family in ('Trees','MLP','GRU'):
   keys=[f'{family}-n{n}-s{s}' for s in (11,29)]
   if all(k in methods for k in keys):
    merged={}
    for ident in methods[keys[0]]:
     a,b=(methods[k][ident] for k in keys);r=copy_record(a);r['metrics']={k:(a['metrics'][k]+b['metrics'][k])/2 if a['metrics'][k] is not None and b['metrics'][k] is not None else None for k in a['metrics']}
     for axis in ('by_game','by_stage'):
      r[axis]={g:{k:(a[axis][g][k]+b[axis][g][k])/2 if a[axis][g][k] is not None and b[axis][g][k] is not None else None for k in a[axis][g]} for g in a[axis] if g in b[axis]}
     merged[ident]=r
    methods[f'{family}-n{n}-seedmean']=merged
 keys=sorted(methods);metric_keys=list(METRICS);allkeys=metric_keys+[f'bin{j}_{x}' for j in range(10) for x in ('count','pred','label')];cols=[(method,k) for method in keys for k in metric_keys];rng=np.random.default_rng(SEED);boot={p:np.zeros((replicates,len(cols))) for p in ('eligible','full_field')};den={p:np.zeros(len(cols)) for p in boot};point={p:{m:{} for m in keys} for p in boot};counters={p:{m:collections.Counter() for m in keys} for p in boot};weights={p:{m:collections.Counter() for m in keys} for p in boot};perdeck=[]
 entries=[x for x in read(OUT/'split-manifest.json')['entries'] if x['partition']=='validation']
 for deck,info in prior().items():
  pd=dict(deck=deck,archetype=info['archetype'],entries=info['entries'],group=info['group'],methods={})
  for seat in (0,1):
   ids=sorted(x['id'] for x in entries if x['opponent_list']==deck and x['chooser']==seat);assert len(ids)==8
   counts=rng.multinomial(8,np.full(8,1/8),size=replicates)/8
   values=np.array([[methods[m][i]['metrics'].get(k,np.nan) if methods[m][i]['metrics'].get(k) is not None else np.nan for m,k in cols] for i in ids]);valid=np.isfinite(values).all(0);resampled=counts@np.nan_to_num(values)
   for pop in boot:
    w=info['entries']/(100 if pop=='full_field' else 82)/2 if pop=='full_field' or info['group']=='passing' else 0
    if not w:continue
    boot[pop]+=w*resampled;den[pop]+=w*valid
    for method in keys:
     for k in allkeys:
      vals=[methods[method][i]['metrics'].get(k) for i in ids]
      if all(v is not None and np.isfinite(v) for v in vals):counters[pop][method][k]+=w*float(np.mean(vals));weights[pop][method][k]+=w
  for method in keys:pd['methods'][method]={k:float(np.mean([r['metrics'][k] for r in methods[method].values() if r['opponent_list']==deck and r['metrics'].get(k) is not None])) for k in metric_keys if all(r['metrics'].get(k) is not None for r in methods[method].values() if r['opponent_list']==deck)}
  perdeck.append(pd)
 intervals={p:{m:{} for m in keys} for p in boot};contrasts={p:{} for p in boot};coverage={p:{} for p in boot}
 for pop in boot:
  for m in keys:
   point[pop][m]={k:counters[pop][m][k]/weights[pop][m][k] for k in counters[pop][m] if weights[pop][m][k]};point[pop][m]['ECE']=sum(abs(point[pop][m].get(f'bin{j}_pred',0)-point[pop][m].get(f'bin{j}_label',0)) for j in range(10));coverage[pop][m]=dict(weights[pop][m])
   for k in metric_keys:
    col=cols.index((m,k))
    if den[pop][col]>=1-1e-8:intervals[pop][m][k]=np.quantile(boot[pop][:,col],[.025,.975]).tolist()
   if m!='A' and 'A' in keys:
    contrasts[pop][m]={}
    for k in metric_keys:
     i,j=cols.index((m,k)),cols.index(('A',k))
     if den[pop][i]>=1-1e-8 and den[pop][j]>=1-1e-8:contrasts[pop][m][k]=dict(difference=point[pop][m][k]-point[pop]['A'][k],CI95=np.quantile(boot[pop][:,i]-boot[pop][:,j],[.025,.975]).tolist())
 breakdowns={}
 for axis in ('by_game','by_stage'):
  groups=sorted({g for rows in methods.values() for r in rows.values() for g in r[axis]})
  for group in groups:
   breakdowns[axis+':'+group]={}
   for pop in boot:
    table={}
    for m in keys:
     num=collections.Counter();weight=collections.Counter();n=0
     for r in methods[m].values():
      info=prior()[r['opponent_list']]
      if pop=='eligible' and info['group']!='passing':continue
      if group not in r[axis]:continue
      w=info['entries']/(100 if pop=='full_field' else 82)/16;n+=1
      for k,v in r[axis][group].items():
       if v is not None:num[k]+=w*v;weight[k]+=w
     table[m]=dict(point={k:num[k]/weight[k] for k in num if weight[k]},histories=n,coverage=dict(weight))
    breakdowns[axis+':'+group][pop]=table
 return dict(at_utc=iso(),point=point,CI95=intervals,minus_A=contrasts,coverage=coverage,per_deck=perdeck,breakdowns=breakdowns,completed_model_evaluations=len(results),expected_model_evaluations=15,validation_BO3=592,bootstrap_replicates=replicates,unit='paired wholeBO3 within deck/chooser; same resampling across methods/sizes/seeds',seedmeans='mean of metrics inside BO3; not an evaluated policy ensemble',conditional_on_fitted_models=True,locked_test_scored=False,selection_evidence=True)

def copy_record(x):
 import copy
 return copy.deepcopy(x)

def run():
 start=time.process_time();results=[read(p) for p in (OUT/'validation-results').glob('*.json')];result=summarize(results);result['analysis_CPU_seconds']=time.process_time()-start;write(OUT/'analysis.json',result);plot(result);return result

def plot(a):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4));point=a['point']['eligible'];styles=dict(Bayes='C0',Trees='C1',MLP='C2',GRU='C3')
 for ax,metric,title in zip(axes,('uncertain_brier','deck_accuracy'),('Uncertain-hand Brier (lower is better)','Deck identification accuracy')):
  if 'A' in point and metric in point['A']:ax.axhline(point['A'][metric],color='black',linestyle='--',label='A')
  for family,color in styles.items():
   xs=[];ys=[]
   for n in SIZES:
    key=f'Bayes-n{n}-s0' if family=='Bayes' else f'{family}-n{n}-seedmean'
    if key in point and metric in point[key]:xs.append(37*n);ys.append(point[key][metric])
    if family!='Bayes':
     for seed in (11,29):
      k=f'{family}-n{n}-s{seed}'
      if k in point and metric in point[k]:ax.scatter([37*n],[point[k][metric]],color=color,alpha=.3,s=15)
   if xs:ax.plot(xs,ys,marker='o',label=family,color=color)
  ax.set_xscale('log',base=2);ax.set_xlabel('Independent training BO3');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Fixed validation set; n/82 opponent weighting; locked test unscored');fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)
