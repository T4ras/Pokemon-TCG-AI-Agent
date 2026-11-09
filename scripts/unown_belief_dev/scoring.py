"""Frozen scoring: no policy actions, truth-only joins, whole-BO3 aggregation."""
from unown_belief_dev.common import *
import numpy as np
METHODS=('A','B_plus_fallback','Full_plus_fallback')

def cell_scores(predictions,truth,known,query_spec,supported):
 queries=query_spec['card_ids'];info=query_spec['cards'] if 'cards' in query_spec else query_spec.get('metadata',{})
 if not info:
  info=next((v for v in query_spec.values() if isinstance(v,dict) and v and 'categories' in next(iter(v.values()),{})),{})
 H={int(c):n for c,n in truth['hand'].items()};D={int(c):n for c,n in truth['deck'].items()};P={int(c):n for c,n in truth['prize'].items()}
 if any(n>H.get(c,0) for c,n in known.items()):raise InvariantFailure('Certified known hand exceeds diagnostic truth')
 groups=collections.defaultdict(list)
 for c in queries:
  groups['all'].append(c)
  if .1<=predictions['A'][c]['p_unknown_hand']<=.9:groups['uncertain'].append(c)
  for category in info.get(str(c),{}).get('categories',[]):groups['category:'+category].append(c)
 result={}
 for scope in (['all_in','supported_only'] if supported else ['all_in']):
  for group,cs in groups.items():
   if not cs:continue
   for method,pr in predictions.items():
    values=[]
    for c in cs:
     v=pr[c];p=v['p_unknown_hand'];y=float(H.get(c,0)>known.get(c,0));hu=H.get(c,0)-known.get(c,0)
     if not(-1e-8<=p<=1+1e-8) or v['hand_unknown_mean']<-1e-8:raise InvariantFailure('Invalid belief probability/count')
     row=dict(brier=(p-y)**2,hand_MAE=abs(v['hand_unknown_mean']-hu),deck_MAE=abs(v['deck_mean']-D.get(c,0)),prize_MAE=abs(v['prize_mean']-P.get(c,0)))
     bin=min(9,int(max(0,p)*10))
     for j in range(10):row.update({f'bin{j}_count':float(j==bin),f'bin{j}_pred':p if j==bin else 0.,f'bin{j}_label':y if j==bin else 0.})
     values.append(row)
    for field in values[0]:result[f'{scope}/{group}/{method}/{field}']=sum(v[field] for v in values)/len(values)
 result['coverage/full_supported']=float(supported);result['coverage/known_hand_correct']=1.;return result

def hierarchical(decisions):
 """Each game receives equal weight inside its BO3, irrespective of length."""
 games=collections.defaultdict(list)
 for e in decisions:games[e['game']].append(e['metrics'])
 pergame=[]
 for rows in games.values():
  keys=set().union(*(r.keys() for r in rows));pergame.append({k:sum(r[k] for r in rows if k in r)/sum(k in r for r in rows) for k in keys})
 keys=set().union(*(r.keys() for r in pergame)) if pergame else set()
 return {k:sum(r[k] for r in pergame if k in r)/sum(k in r for r in pergame) for k in keys}

def summarize(records,partition='development',replicates=20000):
 if partition!='development':raise ValueError('Version003 is development-only; held-out access forbidden')
 rows=[r for r in records if r.get('partition')==partition];p=prior();keys=sorted(set().union(*(r.get('belief_metrics',{}).keys() for r in rows))) if rows else []
 point={};coverage={};perlist={};rng=np.random.default_rng(SEED);bootkeys=[k for k in keys if k.endswith(('/brier','/hand_MAE','/deck_MAE','/prize_MAE')) and ('/all/' in k or '/uncertain/' in k)]+['coverage/full_supported']+[k for k in keys if k.startswith('native/')];bootkeys=[k for k in bootkeys if k in keys]
 boot={field:np.zeros((replicates,len(bootkeys))) for field in ('full_field','eligible')};den={field:np.zeros((replicates,len(bootkeys))) for field in boot};num={field:collections.Counter() for field in boot};weight={field:collections.Counter() for field in boot}
 for d in p:
  perlist[d]={}
  for seat in (0,1):
   group=[r for r in rows if r['opponent_list']==d and r['anchor_seat']==seat];perlist[d][str(seat)]={'attempts':len(group),'completed':sum(r['status']=='DONE' for r in group)}
   if not group:continue
   vals={k:[r.get('belief_metrics',{}).get(k,float('nan')) for r in group] for k in keys}
   sampled=rng.integers(0,len(group),size=(replicates,len(group)))
   for field in boot:
    w=p[d]['entries']/(100 if field=='full_field' else 82)/2 if field=='full_field' or p[d]['group']=='passing' else 0.
    if not w:continue
    for k in keys:
     v=[x for x in vals[k] if math.isfinite(x)]
     if v:num[field][k]+=w*sum(v)/len(group);weight[field][k]+=w*len(v)/len(group)
    if bootkeys:
     arr=np.array([[r.get('belief_metrics',{}).get(k,np.nan) for k in bootkeys] for r in group]);resampled=arr[sampled];valid=np.isfinite(resampled).sum(axis=1);averages=np.nansum(resampled,axis=1)/np.maximum(valid,1);has=valid>0;boot[field]+=w*averages*valid/len(group);den[field]+=w*valid/len(group)
   perlist[d][str(seat)]['metrics']={k:float(np.nanmean(v)) for k,v in vals.items() if any(math.isfinite(x) for x in v)}
 for field in boot:
  point[field]={k:num[field][k]/weight[field][k] for k in keys if weight[field][k]};coverage[field]=dict(weight[field]);boot[field]/=np.maximum(den[field],1e-300)
 intervals={}
 for field in boot:
  intervals[field]={k:np.quantile(boot[field][:,i],[.025,.975]).tolist() for i,k in enumerate(bootkeys) if np.all(den[field][:,i]>0)}
  for scope in ('all_in','supported_only'):
   for group in ('all','uncertain'):
    f=f'{scope}/{group}/Full_plus_fallback/brier'
    for baseline in ('A','B_plus_fallback'):
     a=f'{scope}/{group}/{baseline}/brier';label=f'{scope}/{group}/{baseline}_minus_Full_brier'
     if f in point[field] and a in point[field]:
      point[field][label]=point[field][a]-point[field][f]
      if a in bootkeys and f in bootkeys:intervals[field][label]=np.quantile(boot[field][:,bootkeys.index(a)]-boot[field][:,bootkeys.index(f)],[.025,.975]).tolist()
  for prefix in {k.rsplit('/',1)[0] for k in point[field] if k.endswith('/bin0_count')}:
   point[field][prefix+'/ECE']=sum(abs(point[field].get(prefix+f'/bin{j}_pred',0)-point[field].get(prefix+f'/bin{j}_label',0)) for j in range(10))
 return dict(at_utc=iso(),partition=partition,attempts=len(rows),native_completed=sum(r['status']=='DONE' for r in rows),point=point,CI95=intervals,population_weight_coverage=coverage,per_list=perlist,bootstrap_replicates=replicates,bootstrap_unit='whole BO3 within list/initial-chooser strata; identical sampled indices for all methods',native_deals='independent',missing_scores='explicit metric-specific population coverage; never assert complete-field estimates when a stratum is absent')
