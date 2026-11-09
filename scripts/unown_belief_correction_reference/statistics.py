"""Independent whole-match bootstrap in frozen deck/chooser strata."""
from .common import *
import numpy as np

def wilson(w,n):
 if not n:return [0.,1.]
 z=1.959963984540054;p=w/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0,c-r),min(1,c+r)]
def combine(values,weights,cis,draws):
 if any(x is None for x in values):return dict(rate=None,CI95=[0.,1.],reason='Missing completed stratum; no reweighting'),np.full(20000,np.nan)
 p=float(np.dot(values,weights));lower=max(0,p-math.sqrt(sum((w*(v-ci[0]))**2 for w,v,ci in zip(weights,values,cis))));upper=min(1,p+math.sqrt(sum((w*(ci[1]-v))**2 for w,v,ci in zip(weights,values,cis))));boot=np.sum(np.asarray(draws)*np.asarray(weights)[:,None],0);finite=boot[np.isfinite(boot)];bc=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [0.,1.]
 return dict(rate=p,CI95=[min(lower,bc[0]),max(upper,bc[1])],bootstrap_CI95=bc,Wilson_MOVER_CI95=[lower,upper]),boot

def analyze(allrows,decks):
 summaries={};boots={};weights=[];B=20000
 for agent_id,agent in enumerate(('original','state','belief')):
  rng=np.random.default_rng(np.random.SeedSequence([904097,agent_id]));values={k:[] for k in ('BO3','first_game')};cis={k:[] for k in values};draws={k:[] for k in values};cells=[];weights=[]
  for deck in decks['passing']:
   for seat in (0,1):
    sample=[x for x in allrows if x['agent']==agent and x['opponent_deck']==deck and x['learner_seat']==seat];indices=rng.integers(len(sample),size=(B,len(sample))) if sample else None;cell=dict(deck=deck,chooser=seat==0,attempts=len(sample),failures=sum(x['status']!='DONE' for x in sample),actual_start=collections.defaultdict(lambda:dict(W=0,D=0,L=0)),termination_reasons=dict(collections.Counter(g.get('finish_reason','unavailable') for x in sample for g in x.get('games',[]))))
    for metric in values:
     valid=np.array([x['status']=='DONE' if metric=='BO3' else bool(x.get('games')) and x['games'][0].get('completion_status')=='DONE' for x in sample]);outcomes=[x.get('result') if metric=='BO3' else x.get('games',[{}])[0].get('result') if x.get('games') else None for x in sample];win=np.array([bool(v and o==seat) for v,o in zip(valid,outcomes)]);n=int(valid.sum());w=int(win.sum());dr=sum(v and o==2 for v,o in zip(valid,outcomes));ci=wilson(w,n);values[metric].append(w/n if n else None);cis[metric].append(ci)
     cell[metric]=dict(W=w,D=int(dr),L=n-w-int(dr),completed=n,attempts=len(sample),rate=w/n if n else None,CI95=ci)
     with np.errstate(divide='ignore',invalid='ignore'):bs=win[indices].sum(1)/valid[indices].sum(1) if indices is not None else np.full(B,np.nan)
     draws[metric].append(bs)
    for x in sample:
     if x.get('games') and x['games'][0].get('completion_status')=='DONE':
      g=x['games'][0];a=cell['actual_start']['first' if g.get('first_player')==seat else 'second'];a['W']+=g['result']==seat;a['D']+=g['result']==2;a['L']+=g['result'] not in (seat,2)
    cell['actual_start']=dict(cell['actual_start']);cell['all_game_counts']=dict(W=sum(g.get('result')==seat for x in sample for g in x.get('games',[]) if g.get('completion_status')=='DONE'),D=sum(g.get('result')==2 for x in sample for g in x.get('games',[]) if g.get('completion_status')=='DONE'),L=sum(g.get('result') not in (seat,2) for x in sample for g in x.get('games',[]) if g.get('completion_status')=='DONE'));cells.append(cell);weights.append(decks['training_probabilities'][deck]/2)
  rows=[x for x in allrows if x['agent']==agent];s=dict(attempts=len(rows),completed=sum(x['status']=='DONE' for x in rows),failures=sum(x['status']!='DONE' for x in rows),cells=cells);boots[agent]={}
  for metric in values:s[metric],boots[agent][metric]=combine(values[metric],weights,cis[metric],draws[metric])
  summaries[agent]=s
 contrasts={}
 for a,b in (('belief','state'),('belief','original'),('state','original')):
  contrasts[a+'-'+b]={}
  for metric in ('BO3','first_game'):
   x=summaries[a][metric];y=summaries[b][metric]
   if x['rate'] is None or y['rate'] is None:contrasts[a+'-'+b][metric]=dict(difference=None,CI95=[-1,1]);continue
   diff=x['rate']-y['rate'];bc=np.nanquantile(boots[a][metric]-boots[b][metric],[.025,.975]);lo=diff-math.sqrt((x['rate']-x['Wilson_MOVER_CI95'][0])**2+(y['Wilson_MOVER_CI95'][1]-y['rate'])**2);hi=diff+math.sqrt((x['Wilson_MOVER_CI95'][1]-x['rate'])**2+(y['rate']-y['Wilson_MOVER_CI95'][0])**2);contrasts[a+'-'+b][metric]=dict(difference=diff,CI95=[max(-1,min(lo,float(bc[0]))),min(1,max(hi,float(bc[1])))],bootstrap_CI95=bc.tolist())
 return dict(at_utc=iso(),agents=summaries,contrasts=contrasts,method='20000 whole-BO3 bootstrap draws within agent/deck/chooser strata; agents independently resampled, shared original draws retained across contrasts; frozen n/82 deck weights/equal chooser; percentile interval enveloped with weighted Wilson/MOVER to avoid zero-width boundary intervals',native_deals='Independent, not paired',strength_limitations='Development panel overlaps training; single pilot, no leaderboard or equivalence claim')
