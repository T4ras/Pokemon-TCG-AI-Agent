"""Frozen independent whole-match bootstrap; no extra matches or selection."""
from .common import *

def analyze():
 grouped={stage:records(OUT/f'evaluation-{stage}-records.jsonl') for stage in ('initial','final')};rng=np.random.default_rng(SEED);B=20000;cells=[];boot={};point={}
 for stage,rows in grouped.items():
  for deck in ('D10','D04'):
   for opponent in ('Hydro','D02','D01','D03'):
    row=dict(checkpoint=stage,deck=deck,opponent=opponent,strata=[])
    for chooser in (0,1):
     sample=[r for r in rows if r['learner_deck']==deck and r['opponent_deck']==opponent and r['learner_seat']==chooser];done=[r for r in sample if r['status']=='DONE']
     W=sum(r['result']==r['learner_seat'] for r in done);D=sum(r['result']==2 for r in done);L=len(done)-W-D
     first=[r for r in sample if r.get('games') and r['games'][0].get('completion_status')=='DONE'];FW=sum(r['games'][0]['result']==r['learner_seat'] for r in first);FD=sum(r['games'][0]['result']==2 for r in first)
     strata=dict(chooser=chooser,attempts=len(sample),completions=len(done),failures=len(sample)-len(done),W=W,D=D,L=L,first_attempts=len(sample),first_completions=len(first),first_W=FW,first_D=FD,first_L=len(first)-FW-FD,actual_starters=dict(collections.Counter(str(r['games'][0].get('first_player')) for r in first)))
     row['strata'].append(strata);key=(stage,deck,opponent,chooser)
     if done and sample:
      # One sampled whole-match index supplies all outcomes for this view.
      indices=rng.integers(len(sample),size=(B,len(sample)));done_mask=np.array([r['status']=='DONE' for r in sample],np.float64);first_mask=np.array([bool(r.get('games')) and r['games'][0].get('completion_status')=='DONE' for r in sample],np.float64)
      w=np.array([r['status']=='DONE' and r['result']==r['learner_seat'] for r in sample],np.float64);f=np.array([bool(r.get('games')) and r['games'][0].get('completion_status')=='DONE' and r['games'][0]['result']==r['learner_seat'] for r in sample],np.float64)
      with np.errstate(divide='ignore',invalid='ignore'):
       boot[key]=dict(BO3=w[indices].sum(1)/done_mask[indices].sum(1),first_game=f[indices].sum(1)/first_mask[indices].sum(1))
      point[key]=dict(BO3=W/len(done),first_game=FW/len(first) if first else None)
    for metric in ('BO3','first_game'):
     keys=[(stage,deck,opponent,chooser) for chooser in (0,1)]
     if all(k in boot for k in keys):
      estimates=np.mean([boot[k][metric] for k in keys],axis=0);row[metric]=dict(rate=float(np.mean([point[k][metric] for k in keys])),interval95=list(map(float,np.nanquantile(estimates,[.025,.975]))))
    cells.append(row)
 def contrast(decks,opponents,metric):
  wanted=[(deck,opp,chooser) for deck in decks for opp in opponents for chooser in (0,1)];a={};b={}
  for stage in ('initial','final'):
   keys=[(stage,*key) for key in wanted]
   if any(key not in boot for key in keys):return dict(status='UNAVAILABLE',reason='Missing completed stratum; no replacement or extra sample')
   a[stage]=float(np.mean([point[key][metric] for key in keys]));b[stage]=np.mean([boot[key][metric] for key in keys],axis=0)
  delta=b['final']-b['initial'];return dict(status='COMPLETE',initial=a['initial'],final=a['final'],final_minus_initial=a['final']-a['initial'],interval95=list(map(float,np.quantile(delta,[.025,.975]))),initial_interval95=list(map(float,np.quantile(b['initial'],[.025,.975]))),final_interval95=list(map(float,np.quantile(b['final'],[.025,.975]))))
 contrasts={}
 for metric in ('BO3','first_game'):
  contrasts[metric]=dict(primary=contrast(('D10','D04'),('Hydro','D02','D01','D03'),metric),D10=contrast(('D10',),('Hydro','D02','D01','D03'),metric),D04=contrast(('D04',),('Hydro','D02','D01','D03'),metric),seen=contrast(('D10','D04'),('Hydro','D02'),metric),held_out=contrast(('D10','D04'),('D01','D03'),metric))
 value=dict(at_utc=iso(),cells=cells,contrasts=contrasts,method='20000 independent whole-match draws inside checkpoint/deck/opponent/chooser strata, equal weights, percentile95%; secondary pointwise intervals; native deals unpaired',seed=SEED,failures=sum(r['status']!='DONE' for rows in grouped.values() for r in rows),limitations='D01/D03 are held-out decklists used with the same frozen policy family; these are local Unown agents, not leaderboard agents. Degenerate empirical intervals in all-win/all-loss strata are not certainty.')
 write(OUT/'evaluation-analysis.json',value);return value

if __name__=='__main__':analyze()
