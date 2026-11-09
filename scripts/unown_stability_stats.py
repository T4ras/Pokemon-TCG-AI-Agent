"""Tensor-gated once-only strength evaluation and nondegenerate intervals."""
import collections
import numpy as np
from scipy.stats import beta
from unown_stability_common import *

def cp(wins,n,confidence=.95):
 if not n:return [None,None]
 alpha=(1-confidence)/2
 return [0. if wins==0 else float(beta.ppf(alpha,wins,n-wins+1)),1. if wins==n else float(beta.ppf(1-alpha,wins+1,n-wins))]

def analyze_records(records):
 rng=np.random.default_rng(SEED+700);results={};values={}
 for label,rows in records.items():
  result=dict(attempted=len(rows),completed=sum(r['status']=='DONE' for r in rows),failures=sum(r['status']!='DONE' for r in rows),runtime_seconds=sum(r['wall_seconds'] for r in rows),metrics={})
  for metric in ['BO3','first_game']:
   strata=[];arrays=[]
   for seat in [0,1]:
    group=[r for r in rows if r['learner_seat']==seat];outcomes=[];starters=collections.Counter();finishes=collections.Counter()
    for r in group:
     outcome=r['result'] if metric=='BO3' and r['status']=='DONE' else r['games'][0]['result'] if metric=='first_game' and r['games'] else None
     if outcome is not None:outcomes.append('W' if outcome==seat else 'D' if outcome==2 else 'L')
     if r['games']:starters['learner_first' if r['games'][0].get('first_player')==seat else 'learner_second']+=1
     for game in r['games']:finishes[str(game.get('finish_reason','not recorded'))]+=1
    arr=np.array([x=='W' for x in outcomes],float);arrays.append(arr)
    strata.append(dict(learner_initial_chooser=seat==0,attempted=len(group),valid=len(arr),wins=outcomes.count('W'),draws=outcomes.count('D'),losses=outcomes.count('L'),
     win_rate=float(arr.mean()) if len(arr) else None,CI95=cp(outcomes.count('W'),len(arr)),CP97_5=cp(outcomes.count('W'),len(arr),.975),CP98_75=cp(outcomes.count('W'),len(arr),.9875),actual_starters=dict(starters),finish_reasons=dict(finishes)))
   summary=dict(strata=strata)
   if all(len(a) for a in arrays):
    summary.update(equal_chooser_win_rate=float(np.mean([a.mean() for a in arrays])),CI95_conservative=[float(np.mean([s['CP97_5'][i] for s in strata])) for i in [0,1]])
    bs=np.mean([a[rng.integers(0,len(a),size=(20000,len(a)))].mean(1) for a in arrays],axis=0)
    values[label,metric]=bs;summary['bootstrap_CI95']=np.quantile(bs,[.025,.975]).tolist();summary['degenerate_bootstrap']=bool(bs.max()==bs.min())
   result['metrics'][metric]=summary
  results[label]=result
 differences={}
 for metric in ['BO3','first_game']:
  if ('initial',metric) in values and ('final',metric) in values:
   initial=results['initial']['metrics'][metric];final=results['final']['metrics'][metric];boot=values['final',metric]-values['initial',metric]
   lo=float(np.mean([s['CP98_75'][0] for s in final['strata']])-np.mean([s['CP98_75'][1] for s in initial['strata']]))
   hi=float(np.mean([s['CP98_75'][1] for s in final['strata']])-np.mean([s['CP98_75'][0] for s in initial['strata']]))
   differences[metric]=dict(final_minus_initial=final['equal_chooser_win_rate']-initial['equal_chooser_win_rate'],CI95_conservative=[lo,hi],bootstrap_CI95=np.quantile(boot,[.025,.975]).tolist(),degenerate_bootstrap=bool(boot.max()==boot.min()))
 return dict(checkpoints=results,difference=differences,method='Primary simultaneous exact CP stratum bounds (Bonferroni across2 strata for each rate and4 for difference). Secondary20000 independent whole-match chooser-stratified bootstrap; degenerate empirical intervals are flagged, never certainty. Draws count as nonwins. Native deals independent.')

def evaluate_once():
 from unown_stability_core import load_model,state_hash,tensor_change,runtime
 completion=read(OUT/'training-completion.json')
 init=load_model(PRIVATE/'initialization.pt');final=load_model(PRIVATE/'final.pt')
 gate=tensor_change(state_hash(init),state_hash(final),completion.get('accepted_updates',0));write(OUT/'evaluation-gate.json',gate)
 if not gate['evaluate']:
  write(OUT/'evaluation-analysis.json',dict(status='SKIPPED',reason='no retained model change',gate=gate,new_native_matches=0));return
 opponent=load_model(PRIVATE/'initialization.pt');runtime.freeze_inference(opponent);records={}
 for label,model in [('initial',init),('final',final)]:
  runtime.freeze_inference(model);records[label]=[]
  for job in read(OUT/f'evaluation-{label}-schedule.json'):
   check_deadline();row,_=runtime.play(job,model,opponent,collect=False,limit=__import__('time').monotonic()+max(0,(deadline()-now()).total_seconds()))
   records[label].append(row)
   if row['status']!='DONE':raise RuntimeError('Evaluation failure; no replacement: '+str(row['error']))
   progress('EVALUATION',checkpoint=label,attempts=len(records[label]))
 result=analyze_records(records);result.update(status='COMPLETE',gate=gate);write(OUT/'evaluation-analysis.json',result)

if __name__=='__main__':evaluate_once()
