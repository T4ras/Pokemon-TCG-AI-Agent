from .common import *
import numpy as np
from unown_belief_full_training.fit import predict
from unown_belief_full_training.models import point_weights
from unown_belief_full_training.scoring import decision_scores,aggregate,METRICS
from unown_belief_deck_audit.diagnostics import Accumulator,initial_counts

EXTRA=tuple(f'count_NLL_{z}' for z in ('hand','draw_pile','prizes'))+tuple(f'{m}_{z}' for m in ('presence','uncertain_presence') for z in ('hand','draw_pile','prizes'))
ALL_METRICS=tuple(METRICS)+EXTRA

def score(f,t,d,p):
 v,_=decision_scores(f,t,d,p);target=np.take_along_axis(p,t['counts'][...,None],axis=-1)[...,0];zone=-np.log(np.maximum(target,1e-30)).mean(1);presence=1-p[...,0];truth=t['counts']>0;bs=(presence-truth)**2;A=1-f['A'][...,0];uncertain=(A>=.1)&(A<=.9);den=uncertain.sum(1);ub=np.divide((bs*uncertain).sum(1),den,out=np.full(den.shape,np.nan),where=den>0);values=np.column_stack((v,zone,bs.mean(1),ub));result=dict(zip(ALL_METRICS,aggregate(values,f['games'])))
 if abs(result['count_NLL']-sum(result['count_NLL_'+z] for z in ('hand','draw_pile','prizes'))/3)>5e-6:raise InvariantFailure('Zone NLL reduction drift')
 if abs(result['all_brier']-result['presence_hand'])>1e-8:raise InvariantFailure('Hand Brier semantics drift')
 return result

def evaluate(model,partition,check,initial=False,save_baseline=False):
 items=entries(partition);acc=Accumulator(items);base=Accumulator(items) if save_baseline else None;records=[];base_records=[];parity=dict(max_deck_vs_A=0.,max_counts_vs_floored_A=0.,max_counts_vs_raw_A=0.);started=time.process_time()
 for item in items:
  check();f,t=load(item['id']);d,p=predict(model,f,'MLP')
  if initial:
   expected=initial_counts(f);np.testing.assert_allclose(d,f['deck_A'],atol=2e-7,rtol=2e-5);np.testing.assert_allclose(p,expected,atol=2e-6,rtol=2e-5);parity['max_deck_vs_A']=max(parity['max_deck_vs_A'],float(np.max(np.abs(d-f['deck_A']))));parity['max_counts_vs_floored_A']=max(parity['max_counts_vs_floored_A'],float(np.max(np.abs(p-expected))));parity['max_counts_vs_raw_A']=max(parity['max_counts_vs_raw_A'],float(np.max(np.abs(p-f['A']))))
  w=point_weights(f['games'])
  if partition=='validation':
   m=score(f,t,d,p);records.append(dict(id=item['id'],opponent_list=item['opponent_list'],chooser=item['chooser'],metrics=m));acc.add(item,m,w@d)
   if save_baseline:
    bd,bp=predict(None,f,'A');bm=score(f,t,bd,bp);base.add(item,bm,w@bd);base_records.append(dict(id=item['id'],opponent_list=item['opponent_list'],chooser=item['chooser'],metrics=bm))
  else:
   loss=float(w@(-np.log(np.maximum(d[np.arange(len(d)),t['deck']],1e-30))));acc.add(item,dict(deck_log_loss=loss),w@d)
 result=dict(partition=partition,BO3=len(items),summary=acc.result(),records=records,initial_parity=parity if initial else None,CPU_seconds=time.process_time()-started)
 if save_baseline:
  result['A_summary']=base.result();result['A_records']=base_records
  # Baseline paired rows are needed for bootstrap contrasts on identical matches.
  # Reuse scores computed in this pass, retained by the validation branch below.
 return result
