"""Population-weighted, paired whole-BO3 cross-validation analysis."""
from .common import *
from .scoring import summarize
import numpy as np

PRIMARY='all_in/uncertain/prediction/brier'

def flatten(methods):return {m+'/'+k:v for m,x in methods.items() for k,v in x.items()}
def analysis(records,common_records,check=lambda:None):
 check();planned=read(OUT/'inputs.json');byid={r['id']:r for r in records};wrapped=[]
 for item in planned:
  r=byid.get(item['id']);ok=bool(r and r['status']=='COMPLETE');wrapped.append(dict(id=item['id'],partition='development',opponent_list=item['opponent_list'],anchor_seat=item['chooser'],status='DONE' if ok else 'FAILED',belief_metrics=flatten(r['metrics']) if ok else {}))
 result=summarize(wrapped,'development',20000);result['scores_completed']=result.pop('native_completed');result.update(expected_BO3=296,expected_points=sum(x['corpus']['selected_decisions'] for x in planned),scored_points=sum(r['scored'] for r in records if r['status']=='COMPLETE'),all_strata_complete=len(records)==296 and all(r['status']=='COMPLETE' for r in records),exploratory_development=True,intervals_condition_on_fitted_models=True,retraining_uncertainty_fully_quantified=False,known_deck_diagnostic_scorer_only=True,new_native_matches=0,held_out_accessed=False)
 breakdown={}
 for axis in ('by_game','by_stage'):
  groups=sorted({g for r in records for m in r[axis].values() for g in m})
  for group in groups:
   table={}
   for population in ('full_field','eligible'):
    means={};coverage={}
    for method in ('A','B','known_deck_diagnostic'):
     counters=collections.Counter();weights=collections.Counter();matched=0
     for r in records:
      info=prior()[r['opponent_list']]
      if population=='eligible' and info['group']!='passing':continue
      metrics=r[axis][method].get(group)
      if not metrics:continue
      matched+=1;w=info['entries']/(100 if population=='full_field' else 82)/8
      for k,v in metrics.items():counters[k]+=w*v;weights[k]+=w
     means[method]={k:counters[k]/weights[k] for k in counters};coverage[method]=dict(histories=matched,primary_population_weight=weights.get(PRIMARY,0))
    table[population]=dict(point=means,coverage=coverage)
   breakdown[axis+':'+group]=table
 result['breakdowns']=breakdown
 result['primary_population_complete']=all(result['population_weight_coverage'][pop].get(m+'/'+PRIMARY,0)>=1-1e-9 for pop in ('eligible','full_field') for m in ('A','B','known_deck_diagnostic'))
 common={r['id']:r for r in common_records};perdeck=[]
 for d,info in prior().items():
  rr=[r for r in records if r['opponent_list']==d];perdeck.append(dict(deck=d,archetype=info['archetype'],group=info['group'],entries=info['entries'],completed=len(rr),expected=8,methods={m:{key:float(np.mean([r['metrics'][m][key] for r in rr if key in r['metrics'][m]])) for key in (PRIMARY,'deck/log_loss','deck/top1_accuracy','deck/true_probability','deck/remaining_composition_MAE') if any(key in r['metrics'][m] for r in rr)} for m in ('A','B','known_deck_diagnostic')}))
 result['per_deck']=perdeck
 fit=[read(p) for p in (OUT/'model-selection').glob('*.json')];result['costs']=dict(statistical_fitting_inner_validation_CPU_seconds=sum(x['CPU_seconds'] for x in fit),common_observer_allocation_CPU_seconds=sum(x['CPU_seconds'] for x in common_records),A_incremental_CPU_seconds=sum(r['timing']['A_weight']+r['timing']['A_readout'] for r in records),B_incremental_CPU_seconds=sum(r['timing']['B_weight']+r['timing']['B_readout'] for r in records),scoring_export_CPU_seconds=sum(r['scoring_and_export_CPU_seconds'] for r in records),peak_worker_RSS_MiB=max((x['peak_RSS_kib']/1024 for x in common_records),default=None))
 n=len(records)
 if n:
  result['costs'].update(A_complete_CPU_seconds_per_BO3=sum(common[r['id']]['CPU_seconds']+r['timing']['A_weight']+r['timing']['A_readout'] for r in records)/n,B_complete_CPU_seconds_per_BO3=sum(common[r['id']]['CPU_seconds']+r['timing']['B_weight']+r['timing']['B_readout'] for r in records)/n,common_update_CPU_p50_p95=np.quantile([v for x in common_records for v in x['update_CPU_samples']],[.5,.95]).tolist(),common_allocation_CPU_p50_p95=np.quantile([v for x in common_records for v in x['readout_CPU_samples']],[.5,.95]).tolist(),A_weight_and_mixture_CPU_p50_p95=np.quantile([x['A_weight']+x['A_readout'] for r in records for x in r['mixture_latency_samples']],[.5,.95]).tolist(),B_weight_and_mixture_CPU_p50_p95=np.quantile([x['B_weight']+x['B_readout'] for r in records for x in r['mixture_latency_samples']],[.5,.95]).tolist())
 write(OUT/'analysis.json',result);return result
