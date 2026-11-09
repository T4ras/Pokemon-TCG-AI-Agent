"""Matched, seed-averaged whole-BO3 exploratory analysis on sixteen fixed histories."""
from .common import *
from .replay import matrix_prediction
from .scoring import cell_scores,hierarchical
import gzip,itertools,numpy as np
CONFIGS=('A','PF512','PF2048','RB256','RB512')
def config(r):return 'A' if r['method']=='A' else r['method']+str(r['particles'])
def avg_dict(rows):
 keys=set().union(*(r.keys() for r in rows)) if rows else set()
 return {k:sum(r[k] for r in rows if k in r)/sum(k in r for r in rows) for k in keys}
def calibration(metrics):
 for scope in ('all_in','supported_only'):
  for group in ('all','uncertain'):
   pre=f'{scope}/{group}/Full_plus_fallback/'
   if pre+'bin0_count' in metrics:metrics[pre+'ECE']=sum(abs(metrics.get(pre+f'bin{i}_pred',0)-metrics.get(pre+f'bin{i}_label',0)) for i in range(10))
 return metrics
def analyze(records,check=lambda:None):
 check();inputs=read(OUT/'development-inputs.json');ids=[r['id'] for r in inputs];groups={(i,c):[r for r in records if r['id']==i and config(r)==c] for i in ids for c in CONFIGS};metrics={c:{} for c in CONFIGS};common={c:{} for c in CONFIGS};costs={c:{} for c in CONFIGS};coverage={c:{} for c in CONFIGS};variability={c:{} for c in CONFIGS};per_history=[];spec=read(OUT/'query-set.json');q=spec['card_ids'];common_counts={};failures=collections.Counter();denominators={}
 for item in inputs:
  check();ident=item['id'];preds={};allcomplete=True
  for c in CONFIGS:
   rr=groups[ident,c];planned=1 if c=='A' else 3;valid=len(rr)==planned and all(r['status']=='COMPLETE' for r in rr);allcomplete &= valid
   if not valid:continue
   metrics[c][ident]=avg_dict([r['method_metrics'] for r in rr]);costs[c][ident]=sum(r['CPU_seconds'] for r in rr)/planned;coverage[c][ident]=sum(r['nonfallback_fraction'] for r in rr)/planned
   preds[c]=[{p['index']:p for p in json.load(gzip.open(R/r['prediction_path'],'rt'))['predictions']} for r in rr]
   if c!='A':
    diffs=[]
    for a,b in itertools.combinations(preds[c],2):
     pointset=sorted(a.keys()&b.keys());diffs.append(float(np.mean([np.abs(np.array(a[i]['prediction'])[:,0]-np.array(b[i]['prediction'])[:,0]).mean() for i in pointset])))
    variability[c][ident]=float(np.mean(diffs))
  expected=item['corpus']['selected_decisions'];common_ids=set()
  if allcomplete:
   common_ids=set(preds['A'][0])
   for maps in preds.values():
    for m in maps:common_ids &= {i for i,p in m.items() if p['supported']}
   labels={tuple(t['key']):t for t in json.load(gzip.open(R/item['corpus']['truth_path'],'rt'))}
   for c,maps in preds.items():
    seedmetrics=[]
    for m in maps:
     decisions=[]
     for i in sorted(common_ids):
      point=m[i];A=matrix_prediction(preds['A'][0][i]['prediction'],q);pr=matrix_prediction(point['prediction'],q);truth=labels[tuple(point['key'])];known={int(k):v for k,v in point['known'].items()};sc=cell_scores(dict(A=A,Full_plus_fallback=pr),truth,known,spec,True);decisions.append(dict(game=point['game'],metrics={k:v for k,v in sc.items() if '/Full_plus_fallback/' in k}))
     seedmetrics.append(hierarchical(decisions))
    common[c][ident]=avg_dict(seedmetrics)
  common_counts[ident]=dict(common_points=len(common_ids),expected=expected,all_configurations_complete=bool(allcomplete))
  per_history.append(dict(id=ident,family=item['family'],expected=expected,common_supported=common_counts[ident],configs={c:dict(completed=len([r for r in groups[ident,c] if r['status']=='COMPLETE']),planned=1 if c=='A' else 3,metrics=calibration(dict(metrics[c].get(ident,{}))),CPU_seconds=costs[c].get(ident),nonfallback_coverage=coverage[c].get(ident),seed_mean_abs_probability_difference=variability[c].get(ident)) for c in CONFIGS}))
 summary={};keys=('brier','hand_MAE','deck_MAE','prize_MAE','ECE')
 for c in CONFIGS:
  rr=[r for r in records if config(r)==c];planned=16 if c=='A' else 48;denominators[c]=dict(expected_replays=planned,attempted=len(rr),completed=sum(r['completed'] for r in rr),failed=sum(r['status']!='COMPLETE' for r in rr),scored_replays=sum(r['status']=='COMPLETE' for r in rr),expected_selected_points=sum(x['corpus']['selected_decisions'] for x in inputs)*(1 if c=='A' else 3),scored_selected_points=sum(r['scored'] for r in rr),complete_seed_averaged_histories=len(metrics[c]));summary[c]=dict(metrics=calibration(avg_dict(list(metrics[c].values()))),common_supported_metrics=calibration(avg_dict(list(common[c].values()))),mean_CPU_seconds=float(np.mean(list(costs[c].values()))) if costs[c] else None,nonfallback_coverage=float(np.mean(list(coverage[c].values()))) if coverage[c] else None,seed_mean_abs_probability_difference=float(np.mean(list(variability[c].values()))) if variability[c] else None,depletions=sum(r['depletion_count'] for r in rr),recoveries=sum(r['recovery_attempts'] for r in rr),recovery_successes=sum(r['recovery_successes'] for r in rr),unsupported_replays=sum(r['unsupported_count'] for r in rr),exact_integration_CPU_seconds=sum(r['exact_integration_CPU_seconds'] for r in rr),max_RSS_MiB=max((r['peak_RSS_kib']/1024 for r in rr),default=None),update_CPU_p50_p95=np.quantile([v for r in rr for v in r['update_CPU_samples']],[.5,.95]).tolist() if any(r['update_CPU_samples'] for r in rr) else None,readout_CPU_p50_p95=np.quantile([v for r in rr for v in r['readout_CPU_samples']],[.5,.95]).tolist() if any(r['readout_CPU_samples'] for r in rr) else None)
  for r in rr:
   if r['analysis_error']:failures[c+':'+r['analysis_error']['type']+':'+r['analysis_error']['message'][:120]]+=1
 primary='all_in/uncertain/Full_plus_fallback/brier';contrasts=[];rng=np.random.default_rng(900004)
 for cand in ('RB256','RB512'):
  for ref in ('A','PF512','PF2048'):
   valid=[i for i in ids if primary in metrics[cand].get(i,{}) and primary in metrics[ref].get(i,{})];check()
   if not valid:continue
   delta=np.array([metrics[ref][i][primary]-metrics[cand][i][primary] for i in valid]);samples=rng.integers(0,len(valid),size=(20000,len(valid)));interval=np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist();gain=float(delta.mean());candidate_cpu=float(np.mean([costs[cand][i] for i in valid]));reference_cpu=float(np.mean([costs[ref][i] for i in valid]));coverage_delta=float(np.mean([coverage[cand][i]-coverage[ref][i] for i in valid]));accuracy=gain>=.005 and candidate_cpu<=reference_cpu;efficiency=reference_cpu>=2*candidate_cpu and gain>=-.002
   concentrated=[dict(id=i,Brier_gain=metrics[ref][i][primary]-metrics[cand][i][primary],coverage_delta=coverage[cand][i]-coverage[ref][i]) for i in valid if metrics[cand][i][primary]-metrics[ref][i][primary]>.01 or coverage[cand][i]-coverage[ref][i]<-.10]
   contrasts.append(dict(candidate=cand,reference=ref,paired_histories=len(valid),Brier_gain=gain,CI95=interval,candidate_CPU_seconds=candidate_cpu,reference_CPU_seconds=reference_cpu,CPU_speedup=reference_cpu/candidate_cpu,coverage_delta=coverage_delta,accuracy_screen=bool(accuracy),efficiency_screen=bool(efficiency),screen_pass=bool((accuracy or efficiency) and coverage_delta>=-.02 and len(valid)==16),concentrated_regressions=concentrated))
 families={f:[r for r in per_history if r['family']==f] for f in sorted({r['family'] for r in per_history})};family_summary={f:{c:dict(histories=len(rr),metrics=calibration(avg_dict([r['configs'][c]['metrics'] for r in rr if r['configs'][c]['metrics']]))) for c in CONFIGS} for f,rr in families.items()}
 output=dict(at_utc=iso(),scope='sixteen selected development histories; equal BO3 weight; no representative full-field claim',seed_averaging='seeds averaged inside BO3 before comparisons; bootstrap resamples whole BO3 with shared indices',denominators=denominators,methods=summary,contrasts=contrasts,per_history=per_history,families=family_summary,common_support=common_counts,failure_signatures=dict(failures),new_native_matches=0,held_out_accessed=False)
 write(OUT/'analysis.json',output);return output
