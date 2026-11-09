"""Measured readiness gate; projections are assumptions, never strength evidence."""
from .common import *
import math

def forecast():
 reg=read(OUT/'registration.json');rule=read(OUT/'forecast-rules.json');result=dict(at_utc=iso(),GPU_sessions_launched=0,method=rule,arms={},limitations=['Only representative readiness families, not all30, have been measured live with the correction.','Historical GPU cycle is a different frozen source checkpoint; no speedup from frozen-backbone updates is assumed without a complete GPU measurement.','CPU observer work is added conservatively to historical complete-cycle wall time; components may overlap in a real concurrent run.','Native lengths and policy changes can alter decision and game costs. This is a budget admission forecast, not a proof that a faster implementation is impossible.'])
 for arm in ('state','belief'):
  p=OUT/f'readiness-{arm}.json'
  if not p.exists():raise RuntimeError('Complete collection/update measurement unavailable: '+arm)
  x=read(p);n=x['fresh_readiness_decisions'];matches=x['raw_matches'];N=len(matches);target=250000;overshoot=16*max(m['meaningful_decisions'] for m in matches);fresh=target+overshoot;cpu=sum(m['complete_worker_CPU_seconds'] for m in matches);belief=sum(x['belief_costs'].get(k,0) for k in ('observer_CPU_seconds','A_CPU_seconds','MLP_CPU_seconds'))
  historical=fresh*rule['historical_complete_GPU_cycle_seconds']/rule['historical_cycle_decisions'];extra=fresh*belief/n/rule['GPU_session_CPU_workers'];gpu=rule['startup_and_export_seconds']+rule['margin']*(historical+extra)
  eval_cpu_per_BO3=(x['parent_CPU_seconds']+cpu)/N # Includes readiness checks/update: deliberately conservative CPU envelope.
  result['arms'][arm]=dict(readiness_decisions=n,readiness_BO3=N,measured_collection_seconds=x['collection_wall_seconds'],measured_four_pass_workload_seconds=x['update_wall_seconds'],effective_passes=x['effective_passes'],measured_worker_CPU_seconds=cpu,measured_belief_CPU_seconds=belief,measured_belief_CPU_seconds_per_decision=belief/n,projected_target_BO3=target*N/n,projected_attempts_with_overshoot=math.ceil(fresh*N/n),projected_fresh_with_cohort_reserve=fresh,historical_GPU_cycle_seconds=historical,additional_belief_seconds_conservative=extra,conservative_GPU_session_seconds=gpu,GPU_cap_seconds=10800,GPU_fits=gpu<=10800,attempt_cap_fits=math.ceil(fresh*N/n)<=2048,local_CPU_envelope_per_BO3=eval_cpu_per_BO3)
 # Evaluation has no PPO updates. Keep an update-free bound from the measured collector service and worker CPU.
 rates={}
 for arm in ('state','belief'):
  x=read(OUT/f'readiness-{arm}.json');worker=sum(m['complete_worker_CPU_seconds'] for m in x['raw_matches']);service=x['collection']['service'];timers=service.get('timers',{});parent=x['collection_wall_seconds'] # serial one-worker collection upper-bounds parent+worker CPU only if summed explicitly.
  rates[arm]=(parent+worker)/len(x['raw_matches'])
 evaluation=rule['margin']*480*(2*rates['state']+rates['belief']);readiness=read(OUT/'readiness-completion.json');local=readiness['CPU_seconds']+evaluation+600
 result.update(projected_fixed_evaluation_CPU_seconds=evaluation,conservative_local_CPU_seconds=local,local_CPU_cap_seconds=43200,local_CPU_fits=local<=43200,complete_plan_fits=all(x['GPU_fits'] and x['attempt_cap_fits'] for x in result['arms'].values()) and local<=43200)
 write(OUT/'forecast.json',result);return result
