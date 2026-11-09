"""Unchanged frozen evaluation with an explicitly authorized recovery deadline."""
import os,re,resource,shutil,time,traceback
import unown_corrected_ppo_common as common
from unown_corrected_ppo_common import *
REC=OUT/'cpu-evaluation-recovery'
def main():
 start=time.monotonic();cpu=time.process_time();reg=read(REC/'registration.json');common.deadline=lambda:parse(reg['deadline_utc'])
 import unown_corrected_ppo_local as local
 local.deadline=common.deadline;local.check_time=common.check_time
 for name,digest in reg['schedule_SHA256'].items():assert sha(OUT/name)==digest
 for name,digest in read(OUT/'source-freeze.json')['files'].items():assert sha(ROOT/name)==digest
 exclusive(REC/'evaluation-intent.json',dict(at_utc=iso(),pid=os.getpid(),native_matches_max=512,unchanged_schedule=True));write(REC/'worker-pid.json',dict(at_utc=iso(),pid=os.getpid()))
 state='FAILED';reason='Evaluation incomplete'
 try:
  local.evaluate();analysis=read(OUT/'evaluation-analysis.json');state=analysis['status'];assert state in ['COMPLETE','SKIPPED'];reason='Original fixed evaluation completed' if state=='COMPLETE' else analysis['reason']
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);write(REC/'worker-failure.json',dict(at_utc=iso(),error=reason,traceback=traceback.format_exc(),no_replacement_matches=True))
 finally:
  costs=dict(at_utc=iso(),elapsed_seconds=time.monotonic()-start,CPU_core_hours=(time.process_time()-cpu)/3600,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU_seconds=0,paid_model_API_requests=0);write(REC/'worker-costs.json',costs);append(OUT/'local-costs.jsonl',dict(stage='authorized-CPU-evaluation-recovery',**costs))
  # Correct GPU accounting without discarding the failed controller's record.
  prior=read(REC/'preserved-local/gpu-costs.json');observed=read(OUT/'manual-retrieval-status.json');hours=observed['account_wide_delta_seconds']/3600
  write(OUT/'gpu-costs.json',dict(prior,at_utc=iso(),quota_delta_seconds=observed['account_wide_delta_seconds'],session_hours_conservative=hours,session_terminal_verified=True,used_device_hours_conservative=hours,allocated_device_hours_conservative=2*hours,remote_receipt_elapsed_with_export_seconds=observed['remote_receipt']['elapsed_with_export_seconds'],limitation='Quota delta account-wide. Actual completed export receipt and quota observed after recovery. Historical8.3976-hour local monitor wall time is not GPU allocation; preserved-local/gpu-costs.json retains it. Recovery adds0GPU usage.'))
  import unown_corrected_ppo_report as report
  report.finalize(state,reason+'; CPU recovery explicitly authorized [removed] after historical proxy failure.')
  recovery=dict(registration=reg,worker=costs,elapsed_in_recovery_window_seconds=(now()-parse(reg['at_utc'])).total_seconds(),historical_failed_controller=read(REC/'preserved-local/completion.json'),GPU_seconds=0)
  existing=read(OUT/'costs.json');existing['original_registration_elapsed_seconds_not_recovery_budget']=existing.pop('elapsed_since_registration_seconds');existing['historical_local_elapsed_estimate_not_recovery_budget']=existing.pop('local_elapsed_excluding_bounded_remote_seconds');existing['authorized_CPU_recovery']=recovery;write(OUT/'costs.json',existing)
  text=REPORT.read_text();text=re.sub(r'UTC total elapsed .*?ceiling14400s\.', 'Original registration-to-report wall time includes the failed monitor and idle gap; it is not the renewed recovery budget. The separately authorized CPU recovery has a14400-second elapsed ceiling; exact costs/deadline are recorded in cpu-evaluation-recovery and costs.json.',text)
  REPORT.write_text(text+'\n\n## [removed] CPU-only recovery\n\nThe user explicitly authorized launching the previously unstarted CPU evaluation after a proxy failure. This recovery used the exact frozen512-match schedules, policies, decoders, inference seeds and original statistical rules. No intermediate checkpoint selection, replacement match, new training, GPU job or upload. New recovery window: '+reg['at_utc']+' through '+reg['deadline_utc']+'. Costs: '+str(costs)+'. Historical failed controller, report and accounting preserved in cpu-evaluation-recovery/preserved-local.\n')
  write(REC/'worker-completion.json',dict(at_utc=iso(),state=state,reason=reason));write(OUT/'completion.json',dict(at_utc=iso(),state=state,reason=reason,CPU_recovery=True,historical_failure_preserved=True,GPU_jobs=1,new_GPU_jobs=0,automatic_followup=False))
 if state=='FAILED':raise RuntimeError(reason)
if __name__=='__main__':main()
