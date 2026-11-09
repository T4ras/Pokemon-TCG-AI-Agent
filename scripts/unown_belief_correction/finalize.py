"""Ordinary readiness/feasibility handoff, with no model calls or GPU launch."""
from .common import *
from .integration import context
from .forecast import forecast
from .report import render
import traceback,resource

def finish():
 readiness=read(OUT/'readiness-completion.json') if (OUT/'readiness-completion.json').exists() else read(OUT/'readiness-failure.json');native=[];intents=[]
 for arm in ('state','belief'):
  c=context(arm);intents.extend((c.out/'native-intents').glob('*.json'))
  for name in ('native-records-gpu.jsonl','native-records.jsonl'):
   p=c.out/name
   if p.exists():native.extend(json.loads(x) for x in p.read_text().splitlines())
 assert len(intents)<=16 and len({x['id'] for x in native})==len(native)
 cpu=readiness['CPU_seconds']+time.process_time()+30 # includes conservative preparation and reporting reserve.
 state='READINESS_FAILED';reason='Readiness has an unresolved correctness/runtime failure. No long training or fixed evaluation was launched.'
 if readiness.get('state')=='COMPLETE':
  f=forecast()
  if f['complete_plan_fits']:
   state='READINESS_PASSED';reason='Measured admission forecast fits. Training has not been launched; GPU deployment remains unverified. This record is not a completed learning comparison.'
  else:
   state='STOPPED_FEASIBILITY';bad=[arm+' '+str(round(v['conservative_GPU_session_seconds']/3600,2))+'h' for arm,v in f['arms'].items() if not v['GPU_fits']];reason='The registered conservative forecast does not fit the fixed pilot caps ('+', '.join(bad)+'). Stopped before long training, without reducing experience or extending budgets. No strength conclusion is available.'
 write(OUT/'compact-readiness-matches.json',native)
 completion=dict(at_utc=iso(),state=state,reason=reason,CPU_seconds=cpu,CPU_accounting='Sampled process-group CPU plus1200-second conservative preparation charge,20-second repaired-startup carry and30-second finalizer reserve; preparation is not an invented actual measurement.',elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds(),native_attempts=len(intents),completed_BO3=sum(x['status']=='DONE' for x in native),failed_attempts=sum(x['status']!='DONE' for x in native),native_attempt_cap=16,pilot_training_fresh_decisions=0,actual_readiness_disposable_decisions=sum(x.get('meaningful_decisions',0) for x in native if x['status']=='DONE'),evaluation_attempts=0,GPU_session_seconds=0,allocated_device_seconds=0,uploads=0,paid_API_calls=0,competition_submissions=0,no_automatic_followup=True)
 write(OUT/'completion.json',completion);render(state,reason);return completion

def main():
 exclusive(OUT/'finalizer-intent.json',dict(at_utc=iso(),pid=os.getpid(),no_native_or_GPU_launch=True));write(OUT/'finalizer-startup.json',dict(at_utc=iso(),pid=os.getpid(),phase='WAIT_READINESS',expected_additional_minutes=10));render('READINESS_RUNNING','Persistent native readiness and disposable update workload are running. The ordinary finalizer will write the budget admission decision and actual costs; no long training has started.')
 while not (OUT/'readiness-completion.json').exists() and not (OUT/'readiness-failure.json').exists():
  guard();pid=read(OUT/'pid.json')['pid']
  if not Path('/proc/'+str(pid)).exists():
   write(OUT/'readiness-failure.json',dict(at_utc=iso(),reason='Readiness process exited without a terminal record; no automatic native replay',CPU_seconds=43200));break
  time.sleep(5)
 try:finish()
 except BaseException as e:
  write(OUT/'finalizer-failure.json',dict(at_utc=iso(),reason=str(e),traceback=traceback.format_exc()));render('FINALIZER_FAILED','Finalization needs repair; retained readiness evidence is unchanged. '+str(e));raise
if __name__=='__main__':main()
