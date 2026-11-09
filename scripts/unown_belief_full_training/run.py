"""Ordinary persistent controller: no language models or external compute."""
from .common import *
import concurrent.futures as cf,multiprocessing as mp,threading,signal,traceback,subprocess,resource,shutil

class BudgetStop(RuntimeError):pass
class FeasibilityStop(RuntimeError):pass

def stop(reason,kind='budget'):
 if not (OUT/'STOP.json').exists():write(OUT/'STOP.json',dict(at_utc=iso(),reason=reason,kind=kind))

class Meter:
 def __init__(self):self.group=os.getpgrp();self.hz=os.sysconf('SC_CLK_TCK');self.seen={};self.lock=threading.Lock()
 def sample(self):
  with self.lock:
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
     if int(v[2])==self.group:
      key=p.name+':'+v[19];self.seen[key]=max(self.seen.get(key,0),(int(v[11])+int(v[12]))/self.hz)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   owner=read(OUT/'owner-meter.json')['CPU_seconds'] if (OUT/'owner-meter.json').exists() else 0
   return sum(self.seen.values())+owner+read(OUT/'preparation-charge.json')['CPU_budget_charge_seconds']

_STORAGE_CACHE={'at':-1e30,'value':(0,0)}
_STORAGE_LOCK=threading.Lock()
def bytes_used():
 with _STORAGE_LOCK:
  if time.monotonic()-_STORAGE_CACHE['at']<30:return _STORAGE_CACHE['value']
 return refresh_storage()

def refresh_storage():
 dataset=details=0
 for p in __import__('itertools').chain.from_iterable(root.rglob('*') for root in PRIVATE_ROOTS):
  if p.is_file():
   try:n=p.stat().st_size
   except FileNotFoundError:continue
   if 'detail-temp' in p.parts:details+=n
   else:dataset+=n
 with _STORAGE_LOCK:_STORAGE_CACHE.update(at=time.monotonic(),value=(dataset,details))
 return dataset,details

def check(meter):
 if (OUT/'STOP.json').exists():raise BudgetStop(read(OUT/'STOP.json')['reason'])
 if meter.sample()>=CPU_CAP-1800:stop('20 CPU-core-hour cap, including30-minute analysis reserve');raise BudgetStop('Aggregate CPU cap')
 if (deadline()-now()).total_seconds()<=600:stop('24-hour UTC cap with10-minute finalization reserve');raise BudgetStop('Elapsed cap')
 size,detail=bytes_used()
 if size>=CURATED_CAP-128*1024**2 or detail>=DETAIL_CAP-32*1024**2:stop('Registered storage cap with export reserve');raise BudgetStop('Storage cap')

def watchdog(meter,done):
 began=None;sent=set()
 while not done.wait(.5):
  try:check(meter)
  except BudgetStop:
   if began is None:began=time.monotonic()
   # All descendants share this controller process group; never signal itself.
   for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
     s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
     if int(v[2])==os.getpgrp():
      sig=signal.SIGKILL if time.monotonic()-began>10 else signal.SIGTERM
      if int(p.name) not in sent or sig==signal.SIGKILL:os.kill(int(p.name),sig);sent.add(int(p.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  except BaseException as e:write(OUT/'watchdog-failure.json',dict(error=str(e)));stop('Resource monitor failed')

def run_fit(job,meter):
 from .fit import fit_key
 parent_wall=time.monotonic();key=fit_key(job);path=OUT/'fit-jobs'/(key+'.json');write(path,job);log=OUT/'fit-logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 used=sum(x['CPU_seconds'] for x in rows(OUT/'fit-cost-ledger.jsonl'))
 if used>=FIT_CAP-180:raise BudgetStop('10 fitting CPU-core-hour cap with export reserve')
 env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',CUDA_VISIBLE_DEVICES='');before=meter.sample();usage=resource.getrusage(resource.RUSAGE_CHILDREN);child_before=usage.ru_utime+usage.ru_stime;proc=None
 try:
  with log.open('w') as output:
   proc=subprocess.Popen([str(R/'.venv/bin/python'),'-m','unown_belief_full_training.fit',str(path)],cwd=R,env=env,stdout=output,stderr=subprocess.STDOUT);write(OUT/'fit-process.json',dict(pid=proc.pid,key=key,at_utc=iso()))
   while proc.poll() is None:
    check(meter)
    if meter.sample()-before>=min(fit_budget(job),FIT_CAP-used)-15:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     break
    time.sleep(.5)
  if proc.returncode:
   resultpath=OUT/'fit-results'/(key+'.json')
   if resultpath.exists():
    rec=read(resultpath);failure=R/rec['model_directory']/'failure.json'
    if failure.exists() and read(failure).get('type')=='FitLimit':return rec
   raise RuntimeError('Supervised fit incomplete/failed: '+key+'; see retained fit log/checkpoint')
  result=read(OUT/'fit-results'/(key+'.json'))
  if result['state'] not in ('COMPLETE','INCOMPLETE'):raise InvariantFailure('Fit status missing')
  return result
 finally:
  if proc is not None:
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   usage=resource.getrusage(resource.RUSAGE_CHILDREN);actual=usage.ru_utime+usage.ru_stime-child_before;append(OUT/'fit-cost-ledger.jsonl',dict(key=key,CPU_seconds=actual,at_utc=iso(),returncode=proc.returncode,elapsed_seconds=time.monotonic()-parent_wall,includes_startup_and_export=True))

def main():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 exclusive(OUT/'runner-intent.json',dict(pid=os.getpid(),at_utc=iso()));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),at_utc=iso()));meter=Meter();done=threading.Event();thread=threading.Thread(target=watchdog,args=(meter,done),daemon=True);thread.start();state='FAILED';reason='Unfinished';phase='startup'
 try:
  source_check();check(meter)
  if not read(OUT/'forecast.json')['complete_plan_fits']:raise FeasibilityStop('Registered full-data plan does not fit measured caps')
  import torch
  torch.set_num_threads(3);torch.set_num_interop_threads(1)
  from .report_full import render
  render('RUNNING','Verifying immutable full-data inputs and refreshing fixed online timing workloads; no native matches or test access.')
  # Verify every referenced cache before training, without copying private arrays.
  audit=read(OUT/'manifest-audit.json')
  if not audit['passed'] or audit['verified_non_test_caches']!=2926:raise InvariantFailure('Full-data audit incomplete')
  phase='online_timing_inputs';write(OUT/'phase-state.json',dict(at_utc=iso(),phase=phase,locked_test_scored=False));from .timing import prepare,evaluate as time_model
  from .fit import predict
  from threadpoolctl import threadpool_limits
  tick=time.process_time()
  with threadpool_limits(limits=3):prepare(read(OUT/'dataset-index.json'),lambda:check(meter));baseline_timing=time_model(None,'A',predict,lambda:check(meter))
  write(OUT/'baseline-current-timing.json',baseline_timing);write(OUT/'preprocessing-cost.json',dict(CPU_seconds=time.process_time()-tick,new_training_features=0,reused_non_test_caches=2926,refreshed_timing_histories=16,training_timing_overlap=False,locked_test_accessed=False))
  write(OUT/'startup.json',dict(at_utc=iso(),passed=True,training_BO3=2334,validation_BO3=592,reused_caches=2926,tree_coverage_verified=True,online_timing_histories=16,native_matches=0,locked_test_scores=0,source_manifest_sha256=sha(OUT/'source-manifest.json')))
  from .scoring import evaluate
  for job in read(OUT/'fit-schedule.json'):
   check(meter);phase='fitting';write(OUT/'phase-state.json',dict(at_utc=iso(),phase=phase,job=job,locked_test_scored=False));result=run_fit(job,meter)
   if result['state']=='COMPLETE':
    phase='validation'
    import torch
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):evaluate(job,lambda:check(meter))
   render('RUNNING','Full-data fits continue; capped incomplete fits remain explicitly incomplete and are not evaluated.')
  phase='analysis';check(meter);source_check();from .analysis import run
  with threadpool_limits(limits=3):run()
  fits=[read(p) for p in (OUT/'fit-results').glob('*.json')];complete=sum(r['state']=='COMPLETE' for r in fits);state='COMPLETE' if complete==7 else 'PARTIAL_FIT_LIMIT';reason=f'{complete}/7 fits complete; all completed fits scored on the fixed592 validation BO3; locked test remains sealed.'
 except BaseException as exc:
  state='BUDGET_EXHAUSTED' if isinstance(exc,(BudgetStop,TimeoutError)) else 'FAILED_CORRECTNESS' if isinstance(exc,(InvariantFailure,AssertionError,UnsupportedEvent)) else 'FAILED';reason=type(exc).__name__+': '+str(exc);write(OUT/'failure.json',dict(at_utc=iso(),state=state,phase=phase,reason=reason,traceback=traceback.format_exc()))
 finally:
  if not(OUT/'analysis.json').exists() and meter.sample()<CPU_CAP-600 and (deadline()-now()).total_seconds()>300:
   try:
    from .analysis import run
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=3):run()
   except BaseException as e:write(OUT/'analysis-failure.json',dict(error=str(e),traceback=traceback.format_exc()))
  fits=[read(p) for p in (OUT/'fit-results').glob('*.json')];size,_=refresh_storage();prep=read(OUT/'preparation-charge.json');cost=dict(at_utc=iso(),state=state,reason=reason,phase=phase,fit_expected=7,fit_complete=sum(r['state']=='COMPLETE' for r in fits),new_validation_complete=sum('-nfull-' in p.stem for p in (OUT/'validation-results').glob('*.json')),CPU_seconds=meter.sample(),measured_execution_CPU_seconds=meter.sample()-prep['CPU_budget_charge_seconds'],measured_preparation_CPU_seconds=prep['measured_CPU_seconds'],preparation_budget_charge_seconds=prep['CPU_budget_charge_seconds'],CPU_cap_seconds=CPU_CAP,fit_CPU_seconds=sum(x['CPU_seconds'] for x in rows(OUT/'fit-cost-ledger.jsonl')),fit_CPU_cap_seconds=FIT_CAP,elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds(),elapsed_cap_seconds=ELAPSED_CAP,additional_private_bytes=size,additional_private_cap_bytes=CURATED_CAP,training_BO3=2334,validation_BO3=592,locked_test_scored=0,native_matches=0,GPU_seconds=0,paid_API_calls=0,uploads=0,submissions=0,PPO_updates=0,automatic_followup=False)
  write(OUT/'completion.json',cost)
  try:
   from .report_full import render
   render(state,reason)
  finally:
   done.set();thread.join(timeout=2);cost.update(final_CPU_seconds=meter.sample(),final_elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds());write(OUT/'completion.json',cost)
 if state not in ('COMPLETE','PARTIAL_FIT_LIMIT'):raise SystemExit(1)
if __name__=='__main__':main()
