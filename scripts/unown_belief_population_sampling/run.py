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
   return sum(self.seen.values())+owner+PREP_CHARGE

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
 if meter.sample()>=CPU_CAP-600:stop('4 CPU-core-hour sampling-experiment cap, including10-minute reporting reserve');raise BudgetStop('Aggregate CPU cap')
 if (deadline()-now()).total_seconds()<=120:stop('6-hour UTC sampling-experiment cap with2-minute reporting reserve');raise BudgetStop('Elapsed cap')
 size,detail=bytes_used()
 if size>=CURATED_CAP-128*1024**2:stop('Registered storage cap with export reserve');raise BudgetStop('Storage cap')

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

def main():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 exclusive(OUT/'runner-intent.json',dict(pid=os.getpid(),at_utc=iso()));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),at_utc=iso()));meter=Meter();done=threading.Event();thread=threading.Thread(target=watchdog,args=(meter,done),daemon=True);thread.start();state='FAILED';reason='Unfinished';active=None
 try:
  source_check();check(meter);assert read(OUT/'forecast.json')['complete_plan_fits'];assert read(OUT/'preflight.json')['passed']
  from .report import render
  render('RUNNING','Registered four-run sampling comparison launched. Exact final endpoint, unchanged objectives, train/validation only.')
  write(OUT/'startup.json',dict(at_utc=iso(),passed=True,preflight_passed=True,training_BO3=2334,validation_BO3=592,test_access=False,runs=4,updates_per_run=14004,source_manifest_sha256=sha(OUT/'source-manifest.json')))
  jobs=read(OUT/'jobs.json')
  for index,job in enumerate(jobs):
   check(meter);write(OUT/'phase-state.json',dict(at_utc=iso(),phase='fit',job=job,run_index=index))
   with (OUT/(job['key']+'.log')).open('w') as log:
    active=subprocess.Popen([str(R/'.venv/bin/python'),'-m','unown_belief_population_sampling.train',str(index)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
    while active.poll() is None:
     check(meter);time.sleep(.5)
    meter.sample()
   if active.returncode:raise RuntimeError('Registered fit incomplete: '+job['key']+'; see retained failure and recovery')
   receipt=read(OUT/'fit-results'/(job['key']+'.json'));assert receipt['state']=='COMPLETE' and receipt['updates']==14004
   write(OUT/'progress.json',dict(at_utc=iso(),completed_runs=index+1,expected_runs=4,CPU_seconds=meter.sample(),completed_optimizer_updates=sum(read(p)['updates'] for p in (OUT/'fit-results').glob('*.json'))))
  check(meter);source_check();write(OUT/'phase-state.json',dict(at_utc=iso(),phase='analysis'))
  from .analysis import run as analyze
  analyze(lambda:check(meter));state='COMPLETE';reason='Allfour runs completed exactly14004updates each. Fixed-final comparisons, initialization checks, secondary selection and exposure diagnostics saved; no test access.'
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);write(OUT/'failure.json',dict(at_utc=iso(),reason=reason,traceback=traceback.format_exc()))
  if active is not None and active.poll() is None:
   active.terminate()
   try:active.wait(timeout=10)
   except subprocess.TimeoutExpired:active.kill();active.wait()
 finally:
  receipts=[read(p) for p in (OUT/'fit-results').glob('*.json')];cost=dict(at_utc=iso(),state=state,reason=reason,CPU_seconds=meter.sample(),measured_execution_CPU_seconds=meter.sample()-PREP_CHARGE,preparation_CPU_reserve_seconds=PREP_CHARGE,CPU_cap_seconds=CPU_CAP,elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['started_at_utc'])).total_seconds(),elapsed_cap_seconds=ELAPSED_CAP,optimizer_updates=sum(r['updates'] for r in receipts),completed_runs=sum(r['state']=='COMPLETE' for r in receipts),expected_runs=4,test_access=False,native_matches=0,GPU_seconds=0,PPO_updates=0,paid_API_calls=0,uploads=0,submissions=0,private_bytes=refresh_storage()[0],automatic_followup=False)
  write(OUT/'completion.json',cost)
  try:
   from .report import render
   render(state,reason)
  finally:done.set();thread.join(timeout=2)
  cost.update(final_CPU_seconds=meter.sample(),final_elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['started_at_utc'])).total_seconds());write(OUT/'completion.json',cost)
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
