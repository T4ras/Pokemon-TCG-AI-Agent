"""Persistent offline controller: eight core-hours AND eight elapsed hours."""
from .common import *
from .replay import run_one,initialize
import concurrent.futures as cf,multiprocessing as mp,threading,signal,traceback,resource

class ResourceStop(RuntimeError):pass

def verify_sources():
 for p,v in read(OUT/'source-manifest.json')['files'].items():
  if sha(R/p)!=v['sha256']:raise InvariantFailure('Frozen input/source changed: '+p)

def process_CPU(stat):
 fields=stat[stat.rfind(')')+2:].split();return int(fields[2]),int(fields[11])+int(fields[12]),fields[19]

class Meter:
 def __init__(self):self.seen={};self.group=os.getpgrp();self.hz=os.sysconf('SC_CLK_TCK');self.lock=threading.Lock()
 def sample(self):
  with self.lock:
   for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():continue
    try:
     group,ticks,started=process_CPU((folder/'stat').read_text())
     if group==self.group:
      key=folder.name+':'+started;self.seen[key]=max(self.seen.get(key,0),ticks/self.hz)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   windows=read(OUT/'owner-meter.json').get('CPU_seconds',0) if (OUT/'owner-meter.json').exists() else 0
   return sum(self.seen.values())+windows

def stop(reason):
 if not (OUT/'STOP.json').exists():write(OUT/'STOP.json',dict(at_utc=iso(),reason=reason))

def check(meter):
 if (OUT/'STOP.json').exists():raise ResourceStop(read(OUT/'STOP.json')['reason'])
 if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>2*1024**3:stop('Registered output storage cap');raise ResourceStop('Storage budget')
 if meter.sample()>=28800-180:stop('8 CPU-core-hour cap: reserved finalization margin');raise ResourceStop('CPU budget')
 if (deadline()-now()).total_seconds()<=180:stop('8 elapsed-hour cap: reserved finalization margin');raise ResourceStop('Elapsed budget')

def watchdog(meter,done):
 signalled=None;sent=set();killed=set()
 while not done.wait(.25):
  try:
   check(meter)
  except ResourceStop:
   if signalled is None:signalled=time.monotonic()
   for path in (OUT/'workers').glob('*.json'):
    pid=read(path)['pid']
    try:
     if pid not in sent:os.kill(pid,signal.SIGTERM);sent.add(pid)
     elif time.monotonic()-signalled>=8 and pid not in killed:os.kill(pid,signal.SIGKILL);killed.add(pid)
    except ProcessLookupError:pass
   write(OUT/'budget-progress.json',dict(at_utc=iso(),aggregate_CPU_seconds=meter.sample(),seconds_to_deadline=(deadline()-now()).total_seconds()))
  except BaseException as exc:write(OUT/'meter-failure.json',dict(error=str(exc),at_utc=iso()));stop('Resource meter failed')

def worker_init():
 initialize()
 from .replay import BudgetExhausted
 def terminate(*_):raise BudgetExhausted('Resource controller terminated this worker')
 signal.signal(signal.SIGTERM,terminate)

def batch(pool,jobs,meter,phase):
 # Submit at most four jobs. Intent precedes submission, and results are also
 # persisted atomically by workers so parent interruption cannot erase them.
 pending={};iterator=iter(jobs);exhausted=False
 while pending or not exhausted:
  check(meter)
  while len(pending)<4 and not exhausted:
   try:task=next(iterator)
   except StopIteration:exhausted=True;break
   exclusive(OUT/'intents'/(task['task_id']+'.json'),dict(at_utc=iso(),task=task));pending[pool.submit(run_one,task)]=task
  if not pending:break
  finished,_=cf.wait(pending,timeout=.25,return_when=cf.FIRST_COMPLETED)
  for future in finished:
   task=pending.pop(future);result=future.result();append(OUT/'result-ledger.jsonl',dict(at_utc=iso(),task_id=task['task_id'],status=result['status'],CPU_seconds=result['CPU_seconds']))
   if result.get('shared_correctness_failure'):stop('Shared correctness failure: '+task['task_id']);raise InvariantFailure('Shared correctness problem; no benchmark interpretation')
   if result['status']=='BUDGET_EXHAUSTED':stop('Worker reached resource deadline');raise ResourceStop('Worker resource stop')
  complete=len(list((OUT/'results').glob('*.json')));write(OUT/'progress.json',dict(at_utc=iso(),phase=phase,attempted=len(list((OUT/'intents').glob('*.json'))),completed_result_records=complete,scheduled=208,scheduled_stochastic=192,aggregate_CPU_seconds=meter.sample()))
  if phase=='stochastic' and not (OUT/'startup.json').exists():
   completed=[read(p) for p in (OUT/'results').glob('*.json')];stoch=[r for r in completed if r['method']!='A']
   if stoch:write(OUT/'startup.json',dict(at_utc=iso(),baseline_records=sum(r['method']=='A' and r['status']=='COMPLETE' for r in completed),first_stochastic_task=stoch[0]['task_id'],first_stochastic_status=stoch[0]['status'],passed=stoch[0]['status']=='COMPLETE',new_native_matches=0,held_out_accessed=False))

def main():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 exclusive(OUT/'runner-intent.json',dict(pid=os.getpid(),at_utc=iso()));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),at_utc=iso()));meter=Meter();done=threading.Event();watch=threading.Thread(target=watchdog,args=(meter,done),daemon=True);watch.start();state='FAILED';reason='Unfinished';analysis=None
 try:
  verify_sources();assert read(OUT/'correctness-gate.json')['passed'];schedule=read(OUT/'schedule.json');ctx=mp.get_context('spawn')
  from .report import render
  render('RUNNING','Executing16 deterministic A replays, then192 frozen interleaved stochastic replays.')
  with cf.ProcessPoolExecutor(4,mp_context=ctx,initializer=worker_init) as pool:
   batch(pool,[t for t in schedule if t['method']=='A'],meter,'baseline')
   if any(read(p)['status']!='COMPLETE' for p in (OUT/'results').glob('*.json')):raise InvariantFailure('Baseline correctness gate failed; comparison stopped')
   batch(pool,[t for t in schedule if t['method']!='A'],meter,'stochastic')
  check(meter);verify_sources();records=[read(p) for p in (OUT/'results').glob('*.json')];from .analysis import analyze
  analysis=analyze(records,lambda:check(meter));state='COMPLETE' if len(records)==208 and all(r['status']=='COMPLETE' for r in records) else 'COMPLETE_WITH_FAILURES';reason='Frozen allocation finished; failed analyses are retained, no replacements or automatic follow-up.'
 except BaseException as exc:
  state='BUDGET_EXHAUSTED' if isinstance(exc,ResourceStop) or (OUT/'STOP.json').exists() and 'cap' in read(OUT/'STOP.json')['reason'] else 'FAILED';reason=type(exc).__name__+': '+str(exc);write(OUT/'failure.json',dict(at_utc=iso(),state=state,reason=reason,traceback=traceback.format_exc()))
 finally:
  records=[read(p) for p in (OUT/'results').glob('*.json')]
  # Partial analysis is bounded and cannot turn missing/failed histories into
  # a passing comparison. The finalization reserve is inside both budgets.
  if analysis is None:
   try:
    if meter.sample()<28740 and (deadline()-now()).total_seconds()>30:
     from .analysis import analyze
     from .replay import bounded
     analysis=bounded(lambda:analyze(records),30)
   except BaseException as exc:write(OUT/'partial-analysis-failure.json',dict(at_utc=iso(),error=str(exc)))
  cost=dict(at_utc=iso(),state=state,reason=reason,elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds(),aggregate_CPU_seconds=meter.sample(),CPU_limit_seconds=28800,elapsed_limit_seconds=28800,expected_total_replays=208,expected_stochastic_replays=192,attempted=len(list((OUT/'intents').glob('*.json'))),results=len(records),completed=sum(r['status']=='COMPLETE' for r in records),failed=sum(r['status']!='COMPLETE' for r in records),unresolved_intents=sorted(p.stem for p in (OUT/'intents').glob('*.json') if not (OUT/'results'/(p.stem+'.json')).exists()),new_native_matches=0,held_out_accessed=False,GPU_seconds=0,training_updates=0,paid_model_API_calls=0,uploads=0,submissions=0,private_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()),automatic_followup=False)
  write(OUT/'completion.json',cost)
  from .report import render
  try:render(state,reason)
  finally:
   done.set();watch.join(timeout=2);cost.update(final_aggregate_CPU_seconds=meter.sample(),final_elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds());write(OUT/'completion.json',cost)
   import shutil
   target=Path('/path/to/ptcg-agent')/'experiments'/RUN/'completion.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/'completion.json',target)
 if state not in ('COMPLETE','COMPLETE_WITH_FAILURES'):raise SystemExit(1)
if __name__=='__main__':main()
