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
   return sum(self.seen.values())+owner+300

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
 if meter.sample()>=CPU_CAP-600:stop('4 CPU-core-hour test-evaluation cap, including10-minute reporting reserve');raise BudgetStop('Aggregate CPU cap')
 if (deadline()-now()).total_seconds()<=120:stop('3-hour UTC test-evaluation cap with2-minute reporting reserve');raise BudgetStop('Elapsed cap')
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
 exclusive(OUT/'runner-intent.json',dict(pid=os.getpid(),at_utc=iso()));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),at_utc=iso()));meter=Meter();done=threading.Event();thread=threading.Thread(target=watchdog,args=(meter,done),daemon=True);thread.start();state='FAILED';reason='Unfinished';records=[]
 try:
  source_check();check(meter)
  from .report import render
  render('RUNNING','Frozen all-model test scoring launched; original histories and fitted checkpoints unchanged.')
  from .evaluate import one,initialize
  items=read(OUT/'test-manifest.json');assert len(items)==592
  ctx=mp.get_context('spawn')
  with cf.ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=initialize) as pool:
   active={};todo=iter(items)
   for _ in range(2):
    item=next(todo);active[pool.submit(one,item)]=item['id']
   while active:
    check(meter);finished,_=cf.wait(active,timeout=1,return_when=cf.FIRST_COMPLETED)
    for f in finished:
     ident=active.pop(f)
     try:r=f.result()
     except BaseException as e:r=dict(id=ident,state='FAILED_WORKER',error=type(e).__name__+': '+str(e));write(OUT/'results'/(ident+'.json'),r)
     records.append(r);write(OUT/'progress.json',dict(at_utc=iso(),finished=len(records),completed=sum(x['state']=='COMPLETE' for x in records),failures=sum(x['state']!='COMPLETE' for x in records),expected=592,CPU_seconds=meter.sample(),elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['started_at_utc'])).total_seconds()))
     item=next(todo,None)
     if item is not None:active[pool.submit(one,item)]=item['id']
  check(meter);source_check()
  from .analysis import analyze
  a=analyze(records);state='COMPLETE' if a['completed_BO3']==592 else 'INCOMPLETE';reason=f"Frozen test scores saved for {a['completed_BO3']}/592 BO3 across22 retained methods. No training or checkpoint selection."
 except BaseException as e:reason=type(e).__name__+': '+str(e);write(OUT/'failure.json',dict(at_utc=iso(),reason=reason,traceback=traceback.format_exc()))
 finally:
  write(OUT/'completion.json',dict(at_utc=iso(),state=state,reason=reason,CPU_seconds=meter.sample(),measured_execution_CPU_seconds=meter.sample()-300,preparation_CPU_reserve_seconds=300,CPU_cap_seconds=CPU_CAP,elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'continuation-window.json')['started_at_utc'])).total_seconds(),elapsed_cap_seconds=ELAPSED_CAP,test_BO3_completed=sum(x['state']=='COMPLETE' for x in records),test_BO3_attempted=len(records),native_matches=0,GPU_seconds=0,training_updates=0,paid_API_calls=0,test_opened=(OUT/'test-access.jsonl').exists(),private_bytes=refresh_storage()[0],automatic_followup=False))
  try:
   from .report import render
   render(state,reason)
  finally:done.set();thread.join(timeout=2)
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
