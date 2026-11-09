"""Fixed fresh CPU evaluation, three compute workers and no replacement attempts."""
from .common import *
from .runtime import TrainingRuntime
from .statistics import analyze
import multiprocessing as mp,threading,signal,resource
import torch,numpy as np

def worker(index,jobs,results):
 torch.set_num_threads(1);ctx0=context('belief');ctx=RunContext(ctx0.root,RUN+'-eval-'+str(index),ctx0.assets,ctx0.execution,ctx0.seed,ctx0.recipe,ctx0.limits,'learning');ctx.out.mkdir(parents=True,exist_ok=True);ctx.private.mkdir(parents=True,exist_ok=True);rt=TrainingRuntime(ctx,'belief')
 try:
  d=read(OUT/'decks.json');deckmap={k:dict(cards=v['cards']) for k,v in d['prior'].items()};deckmap['Hydro']=dict(cards=d['prior'][d['anchor']]['cards']);write(ctx.out/'manifest.json',dict(decks=deckmap,registration_sha256=sha(OUT/'registration.json'),evaluation=True))
  models={'original':rt.load_base()}
  for arm in ('state','belief'):models[arm]=rt.load_corrected(PRIVATE/(arm+'-final.pt'));models[arm].pilot_arm=arm
  opponent=rt.load_base();rt.runtime.PilotActor=rt.local_actor_type();versions={a:rt.state_hash(m) for a,m in models.items()}
  while True:
   j=jobs.get()
   if j is None:break
   guard();job=dict(j,stage='evaluation',learner_deck='Hydro');record,_=rt.corrected.play(job,models[j['agent']],opponent,False,time.monotonic()+300);assert rt.state_hash(models[j['agent']])==versions[j['agent']];results.put(dict(kind='match',record=record))
   if record['status']!='DONE':break
 except BaseException as e:results.put(dict(kind='fatal',worker=index,reason=str(e),traceback=traceback.format_exc()))
 finally:rt.close()

def main():
 try:os.setsid()
 except PermissionError:pass
 exclusive(OUT/'evaluation-intent.json',dict(at_utc=iso(),pid=os.getpid(),attempt_cap=1440,no_replacements=True));start=time.monotonic();done=threading.Event();procs=[];state='FAILED';failure=None
 import unown_belief_population_sampling.run as monitor
 monitor.OUT=OUT;monitor.CPU_CAP=LOCAL_CPU_CAP;monitor.PREP_CHARGE=read(ROOT/'experiments'/OLD/'completion.json')['CPU_seconds']+3600;monitor.PRIVATE_ROOTS=(PRIVATE,*(ROOT/'data/private'/(RUN+'-eval-'+str(i)) for i in range(3)));monitor.CURATED_CAP=12*1024**3;monitor.deadline=lambda:parse(read(OUT/'registration.json')['deadline_utc']);meter=monitor.Meter();watch=threading.Thread(target=monitor.watchdog,args=(meter,done),daemon=True);watch.start()
 try:
  verify_inputs();guard()
  for arm in ('state','belief'):
   receipt=read(OUT/(arm+'-training-completion.json'));assert receipt['candidate_eligible'] and receipt['fresh_decisions']>=TARGET;assert sha(PRIVATE/(arm+'-final.pt'))==receipt['final_file_sha256']
  frozen=dict(at_utc=iso(),candidates={arm:read(OUT/(arm+'-training-completion.json'))['final_file_sha256'] for arm in ('state','belief')},original=POLICY_TENSOR,schedule_sha256=sha(OUT/'evaluation-schedule.json'));write(OUT/'evaluation-freeze.json',frozen)
  schedule=read(OUT/'evaluation-schedule.json')['attempts'];assert len(schedule)==1440;ctx=mp.get_context('spawn');jobs=ctx.Queue();results=ctx.Queue()
  for i in range(3):
   p=ctx.Process(target=worker,args=(i,jobs,results));p.start();procs.append(p)
  # Only one match per worker in flight; stop dispatch at the first failure.
  issued=0;finished=0;rows=[]
  for _ in procs:jobs.put(schedule[issued]);issued+=1
  while finished<len(schedule):
   monitor.check(meter);guard()
   try:x=results.get(timeout=1)
   except __import__('queue').Empty:
    if any(p.exitcode is not None and p.exitcode!=0 for p in procs):raise RuntimeError('Evaluation worker exited unexpectedly; no retry')
    continue
   if x['kind']=='fatal':write(OUT/'evaluation-worker-failure.json',x);raise RuntimeError(x['reason'])
   row=x['record'];append(OUT/'evaluation-matches.jsonl',row);rows.append(row);finished+=1
   if row['status']!='DONE':raise RuntimeError('Native evaluation compatibility failure; no replacement')
   if issued<len(schedule):jobs.put(schedule[issued]);issued+=1
   if finished%12==0:progress('evaluation',completed=finished,attempts_issued=issued,expected=1440,CPU_seconds=meter.sample())
  for _ in procs:jobs.put(None)
  for p in procs:p.join(timeout=30)
  if any(p.exitcode!=0 for p in procs):raise RuntimeError('Worker shutdown failed')
  result=analyze(rows,read(OUT/'decks.json'));write(OUT/'analysis.json',result);state='COMPLETE'
 except BaseException as e:failure=str(e);write(OUT/'evaluation-failure.json',dict(at_utc=iso(),reason=failure,traceback=traceback.format_exc(),no_replacements=True))
 finally:
  for p in procs:
   if p.is_alive():p.terminate();p.join(timeout=10)
  all_attempts={x['id']:x for x in records(OUT/'evaluation-matches.jsonl')}
  for i in range(3):
   folder=ROOT/'experiments'/(RUN+'-eval-'+str(i))/'evaluation'
   for x in records(folder/'matches.jsonl'):all_attempts.setdefault(x['id'],x)
   for p in (folder/'intents').glob('*.json'):
    j=read(p)
    if j['id'] not in all_attempts:all_attempts[j['id']]=dict(j,status='FAILED',result=None,games=[],attempt_consumed=True,error='Interrupted after evaluation correctness/runtime/budget stop; no recoverable complete result; no replacement')
  assert len(all_attempts)<=1440;write(OUT/'evaluation-all-attempts.json',list(all_attempts.values()))
  done.set();watch.join(timeout=2);write(OUT/'evaluation-completion.json',dict(at_utc=iso(),state=state,reason=failure,elapsed_seconds=time.monotonic()-start,CPU_seconds=meter.sample(),attempts=len(all_attempts),completed=sum(x['status']=='DONE' for x in all_attempts.values()),failures=sum(x['status']!='DONE' for x in all_attempts.values()),GPU_seconds=0))
 if state!='COMPLETE':raise RuntimeError(failure)
if __name__=='__main__':main()
