"""Persistent four-worker schedule: development, source freeze, held-out, stability."""
from unown_belief_native_common import *
from unown_belief_native_scoring import summarize
from unown_belief_native_corpus import replay
import concurrent.futures as cf,multiprocessing as mp,resource,signal,traceback,gzip,numpy as np

def verify_sources():
 manifest=read(OUT/'source-manifest.json')
 bad=[p for p,d in manifest['files'].items() if sha(R/p)!=d['sha256']]
 if bad:raise InvariantFailure('Frozen source changed: '+str(bad))

def worker_init():
 verify_sources()
 from unown_belief_native_worker import initialize
 initialize()

def one(job):
 from unown_belief_native_worker import one as worker
 return worker(job)

def jobs(partition):
 anchor=next(k for k,v in prior().items() if v['group']=='anchor')
 return [dict(id=r['id'],partition=partition,a=anchor,b=r['opponent_list'],seat=r['anchor_seat'],seed=r['policy_seed'],tracker_seed=r['tracker_seed'],startup_check=(partition=='development' and i<2)) for i,r in enumerate(read(OUT/(partition+'-schedule.json')))]

def storage():return sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
def register_result(result):
 if result.get('detail'):
  src=R/result['detail']['path'];dest=PRIVATE/'details'/src.name;dest.parent.mkdir(exist_ok=True);src.replace(dest);result['detail']['path']=str(dest.relative_to(R))
 append(OUT/'matches.jsonl',result)
 if sum(p.stat().st_size for p in (PRIVATE/'details').glob('*') if p.is_file())>DETAIL_CAP:raise RuntimeError('Registered2GiB detail cap reached')
 if storage()>CORPUS_CAP:raise RuntimeError('Registered8GiB corpus/storage cap reached')
 if result['status']!='DONE':raise RuntimeError('Development/execution failure preserved; no repair/replacement: '+str(result.get('observer_error') or result.get('error')))

def phase(pool,partition):
 schedule=jobs(partition);pending={};index=0;results=[]
 if partition=='development':
  startup=[pool.submit(one,job) for job in schedule[:2]];failure=None
  for future in startup:
   check_time();r=future.result()
   try:register_result(r)
   except BaseException as e:failure=e
   results.append(r);index+=1;progress('STARTUP',attempted=2,completed=index,allocated_only=True)
  if failure:raise failure
  write(OUT/'startup.json',dict(at_utc=iso(),passed=all(r.get('startup_parity') for r in results),native_attempts=2,initial_jobs=[r['id'] for r in results],no_extra_matches=True,mean_native_seconds=float(np.mean([r['awake_seconds'] for r in results])),mean_tracker_CPU_seconds=float(np.mean([r['tracker_CPU_seconds'] for r in results]))))
 while index<len(schedule) or pending:
  check_time()
  while len(pending)<WORKERS and index<len(schedule):
   job=schedule[index];index+=1;pending[pool.submit(one,job)]=job
  completed,_=cf.wait(pending,timeout=10,return_when=cf.FIRST_COMPLETED)
  failure=None
  for future in completed:
   job=pending.pop(future)
   try:r=future.result()
   except BaseException as e:
    r=dict(id=job['id'],partition=partition,opponent_list=job['b'],anchor_seat=job['seat'],status='FAILED',error=type(e).__name__+': '+str(e));append(OUT/'matches.jsonl',r);failure=e;continue
   try:register_result(r);results.append(r)
   except BaseException as e:failure=e
  if failure:
   # Already running scheduled attempts are consumed and their records kept;
   # no new job is submitted after the first implementation failure.
   for future,job in list(pending.items()):
    try:
     r=future.result()
     try:register_result(r)
     except BaseException:pass
    except BaseException as e:append(OUT/'matches.jsonl',dict(id=job['id'],partition=partition,opponent_list=job['b'],anchor_seat=job['seat'],status='FAILED',error=type(e).__name__+': '+str(e)))
   pending.clear();raise failure
  progress(partition,submitted=index,completed=len(results),planned=296,storage_bytes=storage())
 return results

def stability_one(task):
 record,N,seed=task;start=time.process_time();r=replay(record['corpus'],seed,N);pred=r.pop('predictions');target=PRIVATE/'stability'/f'{record["id"]}-{N}-{seed}.json.gz';target.parent.mkdir(exist_ok=True)
 with gzip.open(target,'wt',compresslevel=3) as f:json.dump(pred,f,separators=(',',':'))
 return dict(match_id=record['id'],particles=N,seed=seed,prediction_path=str(target.relative_to(R)),CPU_seconds=time.process_time()-start,metrics=r['belief_metrics'],counts=r['belief_counts'],recoveries=len(r['recoveries']))

def do_stability(pool,records):
 spec=read(OUT/'stability-schedule.json');ids=spec.get('match_ids') or spec.get('development_match_ids') or spec.get('selected_match_ids')
 if ids is None:
  ids=next(v for v in spec.values() if isinstance(v,list) and len(v)==16)
 ids=[x['id'] if isinstance(x,dict) else x for x in ids];selected=[r for r in records if r['id'] in ids];assert len(selected)==16
 tasks=[(r,n,s) for r in selected for n in (512,2048) for s in (11,29,47)];out=[]
 for r in pool.map(stability_one,tasks,chunksize=1):check_time();out.append(r);append(OUT/'stability.jsonl',r);progress('stability',completed=len(out),planned=96)
 discrepancy=[]
 for match in ids:
  arrays=[]
  for r in out:
   if r['match_id']==match:
    pred=json.load(gzip.open(R/r['prediction_path'],'rt'));arrays.append(np.array([v['p_full'] for v in pred]))
  for i,a in enumerate(arrays):
   for b in arrays[i+1:]:discrepancy.append(float(np.mean(np.abs(a-b))))
 result=dict(at_utc=iso(),histories=16,replays=96,particle_budgets=[512,2048],seeds=[11,29,47],no_new_native_matches=True,mean_absolute_probability_discrepancy=float(np.mean(discrepancy)),maximum_history_pair_discrepancy=max(discrepancy),larger_budget='convergence comparison, not ground truth',fallback_included=True,CPU_seconds=sum(r['CPU_seconds'] for r in out));write(OUT/'stability-summary.json',result)

def isolate_process_group():
 try:os.setsid()
 except PermissionError:
  # WSL can already launch Python as the leader of its own process group.
  # That group is already the exact scope the deadline guard needs to stop.
  if os.getpgrp()!=os.getpid():raise
 assert os.getpgrp()==os.getpid()

def main():
 isolate_process_group();write(OUT/'local-pid.json',dict(pid=os.getpid(),process_group=os.getpgrp(),at_utc=iso()));exclusive(OUT/'runner-intent.json',dict(at_utc=iso(),pid=os.getpid(),automatic_restart=False));start=now();cpu=time.process_time();state='INCOMPLETE';reason='Not finished';records=[]
 try:
  verify_sources();assert read(OUT/'native-gate.json')['passed'];assert read(OLD/'completion.json')['actual_native_attempts']==0
  with cf.ProcessPoolExecutor(WORKERS,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
   records+=phase(pool,'development');verify_sources();write(OUT/'development-analysis.json',summarize(records,'development'));write(OUT/'held-out-freeze.json',dict(at_utc=iso(),source_manifest_sha256=sha(OUT/'source-manifest.json'),registration_sha256=sha(OUT/'registration.json'),development_attempts=296,held_out_accessed=False,implementation_changes=False));records+=phase(pool,'held-out');verify_sources();write(OUT/'held-out-analysis.json',summarize(records));do_stability(pool,records)
  from unown_belief_native_gate import assess
  write(OUT/'practical-gates.json',assess(records,read(OUT/'held-out-analysis.json'),read(OUT/'stability-summary.json')))
  state='COMPLETE';reason='Fixed592 native attempts, registered scoring and96 offline stability replays finished; no further experiment authorized.'
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);state='INCOMPLETE' if isinstance(e,TimeoutError) else 'FAILED';write(OUT/'runner-failure.json',dict(at_utc=iso(),reason=reason,traceback=traceback.format_exc(),no_replacement=True))
 finally:
  allrows=rows(OUT/'matches.jsonl');intents=list((OUT/'intents').glob('*.json'));cost=dict(at_utc=iso(),state=state,reason=reason,actual_native_attempts=len(intents),completed_records=sum(r.get('status')=='DONE' for r in allrows),failed_records=sum(r.get('status')!='DONE' for r in allrows),development_attempts=sum(read(p)['job']['partition']=='development' for p in intents),held_out_attempts=sum(read(p)['job']['partition']=='held-out' for p in intents),planned_attempts=592,UTC_wall_seconds=(now()-start).total_seconds(),controller_CPU_seconds=time.process_time()-cpu,worker_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,storage_bytes=storage(),GPU_seconds=0,training_updates=0,submissions=0,model_API_requests=0,power_changes=False,sleep_inhibition=False,automatic_followup=False);write(OUT/'completion.json',cost)
  from unown_belief_native_report import render
  render(state,reason,True)
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
