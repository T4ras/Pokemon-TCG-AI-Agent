"""Persistent CPU-only nested-CV benchmark, with aggregate resource limits."""
from .common import *
from .cache import cache_one,initialize,ResourceStop,OperationLimit,bounded
import concurrent.futures as cf,multiprocessing as mp,threading,signal,traceback,shutil

def verify_sources():
 for p,v in read(OUT/'source-manifest.json')['files'].items():
  if sha(R/p)!=v['sha256']:raise InvariantFailure('Frozen source/input changed: '+p)

def process_CPU(stat):
 x=stat[stat.rfind(')')+2:].split();return int(x[2]),int(x[11])+int(x[12]),x[19]
class Meter:
 def __init__(self):self.seen={};self.group=os.getpgrp();self.hz=os.sysconf('SC_CLK_TCK');self.lock=threading.Lock()
 def sample(self):
  with self.lock:
   for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     group,ticks,start=process_CPU((p/'stat').read_text())
     if group==self.group:self.seen[p.name+':'+start]=max(self.seen.get(p.name+':'+start,0),ticks/self.hz)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   return sum(self.seen.values())+(read(OUT/'owner-meter.json')['CPU_seconds'] if (OUT/'owner-meter.json').exists() else 0.)
class SharedRuntimeFailure(RuntimeError):pass
class SharedCorrectnessFailure(InvariantFailure):pass

def classify_shared_failure(result):
 error=result.get('error') or {};runtime=error.get('type') in ('OperationLimit','MemoryError','TimeoutError','ResourceStop')
 kind='shared_runtime' if runtime else 'shared_correctness'
 detail=error.get('type','Unknown')+': '+error.get('message','No worker diagnostic')
 return kind,('Shared runtime guard' if runtime else 'Shared allocation correctness failure')+': '+result['id']+'; '+detail

def cpu_limit():return read(OUT/'budget.json')['remaining_CPU_seconds']
def stop(reason,kind='resource'):
 if not (OUT/'STOP.json').exists():write(OUT/'STOP.json',dict(at_utc=iso(),reason=reason,kind=kind))
def check(meter):
 if (OUT/'STOP.json').exists():raise ResourceStop(read(OUT/'STOP.json')['reason'])
 if meter.sample()>=cpu_limit()-180:stop('8 CPU-core-hour cap with finalization reserve');raise ResourceStop('CPU cap')
 if (deadline()-now()).total_seconds()<=180:stop('8 elapsed-hour cap with finalization reserve');raise ResourceStop('Elapsed cap')
 if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())>2*1024**3:stop('Registered2GiB output cap');raise ResourceStop('Storage cap')
def watchdog(meter,done):
 signalled=None;sent=set();killed=set()
 while not done.wait(.25):
  try:check(meter)
  except ResourceStop:
   if signalled is None:signalled=time.monotonic()
   for p in (OUT/'workers').glob('*.json'):
    pid=read(p)['pid']
    try:
     if pid not in sent:os.kill(pid,signal.SIGTERM);sent.add(pid)
     elif time.monotonic()-signalled>=8 and pid not in killed:os.kill(pid,signal.SIGKILL);killed.add(pid)
    except ProcessLookupError:pass
  except BaseException as exc:write(OUT/'meter-failure.json',dict(at_utc=iso(),error=str(exc)));stop('Resource meter failed')

def collect_common(meter):
 keys=('legal_path','legal_sha256','selected_decisions','selection');jobs=[dict(id=x['id'],corpus={k:x['corpus'][k] for k in keys}) for x in read(OUT/'inputs.json')];pending={};it=iter(jobs);exhausted=False;records=[]
 with cf.ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
  while pending or not exhausted:
   check(meter)
   while len(pending)<4 and not exhausted:
    try:job=next(it)
    except StopIteration:exhausted=True;break
    exclusive(OUT/'intents'/(job['id']+'.json'),dict(at_utc=iso(),id=job['id'],stage='common_legal_allocation'));pending[pool.submit(cache_one,job)]=job
   if not pending:break
   ready,_=cf.wait(pending,timeout=.25,return_when=cf.FIRST_COMPLETED)
   for future in ready:
    job=pending.pop(future);result=future.result();records.append(result);append(OUT/'common-ledger.jsonl',dict(at_utc=iso(),id=result['id'],status=result['status'],completed_points=result['completed_points']))
    if result['status']!='COMPLETE':
     kind,reason=classify_shared_failure(result);stop(reason,kind)
     raise (SharedRuntimeFailure if kind=='shared_runtime' else SharedCorrectnessFailure)(reason)
   write(OUT/'progress.json',dict(at_utc=iso(),phase='common_observer_allocation',expected=296,issued=len(list((OUT/'intents').glob('*.json'))),completed=len(records),aggregate_CPU_seconds=meter.sample()))
   if len(records)>=2 and not (OUT/'startup.json').exists():write(OUT/'startup.json',dict(at_utc=iso(),passed=all(x['status']=='COMPLETE' for x in records),allocated_common_histories_verified=len(records),new_native_matches=0,held_out_accessed=False,source_manifest_sha256=sha(OUT/'source-manifest.json')))
 return records

def verify_retained_allocations(records):
 """Pre-comparison numerical gate against every completed failed-run cache."""
 import gzip,numpy as np
 historical=R/'experiments/astra-unown-belief-bayesian-deck-inference-001';compared=points=0;maximum=0.;started=time.process_time()
 for record in records:
  path=historical/'common-results'/(record['id']+'.json')
  if not path.exists():continue
  old=read(path)
  if old['status']!='COMPLETE':continue
  for item in (old,record):
   for field in ('feature','allocation'):
    if sha(R/item[field+'_path'])!=item[field+'_sha256']:raise SharedCorrectnessFailure('Retained cache checksum mismatch')
  with np.load(R/old['allocation_path']) as a,np.load(R/record['allocation_path']) as b:
   for key in ('decks','offsets'):
    if not np.array_equal(a[key],b[key]):raise SharedCorrectnessFailure('Historical cache layout changed: '+record['id'])
   if a['allocations'].shape!=b['allocations'].shape or not np.allclose(a['allocations'],b['allocations'],rtol=1e-10,atol=1e-10):raise SharedCorrectnessFailure('Historical allocation feature drift: '+record['id'])
   maximum=max(maximum,float(np.max(np.abs(a['allocations']-b['allocations']))))
  with gzip.open(R/old['feature_path'],'rt') as a,gzip.open(R/record['feature_path'],'rt') as b:
   if json.load(a)!=json.load(b):raise SharedCorrectnessFailure('Legal features changed relative to historical cache')
  compared+=1;points+=record['completed_points']
 write(OUT/'retained-allocation-equivalence.json',dict(at_utc=iso(),passed=True,completed_histories=compared,selected_points=points,max_absolute_error=maximum,rtol=1e-10,atol=1e-10,CPU_seconds=time.process_time()-started,labels_used=False))
 if compared!=220:raise SharedCorrectnessFailure('Expected all220 completed historical allocations to be compared')

def main():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 exclusive(OUT/'runner-intent.json',dict(at_utc=iso(),pid=os.getpid()));write(OUT/'pid.json',dict(at_utc=iso(),pid=os.getpid(),group=os.getpgrp()));meter=Meter();done=threading.Event();thread=threading.Thread(target=watchdog,args=(meter,done),daemon=True);thread.start();state='FAILED';reason='Unfinished';records=[]
 try:
  verify_sources();assert read(OUT/'correctness-gate.json')['passed'];from .report import render
  render('RUNNING','Computing shared legal allocations, then nested statistical fitting and out-of-fold scoring.');common=collect_common(meter);assert len(common)==296;check(meter);verify_retained_allocations(common);check(meter)
  from .cv import load_features,select_parameters
  from .deck import catalogue
  from .evaluate import evaluate_one
  features=load_features(common);inputs={x['id']:x for x in read(OUT/'inputs.json')};decks,*_=catalogue();labels={i:decks.index(x['opponent_list']) for i,x in inputs.items()};folds=read(OUT/'folds.json')['outer_folds'];common_by_id={x['id']:x for x in common}
  contradictions=[dict(id=ident,index=point['index']) for ident,feature in features.items() for point in feature['points'] if labels[ident] not in point['compatible']]
  write(OUT/'allocation-support-gate.json',dict(at_utc=iso(),contradictions=contradictions,passed=not contradictions,identity_used_only_for_diagnostic_validation=True))
  if contradictions:raise SharedCorrectnessFailure('Shared conditional allocation excludes true list; no classifier fitted')
  for outer in range(4):
   check(meter);write(OUT/'progress.json',dict(at_utc=iso(),phase='inner_validation_and_fitting',outer_fold=outer,scored=len(records),expected=296,aggregate_CPU_seconds=meter.sample()));theta,model=select_parameters(outer,features,labels,folds,lambda:check(meter))
   for ident in folds[outer]:
    check(meter);exclusive(OUT/'evaluation-intents'/(ident+'.json'),dict(at_utc=iso(),id=ident,outer_fold=outer,model_sha256=model['model_sha256']));write(OUT/'evaluation-active.json',dict(id=ident,outer_fold=outer,at_utc=iso()));record=evaluate_one(inputs[ident],common_by_id[ident],theta,model['beta'],features[ident]);records.append(record);write(OUT/'progress.json',dict(at_utc=iso(),phase='out_of_fold_scoring',outer_fold=outer,scored=len(records),expected=296,aggregate_CPU_seconds=meter.sample()))
  verify_sources();check(meter);from .analysis import analysis
  analysis(records,common,lambda:check(meter));state='COMPLETE';reason='All296 out-of-fold BO3 predictions scored; no follow-up or held-out access.'
 except BaseException as exc:
  state='FAILED_RUNTIME' if isinstance(exc,(SharedRuntimeFailure,OperationLimit)) else 'FAILED_SHARED' if isinstance(exc,SharedCorrectnessFailure) else 'BUDGET_EXHAUSTED' if isinstance(exc,(ResourceStop,TimeoutError)) and (OUT/'STOP.json').exists() and 'cap' in read(OUT/'STOP.json')['reason'] else 'FAILED';reason=type(exc).__name__+': '+str(exc);write(OUT/'failure.json',dict(at_utc=iso(),state=state,reason=reason,traceback=traceback.format_exc()))
 finally:
  common=[read(p) for p in (OUT/'common-results').glob('*.json')];evaluations=[read(p) for p in (OUT/'evaluation-results').glob('*.json')]
  if not (OUT/'analysis.json').exists() and evaluations and meter.sample()<cpu_limit()-60 and (deadline()-now()).total_seconds()>30:
   try:
    from .analysis import analysis
    bounded(lambda:analysis(evaluations,common),30)
   except BaseException as exc:write(OUT/'partial-analysis-failure.json',dict(at_utc=iso(),error=str(exc)))
  cost=dict(at_utc=iso(),state=state,reason=reason,expected_BO3=296,expected_points=sum(x['corpus']['selected_decisions'] for x in read(OUT/'inputs.json')),common_attempted=len(list((OUT/'intents').glob('*.json'))),common_completed=sum(x['status']=='COMPLETE' for x in common),common_failed=sum(x['status']!='COMPLETE' for x in common),evaluation_attempted=len(list((OUT/'evaluation-intents').glob('*.json'))),evaluation_completed=len(evaluations),scored_points=sum(x['scored'] for x in evaluations),aggregate_CPU_seconds=meter.sample(),elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds(),CPU_limit_seconds=cpu_limit(),original_CPU_limit_seconds=28800,prior_charged_CPU_seconds=read(OUT/'budget.json')['prior_charged_CPU_seconds'],elapsed_limit_seconds=28800,unresolved_common_intents=[p.stem for p in (OUT/'intents').glob('*.json') if not (OUT/'common-results'/(p.stem+'.json')).exists()],unresolved_evaluation_intents=[p.stem for p in (OUT/'evaluation-intents').glob('*.json') if not (OUT/'evaluation-results'/(p.stem+'.json')).exists()],private_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()),held_out_accessed=False,new_native_matches=0,GPU_seconds=0,neural_training=0,PPO_updates=0,uploads=0,submissions=0,paid_model_API_calls=0,automatic_followup=False);write(OUT/'completion.json',cost)
  from .report import render
  try:render(state,reason)
  finally:
   done.set();thread.join(timeout=2);cost.update(final_CPU_seconds=meter.sample(),cumulative_CPU_seconds=meter.sample()+read(OUT/'budget.json')['prior_charged_CPU_seconds'],original_started_at_utc=read(OUT/'budget.json')['original_started_at_utc'],final_elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds());write(OUT/'completion.json',cost);target=Path('/path/to/ptcg-agent')/'experiments'/RUN/'completion.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/'completion.json',target)
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
