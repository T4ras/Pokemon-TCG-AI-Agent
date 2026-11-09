"""Ordinary persistent controller: no language models or external compute."""
from .common import *
from .storage import Shards,import_payload
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
 if meter.sample()>=CPU_CAP-1800:stop('64 CPU-core-hour cap, including30-minute analysis reserve');raise BudgetStop('Aggregate CPU cap')
 if (deadline()-now()).total_seconds()<=600:stop('48-hour UTC cap with10-minute finalization reserve');raise BudgetStop('Elapsed cap')
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

def initialize_actor():
 from .worker import initialize
 initialize()
def collect_one(job):
 from .worker import one
 return one(job)

def preprocess_one(row):
 from .preprocess import one
 return one(row)

def extract(rows,meter):
 eligible=[r for r in rows if r['partition']!='locked-test'];results=[]
 with cf.ProcessPoolExecutor(3,mp_context=mp.get_context('spawn')) as pool:
  pending={};next_index=0
  while pending or next_index<len(eligible):
   check(meter)
   while len(pending)<3 and next_index<len(eligible):
    row=eligible[next_index];next_index+=1;pending[pool.submit(preprocess_one,row)]=row
   ready,_=cf.wait(pending,timeout=.5,return_when=cf.FIRST_COMPLETED)
   for future in ready:
    row=pending.pop(future)
    try:results.append(future.result())
    except BaseException as exc:
     write(OUT/'feature-failure.json',dict(id=row['id'],error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()));stop('Material feature/target correctness failure: '+row['id'],'correctness');raise
 return results

def record_result(result,shards):
 payload=result.pop('payload',None)
 # Outcome-bearing receipts live only in private provenance, especially locked test.
 for field in ('games','result','actor_errors'):result.pop(field,None)
 append(OUT/'matches.jsonl',result);append(OUT/'attempt-ledger.jsonl',dict(event='RESULT',id=result['id'],status=result['status'],at_utc=iso(),partition=result['partition']))
 if result.get('detail_error'):append(OUT/'detail-errors.jsonl',dict(id=result['id'],error=result['detail_error']))
 if payload is not None:shards.append(payload)
 if result['status']!='DONE' or payload is None:raise InvariantFailure('Native collection/integrity failure '+result['id']+': '+str(result.get('error')))

def drain_started(pending,shards,meter):
 """Finish only already-dispatched work before a correctness stop.

 The resource watchdog stays active. No new match is submitted here.
 """
 records=[]
 write(OUT/'drain-progress.json',dict(at_utc=iso(),state='DRAINING_ALREADY_STARTED',ids=[j['id'] for j in pending.values()],new_dispatches=0))
 for future,job in list(pending.items()):
  try:
   while not future.done():check(meter);time.sleep(.1)
   record_result(future.result(),shards);records.append(dict(id=job['id'],status='COMMITTED'))
  except BudgetStop:
   write(OUT/'drain-budget-stop.json',dict(at_utc=iso(),id=job['id'],preserved=records));raise
  except BaseException as exc:
   records.append(dict(id=job['id'],status='FAILED',error=type(exc).__name__+': '+str(exc)))
  finally:pending.pop(future,None)
 shards.flush_all()
 write(OUT/'drain-progress.json',dict(at_utc=iso(),state='FINISHED_ALREADY_STARTED',records=records,new_dispatches=0))
 return records

def collect_batch(pool,batch,shards,meter):
 jobs=batch['jobs'];pending={};next_index=0;results=[]
 while pending or next_index<len(jobs):
  check(meter);startup_done=(OUT/'startup.json').exists()
  limit=3 if startup_done else 2
  while len(pending)<limit and next_index<len(jobs) and (startup_done or next_index<2):
   job=jobs[next_index];next_index+=1;exclusive(OUT/'issued'/(job['id']+'.json'),dict(job=job,at_utc=iso()));append(OUT/'attempt-ledger.jsonl',dict(event='ISSUED',id=job['id'],partition=job['partition'],at_utc=iso()));pending[pool.submit(collect_one,job)]=job
  ready,_=cf.wait(pending,timeout=.5,return_when=cf.FIRST_COMPLETED)
  for future in ready:
   job=pending.pop(future)
   try:
    result=future.result();record_result(result,shards);results.append(result)
   except BaseException as exc:
    write(OUT/'collection-failure.json',dict(id=job['id'],partition=job['partition'],error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
    # A correctness stop prevents new dispatches immediately; let the at-most
    # two already-started frozen BO3 save results before STOP signals workers.
    drain_started(pending,shards,meter)
    stop('Collection stopped after failure: '+job['id'],'correctness');raise
  if not startup_done and len(results)>=2:
   shards.flush_all()
   write(OUT/'startup.json',dict(at_utc=iso(),passed=all(x['startup_parity'] and x['policy_unchanged'] for x in results[:2]),ids=[x['id'] for x in results[:2]],allocated_native_checks=2,extra_native_matches=0,policy_tensor=TENSOR,shard_integrity='first allocated BO3 components committed and roundtrip verified',source_manifest_sha256=sha(OUT/'source-manifest.json')))
  write(OUT/'progress.json',dict(at_utc=iso(),phase='collection',batch=batch['id'],partition=batch['partition'],batch_completed=len(results),batch_expected=len(jobs),total_issued=len(list((OUT/'issued').glob('*.json'))),total_native_complete=sum(r['event']=='RESULT' and r['status']=='DONE' for r in rows(OUT/'attempt-ledger.jsonl'))+len(read(OUT/'recovery-registration.json').get('recovered_ids',[])),CPU_seconds=meter.sample()))
 return results

def feasibility(first,features,meter):
 size,details=refresh_storage();cpu=sum(x['CPU_core_seconds']+x.get('integrity_CPU_seconds',0) for x in first)/len(first);feature_cpu=sum(x['total_CPU_seconds'] for x in features)/len(features);startup=sum(read(p).get('initialization_CPU_seconds',0) for p in (OUT/'workers').glob('*.json'));elapsed_ratio=max(2.,max(x['UTC_seconds']/max(x['CPU_core_seconds'],.1) for x in first));remaining=5556-len(first)
 expected_CPU=meter.sample()+remaining*cpu*1.5+(5328-len(features))*feature_cpu*1.5+FIT_CAP+3600
 expected_elapsed=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds()+remaining*cpu/3*elapsed_ratio+(5328-len(features))*feature_cpu/3*elapsed_ratio+FIT_CAP+3600
 raw_bytes=sum(p.stat().st_size for p in (PRIVATE/'dataset'/'validation').glob('*'))/len(first);feature_bytes=sum((R/x['feature_path']).stat().st_size+(R/x['target_path']).stat().st_size for x in features)/len(features)
 predicted_storage=1.5*(5920*raw_bytes+5328*feature_bytes+36*592*feature_bytes*.5)+256*1024**2
 fit_probe=read(OUT/'neural-workload-probe.json');predictions=[]
 for n in SIZES:
  for family in ('MLP','GRU'):
   value=(fit_probe[family]['training_load_and_update_CPU_seconds']*n*37*6+fit_probe[family]['validation_load_and_inference_CPU_seconds']*592*6)*1.2+30
   predictions.append(dict(family=family,size=n,forecast_CPU_seconds=value,per_fit_cap_seconds=fit_budget(dict(family=family,size=n))))
 predicted_neural=sum(x['forecast_CPU_seconds']*2 for x in predictions);tree_probe=read(OUT/'tree-workload-probe.json');tree_per_fit=tree_probe['CPU_seconds']*32*2+10;predicted_fit=predicted_neural+tree_per_fit*10+600
 okay=expected_CPU<CPU_CAP and expected_elapsed<ELAPSED_CAP and predicted_storage<CURATED_CAP and predicted_fit<FIT_CAP and tree_per_fit<900 and all(x['forecast_CPU_seconds']<x['per_fit_cap_seconds'] for x in predictions)
 forecast=dict(at_utc=iso(),first_batch_BO3=len(first),mean_native_CPU_seconds=cpu,worker_startup_CPU_seconds=startup,mean_feature_CPU_seconds=feature_cpu,power_slowdown_elapsed_factor=elapsed_ratio,conservative_total_CPU_hours=expected_CPU/3600,conservative_elapsed_hours=expected_elapsed/3600,curated_storage_GiB=predicted_storage/1024**3,measured_full_BO3_neural_projection_CPU_hours=predicted_neural/3600,per_fit_neural_forecasts=predictions,tree_per_fit_forecast_CPU_seconds=tree_per_fit,conservative_full_fitting_CPU_hours=predicted_fit/3600,tree_probe_CPU_seconds=tree_probe['CPU_seconds'],fit_cap_CPU_hours=20,complete_plan_fits=okay,storage_forecast_includes_curated_predictions=True,new_native_diagnostic_matches=0)
 write(OUT/'feasibility-forecast.json',forecast)
 if not okay:raise FeasibilityStop('Measured complete plan does not conservatively fit registered caps; no schedule reduction or replacement')

def neural_workload_probe(meter,validation_features):
 """Disposable actual BO3 updates and validation forwards; timing only."""
 import torch
 from .models import make_neural,neural_loss,neural_predict
 from .preprocess import load_features
 from .fit import allowed_items
 torch.set_num_threads(3);torch.set_num_interop_threads(1)
 ids=set(read(OUT/'training-subsets.json')['8']);items=[x for x in allowed_items('training') if x['id'] in ids];selected=[]
 for family in sorted({v['archetype'] for v in prior().values()}):
  for seat in (0,1):selected.append(min([x for x in items if prior()[x['opponent_list']]['archetype']==family and x['chooser']==seat],key=lambda x:stable(SEED,'NN-timing',x['id'])))
 valid=sorted(validation_features,key=lambda x:stable(SEED,'validation-timing',x['id']))[:16];out={};start=time.process_time()
 for family in ('MLP','GRU'):
  model=make_neural(family,11);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);train_time=val_time=0.
  for item in selected:
   check(meter);tick=time.process_time();f,t=load_features(item['id']);optimizer.zero_grad(set_to_none=True);loss=neural_loss(model,f,t);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();train_time+=time.process_time()-tick
  for item in valid:
   check(meter);tick=time.process_time();f,t=load_features(item['id']);neural_predict(model,f);val_time+=time.process_time()-tick
  out[family]=dict(training_load_and_update_CPU_seconds=train_time/len(selected),validation_load_and_inference_CPU_seconds=val_time/len(valid),training_BO3=len(selected),validation_BO3=len(valid));del model,optimizer
 cost=time.process_time()-start;out.update(at_utc=iso(),CPU_seconds=cost,candidate_weights_reused=False,accuracy_scores_examined=False,source='fixed16 training BO3 across eight families/both chooser assignments; fixed16 fresh validation inputs; no new native matches')
 write(OUT/'neural-workload-probe.json',out);append(OUT/'fit-cost-ledger.jsonl',dict(key='registered-disposable-neural-probe',CPU_seconds=cost,at_utc=iso(),candidate_fit=False))

def tree_workload_probe(meter):
 """One disposable full296-BO3 gradient stage; only runtime is inspected."""
 from .models import BoostedTrees
 from .preprocess import load_features
 from .fit import allowed_items
 from unown_belief_bayes_repair.cache import bounded,OperationLimit
 from threadpoolctl import threadpool_limits
 ids=set(read(OUT/'training-subsets.json')['8']);items=[x for x in allowed_items('training') if x['id'] in ids];start=time.process_time();model=BoostedTrees(11)
 try:
  with threadpool_limits(limits=3):bounded(lambda:model.stage(items,0,load_features,lambda:check(meter)),600)
 except OperationLimit:raise FeasibilityStop('Disposable tree stage exceeded600 CPU seconds; complete fitting plan cannot fit')
 finally:
  cost=time.process_time()-start;del model
  write(OUT/'tree-workload-probe.json',dict(at_utc=iso(),training_BO3=len(items),stage=1,CPU_seconds=cost,candidate_weights_reused=False,validation_scores_examined=False,CPU_limit_seconds=600))
  append(OUT/'fit-cost-ledger.jsonl',dict(key='registered-disposable-tree-probe',CPU_seconds=cost,at_utc=iso(),candidate_fit=False))

def run_fit(job,meter):
 from .fit import fit_key
 key=fit_key(job);path=OUT/'fit-jobs'/(key+'.json');write(path,job);log=OUT/'fit-logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 used=sum(x['CPU_seconds'] for x in rows(OUT/'fit-cost-ledger.jsonl'))
 if used>=FIT_CAP-180:raise BudgetStop('20 fitting CPU-core-hour cap with export reserve')
 env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',CUDA_VISIBLE_DEVICES='');before=meter.sample();usage=resource.getrusage(resource.RUSAGE_CHILDREN);child_before=usage.ru_utime+usage.ru_stime;proc=None
 try:
  with log.open('w') as output:
   proc=subprocess.Popen([str(R/'.venv/bin/python'),'-m','unown_belief_learning_source_repair.fit',str(path)],cwd=R,env=env,stdout=output,stderr=subprocess.STDOUT);write(OUT/'fit-process.json',dict(pid=proc.pid,key=key,at_utc=iso()))
   while proc.poll() is None:
    check(meter)
    if meter.sample()-before>=min(fit_budget(job),FIT_CAP-used)-15:raise BudgetStop('Registered per-fit/aggregate fitting cap; incomplete fit preserved')
    time.sleep(.5)
  if proc.returncode:
   resultpath=OUT/'fit-results'/(key+'.json')
   if resultpath.exists():
    rec=read(resultpath);failure=R/rec['model_directory']/'failure.json'
    if failure.exists() and read(failure).get('type')=='FitLimit':raise BudgetStop('Registered fit cap reached: '+key+'; recoverable incomplete checkpoint retained')
   raise RuntimeError('Supervised fit incomplete/failed: '+key+'; see retained fit log/checkpoint')
  result=read(OUT/'fit-results'/(key+'.json'))
  if result['state']!='COMPLETE':raise InvariantFailure('Fit completion missing')
  return result
 finally:
  if proc is not None:
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
   usage=resource.getrusage(resource.RUSAGE_CHILDREN);actual=usage.ru_utime+usage.ru_stime-child_before;append(OUT/'fit-cost-ledger.jsonl',dict(key=key,CPU_seconds=actual,at_utc=iso(),returncode=proc.returncode,includes_startup_and_export=True))

def main():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 exclusive(OUT/'runner-intent.json',dict(pid=os.getpid(),at_utc=iso()));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),at_utc=iso()));meter=Meter();done=threading.Event();thread=threading.Thread(target=watchdog,args=(meter,done),daemon=True);thread.start();shards=Shards();state='FAILED';reason='Unfinished';phase='initialization';dataset_complete=False;fit_complete=False
 try:
  source_check()
  if not read(OUT/'recovery-forecast.json')['fits_original_budget']:raise FeasibilityStop('Recovery does not conservatively fit original caps/deadline')
  from .report import render
  render('RUNNING','Resuming eligible8/16 curves from3451 preserved/recovered records;32/64/128 unavailable; no native replacements or test scoring.');phase='recovery_verification';conversion_start=time.process_time()
  for row in shards.committed:
   check(meter)
   for item in row['paths'].values():
    if sha(R/item['path'])!=item['sha256']:raise InvariantFailure('Inherited shard checksum changed')
  write(OUT/'dataset-index.json',shards.committed);write(OUT/'dataset-shards.json',dict(shards=shards.manifest,committed_BO3=len(shards.committed),inherited_valid_BO3=len(shards.committed)))
  shards.flush_all();write(OUT/'conversion-cost.json',dict(CPU_seconds=time.process_time()-conversion_start,reused_BO3=len(shards.committed),at_utc=iso()));historical_rows=list(shards.committed);schedule=read(OUT/'collection-schedule.json');previous_ids={x['id'] for x in shards.committed}
  # Reuse immutable encoded caches; finish only missing non-test features.
  missing_features=[r for r in shards.committed if r['partition']!='locked-test' and not (OUT/'feature-results'/(r['id']+'.json')).exists()]
  write(OUT/'phase-state.json',dict(at_utc=iso(),phase='recovery_feature_extraction',missing_feature_matches=len(missing_features),dataset_BO3=len(shards.committed),dataset_complete=False,fitting_started=False,locked_test_scored=False))
  extract(missing_features,meter)
  with cf.ProcessPoolExecutor(3,mp_context=mp.get_context('spawn'),initializer=initialize_actor) as pool:
   for batch in schedule:
    phase='collection';first=collect_batch(pool,batch,shards,meter);shards.flush_all();newrows=[x for x in shards.committed if x['id'] not in previous_ids];previous_ids|={x['id'] for x in newrows};phase='feature_extraction';features=extract(newrows,meter)
    if batch['id']==0:
     extract(historical_rows,meter);tree_workload_probe(meter);neural_workload_probe(meter,features);feasibility(first,features,meter)
    write(OUT/'phase-state.json',dict(at_utc=iso(),phase=phase,last_completed_collection_batch=batch['id'],dataset_BO3=len(shards.committed),dataset_complete=False,fitting_started=False,locked_test_scored=False));render('RUNNING','Registered collection and shared feature checks continue; test targets remain sealed.')
  phase='dataset_seal';committed_ids={x['id'] for x in shards.committed}
  required=set(read(OUT/'training-subsets.json')['16']);entries=read(OUT/'split-manifest.json')['entries'];required|={x['id'] for x in entries if x['partition'] in ('validation','locked-test')}
  if not required<=committed_ids:raise InvariantFailure('Eligible nested subsets or fixed validation/test collection incomplete')
  counts=collections.Counter(x['partition'] for x in shards.committed);source_check();dataset_complete=False
  for root in PRIVATE_ROOTS:
   for p in (root/'dataset'/'locked-test').glob('targets-*.jsonl.gz'):p.chmod(0o444)
  write(OUT/'dataset-completion.json',dict(at_utc=iso(),state='PARTIAL_ORIGINAL_PLAN_ELIGIBLE_SUBSETS_COMPLETE',valid=True,original_dataset_complete=False,original_target=5920,BO3=len(shards.committed),partitions=dict(counts),eligible_sizes=list(SIZES),unavailable_sizes=[32,64,128],missing_ids=read(OUT/'recovery-registration.json')['lost_ids'],all_components_roundtrip_verified=True,locked_test_collected=592,locked_test_model_scores=0,split_manifest_sha256=sha(OUT/'split-manifest.json'),shard_manifest_sha256=sha(OUT/'dataset-shards.json')));render('RUNNING','User-approved eligible subsets8/16 complete;32/64/128 unavailable. Fixed supervised fits begin; test sealed.')
  phase='full_online_timing_inputs';from .timing import prepare as prepare_timing
  prepare_timing(shards.committed,lambda:check(meter))
  phase='validation';from .scoring import evaluate
  from threadpoolctl import threadpool_limits
  with threadpool_limits(limits=3):evaluate(dict(family='A',size=0,seed=0),lambda:check(meter))
  for job in read(OUT/'fit-schedule.json'):
   check(meter);phase='fitting';write(OUT/'phase-state.json',dict(at_utc=iso(),phase=phase,job=job,dataset_complete=False,eligible_subsets_complete=True,locked_test_scored=False));run_fit(job,meter);phase='validation'
   import torch
   torch.set_num_threads(3)
   with threadpool_limits(limits=3):evaluate(job,lambda:check(meter))
   render('RUNNING','Supervised fits and fixed validation evaluations continue; no locked-test scoring.')
  fit_complete=True;phase='analysis';check(meter);source_check();from .analysis import run
  with threadpool_limits(limits=3):run()
  state='COMPLETE_ELIGIBLE_CURVES';reason='All14 authorized fits and15 fixed validation evaluations at8/16 completed. Original5920-record plan remains incomplete;32/64/128 unavailable because four interrupted histories were not replaced. Locked test unscored; no automatic follow-up.'
 except BaseException as exc:
  state='PARTIAL_FEASIBILITY' if isinstance(exc,FeasibilityStop) else 'BUDGET_EXHAUSTED' if isinstance(exc,(BudgetStop,TimeoutError)) else 'FAILED_CORRECTNESS' if isinstance(exc,(InvariantFailure,AssertionError,UnsupportedEvent)) else 'FAILED';reason=type(exc).__name__+': '+str(exc);write(OUT/'failure.json',dict(at_utc=iso(),state=state,phase=phase,reason=reason,traceback=traceback.format_exc()))
 finally:
  try:shards.flush_all()
  except BaseException as e:write(OUT/'shard-finalization-failure.json',dict(error=str(e)))
  if (OUT/'validation-results').exists() and not (OUT/'analysis.json').exists() and meter.sample()<CPU_CAP-600 and (deadline()-now()).total_seconds()>300:
   try:
    from .analysis import run
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=3):run()
   except BaseException as e:write(OUT/'analysis-failure.json',dict(error=str(e),traceback=traceback.format_exc()))
  issued=list((OUT/'issued').glob('*.json'));records=rows(OUT/'matches.jsonl');completed_ids={x['id'] for x in records};size,detail=refresh_storage();fits=[read(p) for p in (OUT/'fit-results').glob('*.json')];complete_sizes=[]
  for n in SIZES:
   expected=[x for x in read(OUT/'fit-schedule.json') if x['size']==n]
   if all(any(r['job']==j and r['state']=='COMPLETE' for r in fits) for j in expected):complete_sizes.append(n)
  committed_ids={x['id'] for x in shards.committed};subsets=read(OUT/'training-subsets.json');raw_sizes=[n for n in SIZES if set(subsets[str(n)])<=committed_ids];encoded_ids={p.stem for p in (OUT/'feature-results').glob('*.json')};encoded_sizes=[n for n in SIZES if set(subsets[str(n)])<=encoded_ids];manifest_entries=read(OUT/'split-manifest.json')['entries'];balance={part:2*min(sum(x['id'] in committed_ids and x['partition']==part and x['opponent_list']==d and x['chooser']==seat for x in manifest_entries) for d in prior() for seat in (0,1)) for part in ('training','validation','locked-test')};prep=read(OUT/'preparation-charge.json')
  cost=dict(measured_execution_CPU_seconds=meter.sample()-prep['CPU_budget_charge_seconds']+prep.get('prior_measured_execution_CPU_seconds',0),measured_preparation_CPU_seconds=prep['measured_CPU_seconds'],preparation_budget_charge_seconds=prep['CPU_budget_charge_seconds'],fully_collected_training_subset_sizes=raw_sizes,fully_encoded_training_subset_sizes=encoded_sizes,minimum_balanced_BO3_per_list=balance,at_utc=iso(),state=state,reason=reason,phase=phase,CPU_seconds=meter.sample(),CPU_cap_seconds=CPU_CAP,elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds(),elapsed_cap_seconds=ELAPSED_CAP,failed_pre_native_dispatches=2,native_engine_starts=len(list((OUT/'native-starts').glob('*.json'))),native_expected=3158,original_native_allocation=5556,recovered_offline_ids=read(OUT/'recovery-registration.json').get('recovered_ids',[]),native_issued=len(issued),native_completed=sum(x['status']=='DONE' for x in records),native_failed=sum(x['status']!='DONE' for x in records),unresolved_native_ids=[p.stem for p in issued if p.stem not in completed_ids],dataset_BO3=len(shards.committed),dataset_complete=dataset_complete,dataset_valid=bool((OUT/'dataset-completion.json').exists() and read(OUT/'dataset-completion.json').get('valid')),fit_expected=len(read(OUT/'fit-schedule.json')),fit_complete=sum(x['state']=='COMPLETE' for x in fits),fit_CPU_seconds=sum(x['CPU_seconds'] for x in rows(OUT/'fit-cost-ledger.jsonl')),fit_CPU_cap_seconds=FIT_CAP,completed_balanced_learning_curve_sizes=complete_sizes,validation_complete=fit_complete,locked_test_scored=0,curated_bytes=size,detail_bytes=detail,GPU_seconds=0,paid_API_calls=0,uploads=0,submissions=0,PPO_updates=0,automatic_followup=False)
  write(OUT/'completion.json',cost)
  from .report import render
  try:render(state,reason)
  finally:
   done.set();thread.join(timeout=2);cost.update(final_CPU_seconds=meter.sample(),final_elapsed_seconds=(now()-dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['started_at_utc'])).total_seconds());write(OUT/'completion.json',cost);mirror=Path('/path/to/ptcg-agent')/'experiments'/RUN;shutil.copy2(OUT/'completion.json',mirror/'completion.json')
 if state not in ('COMPLETE','COMPLETE_ELIGIBLE_CURVES'):raise SystemExit(1)
if __name__=='__main__':main()
