"""At most16registered native attempts; disposable updates, no pilot candidates."""
from .common import *
from .integration import PilotRuntime,context,environment_worker
from .correction import configure,frozen_digest,learning_rate
import torch,numpy as np,copy,functools,resource,traceback,signal,threading

def run_arm(arm,check):
 ctx=context(arm);ctx.out.mkdir(parents=True,exist_ok=True);ctx.private.mkdir(parents=True,exist_ok=True);rt=PilotRuntime(ctx,arm);pool=None;started=time.monotonic();cpu=time.process_time()
 try:
  learner=rt.load_corrected();opponent=rt.load_base();opt=configure(learner);initial_frozen=frozen_digest(learner);initial_classifier=read(OUT/'frozen-components.json')['belief_model']['sha256'];deck_info=read(OUT/'decks.json');decks={k:dict(cards=v['cards']) for k,v in deck_info['prior'].items()};decks['Hydro']=dict(cards=deck_info['prior'][deck_info['anchor']]['cards']);manifest=dict(decks=decks,pilot_run=RUN,pilot_arm=arm,policy_tensor=POLICY_TENSOR,belief_classifier=initial_classifier)
  write(ctx.out/'manifest.json',manifest);jobs=[]
  for j in read(OUT/'readiness-schedule.json')['attempts']:
   if j['arm']!=arm or (ctx.out/'native-intents'/(j['id']+'.json')).exists():continue
   jobs.append(dict(j,learner_deck='Hydro'))
  # Four worker startups, one active environment: at most4 local compute threads.
  pool=rt.pool.ConcurrentCollector(4,worker_target=functools.partial(environment_worker,ctx.spec(),arm),manifest=manifest);start=time.monotonic();rows,matches,summary=pool.collect(jobs,dict(learner=learner,opponent=opponent),concurrency=1,deadline=time.monotonic()+3000,safety_check=check,allow_partial=True);collection=time.monotonic()-start;pool.close();pool=None
  if len(matches)!=len(jobs) or any(m['status']!='DONE' for m in matches):raise InvariantFailure('ReadinessBO3allocationincomplete')
  if any(np.any(r['belief']) for r in rows) and arm=='state':raise InvariantFailure('Statearmreceivedbeliefinput')
  if arm=='belief' and not all(np.isfinite(r['belief']).all() and r['belief'].shape==(1479,) for r in rows):raise InvariantFailure('Beliefcoverage failure')
  cohort=rt.epochs.new_cohort(rows,rt.state_hash(learner));path=PRIVATE/f'readiness-{arm}-cohort.pt';torch.save(dict(cohort=cohort,software_verification_only=True,candidate_eligible=False),path);ref=dict(path=str(path.relative_to(R)),sha256=sha(path));check();compat=rt.checks.collected(learner,rows,'readiness-'+arm,ref,'revised');engine=rt.make_engine(learner,opt,cohort,0);reference=engine.reference_check(rows)
  if reference['status']!='PASS':raise InvariantFailure('Eagerbucket64 collectionprobabilitymismatch')
  execution=dict(step=0,original_step=0,phase=0,accepted_used_decisions=0,unique_used_decisions=0,meaningful_unique_used=0,meaningful_exposures=0);rt.epochs.initialize_cursor(execution,len(rows));learning_rate(opt,0);events=[];t=time.monotonic()
  def emit(kind,event):
   check();events.append(dict(kind=kind,**event));append(ctx.out/'readiness-update-events.jsonl',dict(at_utc=iso(),kind=kind,**event))
  try:terminal=rt.epochs.optimize(learner,opt,cohort,execution,transact=engine.transaction,measured=engine.measure,emit=emit,commit=lambda:None,should_stop=lambda:False)
  finally:engine.close()
  update=time.monotonic()-t;assert frozen_digest(learner)==initial_frozen;rt.epochs.verify_cohort(cohort);timers=summary['service']['timers'];belief_costs=collections.Counter()
  for m in matches:belief_costs.update(m.get('belief_costs',{}))
  result=dict(at_utc=iso(),arm=arm,state='COMPLETE',attempted_BO3=len(matches),completed_BO3=len(matches),fresh_readiness_decisions=len(rows),cohort_sha256=sha(path),readiness_experience_not_pilot_training=True,collection_wall_seconds=collection,pool_startup_seconds=summary['pool_startup_seconds'],update_wall_seconds=update,update_CPU_seconds=None,completed_passes=execution['completed_epochs'],accepted_updates=execution['step'],effective_passes=execution['cohort_exposures']/len(rows),update_terminal=terminal,events=events,collection=summary,belief_costs=dict(belief_costs),frozen_policy_identity=True,learner_tensor_hash_before=cohort['collection_policy_hash'],all_old_references_preserved=True,compatibility=compat['status'],bucket64_reference=reference,optimizer_groups=[dict(name=g['name'],lr=g['lr'],betas=g['betas'],eps=g['eps'],weight_decay=g['weight_decay']) for g in opt.param_groups],parent_CPU_seconds=time.process_time()-cpu,complete_arm_wall_seconds=time.monotonic()-started,raw_matches=matches)
  write(OUT/f'readiness-{arm}.json',result);return result
 finally:
  if pool is not None:pool.close()
  rt.close()

def main():
 try:os.setsid()
 except PermissionError:pass
 from unown_belief_population_sampling.run import Meter as OldMeter
 # Reuse the monitor implementation with run-local paths, a12core-hour cap and36h deadline.
 import unown_belief_population_sampling.run as monitor
 monitor.OUT=OUT;monitor.CPU_CAP=43200;monitor.PRIVATE_ROOTS=(PRIVATE,context("state").private,context("belief").private);monitor.CURATED_CAP=6*1024**3;monitor.PREP_CHARGE=1220;monitor.deadline=lambda:dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['deadline_utc']);meter=OldMeter();done=threading.Event();watch=threading.Thread(target=monitor.watchdog,args=(meter,done),daemon=True);watch.start();state='FAILED';results=[]
 def check():guard();monitor.check(meter)
 try:
  exclusive(OUT/'readiness-intent.json',dict(at_utc=iso(),pid=os.getpid(),native_attempt_cap=16));write(OUT/'pid.json',dict(pid=os.getpid(),group=os.getpgrp(),phase='readiness'));assert read(OUT/'preflight.json')['passed'];check()
  for arm in ('state','belief'):results.append(run_arm(arm,check))
  state='COMPLETE';write(OUT/'readiness-completion.json',dict(at_utc=iso(),state=state,arms=[r['arm'] for r in results],native_attempts=sum(len(list((context(a).out/"native-intents").glob("*.json"))) for a in ("state","belief")),CPU_seconds=meter.sample(),GPU_seconds=0,actual_pilot_training_decisions=0,preparation_charge_seconds=1200))
 except BaseException as e:write(OUT/'readiness-failure.json',dict(at_utc=iso(),reason=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),CPU_seconds=meter.sample(),native_attempts=sum(len(list((context(a).out/'native-intents').glob('*.json'))) for a in ('state','belief')),GPU_seconds=0));raise
 finally:done.set();watch.join(timeout=2)
if __name__=='__main__':main()
