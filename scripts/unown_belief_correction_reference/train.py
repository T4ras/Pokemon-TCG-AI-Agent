"""One arm; exact persistent production updates and recovery artifacts."""
from .common import *
from .runtime import PilotRunner,TrainingRuntime
from unown_belief_correction.correction import frozen_digest
import torch,numpy as np,resource

def build_cohorts(arm,schedule):
 if arm not in ('state','belief') or len(schedule)!=128:raise ValueError('Invalid frozen training allocation')
 cohorts=[];seen=set()
 for i,cohort in enumerate(schedule):
  if len(cohort)!=16 or sorted(j['learner_seat'] for j in cohort)!=[0]*8+[1]*8:raise ValueError('Unbalanced registered cohort')
  jobs=[]
  for j in cohort:
   index=j['schedule_index']
   if j['cohort']!=i or index in seen:raise ValueError('Duplicate/misrouted schedule index')
   seen.add(index);jobs.append(dict(j,id=f'{arm}-train-{index:04d}',arm=arm,stage='training',learner_deck='Hydro'))
  cohorts.append(jobs)
 if seen!=set(range(2048)):raise ValueError('Incomplete frozen schedule indices')
 return cohorts[11:126] if arm=='belief' else cohorts

def register_runtime(arm,device,software_only=False):
 ctx=context(arm);scientific=read(OUT/'registration.json');inputs={}
 if software_only:ctx=RunContext(ctx.root,RUN+'-software-'+arm,ctx.assets,ctx.execution,ctx.seed,ctx.recipe,ctx.limits,'software_test')
 for value in ctx.assets.values():
  p=Path(value);files=p.rglob('*') if p.is_dir() else [p]
  for x in files:
   if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc':inputs[str(x.resolve())]=sha(x)
 jobs=build_cohorts(arm,read(OUT/'training-schedule.json')['cohorts'])
 if arm=='belief':
  jobs=[[dict(j,id=('belief-target-' if j['cohort']==11 else 'belief-reference-')+str(j['schedule_index']).zfill(4),original_attempt_id=j['id'],attempt_number=1) for j in cohort] for cohort in jobs]
 decks=read(OUT/'decks.json');deckmap={k:dict(cards=v['cards']) for k,v in decks['prior'].items()};deckmap['Hydro']=dict(cards=decks['prior'][decks['anchor']]['cards'])
 ctx.out.mkdir(parents=True,exist_ok=True);ctx.private.mkdir(parents=True,exist_ok=True)
 reg=dict(context=ctx.spec(),device=device,inputs=inputs,implementation=implementation_hashes(),cohorts=jobs,decks=deckmap,scientific_registration_sha256=sha(OUT/'registration.json'))
 write(ctx.out/'runtime-registration.json',reg);write(ctx.out/'runtime-authorization.json',dict(run_id=ctx.run_id,registration_sha256=sha(ctx.out/'runtime-registration.json'),mode=ctx.mode,scope='local_software_verification_no_native' if software_only else 'explicitly_authorized_training_run',approved=True,human_authority='Local software-only verification, no native or GPU execution' if software_only else 'Explicit current user instruction to continue the saved run to250000decisions and disregard prior study budget constraints; same private Kaggle workflow and assets; no paid model APIs or submission',expires_utc=os.environ.get('UNOWN_ABSOLUTE_DEADLINE',scientific['deadline_utc'])))
 return ctx

def run(arm):
 if arm!='belief':raise PermissionError('Completed state arm must not be trained again')
 start=time.monotonic();runner=None;state='FAILED';reason='Startup incomplete';summary={}
 try:
  verify_inputs();guard();build_cohorts(arm,read(OUT/'training-schedule.json')['cohorts']);from unown_pilot_execution import cuda_startup
  cuda_startup(torch,lambda x:write(OUT/(arm+'-gpu-runtime.json'),x))
  if str(torch.__version__)!='2.11.0+cu128' or torch.version.cuda!='12.8' or sys.version_info[:2]!=(3,13):raise RuntimeError('Pinned backend mismatch')
  from .recovery_gate import run as pending_gate
  pending_gate('cuda')
  from .cloud_profile import run as profile_cache
  profile_cache()
  torch.cuda.manual_seed_all(SEED)
  from .probe import run as gpu_probe
  gpu_probe(arm)
  torch.cuda.manual_seed_all(SEED)
  ctx=register_runtime(arm,'cuda');runner=PilotRunner(ctx,ctx.out/'runtime-registration.json',ctx.out/'runtime-authorization.json');runner.run(False)
  state=runner.state;reason=runner.execution.get('registered_stop','resource_or_attempt_cap')
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);write(OUT/(arm+'-training-failure.json'),dict(at_utc=iso(),reason=reason,traceback=traceback.format_exc(),no_replacement=True));raise
 finally:
  summary=dict(at_utc=iso(),arm=arm,state=state,reason=reason,process_elapsed_seconds=time.monotonic()-start,session_cap_seconds=ARM_CAPS[arm],fresh_decisions=0,candidate_eligible=False)
  if runner and runner.model is not None:
   try:
    frozen=frozen_digest(runner.model);assert frozen==runner.rt.fixed_frozen
    classifier=read(ROOT/'experiments'/OLD/'frozen-components.json')['belief_model'];assert sha(ROOT/classifier['path'])==classifier['sha256']
    e=runner.execution;complete=state=='COMPLETE' and e['additional_collected']>=TARGET;final=PRIVATE/(arm+'-final.pt');runner.rt.legacy.save(final,runner.model,runner.opt,e['step'],extra=dict(execution=e,pilot_arm=arm,software_verification_only=False,candidate_eligible=complete,frozen_policy_digest=frozen,scientific_registration_sha256=sha(OUT/'registration.json')))
    summary.update(fresh_decisions=e['additional_collected'],target_overshoot=max(0,e['additional_collected']-TARGET),candidate_eligible=complete,accepted_updates=e['step'],accepted_exposures=e['accepted_used_decisions'],effective_PPO_passes=e['accepted_used_decisions']/max(1,e['additional_collected']),cohorts=11+runner.phase,prior_completed_trained_BO3=176,attempted_BO3=len(list((runner.context.out/'native-intents').glob('*.json'))),completed_BO3=sum(x['status']=='DONE' for x in records(runner.context.out/'native-records-gpu.jsonl')),prior_attempted_BO3=224,total_attempted_BO3=224+len(list((runner.context.out/'native-intents').glob('*.json'))),prior_completed_untrained_BO3=4,final_file_sha256=sha(final),final_tensor_sha256=runner.rt.state_hash(runner.model),frozen_tensor_digest=frozen,frozen_classifier_file_sha256=classifier['sha256'],execution=e,peak_allocated_VRAM_bytes=torch.cuda.max_memory_allocated(),peak_reserved_VRAM_bytes=torch.cuda.max_memory_reserved())
   except BaseException as e:summary.update(state='FAILED',candidate_eligible=False,export_error=str(e));raise
   finally:runner.rt.close()
  write(OUT/(arm+'-training-completion.json'),summary)
if __name__=='__main__':run(sys.argv[1])
