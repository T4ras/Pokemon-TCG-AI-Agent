"""One owned GPU training process; no uploads, launches or model APIs here."""
import resource,sys,traceback
from .common import *
from .runtime import AdaptationRuntime
from .model import frozen_state,actor_hash,inventory
from unown_runner.runner import Runner

def runtime_registration():
 scientific=read(OUT/'registration.json');ctx=context()
 for relative,entry in read(OUT/'input-manifest.json')['files'].items():
  if sha(ROOT/relative)!=entry['sha256']:raise ValueError('Frozen input changed: '+relative)
 inputs={}
 for value in ctx.assets.values():
  path=Path(value);files=[p for p in path.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts] if path.is_dir() else [path]
  for p in files:inputs[str(p.resolve())]=sha(p)
 reg=dict(context=ctx.spec(),device='cuda',inputs=inputs,implementation=implementation_hashes(),cohorts=scientific['cohorts'],decks=scientific['decks'],scientific_registration_sha256=sha(OUT/'registration.json'),row_manifest_sha256=sha(OUT/'trainable-rows.json'))
 write(OUT/'runtime-registration.json',reg)
 write(OUT/'runtime-authorization.json',dict(run_id=RUN,registration_sha256=sha(OUT/'runtime-registration.json'),mode='learning',scope='explicitly_authorized_training_run',approved=True,human_authority='Current user request plus exact private-upload approval; One explicit repair retry, remaining original session budget; no further replacements',expires_utc=os.environ['UNOWN_ABSOLUTE_DEADLINE']))
 return ctx

def run():
 from unown_pilot_execution import cuda_startup
 start=time.monotonic();cpu=time.process_time();runner=None;status='FAILED';reason='Startup incomplete'
 try:
  assert read(OUT/'readiness.json')['status']=='PASS' and read(OUT/'runner-regression.json')['returncode']==0
  assert read(OUT/'evaluation-initial-completion.json')['attempts']==512 and read(OUT/'evaluation-initial-completion.json')['failures']==0
  cuda_startup(torch,lambda data:write(OUT/'gpu-runtime.json',data))
  if str(torch.__version__)!='2.11.0+cu128' or torch.version.cuda!='12.8' or sys.version_info[:2]!=(3,13):raise RuntimeError('Pinned GPU environment differs; no upgrade or replacement')
  ctx=runtime_registration();runner=Runner(ctx,OUT/'runtime-registration.json',OUT/'runtime-authorization.json',device='cuda',runtime_factory=AdaptationRuntime)
  print('EMBEDDING_STARTUP_PASS '+json.dumps(dict(initial_tensor=INITIAL_TENSOR,used_GPU=1,allocated_GPU=2,compiled=False)),flush=True)
  runner.run(resume=False);status=runner.state;reason=runner.execution.get('registered_stop','cohort_cap' if runner.phase==MAX_COHORTS else 'resource_deadline')
 except BaseException as error:
  reason=type(error).__name__+': '+str(error);write(OUT/'training-failure.json',dict(at_utc=iso(),error=reason,traceback=traceback.format_exc(),no_replacement=True));raise
 finally:
  summary=dict(at_utc=iso(),state=status,reason=reason,session_seconds_cap=SESSION_CAP,process_elapsed_seconds=time.monotonic()-start,CPU_core_hours=(time.process_time()-cpu)/3600,paid_model_API_requests=0,no_replacement=True)
  if runner and runner.model is not None:
   try:
    if runner.rt.scalar.digest(frozen_state(runner.model))!=runner.rt.frozen_digest:raise RuntimeError('Final frozen actor/buffer hash changed')
    if runner.anchor is not None:runner.rt.policy_diagnostics(runner)
    runner.rt.export(runner.model,runner.execution,PRIVATE/'final.pt');canonical=runner.rt.state_hash_from_canonical(runner.model)
    actor_changed=actor_hash(runner.model,runner.rt.scalar.digest)!=runner.rt.actor_initial_digest
    summary.update(final_file_sha256=sha(PRIVATE/'final.pt'),final_tensor_hash=canonical,actor_changed=actor_changed,fresh_decisions=runner.execution['additional_collected'],accepted_updates=runner.execution['step'],accepted_actor_updates=runner.execution.get('accepted_actor_updates',0),accepted_exposures=runner.execution['accepted_used_decisions'],effective_PPO_passes=runner.execution['accepted_used_decisions']/max(1,runner.execution['additional_collected']),cohorts=runner.phase,execution=runner.execution,evaluate=actor_changed and status in ('COMPLETE','INCOMPLETE'),training_attempts=len(list((OUT/'native-intents').glob('*.json'))),peak_allocated_VRAM_bytes=torch.cuda.max_memory_allocated(),peak_reserved_VRAM_bytes=torch.cuda.max_memory_reserved())
    summary['skip_final_evaluation']=None if summary['evaluate'] else 'No retained actor change' if not actor_changed else 'Training execution failed; final candidate not eligible'
   except BaseException as error:summary.update(state='FAILED',export_error=type(error).__name__+': '+str(error))
   finally:runner.rt.close()
  write(OUT/'training-completion.json',summary)

if __name__=='__main__':run()
