"""Disposable saved-optimizer, cursor and rejection checks; no native games."""
import copy,datetime as dt,shutil,types
import pytest,numpy as np
from unown_embedding_continuation.common import *
from unown_embedding_continuation.recovery import *

class NoNativeCollector:
 software_only=True
 def collect(self,*a,**kw):raise AssertionError('No native collection in software checks')
 def close(self):pass

@pytest.fixture(scope='module')
def pinned():
 assets=context().assets;files={}
 for name in assets.values():
  p=Path(name)
  for f in ([p] if p.is_file() else (v for v in p.rglob('*') if v.is_file() and v.suffix!='.pyc' and '__pycache__' not in v.parts)):files[str(f.resolve())]=sha(f)
 return assets,files

def make_runner(tmp_path,pinned):
 assets,inputs=pinned
 ctx=RunContext(tmp_path,'software-continuation',assets,Execution('bucket64'),SEED,dict(FIXED_RECIPE,lr=1e-4),dict(max_cohorts=255,native_attempts=0,session_seconds=1800,export_reserve_seconds=60,storage_bytes=2*1024**3,log_bytes=256*1024**2),'software_test')
 ctx.out.mkdir(parents=True);ctx.private.mkdir(parents=True)
 for name in ('trainable-rows.json','parameter-verification.json'):shutil.copy2(OUT/name,ctx.out/name)
 reg=tmp_path/'reg.json';auth=tmp_path/'auth.json'
 write(reg,dict(context=ctx.spec(),device='cpu',inputs=inputs,implementation=implementation_hashes(),cohorts=read(OUT/'training-schedule.json'),decks=read(OUT/'decks.json')))
 write(auth,dict(run_id=ctx.run_id,registration_sha256=sha(reg),mode='software_test',scope='local_software_verification_no_native',approved=True,expires_utc=(now()+dt.timedelta(hours=1)).isoformat()))
 return ContinuationRunner(ctx,reg,auth,device='cpu',collector=NoNativeCollector(),runtime_factory=ContinuationRuntime)

def test_saved_optimizer_RNG_and_pending_cursor_adopt_exactly(tmp_path,pinned):
 runner=make_runner(tmp_path,pinned)
 try:
  adopt_recovery(runner)
  saved=torch.load(PRIVATE/'parent-checkpoint.pt',map_location='cpu',weights_only=False)
  assert runner.rt.scalar.digest(runner.opt.state_dict())==runner.rt.scalar.digest(saved['optimizer'])
  assert len(runner.opt.state)==10 and runner.execution['step']==14132 and runner.phase==174
  assert runner.execution['next_minibatch']==0 and runner.execution['additional_collected']==444018
  expected=runner.rt.scalar.digest(dict(model=runner.model.state_dict(),optimizer=runner.opt.state_dict(),rng=torch.get_rng_state()))
  runner.restore()
  assert runner.rt.scalar.digest(dict(model=runner.model.state_dict(),optimizer=runner.opt.state_dict(),rng=torch.get_rng_state()))==expected
  assert runner.active['immutable_digest']==runner.reference['immutable_digest']
  write(OUT/'continuation-verification.json',dict(status='PASS',saved_optimizer_exact=True,saved_model_exact=True,CPU_RNG_exact=True,CUDA_RNG_retained_in_parent=True,pending_cursor_exact=True,new_manifest_lineage_explicit=True,native_attempts=0,GPU_seconds=0))
 finally:runner.rt.close()

def historical_proof():return read(ROOT/'experiments/astra-unown-new-card-embedding-adaptation-retry-001/phase-000174-compatibility.json')

def test_known_numerical_failure_rejects_without_updating_and_survives_resume(tmp_path,pinned):
 runner=make_runner(tmp_path,pinned)
 try:
  adopt_recovery(runner);proof=historical_proof();assert numerical_only_rejection(proof)
  original=runner.rt.checks.collected
  def reject(*args,**kw):
   write(runner.context.out/'phase-000174-compatibility.json',proof)
   raise AssertionError('Software replay of retained probability-only failure')
  runner.rt.checks.collected=reject
  before=runner.rt.scalar.digest(dict(model=runner.model.state_dict(),optimizer=runner.opt.state_dict(),rng=torch.get_rng_state()))
  runner.validate_cohort();assert runner.numerical_rejection_phase==174
  assert runner.train_active()=='numerical_compatibility_cohort_rejected'
  runner.state='UPDATING';runner.commit();runner.restore()
  assert runner.train_active()=='numerical_compatibility_cohort_rejected'
  after=runner.rt.scalar.digest(dict(model=runner.model.state_dict(),optimizer=runner.opt.state_dict(),rng=torch.get_rng_state()))
  assert before==after and runner.execution['step']==14132
 finally:runner.rt.close()

def test_structural_or_identity_failures_are_never_softened():
 proof=historical_proof();assert numerical_only_rejection(proof)
 for key in ('identity','masks','actions','KL','values','TV'):
  changed=copy.deepcopy(proof);changed['gates'][key]=False;assert not numerical_only_rejection(changed)
 changed=copy.deepcopy(proof);changed['precision_envelope']['checks'][0]['double_path_difference']=1e-5;assert not numerical_only_rejection(changed)

def test_KL_actor_zero_streak_does_not_stop_whole_run(monkeypatch):
 monkeypatch.setattr(AdaptationRuntime,'cohort_finished',lambda *a:'three_zero_actor_cohorts')
 rt=ContinuationRuntime.__new__(ContinuationRuntime);runner=types.SimpleNamespace(phase=174,execution=dict(consecutive_zero_actor_cohorts=3,additional_collected=444018,step=14132),numerical_rejection_phase=None)
 for reason in ('phase_preupdate_drift_stop','phase_postupdate_drift_rollback_stop'):
  assert rt.cohort_finished(runner,reason) is None
 for i in range(3):result=rt.cohort_finished(runner,'numerical_compatibility_cohort_rejected')
 assert result=='three_numerically_rejected_cohorts'

@pytest.mark.parametrize('kind',['pre','post'])
def test_KL_rejected_transaction_preserves_optimizer_and_model(tmp_path,pinned,monkeypatch,kind):
 runner=make_runner(tmp_path,pinned)
 try:
  rt=runner.rt;model,opt,_=rt.restore(PRIVATE/'parent-checkpoint.pt','cpu')
  rows=copy.deepcopy(torch.load(PRIVATE/'new-probes.pt',map_location='cpu',weights_only=False)[:17])
  service=rt.revised.Revised(dict(learner=model),dict(learner=rt.state_hash(model)))
  for row in rows:
   response=service.predict('learner',[row['packet']])[0]
   row.update(old_probs=response['probs'],log_probability=float(response['log_probs'][row['action']]),value=response['value'])
  cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));engine=rt.make_engine(model,opt,cohort,0);execution=dict(step=14132)
  method='gradients' if kind=='pre' else 'diagnose';original=getattr(engine,method)
  def force(*a,**kw):
   result=original(*a,**kw)
   if kind=='pre':result[1]['phase_KL']=.021
   else:result['phase_KL']=.031
   return result
  monkeypatch.setattr(engine,method,force)
  before=rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,execution))
  result=engine.transaction(model,opt,rows,cohort['advantages'],execution,'software-KL-'+kind)
  assert not result['accepted'] and execution['step']==14132
  assert before==rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,execution))
  engine.close()
 finally:runner.rt.close()
