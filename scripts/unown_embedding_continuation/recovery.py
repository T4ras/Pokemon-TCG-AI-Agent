"""Explicit recovery fork and cohort-local rejection; strict checks stay intact."""
import copy,fcntl,shutil
from .common import *
from .runtime import AdaptationRuntime
from unown_runner.runner import Runner,RecoveryConflict

def numerical_only_rejection(proof):
 gates=proof.get('gates',{})
 envelope=proof.get('precision_envelope') or {}
 checks=envelope.get('checks',[])
 return (proof.get('accepted') is False and proof.get('violated_original_criteria')==['probability']
  and gates.get('probability') is False and all(gates.get(k) is True for k in ('TV','KL','values','masks','actions','identity'))
  and 0<len(checks)<=8 and all(math.isfinite(c.get('double_path_difference',math.inf)) and c['double_path_difference']<=1e-10 for c in checks))

class ContinuationRuntime(AdaptationRuntime):
 def save_extra(self,runner):
  return dict(super().save_extra(runner),numerical_rejection_phase=getattr(runner,'numerical_rejection_phase',None))
 def cohort_finished(self,runner,terminal):
  result=super().cohort_finished(runner,terminal)
  print('EMBEDDING_CONTINUATION_PROGRESS '+json.dumps(dict(next_phase=runner.phase+1,fresh=runner.execution['additional_collected'],accepted_updates=runner.execution['step'],cohort_stop=terminal)),flush=True)
  rejected=terminal=='numerical_compatibility_cohort_rejected'
  runner.execution['consecutive_numerical_rejections']=(runner.execution.get('consecutive_numerical_rejections',0)+1) if rejected else 0
  runner.numerical_rejection_phase=None
  if result=='fresh_decision_target':return result
  if runner.execution['consecutive_numerical_rejections']>=3:return 'three_numerically_rejected_cohorts'
  # Original per-cohort KL limits and rollback remain. Three actor-zero cohorts
  # must not turn a local KL stop into a global stop in this continuation.
  return None

class ContinuationRunner(Runner):
 def restore(self):
  super().restore()
  self.numerical_rejection_phase=torch.load(self.latest,map_location='cpu',weights_only=False)['extra'].get('numerical_rejection_phase')
 def validate_cohort(self):
  self.numerical_rejection_phase=None
  before=self.rt.batch.snapshot(self.model,self.opt,self.execution)
  try:return super().validate_cohort()
  except AssertionError:
   path=self.context.out/f'phase-{self.phase:06d}-compatibility.json'
   proof=read(path) if path.exists() else {}
   if not numerical_only_rejection(proof):raise
   expected=self.reference['sha256']
   if proof.get('rollout',{}).get('sha256')!=expected:raise RecoveryConflict('Numerical rejection does not identify current immutable cohort')
   after=self.rt.batch.snapshot(self.model,self.opt,self.execution)
   if self.rt.scalar.state_fingerprints(before)!=self.rt.scalar.state_fingerprints(after):raise RecoveryConflict('Rejected validation changed model, optimizer, execution or RNG')
   self.numerical_rejection_phase=self.phase
   self.event('numerical-cohort-rejected',phase=self.phase,cohort_sha256=expected,thresholds_unchanged=True,model_optimizer_RNG_unchanged=True,metrics=proof['metrics'])
 def train_active(self):
  if getattr(self,'numerical_rejection_phase',None)==self.phase:
   self.execution['cohort_terminal']='numerical_compatibility_cohort_rejected'
   return self.execution['cohort_terminal']
  return super().train_active()

def adopt_recovery(runner,checkpoint=None,cohort=None):
 """New manifest/journal lineage; exact old learning state, never reset momentum."""
 checkpoint=checkpoint or PRIVATE/'parent-checkpoint.pt';cohort=cohort or PRIVATE/'parent-cohort.pt'
 lineage=read(OUT/'lineage.json')
 assert sha(checkpoint)==lineage['parent_checkpoint_sha256'] and sha(cohort)==lineage['parent_cohort_sha256']
 if runner.latest.exists() or runner.journal.exists():raise RecoveryConflict('Recovery fork already exists; do not duplicate adoption')
 runner.acquire()
 try:
  runner.manifest();runner.model,runner.opt,saved=runner.rt.restore(checkpoint,runner.device);extra=saved['extra']
  assert extra['runner_phase']==174 and extra['runner_state']=='VALIDATING'
  assert extra['execution']['next_minibatch']==0 and extra['execution']['epoch']==0
  assert saved['step']==14132
  assert runner.rt.state_hash_from_canonical(runner.model)==lineage['parent_verified']['canonical_tensor_hash']
  assert runner.rt.scalar.digest(runner.opt.state_dict())==runner.rt.scalar.digest(saved['optimizer'])
  runner.execution=copy.deepcopy(extra['execution']);runner.phase=174;runner.state='VALIDATING';runner.reference=copy.deepcopy(extra['cohort_reference'])
  runner.execution.pop('cohort_update_clock_start',None) # process-local timer only
  target=runner.context.private/runner.reference['path'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():raise RecoveryConflict('Pending cohort already adopted')
  shutil.copy2(cohort,target);runner.active=runner.read_cohort(runner.reference)
  assert runner.active['collection_policy_hash']==runner.rt.state_hash(runner.model)
  runner.numerical_rejection_phase=None
  runner.event('recovery-fork',parent_checkpoint_sha256=sha(checkpoint),parent_step=14132,parent_phase=174,parent_cohort_sha256=sha(cohort),model_optimizer_RNG_preserved=True)
  runner.commit()
  migrated=torch.load(runner.latest,map_location='cpu',weights_only=False)
  assert runner.rt.scalar.digest(migrated['state_dict'])==runner.rt.scalar.digest(saved['state_dict'])
  assert runner.rt.scalar.digest(migrated['optimizer'])==runner.rt.scalar.digest(saved['optimizer'])
  for key in ('torch_rng','numpy_rng','python_rng'):
   assert runner.rt.scalar.digest(migrated[key])==runner.rt.scalar.digest(saved[key])
  if str(runner.device).startswith('cuda'):
   assert runner.rt.scalar.digest(migrated['CUDA_rng'])==runner.rt.scalar.digest(saved['CUDA_rng'])
  write(runner.context.out/'recovery-adoption.json',dict(at_utc=iso(),status='PASS',parent_step=14132,phase=174,model_exact=True,optimizer_exact=True,pending_cohort_exact=True,new_manifest_and_journal_explicit=True))
 finally:
  if runner.lock:fcntl.flock(runner.lock,fcntl.LOCK_UN);runner.lock.close();runner.lock=None
