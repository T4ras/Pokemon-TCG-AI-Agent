"""Pilot-specific hooks around the validated persistent runner and bucket64."""
from .common import *
from unown_belief_correction.integration import PilotRuntime
from unown_belief_correction.correction import learning_rate,frozen_digest,configure
from unown_runner.runtime import Runtime
from unown_runner.runner import Runner
import functools,numpy as np,torch

class TrainingRuntime(PilotRuntime):
 def __init__(self,ctx,arm=None):
  from .math_cache import install
  from .deadline_hooks import install as deadline_install
  deadline_install();self.fixed_cache=install();super().__init__(ctx,arm or ctx.run_id.rsplit('-',1)[1]);self.fixed_frozen=None;self.update_started=None
  played=self.corrected.play
  def checked_play(*args,**kwargs):
   result,rows=played(*args,**kwargs);assert self.fixed_cache.digest()==self.fixed_cache.initial;result['fixed_cache_digest']=self.fixed_cache.initial;return result,rows
  self.corrected.play=checked_play
  reconstructed=self.revised.Recorded.reconstruct
  def durable_reconstruct(service,reduced):
   rows=reconstructed(service,reduced)
   if rows:
    begin=self.clock();path=self.persist_completed_match(rows,service.versions);service.metrics['durable_rollout_serialization_seconds']+=self.clock()-begin;service.metrics['durable_rollout_bytes']+=path.stat().st_size
   return rows
  self.revised.Recorded.reconstruct=durable_reconstruct
 def persist_completed_match(self,rows,versions):
  identities=[tuple(r['request_id']) for r in rows];job=identities[0][0]
  if not isinstance(job,str) or not job or not all(c.isalnum() or c in '-_' for c in job):raise ValueError('Invalid completed-match identity')
  if any(i[0]!=job or i[1]!='learner' for i in identities):raise ValueError('Mixed completed-match rollout')
  folder=self.context.private/'pending-match-rollouts';folder.mkdir(parents=True,exist_ok=True);target=folder/(job+'.pt')
  if target.exists():raise RuntimeError('Duplicate completed-match rollout persistence')
  data=dict(rows=rows,versions=dict(versions),match_id=job,row_digest=self.epochs.immutable_digest(rows,np.empty(0,np.float32)),software_verification_only=self.context.mode=='software_test',candidate_eligible=False,normalization='Complete cohort normalizes advantages once; no normalized per-match advantages are reused')
  temp=target.with_suffix('.tmp')
  with temp.open('xb') as stream:torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
  temp.replace(target);return target
 def release_match_duplicates(self):
  folder=(self.context.private/'pending-match-rollouts').resolve()
  for p in folder.glob('*.pt'):
   if p.resolve().parent!=folder:raise RuntimeError('Invalid duplicate rollout path')
   p.unlink()
 def load_base(self,device='cpu'):
  m=Runtime.load_model(self,self.context.assets['base_policy_checkpoint'],device)
  if self.state_hash(m)!=POLICY_TENSOR:raise ValueError('Wrong current policy')
  for p in m.parameters():p.requires_grad_(False)
  return m
 def restore(self,path,device='cpu',allow_disposable=False):
  m,o,s=super().restore(path,device,allow_disposable)
  self.fixed_frozen=read(OUT/'initialization.json')['frozen_tensor_digest']
  if frozen_digest(m)!=self.fixed_frozen:raise ValueError('Frozen tensor identity mismatch')
  return m,o,s
 def save_extra(self,runner):
  return dict(pilot_arm=self.arm,frozen_policy_digest=self.fixed_frozen,belief_checkpoint_sha256=read(OUT/'registration.json')['belief_checkpoint_sha256'],scientific_registration_sha256=sha(OUT/'registration.json'))
 def may_start_cohort(self,runner):
  if runner.execution['additional_collected']>=TARGET:return False
  last=runner.execution.get('last_complete_cycle_seconds',0)
  return self.clock()+max(600,1.5*last+60)<runner.deadline
 def begin_cohort_update(self,runner):
  runner.execution['phase']=9+runner.phase
  fresh_before=runner.execution['additional_collected']-len(runner.active['rows']);lr=learning_rate(runner.opt,fresh_before);self.epochs.LR=lr;self.update_started=self.clock()
  runner.execution['current_actor_lr']=lr;runner.execution['current_value_head_lr']=1e-5
 def cohort_finished(self,runner,terminal):
  if frozen_digest(runner.model)!=self.fixed_frozen:raise RuntimeError('Frozen actor changed')
  update_seconds=self.clock()-self.update_started;rows=runner.active['rows'];indices=np.linspace(0,len(rows)-1,min(128,len(rows)),dtype=int);sample=[rows[i] for i in indices]
  metrics=self.policy_diagnostics(runner.model,runner.anchor,sample);exposures=runner.execution['cohort_exposures'];runner.execution['last_complete_cycle_seconds']=runner.execution.get('last_collection_seconds',0)+update_seconds
  record=dict(at_utc=iso(),arm=self.arm,phase=runner.phase,fresh=runner.execution['additional_collected'],cohort_decisions=len(rows),accepted_updates=runner.execution['step'],effective_passes=exposures/len(rows),update_seconds=update_seconds,update_terminal=terminal,actor_lr=runner.execution['current_actor_lr'],diagnostics=metrics)
  append(self.context.out/'cohort-diagnostics.jsonl',record);progress('training',**record)
  self.release_match_duplicates()
  if runner.execution['additional_collected']>=TARGET:return 'fresh_decision_target'
 @torch.no_grad()
 def policy_diagnostics(self,model,base,rows):
  sums=collections.Counter();values=[];returns=[];device=next(model.parameters()).device
  self.runtime.freeze_inference(model);self.runtime.freeze_inference(base)
  for _,group in self.batch.groups(rows):
   p=self.prefix.DevicePacket(group,device);s,_=model.core(p);b,_=base.core(p);lp=s['logits'].log_softmax(-1);blp=b['logits'].log_softmax(-1);prob=lp.exp();bp=blp.exp();mask=s['pilot_mask'];meaningful=mask.sum(-1)>1
   sums['base_to_candidate_KL']+=float((bp*(blp-lp)).sum());sums['candidate_to_base_KL']+=float((prob*(lp-blp)).sum());sums['greedy_disagreements']+=int(((lp.argmax(-1)!=blp.argmax(-1))&meaningful).sum());sums['meaningful']+=int(meaningful.sum());sums['delta_abs']+=float(s['pilot_delta'][mask].abs().sum());sums['legal_candidates']+=int(mask.sum());sums['delta_abs_max']=max(sums['delta_abs_max'],float(s['pilot_delta'][mask].abs().max()));values.extend(s['pilot_value'].cpu().tolist());returns.extend(x['return_target'] for x in group)
  variance=np.var(returns);return dict(sample_rows=len(rows),sample_rule='At most128 evenly spaced collection rows per cohort; off-policy after update',base_to_candidate_KL=sums['base_to_candidate_KL']/len(rows),candidate_to_base_KL=sums['candidate_to_base_KL']/len(rows),meaningful_greedy_disagreement=sums['greedy_disagreements']/max(1,sums['meaningful']),mean_absolute_delta=sums['delta_abs']/sums['legal_candidates'],max_absolute_delta=sums['delta_abs_max'],critic_explained_variance=float(1-np.var(np.asarray(returns)-values)/variance) if variance>0 else None,frozen_identity=True)

class NativeCollector:
 software_only=False
 def __init__(self,rt):self.rt=rt;self.pool=None
 def collect(self,jobs,models,deadline,safety_check):
  if len(list((self.rt.context.out/'native-intents').glob('*.json')))+len(jobs)>read(OUT/'registration.json')['recovery_max_new_native_attempts']:raise RuntimeError('Original native-attempt ceiling; no extra matches')
  if self.pool is None:self.pool=self.rt.pool.ConcurrentCollector(16,worker_target=functools.partial(environment_worker,self.rt.context.spec(),self.rt.arm),manifest=read(self.rt.context.out/'manifest.json'))
  rows,matches,summary=self.pool.collect(jobs,models,concurrency=16,deadline=deadline,safety_check=safety_check)
  append(self.rt.context.out/'collection-timings.jsonl',dict(at_utc=iso(),phase=jobs[0]['cohort'],summary=summary,belief_costs=dict(sum((collections.Counter(m.get('belief_costs',{})) for m in matches),collections.Counter()))))
  return rows,matches,summary
 def close(self):
  if self.pool:self.pool.close()

def environment_worker(spec,arm,connection,index,manifest):
 os.environ['CUDA_VISIBLE_DEVICES']='';rt=TrainingRuntime(RunContext.from_spec(spec),arm)
 try:rt.worker.run_worker(connection,index,manifest)
 finally:rt.close()

class PilotRunner(Runner):
 def seed_boundary(self):
  """Adopt a verified completed boundary under a new manifest; no learning here."""
  if self.latest.exists() or self.journal.exists():raise RuntimeError('Continuation already seeded; explicit recovery required')
  self.rt.configure_precision();self.manifest()
  saved=torch.load(self.context.assets['start_checkpoint'],map_location='cpu',weights_only=False)
  old=saved['extra'];init=read(OUT/'initialization.json')
  if sha(self.context.assets['start_checkpoint'])!=init['source_boundary_sha256']:raise ValueError('Wrong saved boundary')
  if old['runner_state']!='VALIDATING' or old['cohort_reference'] is None or old['runner_phase']!=9 or saved['step']!=1052 or old['execution']['cohort_exposures']!=0:raise ValueError('Unsafe pending-cohort cursor')
  source=Path(self.context.assets['pending_cohort']);cohort=torch.load(source,map_location='cpu',weights_only=False);self.rt.epochs.verify_cohort(cohort)
  if sha(source)!=old['cohort_reference']['sha256'] or cohort['collection_policy_hash']!=init['tensor_sha256']:raise ValueError('Pending rollout identity changed')
  folder=self.context.private/'cohorts';folder.mkdir(parents=True,exist_ok=True);target=folder/'000000.pt'
  if target.exists():raise RuntimeError('Pending cohort already transplanted')
  shutil.copy2(source,target)
  saved=copy.deepcopy(saved);saved['extra'].update(runner_phase=0,runner_state='VALIDATING',run_manifest_sha256=self.manifest_sha,journal_bytes=0,journal_sha256=hashlib.sha256(b'').hexdigest(),journal_seq=0,source_boundary_sha256=init['source_boundary_sha256'])
  saved['extra']['cohort_reference']=dict(old['cohort_reference'],path='cohorts/000000.pt')
  # Preserve phase9 for the pending shuffle; begin_cohort_update advances with9+local phase.
  self.latest.parent.mkdir(parents=True,exist_ok=True)
  with self.latest.open('xb') as stream:torch.save(saved,stream);stream.flush();os.fsync(stream.fileno())
 def run(self,resume=False):
  if not resume:self.seed_boundary()
  return super().run(True)

 def __init__(self,ctx,reg,auth,device='cuda'):
  super().__init__(ctx,reg,auth,device=device,runtime_factory=TrainingRuntime)
  self.collector=NativeCollector(self.rt)
 def train_active(self):
  # New branches and detached value head are the only parameters requiring gradients.
  return super().train_active()
