"""Pilot-specific hooks around the validated persistent runner and bucket64."""
from .common import *
from unown_belief_correction.integration import PilotRuntime
from unown_belief_correction.correction import learning_rate,frozen_digest,configure
from unown_runner.runtime import Runtime
from unown_runner.runner import Runner
import functools,numpy as np,torch

class TrainingRuntime(PilotRuntime):
 def __init__(self,ctx,arm=None):
  super().__init__(ctx,arm or ctx.run_id.rsplit('-',1)[1]);self.fixed_frozen=None;self.update_started=None
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
  fresh_before=runner.execution['additional_collected']-len(runner.active['rows']);lr=learning_rate(runner.opt,fresh_before);self.epochs.LR=lr;self.update_started=self.clock()
  runner.execution['current_actor_lr']=lr;runner.execution['current_value_head_lr']=1e-5
 def cohort_finished(self,runner,terminal):
  if frozen_digest(runner.model)!=self.fixed_frozen:raise RuntimeError('Frozen actor changed')
  update_seconds=self.clock()-self.update_started;rows=runner.active['rows'];indices=np.linspace(0,len(rows)-1,min(128,len(rows)),dtype=int);sample=[rows[i] for i in indices]
  metrics=self.policy_diagnostics(runner.model,runner.anchor,sample);exposures=runner.execution['cohort_exposures'];runner.execution['last_complete_cycle_seconds']=runner.execution.get('last_collection_seconds',0)+update_seconds
  record=dict(at_utc=iso(),arm=self.arm,phase=runner.phase,fresh=runner.execution['additional_collected'],cohort_decisions=len(rows),accepted_updates=runner.execution['step'],effective_passes=exposures/len(rows),update_seconds=update_seconds,update_terminal=terminal,actor_lr=runner.execution['current_actor_lr'],diagnostics=metrics)
  append(self.context.out/'cohort-diagnostics.jsonl',record);progress('training',**record)
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
 def __init__(self,ctx,reg,auth,device='cuda'):
  super().__init__(ctx,reg,auth,device=device,runtime_factory=TrainingRuntime)
  self.collector=NativeCollector(self.rt)
 def train_active(self):
  # New branches and detached value head are the only parameters requiring gradients.
  return super().train_active()
