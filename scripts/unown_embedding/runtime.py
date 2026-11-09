"""Experiment-specific hooks over the reusable eager bucket64 runner."""
import copy,gc,os,time
from pathlib import Path
import numpy as np
import torch
from unown_runner.runtime import Runtime
from unown_runner.context import read,write,append,iso,sha
from .common import INITIAL_TENSOR,embedding_lr,TARGET
from .model import install,optimizer,canonical_state,actor_parameters,actor_hash,frozen_state,gradient_norm,bind_detached_value,inventory

class AdaptationRuntime(Runtime):
 def __init__(self,context):
  super().__init__(context);self.scope=read(context.out/'trainable-rows.json')['rows'];self.frozen_digest=None;self.actor_initial_digest=None
  self.batch.optimizer=lambda model:optimizer(model,context.recipe['lr'])
  bind_detached_value(self)
  # Keep all rejection metrics; bound detailed before/rejected tensor pairs.
  runtime_self=self
  class DiagnosticTorch:
   def __getattr__(self,name):return getattr(torch,name)
   def save(self,value,path):
    path=runtime_self.context.private/Path(path).name;signature='numerical' if 'numerical' in path.name else 'KL-rollback'
    ledger=runtime_self.context.out/'diagnostic-snapshots.jsonl';saved=[r for r in __import__('unown_runner.context',fromlist=['records']).records(ledger) if r['signature']==signature and r.get('saved')]
    allocated=sum(p.stat().st_size for p in runtime_self.context.private.glob('*-before.pt'))+sum(p.stat().st_size for p in runtime_self.context.private.glob('*-rejected.pt'))
    keep=len(saved)<20 and allocated+80*1024**2<=2*1024**3
    if keep:torch.save(value,path)
    append(ledger,dict(at_utc=iso(),path=str(path.relative_to(runtime_self.context.root)),signature=signature,saved=keep,dropped_detail=not keep,bytes=path.stat().st_size if keep else 0))
  self.lean.torch=DiagnosticTorch()
 def load_model(self,path,device):
  if path is not None:
   raw=torch.load(path,map_location='cpu',weights_only=False)
   if not raw.get('extra',{}).get('embedding_adaptation'):
    return super().load_model(path,device)
  model=super().load_model(self.context.assets['start_checkpoint'],device)
  if self.state_hash(model)!=INITIAL_TENSOR:raise ValueError('Wrong experimental initialization; later PPO substitution forbidden')
  install(model,self.scope)
  model._initial_actor_digest=actor_hash(model,self.scalar.digest)
  if path is not None:model.load_state_dict(raw['state_dict'],strict=True)
  return model
 def restore(self,path,device='cpu'):
  saved=torch.load(path,map_location=device,weights_only=False)
  if saved.get('extra',{}).get('software_verification_only') and self.context.mode=='learning':raise PermissionError('Probe checkpoint is ineligible')
  model=self.load_model(path if saved.get('extra',{}).get('embedding_adaptation') else None,device)
  opt=optimizer(model,self.context.recipe['lr'])
  if saved.get('extra',{}).get('embedding_adaptation'):
   opt.load_state_dict(copy.deepcopy(saved['optimizer']));self.restore_rng(saved,device)
  else:
   if self.state_hash_from_canonical(model)!=INITIAL_TENSOR:raise ValueError('Initialization effective tensor mismatch')
   np.random.seed(self.context.seed);torch.manual_seed(self.context.seed)
   import random;random.seed(self.context.seed)
   if torch.device(device).type=='cuda':torch.cuda.manual_seed_all(self.context.seed)
   saved=dict(saved,step=0)
  self.frozen_digest=self.scalar.digest(frozen_state(model));self.actor_initial_digest=model._initial_actor_digest
  expected=read(self.context.out/'parameter-verification.json').get('frozen_digest') if (self.context.out/'parameter-verification.json').exists() else None
  if expected and expected!=self.frozen_digest:raise ValueError('Registered frozen parameters or buffers changed')
  if saved.get('extra',{}).get('embedding_adaptation') and saved['extra']['frozen_digest']!=self.frozen_digest:raise ValueError('Recovery frozen digest mismatch')
  return model,opt,saved
 def state_hash_from_canonical(self,model):
  import hashlib
  h=hashlib.sha256()
  for k,v in sorted(canonical_state(model).items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
  return h.hexdigest()
 def make_engine(self,model,opt,cohort,phase):
  engine=super().make_engine(model,opt,cohort,phase);original=engine.gradients
  def gradients(*args):
   metrics,pre=original(*args)
   metrics['actor_gradient_norm']=gradient_norm(actor_parameters(model));metrics['value_gradient_norm']=gradient_norm(model.value_mlp.parameters())
   if any(p.grad is not None for name,p in model.named_parameters() if not p.requires_grad):raise RuntimeError('Frozen parameter received gradient')
   return metrics,pre
  engine.gradients=gradients;transaction=engine.transaction
  def checked_transaction(model,opt,rows,adv,execution,label):
   before=actor_hash(model,self.scalar.digest);result=transaction(model,opt,rows,adv,execution,label)
   actor_updated=result['accepted'] and actor_hash(model,self.scalar.digest)!=before
   result['accepted_actor_update']=actor_updated
   if actor_updated:execution['accepted_actor_updates']=execution.get('accepted_actor_updates',0)+1
   return result
  engine.transaction=checked_transaction;return engine
 def begin_cohort_update(self,runner):
  runner.execution.setdefault('cohort_update_clock_start',self.clock())
  lr=embedding_lr(runner.execution['additional_collected'])
  for group in runner.opt.param_groups:group['lr']=lr if group['name']=='eligible_identity_and_text_corrections' else 1e-5
  self.epochs.LR=lr
  append(self.context.out/'learning-rates.jsonl',dict(at_utc=iso(),phase=runner.phase,fresh=runner.execution['additional_collected'],embedding_lr=lr,value_lr=1e-5,index='fresh collected at cohort end; held constant through all four epochs'))
  runner.execution.setdefault('actor_before_cohort',actor_hash(runner.model,self.scalar.digest))
  runner.execution.setdefault('actor_steps_before_cohort',runner.execution.get('accepted_actor_updates',0))
 def save_extra(self,runner):return dict(embedding_adaptation=True,initial_tensor_hash=INITIAL_TENSOR,trainable_rows_SHA256=sha(self.context.out/'trainable-rows.json'),frozen_digest=self.frozen_digest)
 def may_start_cohort(self,runner):
  measured=runner.execution.get('last_collection_seconds',0)+runner.execution.get('last_update_seconds',0)
  reserve=max(600,1.5*measured+60)
  if runner.deadline-self.clock()<reserve:
   runner.execution['registered_stop']='resource_deadline_before_new_cohort';return False
  return True
 def cohort_finished(self,runner,terminal):
  runner.execution['last_update_seconds']=self.clock()-runner.execution.pop('cohort_update_clock_start')
  current=self.scalar.digest(frozen_state(runner.model))
  if current!=self.frozen_digest:raise RuntimeError('Frozen actor/buffer bytes changed')
  actor=actor_hash(runner.model,self.scalar.digest);changed=actor!=runner.execution.pop('actor_before_cohort');steps=runner.execution.get('accepted_actor_updates',0)-runner.execution.pop('actor_steps_before_cohort')
  runner.execution['consecutive_zero_actor_cohorts']=0 if steps else runner.execution.get('consecutive_zero_actor_cohorts',0)+1
  append(self.context.out/'adaptation-cohorts.jsonl',dict(at_utc=iso(),phase=runner.phase,fresh=runner.execution['additional_collected'],actor_changed=changed,accepted_actor_updates=steps,actor_hash=actor,frozen_digest=current,zero_actor_streak=runner.execution['consecutive_zero_actor_cohorts'],terminal=terminal,embedding_lr=runner.opt.param_groups[0]['lr'],collection_seconds=runner.execution.get('last_collection_seconds'),update_seconds=runner.execution['last_update_seconds'],component_timers=dict(self.lean.METERS)))
  self.policy_diagnostics(runner)
  if runner.phase==1:print('EMBEDDING_FIRST_TWO_COHORTS_PASS '+__import__('json').dumps(dict(native_matches=32,fresh=runner.execution['additional_collected'],frozen_exact=True,old_only_exact=True,current_committed_policy_used=True)),flush=True)
  self.export(runner.model,runner.execution,self.context.private/'final.pt')
  if runner.execution['additional_collected']>=TARGET:return 'fresh_decision_target'
  if runner.execution['consecutive_zero_actor_cohorts']>=3:return 'three_zero_actor_cohorts'
  return None
 def policy_diagnostics(self,runner):
  from .model import CHANNELS
  from .common import PRIVATE
  result=dict(at_utc=iso(),phase=runner.phase,label='Fixed retained off-policy diagnostics; not strength evidence',channels={})
  for kind,(identity,text) in CHANNELS.items():
   a=getattr(runner.model,identity).parametrizations.weight[0].delta.detach();b=runner.model.parametrizations[text][0].delta.detach()
   result['channels'][kind]=dict(identity_delta_L2=float(a.double().norm()),text_delta_L2=float(b.double().norm()),identity_max_abs=float(a.abs().max()),text_max_abs=float(b.abs().max()))
  for label in ('old','new'):
   rows=torch.load(self.context.private/(label+'-probes.pt'),map_location='cpu',weights_only=False)[:64]
   cohort=self.epochs.new_cohort(rows,'off-policy-diagnostic');outputs=[]
   for model in (runner.anchor,runner.model):
    engine=self.engine.Engine(model,cohort,'bucket64',phase=0)
    try:outputs.append(engine.outputs(rows))
    finally:engine.close()
   kl=[];tv=[];disagree=0;exact=True
   for i in range(len(rows)):
    a,b=outputs[0][i][0],outputs[1][i][0];p=a.softmax(-1);q=b.softmax(-1)
    exact=exact and torch.equal(a,b);kl.append(float((p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum()));tv.append(float((p-q).abs().sum())/2);disagree+=int(a.argmax()!=b.argmax())
   result[label]=dict(n=len(rows),exact_actor_logits=exact,KL_initial_to_current_mean=float(np.mean(kl)),TV_mean=float(np.mean(tv)),greedy_disagreements=disagree)
   if label=='old' and not exact:raise RuntimeError('Old-only actor equivalence failed after learning')
  append(self.context.out/'policy-diagnostics.jsonl',result)
 def export(self,model,execution,path):
  state=canonical_state(model);value=dict(state_dict=state,step=execution['step'],extra=dict(embedding_adaptation_export=True,software_verification_only=self.context.mode=='software_test',candidate_eligible=False,actor_changed=actor_hash(model,self.scalar.digest)!=self.actor_initial_digest,execution=copy.deepcopy(execution),initial_tensor_hash=INITIAL_TENSOR,eligible_rows=self.scope,corrections_folded=True))
  from pathlib import Path
  path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);temp=path.with_suffix('.tmp')
  with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  temp.replace(path)
