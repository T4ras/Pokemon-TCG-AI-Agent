"""Batched PPO; legacy scalar arithmetic remains an independent reference."""
import copy, random, time, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from unown_batched_common import *
import unown_stability_core as scalar
runtime=scalar.runtime; legacy=scalar.legacy
scalar.OUT=OUT;scalar.PRIVATE=PRIVATE
runtime.OUT=OUT;runtime.PRIVATE=PRIVATE;runtime.rt.OUT=OUT;runtime.rt.PRIVATE=PRIVATE
legacy.OUT=OUT;legacy.PRIVATE=PRIVATE
load_model=runtime.load_model;state_hash=runtime.state_hash
advantages=scalar.advantages;digest=scalar.digest;snapshot=scalar.snapshot;restore=scalar.restore

def deterministic_backend():
 torch.set_num_threads(1)
 torch.use_deterministic_algorithms(True)
 torch.autograd.set_multithreading_enabled(False)

def optimizer(model):return scalar.optimizer(model,1e-5)
def restore_checkpoint(path,device='cpu',rng=True):
 deterministic_backend()
 m=load_model(PRIVATE/'initialization.pt',device);o=optimizer(m)
 s=torch.load(path,map_location=device,weights_only=False);m.load_state_dict(s['state_dict']);o.load_state_dict(s['optimizer'])
 assert all(g['lr']==1e-5 and g['betas']==(.95,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in o.param_groups)
 if rng:
  torch.set_rng_state(s['torch_rng'].cpu());np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
  if device!='cpu' and s.get('CUDA_rng'):torch.cuda.set_rng_state_all([v.cpu() for v in s['CUDA_rng']])
 return m,o,s

class BatchPacket:
 def __init__(self,rows):
  self.layout=rows[0]['packet'].layout
  assert all(r['packet'].layout==self.layout for r in rows)
  self.rows=rows
 def tensor(self,device):return torch.as_tensor(np.stack([r['packet'].row for r in self.rows]),device=device)

def token_count(r):
 p=r['packet'];a=p.row[slice(*p.layout.offsets['tok_int'])].reshape(p.layout.tokens,3+p.layout.tools)
 return int((a[:,1]>0).sum())

def groups(rows,physical=16,exact_length=True):
 buckets={}
 for i,r in enumerate(rows):
  key=(r['packet'].layout,token_count(r) if exact_length else None)
  buckets.setdefault(key,[]).append(i)
 for indices in buckets.values():
  for start in range(0,len(indices),physical):
   ii=indices[start:start+physical];yield ii,[rows[i] for i in ii]

def scalar_outputs(model,rows):
 faithful=[];values=[]
 for row in rows:
  packet=row.get('scalar_reference_packet',row['packet'])
  s,_=model.core(packet);faithful.append(s['logits']);values.append(model.value_mlp(torch.cat([s['r'],s['g']],-1)).squeeze(-1))
 ref=torch.cat(faithful);rv=torch.cat(values);options=rows[0]['packet'].layout.options
 if ref.shape[1]!=options:
  expanded=torch.full((len(rows),options),-1e9,device=ref.device,dtype=ref.dtype)
  expanded[:,:ref.shape[1]-1]=ref[:,:-1];expanded[:,-1]=ref[:,-1];ref=expanded
 return ref,rv

def forward(model,rows,raw=False):
 # Preserve collection's float32 scalar forward values, while differentiating
 # the same operations in batches. This corrects backend GEMM roundoff, not the
 # stored old probabilities. The independently checked gradient is batched.
 prior=getattr(model,'_frozen_tables',None)
 tables=model._card_table();model._frozen_tables=tables
 try:
  if not torch.is_grad_enabled() and not raw:
   # LP/KL reductions are vectorized over faithful scalar readouts. No
   # derivative is needed, so the discarded batched transformer is unnecessary.
   return scalar_outputs(model,rows)
  c,_=model.core(BatchPacket(rows));logits=c['logits'];value=model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze(-1)
  if raw:return logits,value
  model._frozen_tables=tuple(x.detach() for x in tables)
  with torch.no_grad():
   ref,rv=scalar_outputs(model,rows)
  return ref+(logits-logits.detach()),rv+(value-value.detach())
 finally:
  if prior is None:clear(model)
  else:model._frozen_tables=prior

def clear(model):
 if hasattr(model,'_frozen_tables'):del model._frozen_tables

def losses(logits,value,rows,adv):
 lp=logits.log_softmax(-1);prob=lp.exp();device=lp.device
 action=torch.tensor([r['action'] for r in rows],device=device)
 old=torch.tensor([r['log_probability'] for r in rows],device=device)
 target=torch.tensor([r['return_target'] for r in rows],device=device)
 a=torch.as_tensor(np.asarray(adv),device=device,dtype=lp.dtype)
 ratio=(lp.gather(1,action[:,None]).squeeze(1)-old).exp()
 actor=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)
 val=.5*(value-target).square();ent=-(prob*lp).sum(1)
 return actor,val,ent,((ratio-1).abs()>.2)

def gradients(model,rows,adv,batched=True):
 clear(model);model.train();model.zero_grad(set_to_none=True)
 sums=dict(policy_loss=0.,value_loss=0.,entropy=0.,clip_fraction=0.)
 iterator=groups(rows) if batched else (([i],[r]) for i,r in enumerate(rows))
 for ii,group in iterator:
  logits,v=forward(model,group) if batched else legacy.forward(model,group)
  a,vl,e,clip=losses(logits,v,group,np.asarray(adv)[ii]);loss=(a+2*vl-.01*e).sum()/len(rows)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite logical PPO loss')
  loss.backward()
  for k,x in zip(sums,[a,vl,e,clip]):sums[k]+=float(x.detach().sum())/len(rows)
 return sums

def step(model,opt,rows,adv,batched=True):
 metrics=gradients(model,rows,adv,batched)
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True)
 opt.step();model.eval();metrics['gradient_norm']=float(norm);return metrics

@torch.no_grad()
def diagnose(model,rows,batched=True):
 runtime.freeze_inference(model);points=[];chosen=0.;maximum=0.;maxkl=0.;n=0
 iterator=groups(rows) if batched else (([i],[r]) for i,r in enumerate(rows))
 for ii,group in iterator:
  logits,value=forward(model,group) if batched else legacy.forward(model,group)
  lp=logits.log_softmax(-1);p=lp.exp()
  if not torch.isfinite(lp).all() or not torch.isfinite(value).all():raise FloatingPointError('Nonfinite policy diagnostics')
  for j,r in enumerate(group):
   mask=r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0
   old=torch.as_tensor(r['old_probs'],device=lp.device);init=torch.as_tensor(r['initial_probs'],device=lp.device)
   assert np.all(r['old_probs'][~mask]==0) and abs(float(old.sum())-1)<1e-5
   assert float(p[j,~mask].sum())==0
   phase=float((old*(old.clamp_min(1e-30).log()-lp[j])).sum())
   initkl=float((init*(init.clamp_min(1e-30).log()-lp[j])).sum())
   diff=abs(float(lp[j,r['action']])-r['log_probability']);maximum=max(maximum,diff);maxkl=max(maxkl,phase)
   ratio=(lp[j,r['action']]-r['log_probability']).exp()
   points.append((phase,initkl,float(-(p[j]*lp[j]).sum()),float((ratio-1).abs()>.2)))
   n+=1
 clear(model);a=np.array(points)
 return dict(phase_KL=float(a[:,0].mean()),initialization_KL=float(a[:,1].mean()),entropy=float(a[:,2].mean()),clip_fraction=float(a[:,3].mean()),
  initialization_KL_p95=float(np.quantile(a[:,1],.95)),initialization_KL_p99=float(np.quantile(a[:,1],.99)),initialization_KL_max=float(a[:,1].max()),
  max_chosen_logprob_difference=maximum,max_old_policy_KL=maxkl,decisions=n)

@torch.no_grad()
def attach_initial(rows,anchor):
 runtime.freeze_inference(anchor)
 for _,group in groups(rows):
  logits,_=forward(anchor,group)
  for r,p in zip(group,logits.softmax(-1).cpu().numpy()):r['initial_probs']=p.astype(np.float32)
 clear(anchor)

def transaction(model,opt,rows,adv,execution,label):
 before=snapshot(model,opt,execution);pre=diagnose(model,rows)
 if pre['phase_KL']>.02:return dict(accepted=False,reason='phase_preupdate_drift_stop',pre=pre)
 try:
  metrics=step(model,opt,rows,adv);post=diagnose(model,rows)
  accepted=post['phase_KL']<=.03
 except BaseException as error:
  torch.save(before,PRIVATE/(label+'-before.pt'));write(OUT/(label+'-numerical-failure.json'),dict(error_type=type(error).__name__,error=str(error)))
  restore(model,opt,before,execution);raise
 result=dict(pre=pre,post=post,metrics=metrics,accepted=accepted)
 if not accepted:
  result['reason']='phase_postupdate_drift_rollback_stop'
  torch.save(before,PRIVATE/(label+'-before.pt'));torch.save(snapshot(model,opt,execution),PRIVATE/(label+'-rejected.pt'))
  write(OUT/(label+'-rejected.json'),result);restore(model,opt,before,execution)
  assert scalar.state_fingerprints(snapshot(model,opt,execution))==scalar.state_fingerprints(before)
  result['rollback_exact']=True
 else:execution['step']+=1
 return result

