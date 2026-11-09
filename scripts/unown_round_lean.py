"""Disposable raw-batched PPO arithmetic; scalar implementation stays unchanged."""
from unown_round_core import *
METERS=collections.Counter()

def summary(logits,values,rows):
 lp=logits.log_softmax(-1);p=lp.exp();old=torch.as_tensor(np.stack([r['old_probs'] for r in rows]),device=lp.device);mask=torch.as_tensor(np.stack([r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0 for r in rows]),device=lp.device)
 actions=torch.tensor([r['action'] for r in rows],device=lp.device);saved=torch.tensor([r['log_probability'] for r in rows],device=lp.device);chosen=lp.gather(1,actions[:,None]).squeeze(1)
 if not torch.isfinite(lp).all() or not torch.isfinite(values).all() or not mask.any(1).all() or not mask.gather(1,actions[:,None]).all() or torch.any(p.masked_select(~mask)!=0) or torch.any(old.masked_select(~mask)!=0) or torch.any((old.sum(1)-1).abs()>1e-5):raise FloatingPointError('Invalid/nonfinite batched distribution/mask')
 kl=(old*(old.clamp_min(1e-30).log()-lp)).sum(1);ent=-(p*lp).sum(1);meaningful=mask.sum(1)>1
 data=torch.stack([kl.sum(),ent.sum(),((chosen-saved).exp().sub(1).abs()>.2).float().sum(),(chosen-saved).abs().max(),kl.max(),ent[meaningful].sum(),meaningful.sum()]).detach().cpu().tolist()
 return dict(n=len(rows),KL_sum=data[0],entropy_sum=data[1],clip_sum=data[2],max_chosen_logprob_difference=data[3],max_old_policy_KL=data[4],meaningful_entropy_sum=data[5],meaningful_decisions=data[6])
def combine(items):
 n=sum(x['n'] for x in items);meaningful=sum(x['meaningful_decisions'] for x in items)
 return dict(phase_KL=sum(x['KL_sum'] for x in items)/n,entropy=sum(x['entropy_sum'] for x in items)/n,clip_fraction=sum(x['clip_sum'] for x in items)/n,max_chosen_logprob_difference=max(x['max_chosen_logprob_difference'] for x in items),max_old_policy_KL=max(x['max_old_policy_KL'] for x in items),meaningful_decisions=meaningful,meaningful_action_entropy=sum(x['meaningful_entropy_sum'] for x in items)/meaningful if meaningful else None)
@torch.no_grad()
def diagnose(model,rows):
 runtime.freeze_inference(model);items=[]
 for _,group in batch.groups(rows):l,v=batch.forward(model,group,raw=True);items.append(summary(l,v,group))
 batch.clear(model);return combine(items)
def gradients(model,rows,adv):
 batch.clear(model);model.train();model.zero_grad(set_to_none=True);metrics=dict(policy_loss=0.,value_loss=0.,entropy=0.,clip_fraction=0.);pre=[]
 for ii,group in batch.groups(rows):
  l,v=batch.forward(model,group,raw=True);t=time.monotonic();pre.append(summary(l.detach(),v.detach(),group));METERS['pre_metrics_seconds']+=time.monotonic()-t;a,vl,e,cl=batch.losses(l,v,group,np.asarray(adv)[ii]);loss=(a+2*vl-.01*e).sum()/len(rows)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite raw PPO objective')
  loss.backward()
  vals=torch.stack([a.sum(),vl.sum(),e.sum(),cl.sum()]).detach().cpu().tolist()
  for k,x in zip(metrics,vals):metrics[k]+=x/len(rows)
 return metrics,combine(pre)
def transaction(model,opt,rows,adv,execution,label):
 t=time.monotonic();before=batch.snapshot(model,opt,execution);METERS['snapshot_seconds']+=time.monotonic()-t
 try:
  t=time.monotonic();metrics,pre=gradients(model,rows,adv);METERS['gradient_seconds']+=time.monotonic()-t
  if pre['phase_KL']>.02:
   batch.restore(model,opt,before,execution);return dict(accepted=False,reason='phase_preupdate_drift_stop',pre=pre)
  t=time.monotonic();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True);opt.step();model.eval();metrics['gradient_norm']=float(norm);METERS['optimizer_seconds']+=time.monotonic()-t;t=time.monotonic();post=diagnose(model,rows);METERS['post_diagnostic_seconds']+=time.monotonic()-t;accepted=post['phase_KL']<=.03
  result=dict(pre=pre,post=post,metrics=metrics,accepted=accepted)
  if not accepted:
   torch.save(before,PRIVATE/(label+'-before.pt'));torch.save(batch.snapshot(model,opt,execution),PRIVATE/(label+'-rejected.pt'));write(OUT/(label+'-rejected.json'),result);batch.restore(model,opt,before,execution)
   assert batch.scalar.state_fingerprints(batch.snapshot(model,opt,execution))==batch.scalar.state_fingerprints(before);result.update(reason='phase_postupdate_drift_rollback_stop',rollback_exact=True)
  else:execution['step']+=1
  return result
 except BaseException as e:
  torch.save(before,PRIVATE/(label+'-numerical-before.pt'));write(OUT/(label+'-numerical-failure.json'),dict(type=type(e).__name__,error=str(e)));batch.restore(model,opt,before,execution);raise
