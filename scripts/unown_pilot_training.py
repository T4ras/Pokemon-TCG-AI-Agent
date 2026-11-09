"""Single conservative PPO pilot; no oracle/aux losses or text updates."""
import copy,random,time
from collections import defaultdict
import numpy as np
import torch
from unown_pilot_common import *
from unown_pilot_runtime import load_model,freeze_inference,play,state_hash
from unown_pilot_execution import cuda_startup
import unown_v2_runtime as rt


class NumericalFailure(RuntimeError):pass
class ExcessiveDrift(RuntimeError):pass


class BatchPacket:
 def __init__(self,rows):self.layout=rows[0]['packet'].layout;self.rows=rows
 def tensor(self,device):return torch.as_tensor(np.stack([r['packet'].row for r in self.rows]),device=device)


def configure(model):
 disabled=['oracle_proj.','critic_readout','critic_blocks.','oracle_value_mlp.','aux_hand.','aux_prize.']
 for name,p in model.named_parameters():p.requires_grad_(not any(name.startswith(x) for x in disabled))
 groups=[dict(params=[p for p in model.parameters() if p.requires_grad],weight_decay=0.)]
 return torch.optim.AdamW(groups,lr=3e-5,betas=(.95,.999),eps=1e-8,foreach=False)


def forward(model,rows):
 # Match collection's single-packet numerical path. A logical PPO minibatch
 # still contains128 decisions; batch_step accumulates its gradients before
 # one optimizer step. No rebasing of stored probabilities or relaxed gate.
 logits=[];values=[]
 for row in rows:
  c,_=model.core(row['packet'])
  logits.append(c['logits'])
  values.append(model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze(-1))
 return torch.cat(logits,0),torch.cat(values,0)


def grouped(rows):
 groups={}
 for r in rows:groups.setdefault(r['packet'].layout,[]).append(r)
 return [group[i:i+128] for group in groups.values() for i in range(0,len(group),128)]


def likelihood_check(model,rows):
 maximum=0.;maxkl=0.
 with torch.no_grad():
  for group in grouped(rows):
   logits,v=forward(model,group);lp=torch.log_softmax(logits,-1).cpu().numpy();prob=np.exp(lp)
   if not np.isfinite(lp).all() or not torch.isfinite(v).all():raise NumericalFailure('Nonfinite initial likelihood/value')
   for i,r in enumerate(group):
    mask=r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0
    if np.any(r['old_probs'][~mask]!=0) or not abs(r['old_probs'].sum()-1)<1e-5:raise NumericalFailure('Saved probability/mask mismatch')
    maximum=max(maximum,abs(float(lp[i,r['action']])-r['log_probability']))
    p=r['old_probs'];maxkl=max(maxkl,float(np.sum(p*(np.log(p.clip(1e-30))-lp[i]))))
 if maximum>1e-5 or maxkl>1e-6:raise NumericalFailure(f'Rollout/batched training likelihood mismatch: {maximum},{maxkl}')
 return dict(max_abs_logprob_difference=maximum,max_old_policy_KL=maxkl,exact_legal_masks=True)


def drift(model,rows):
 sums=dict(phase_KL=0.,initialization_KL=0.,entropy=0.,clip_fraction=0.);n=0
 with torch.no_grad():
  for group in grouped(rows):
   logits,v=forward(model,group);lp=torch.log_softmax(logits,-1);prob=lp.exp()
   old=torch.as_tensor(np.stack([r['old_probs'] for r in group]),device=lp.device)
   initial=torch.as_tensor(np.stack([r['initial_probs'] for r in group]),device=lp.device)
   actions=torch.tensor([r['action'] for r in group],device=lp.device)
   ratio=(lp.gather(1,actions[:,None]).squeeze(1)-torch.tensor([r['log_probability'] for r in group],device=lp.device)).exp()
   sums['phase_KL']+=float((old*(old.clamp_min(1e-30).log()-lp)).sum())
   sums['initialization_KL']+=float((initial*(initial.clamp_min(1e-30).log()-lp)).sum())
   sums['entropy']+=float(-(prob*lp).sum());sums['clip_fraction']+=float(((ratio-1).abs()>.2).sum());n+=len(group)
 return {k:v/n for k,v in sums.items()}


def save(path,model,opt,step,extra=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 state=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),step=step,
  torch_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
  numpy_rng=np.random.get_state(),python_rng=random.getstate(),experimental_approximate_initialization=True,
  text_tables_fixed=True,disabled_objectives=['oracle value','hand auxiliary','prize auxiliary'],extra=extra or {})
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path);return sha(path)


def batch_step(model,opt,rows,advantages):
 opt.zero_grad(set_to_none=True);model.train()
 # Backpropagate one sample at a time so this arithmetic matches collection
 # and the graph stays bounded. Parameters/optimizer are fixed until all
 #128 (or the registered final remainder) contributions have accumulated.
 buckets={i:[(r,float(a))] for i,(r,a) in enumerate(zip(rows,advantages))}
 metrics=dict(policy_loss=0.,value_loss=0.,entropy=0.,clip_fraction=0.)
 for entries in buckets.values():
  group=[r for r,a in entries];logits,value=forward(model,group);lp=torch.log_softmax(logits,-1);prob=lp.exp()
  actions=torch.tensor([r['action'] for r in group],device=lp.device)
  current=lp.gather(1,actions[:,None]).squeeze(1);old=torch.tensor([r['log_probability'] for r in group],device=lp.device)
  adv=torch.tensor([a for r,a in entries],device=lp.device);target=torch.tensor([r['return_target'] for r in group],device=lp.device)
  ratio=(current-old).exp();actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
  value_loss=.5*(value-target).square();entropy=-(prob*lp).sum(1)
  loss=(actor+2*value_loss-.01*entropy).sum()/len(rows)
  if not torch.isfinite(loss):raise NumericalFailure('Nonfinite PPO loss')
  loss.backward()
  metrics['policy_loss']+=float(actor.detach().sum())/len(rows);metrics['value_loss']+=float(value_loss.detach().sum())/len(rows)
  metrics['entropy']+=float(entropy.detach().sum())/len(rows);metrics['clip_fraction']+=float(((ratio-1).abs()>.2).sum())/len(rows)
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True)
 if not torch.isfinite(norm):raise NumericalFailure('Nonfinite gradients')
 opt.step();model.eval();metrics['gradient_norm']=float(norm);return metrics


def backend_and_recovery(model,initial,opt,rows,device):
 """No native matches: same saved packets, disposable update, explicit recovery."""
 sample=rows[:min(16,len(rows))]
 if not sample:raise NumericalFailure('No representative validation packets')
 cpu=load_model(PRIVATE/'initialization.pt','cpu');maxdiff=0.;maxtv=0.
 with torch.no_grad():
  for r in sample:
   a=cpu.policy(r['packet']).cpu();b=model.policy(r['packet']).cpu()
   maxdiff=max(maxdiff,float((a-b).abs().max()));maxtv=max(maxtv,float((a.softmax(-1)-b.softmax(-1)).abs().sum()/2))
   torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
 if maxtv>1e-4:raise NumericalFailure('CPU/GPU probability disagreement')
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 likelihood=likelihood_check(model,sample)
 probe=copy.deepcopy(model);popt=configure(probe);adv=np.array([r['advantage'] for r in sample]);adv=(adv-adv.mean())/(adv.std()+1e-8)
 batch_step(probe,popt,sample,adv)
 path=PRIVATE/'recovery-probe.pt';save(path,probe,popt,1)
 state=torch.load(path,map_location=device,weights_only=False);resume=load_model(PRIVATE/'initialization.pt',device);ropt=configure(resume)
 resume.load_state_dict(state['state_dict']);ropt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'].cpu())
 np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
 if device!='cpu':torch.cuda.set_rng_state_all([x.cpu() for x in state['CUDA_rng']])
 batch_step(probe,popt,sample,adv);batch_step(resume,ropt,sample,adv)
 delta=0.
 for k,v in probe.state_dict().items():torch.testing.assert_close(v,resume.state_dict()[k],atol=1e-6,rtol=1e-5);delta=max(delta,float((v-resume.state_dict()[k]).abs().max()))
 write(OUT/'backend-recovery.json',dict(status='PASS',device=device,CPU_backend_max_logit_difference=maxdiff,max_probability_TV=maxtv,
  same_backend_resumed_max_difference=delta,initial_likelihood=likelihood,probe_updates=3,not_promoted=True))
 del probe,resume,cpu


def train(device='cuda',cap_seconds=TRAINING_CAP_SECONDS):
 started=time.monotonic();deadline=started+cap_seconds
 if device=='cuda':
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  cuda_startup(torch,lambda value:write(OUT/'gpu-runtime.json',dict(value,at_utc=iso())))
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 model=load_model(PRIVATE/'initialization.pt',device);initial=load_model(PRIVATE/'initialization.pt',device);opt=configure(model)
 freeze_inference(initial);text_before={k:v.clone() for k,v in model.named_buffers()};unused_before={k:v.clone() for k,v in model.named_parameters() if not v.requires_grad}
 step=0;completed=0;reason='fixed_four_phases_complete';decisions=0;collection_seconds=0.;update_seconds=0.;backend_checked=False
 save(PRIVATE/'latest.pt',model,opt,step);jobs=read(OUT/'training-schedule.json')
 for phase in range(4):
  if time.monotonic()+300>=deadline:reason='registered_time_stop_before_cohort';break
  freeze_inference(model);rows=[];begin=time.monotonic();phasejobs=[j for j in jobs if j['phase']==phase]
  for job in phasejobs:
   record,experience=play(job,model,initial,collect=True,limit=deadline)
   if record['status']!='DONE':raise NumericalFailure('Training collection failed; no replacement: '+str(record['error']))
   completed+=1;rows+=experience
  # Rollout and exact phase input checkpoint precede every optimizer update.
  torch.save(rows,PRIVATE/f'phase-{phase:02d}-rollout.pt');save(PRIVATE/f'phase-{phase:02d}-input.pt',model,opt,step)
  replay=torch.load(PRIVATE/f'phase-{phase:02d}-rollout.pt',map_location='cpu',weights_only=False)
  assert len(replay)==len(rows) and all(a['packet'].layout==b['packet'].layout and a['packet'].row.tobytes()==b['packet'].row.tobytes() and np.array_equal(a['old_probs'],b['old_probs']) and a['log_probability']==b['log_probability'] and a['action']==b['action'] for a,b in zip(rows,replay))
  collection_seconds+=time.monotonic()-begin;decisions+=len(rows)
  if not backend_checked:backend_and_recovery(model,initial,opt,rows,device);backend_checked=True
  if hasattr(model,'_frozen_tables'):del model._frozen_tables
  likelihood=likelihood_check(model,rows);write(OUT/f'phase-{phase:02d}-initial-likelihood.json',likelihood)
  with torch.no_grad():
   for group in grouped(rows):
    logits,_=forward(initial,group);probs=logits.softmax(-1).cpu().numpy()
    for r,p in zip(group,probs):r['initial_probs']=p.astype(np.float32)
  advantages=np.array([r['advantage'] for r in rows],np.float64);advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
  order=np.random.default_rng(SEED+phase).permutation(len(rows));begin=time.monotonic();stopped=False
  for offset in range(0,len(rows),128):
   if time.monotonic()+30>=deadline:reason='registered_time_stop_between_updates';stopped=True;break
   idx=order[offset:offset+128];batch=[rows[i] for i in idx];pre=drift(model,batch)
   if pre['phase_KL']>.02 or pre['initialization_KL']>.10:reason='registered_preupdate_policy_drift_stop';stopped=True;break
   backup={k:v.detach().clone() for k,v in model.state_dict().items()};backup_opt=copy.deepcopy(opt.state_dict())
   exclusive(OUT/'update-intents'/f'{step+1:05d}.json',dict(step=step+1,phase=phase,at_utc=iso(),rows=len(batch)))
   metrics=batch_step(model,opt,batch,advantages[idx]);post=drift(model,batch)
   if not all(np.isfinite(x) for x in post.values()):raise NumericalFailure('Nonfinite postupdate policy diagnostics')
   if post['phase_KL']>.03 or post['initialization_KL']>.10:
    model.load_state_dict(backup);opt.load_state_dict(backup_opt)
    write(OUT/'drift-stop.json',dict(phase=phase,rejected_step=step+1,pre=pre,post=post,last_valid_step=step));reason='registered_postupdate_drift_rollback_stop';stopped=True;break
   step+=1;checksum=save(PRIVATE/'latest.pt',model,opt,step)
   append(OUT/'updates.jsonl',dict(step=step,phase=phase,rows=len(batch),at_utc=iso(),pre=pre,post=post,metrics=metrics,checkpoint_sha256=checksum))
  update_seconds+=time.monotonic()-begin
  for k,v in model.named_buffers():
   if not torch.equal(v,text_before[k]):raise NumericalFailure('Fixed text/static buffer changed')
  for k,v in model.named_parameters():
   if k in unused_before and not torch.equal(v,unused_before[k]):raise NumericalFailure('Disabled oracle/auxiliary parameter changed')
  save(PRIVATE/f'phase-{phase:02d}-valid.pt',model,opt,step)
  progress('PILOT_PHASE',phase=phase,completed_BO3=completed,updates=step,learner_decisions=decisions)
  if stopped:break
 final=PRIVATE/'final.pt';checksum=save(final,model,opt,step,extra=dict(stop_reason=reason))
 write(OUT/'training-completion.json',dict(status='COMPLETE',stop_reason=reason,completed_BO3=completed,planned_BO3=64,learner_decisions=decisions,
  optimizer_updates=step,checkpoint_sha256=checksum,final_tensor_hash=state_hash(model),device=device,
  collection_seconds=collection_seconds,update_checkpoint_seconds=update_seconds,
  complete_elapsed_seconds=time.monotonic()-started,useful_decisions_per_second=decisions/(time.monotonic()-started),
  GPU_session_hours_script=(time.monotonic()-started)/3600 if device=='cuda' else 0,
  objective_losses=['clipped policy surrogate','blind value regression','entropy'],oracle_auxiliary_disabled=True,text_tables_unchanged=True,
  numerical_path='Scalar packet core and value; gradient accumulation to logical minibatch128',
  torch_version=torch.__version__,python_version=__import__('sys').version,numpy_version=np.__version__,
  kaggle_environments_version=__import__('importlib.metadata',fromlist=['version']).version('kaggle-environments'),
  CUDA_version=torch.version.cuda,GPU_name=torch.cuda.get_device_name(0) if device=='cuda' else None,
  maximum_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
  maximum_GPU_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0))


if __name__=='__main__':
 import traceback
 torch.set_num_threads(2)
 try:train()
 except BaseException as error:
  write(OUT/'training-failure.json',dict(at_utc=iso(),state='FAILED',error=str(error),traceback=traceback.format_exc(),automatic_restart=False,last_valid_checkpoint=str(PRIVATE/'latest.pt')))
  raise
