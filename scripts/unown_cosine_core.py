"""Bind unchanged verified policy/fast arithmetic to this new experiment."""
import collections,copy,gzip,random
import numpy as np,torch
import unown_corrected_ppo_core as core
import unown_round_lean as lean
import unown_round_benchmark as benchmark
from unown_cosine_common import *
batch=core.batch;runtime=core.runtime;legacy=core.legacy
for module in [core,batch,batch.scalar,legacy,runtime,runtime.rt,lean,benchmark]:module.OUT=OUT;module.PRIVATE=PRIVATE
load_model=core.load_model;state_hash=core.state_hash
def restore_rng(s,device):
 torch.set_rng_state(s['torch_rng'].cpu());np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
 if device!='cpu' and s.get('CUDA_rng'):torch.cuda.set_rng_state_all([x.cpu() for x in s['CUDA_rng']])
def restore(path,device='cpu',rng=True):
 m=load_model(PRIVATE/'initialization.pt',device);o=batch.optimizer(m);s=torch.load(path,map_location=device,weights_only=False);m.load_state_dict(s['state_dict']);o.load_state_dict(s['optimizer'])
 for group in o.param_groups:assert group['betas']==(.95,.999) and group['eps']==1e-8 and group['weight_decay']==0 and 3e-6-1e-12<=group['lr']<=1e-5+1e-12
 if rng:restore_rng(s,device)
 return m,o,s
def jobs(phase):
 seats=np.array([0]*8+[1]*8);np.random.default_rng(SEED+phase).shuffle(seats)
 return [dict(id=f'train-{phase:04d}-{i:02d}',stage='training',phase=phase,learner_seat=int(s),learner_deck='Hydro',opponent_deck='Hydro',seed=SEED+100000+phase*100+i,retain_history=phase in [0,32,64,96] and i<2) for i,s in enumerate(seats)]
@torch.no_grad()
def live_distribution(model,rows):
 runtime.freeze_inference(model);probs=[];kls=[];maxabs=0.;maxlp=0.;value_scaled=0.;meaningful=0
 for _,group in batch.groups(rows):
  logits,values=batch.forward(model,group,raw=True);lp=logits.log_softmax(-1);p=lp.exp()
  lean.summary(logits,values,group) # Exact legality/finiteness assertions, no rebasing.
  old=torch.as_tensor(np.stack([r['old_probs'] for r in group]),device=lp.device)
  tv=(p-old).abs().sum(1)/2;kl=(old*(old.clamp_min(1e-30).log()-lp)).sum(1)
  probs+=tv.cpu().tolist();kls+=kl.cpu().tolist();maxabs=max(maxabs,float((p-old).abs().max()))
  for i,r in enumerate(group):
   assert r['feature_semantics']=='starting-order-v1' and r['seat']==r['physical_seat']
   expected=r['seat'] if r['public_first'] is None else int(r['seat']!=r['public_first']);lo,_=r['packet'].layout.offsets['global_f'];assert r['packet'].row[lo+31]==expected
   maxlp=max(maxlp,abs(float(lp[i,r['action']])-r['log_probability']));value_scaled=max(value_scaled,float((values[i]-r['value']).abs()/(1e-4+1e-4*abs(r['value']))));meaningful+=int((r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0).sum()>1)
 result=dict(n=len(rows),TV_max=max(probs),TV_mean=float(np.mean(probs)),KL_max=max(kls),KL_mean=float(np.mean(kls)),probability_max_abs=maxabs,chosen_LP_max=maxlp,value_scaled_error=value_scaled,meaningful_decisions=meaningful,recorded_probability_digest=batch.digest([r['old_probs'] for r in rows]))
 assert result['TV_max']<=1e-4 and maxabs<=5e-5 and result['KL_max']<=1e-4 and result['KL_mean']<=1e-6 and value_scaled<=1,result
 batch.clear(model);return result
def live_check(model,opt,rows,execution):
 before=batch.snapshot(model,opt,execution);fingerprints=batch.scalar.state_fingerprints(before);prob_digest=batch.digest([r['old_probs'] for r in rows]);results=[]
 try:
  likelihood=live_distribution(model,rows);adv=core.advantages(rows)
  benchmark.loaded=lambda case,device:restore(PRIVATE/'phase-input.pt',device,rng=True)[:2]
  cases=[('first128',np.arange(min(128,len(rows))))];remainder=np.arange(len(rows)//128*128,len(rows))
  if len(remainder):cases.append(('actual-remainder',remainder))
  for name,ii in cases:
   result=benchmark.verify(dict(name='live-'+name,phase=0,rows=[rows[i] for i in ii],adv=adv[ii]),'cuda');results.append(result);assert result['status']=='PASS',result['gates']
  return dict(status='PASS',likelihood=likelihood,update_checks=results,disposable_weights_not_promoted=True)
 finally:
  batch.restore(model,opt,before,execution);assert batch.scalar.state_fingerprints(batch.snapshot(model,opt,execution))==fingerprints;assert batch.digest([r['old_probs'] for r in rows])==prob_digest
