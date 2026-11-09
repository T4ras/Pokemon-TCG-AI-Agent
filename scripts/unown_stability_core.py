"""Same PPO arithmetic; explicit transactional rollback and loss attribution."""
import copy,random,time
import numpy as np
import torch
from unown_stability_common import *
import unown_pilot_runtime as runtime
import unown_pilot_training as legacy
runtime.OUT=OUT;runtime.PRIVATE=PRIVATE;runtime.rt.OUT=OUT;runtime.rt.PRIVATE=PRIVATE
legacy.OUT=OUT;legacy.PRIVATE=PRIVATE;legacy.SEED=SEED
load_model=runtime.load_model;state_hash=runtime.state_hash

def optimizer(model,lr):
 opt=legacy.configure(model)
 for group in opt.param_groups:group['lr']=lr
 return opt

def fresh(lr,device='cpu',saved=False):
 model=load_model(PRIVATE/'initialization.pt',device);opt=optimizer(model,lr)
 if saved:
  state=torch.load(PRIVATE/'historical-phase-input.pt',map_location=device,weights_only=False)
  model.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer'])
  for group in opt.param_groups:group['lr']=lr
 return model,opt

def digest(value):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str((type(k).__name__,k)).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(type(v).__name__.encode())
   for x in v:visit(x)
  else:h.update(repr(v).encode())
 visit(value);return h.hexdigest()

def snapshot(model,opt,execution):
 return dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),
  gradients={k:None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()},training=model.training,
  torch_rng=torch.get_rng_state().clone(),CUDA_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
  numpy_rng=np.random.get_state(),python_rng=random.getstate(),execution=copy.deepcopy(execution))

def restore(model,opt,saved,execution):
 model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);model.train(saved['training'])
 for k,p in model.named_parameters():p.grad=None if saved['gradients'][k] is None else saved['gradients'][k].clone()
 torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
 if saved['CUDA_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in saved['CUDA_rng']])
 execution.clear();execution.update(copy.deepcopy(saved['execution']))
 if hasattr(model,'_frozen_tables'):del model._frozen_tables

def state_fingerprints(saved):return {k:digest(saved[k]) for k in ['model','optimizer','gradients','torch_rng','CUDA_rng','numpy_rng','python_rng','execution','training']}

def advantages(rows):
 raw=np.array([r['advantage'] for r in rows],np.float64)
 return (raw-raw.mean())/(raw.std()+1e-8)

def attach_initial(rows,model):
 runtime.freeze_inference(model)
 with torch.no_grad():
  for group in legacy.grouped(rows):
   logits,_=legacy.forward(model,group)
   for r,p in zip(group,logits.softmax(-1).cpu().numpy()):r['initial_probs']=p.astype(np.float32)
 if hasattr(model,'_frozen_tables'):del model._frozen_tables

def evaluate(model,rows,reference_model=None):
 runtime.freeze_inference(model)
 result=legacy.drift(model,rows)
 if reference_model is not None:
  tv=0.;disagree=0
  with torch.no_grad():
   for group in legacy.grouped(rows):
    p=legacy.forward(model,group)[0].softmax(-1);q=legacy.forward(reference_model,group)[0].softmax(-1)
    tv+=float((p-q).abs().sum()/2);disagree+=int((p.argmax(-1)!=q.argmax(-1)).sum())
  result.update(mean_probability_TV=tv/len(rows),greedy_disagreement=disagree/len(rows))
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 return result

def transaction(model,opt,rows,adv,execution,label,allrows=None,persist=True):
 before=snapshot(model,opt,execution);pre=evaluate(model,rows)
 if pre['phase_KL']>.02 or pre['initialization_KL']>.10:return dict(accepted=False,reason='preupdate_drift_stop',pre=pre)
 try:metrics=legacy.batch_step(model,opt,rows,adv);post=evaluate(model,rows)
 except BaseException as error:
  if persist:
   write(OUT/(label+'-numerical-failure.json'),dict(error_type=type(error).__name__,error=str(error),before=state_fingerprints(before)))
   torch.save(before,PRIVATE/(label+'-before.pt'))
  restore(model,opt,before,execution)
  raise
 result=dict(pre=pre,post=post,metrics=metrics,accepted=post['phase_KL']<=.03 and post['initialization_KL']<=.10)
 if allrows is not None:result['full_cohort']=evaluate(model,allrows)
 if not all(np.isfinite(v) for v in post.values()):result.update(accepted=False,reason='nonfinite')
 if not result['accepted']:
  result.setdefault('reason','postupdate_drift_rollback_stop')
  # Save diagnostics and the rejected optimizer state BEFORE rollback.
  if persist:
   torch.save(before,PRIVATE/(label+'-before.pt'));torch.save(snapshot(model,opt,execution),PRIVATE/(label+'-rejected.pt'))
   write(OUT/(label+'-rejected.json'),dict(result,before=state_fingerprints(before),rejected_tensor_hash=state_hash(model)))
  restore(model,opt,before,execution);after=state_fingerprints(snapshot(model,opt,execution))
  expected=state_fingerprints(before);assert after==expected,'Rollback failed to restore execution state'
  result['rollback']=dict(all_state_exact=True,restored=after)
  if persist:write(OUT/(label+'-rollback.json'),result['rollback'])
 else:execution['step']=execution.get('step',0)+1
 return result

def loss_gradients(model,opt,rows,adv,term='all',take_step=False):
 opt.zero_grad(set_to_none=True);model.train();losses=dict(policy=0.,value=0.,entropy=0.)
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 for r,a in zip(rows,adv):
  logits,value=legacy.forward(model,[r]);lp=logits[0].log_softmax(-1);p=lp.exp();ratio=(lp[r['action']]-r['log_probability']).exp()
  actor=-torch.minimum(ratio*float(a),ratio.clamp(.8,1.2)*float(a));val=.5*(value[0]-r['return_target']).square();ent=-(p*lp).sum()
  terms=dict(policy=actor,value=2*val,entropy=-.01*ent)
  selected=sum(terms.values()) if term=='all' else terms[term]
  (selected/len(rows)).backward()
  for k,v in terms.items():losses[k]+=float(v.detach())/len(rows)
 gradients={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
 norm=float(sum(g.double().square().sum() for g in gradients.values()).sqrt())
 if take_step:torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True);opt.step()
 model.eval();return losses,gradients,norm

def calibration(rows,lr,device,label):
 model,opt=fresh(lr,device,saved=True);anchor,_=fresh(lr,device,saved=True);runtime.freeze_inference(anchor)
 adv=advantages(rows);order=np.random.default_rng(OLD_SEED).permutation(len(rows));execution=dict(step=0,offset=0,order=order)
 windows=[];started=time.monotonic();complete=True
 for offset in range(0,len(rows),128):
  if device=='cpu':check_deadline()
  idx=order[offset:offset+128];execution['offset']=offset
  result=transaction(model,opt,[rows[i] for i in idx],adv[idx],execution,label+f'-step{len(windows)+1:02d}',allrows=rows)
  append(OUT/(label+'-windows.jsonl'),dict(window=len(windows)+1,offset=offset,**result));windows.append(result)
  if not result['accepted']:complete=False;break
 result=dict(lr=lr,complete_epoch=complete,accepted_updates=execution['step'],windows=len(windows),elapsed_seconds=time.monotonic()-started,
  final_cohort=evaluate(model,rows,anchor),minibatch_gates='pre phaseKL<=.02,post<=.03,initialKL<=.10',file_is_disposable=True)
 write(OUT/(label+'-result.json'),result);return result

def tensor_change(initial_hash,final_hash,accepted):
 if accepted==0 and initial_hash!=final_hash:raise RuntimeError('Zero accepted updates but model tensors changed: rollback defect')
 return dict(model_tensors_changed=initial_hash!=final_hash,accepted_updates=accepted,evaluate=accepted>0 and initial_hash!=final_hash,
  file_hashes_are_not_tensor_hashes=True,interpretation='retained model change' if initial_hash!=final_hash else 'no retained model change')
