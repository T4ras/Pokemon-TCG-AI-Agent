"""Small learner-seat PPO correctness path; never a substantive trainer command."""
import copy, math, os, random, time
import numpy as np
import torch
from unown_v2_common import *
from unown_v2_runtime import make_model


def optimizer(model):
 one=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
 matrices=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
 return torch.optim.AdamW([dict(params=one,weight_decay=.01),dict(params=matrices,weight_decay=0.)],
  lr=.001,betas=(.95,.999),eps=1e-12,foreach=False)


def state_hash(model):
 import hashlib
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def completed_game_returns(records,games,learner_seat):
 """No cross-seat negamax: every retained row belongs to the same learner."""
 output=[]
 for game in games:
  if game.get('completion_status')!='DONE' or game.get('result') not in [0,1,2]:raise ValueError('Failure/truncation is not an outcome')
  rows=[copy.copy(r) for r in records if r['game']==game['round']]
  if any(r['seat']!=learner_seat for r in rows):raise ValueError('Mixed learner perspectives')
  if not rows:continue
  reward=0. if game['result']==2 else 1. if game['result']==learner_seat else -1.
  gae=0.;next_value=0.
  for i in reversed(range(len(rows))):
   terminal=i==len(rows)-1;r=reward if terminal else 0.;mask=0. if terminal else 1.
   delta=r+mask*next_value-rows[i]['value'];gae=delta+mask*.95*gae
   rows[i].update(reward=r,bootstrap_mask=mask,advantage=gae,return_target=gae+rows[i]['value'],episode_definition='completed game')
   next_value=rows[i]['value']
  output+=rows
 return output


def update(model,opt,rows,normalize_advantages=True):
 if not rows:raise ValueError('Empty disposable cohort')
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 model.train();opt.zero_grad(set_to_none=True)
 advantages=np.array([r['advantage'] for r in rows],np.float64)
 if normalize_advantages:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
 diagnostics=[]
 groups={'actor_score':['q_mlp.','score_mlp.'],'shared':['blocks.','card_id_emb.','attack_id_emb.','skill_id_emb.'],
  'blind_value':['value_mlp.'],'oracle_value':['oracle_value_mlp.','oracle_proj.','critic_blocks.'],
  'auxiliary':['aux_hand.','aux_prize.']}
 before={name:p.detach().clone() for name,p in model.named_parameters()}
 for r,adv in zip(rows,advantages):
  out=model.train_packet(r['packet']);logits=out['logits'][0]
  dist=torch.distributions.Categorical(logits=logits);lp=dist.log_prob(torch.tensor(r['action']))
  ratio=(lp-r['log_probability']).exp();a=torch.as_tensor(float(adv),dtype=lp.dtype)
  actor=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)
  target=torch.as_tensor(r['return_target'],dtype=lp.dtype)
  blind=(out['blind_v'].squeeze()-target).square()*.5
  oracle=(out['oracle_v'].squeeze()-target).square()*.5
  entropy=dist.entropy();aux=lp*0.
  # Real rollout collection never invents hidden labels. Explicit separate
  # training-only targets are allowed for synthetic interface tests.
  if r.get('aux_hand_truth') is not None:
   truth=torch.as_tensor(r['aux_hand_truth'],dtype=lp.dtype).reshape_as(out['hand_pred'])
   aux=torch.nn.functional.mse_loss(out['hand_pred'],truth)
  if r.get('aux_prize_truth') is not None:
   truth=torch.as_tensor(r['aux_prize_truth'],dtype=lp.dtype).reshape_as(out['prize_logits'])
   aux=aux+torch.nn.functional.binary_cross_entropy_with_logits(out['prize_logits'],truth)
  loss=(actor+2*blind+2*oracle-.01*entropy+aux)/len(rows)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  loss.backward()
  diagnostics.append(dict(actor=float(actor.detach()),blind_value=float(blind.detach()),oracle_value=float(oracle.detach()),
   auxiliary=float(aux.detach()),entropy=float(entropy.detach()),initial_ratio=float(ratio.detach()),old_policy_log_difference=float(lp.detach())-r['log_probability']))
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.5,error_if_nonfinite=True)
 if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradients')
 gradient_groups={label:float(sum((p.grad.detach().double().square().sum() for name,p in model.named_parameters() if p.grad is not None and any(name.startswith(prefix) for prefix in prefixes)),torch.tensor(0.,dtype=torch.float64)).sqrt()) for label,prefixes in groups.items()}
 opt.step();model.eval()
 changed_groups={label:sum(not torch.equal(before[name],p) for name,p in model.named_parameters() if any(name.startswith(prefix) for prefix in prefixes)) for label,prefixes in groups.items()}
 return dict(rows=len(rows),gradient_norm=float(norm),means={k:float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]},
  max_ratio_deviation=max(abs(d['initial_ratio']-1) for d in diagnostics),gradient_norm_groups=gradient_groups,changed_parameter_tensors_by_group=changed_groups,per_decision=diagnostics)


def save_recovery(path,model,opt,rows,step,extra=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 state=dict(schema=SCHEMA,model=model.state_dict(),optimizer=opt.state_dict(),rows=rows,step=step,
  torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
  technical_probe=bool(model.technical_probe),extra=extra or {})
 with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
 return sha(path)


def restore_recovery(path,model,opt):
 state=torch.load(path,map_location='cpu',weights_only=False)
 if state['schema']!=SCHEMA:raise ValueError('Recovery schema mismatch')
 model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
 torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 return state


def transactional_update(directory,model,opt,rows,step):
 """A stale intent is actionable evidence, never permission to repeat an update."""
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 commit=directory/f'step-{step:03d}.json';intent=directory/f'step-{step:03d}.intent.json'
 if commit.exists():
  saved=read(commit)
  if state_hash(model)!=saved['model_hash']:raise RuntimeError('Committed update exists; explicitly restore its checkpoint before continuing')
  return saved
 exclusive(intent,dict(at_utc=iso(),step=step,base_model_hash=state_hash(model),rows=len(rows)))
 metrics=update(model,opt,rows)
 path=directory/f'step-{step:03d}.pt';checksum=save_recovery(path,model,opt,rows,step)
 value=dict(at_utc=iso(),step=step,checkpoint_sha256=checksum,checkpoint=str(path),model_hash=state_hash(model),metrics=metrics)
 write(commit,value);return value


def export_candidate(path,model,text_path):
 if model.technical_probe:raise ValueError('A nonsemantic technical probe cannot be exported as a candidate')
 gate=read(OUT/'text-mapping-results.json')
 if not gate.get('accepted'):raise ValueError('Text mapping has not passed the frozen gate')
 parent=torch.load(UP/'submission/model.pt',map_location='cpu',weights_only=False)
 for k,old in parent.items():
  new=model.state_dict()[k]
  if new.shape==old.shape:
   if new.numpy().tobytes()!=old.numpy().tobytes():raise ValueError('Parent tensor changed: '+k)
  elif new[:len(old)].numpy().tobytes()!=old.numpy().tobytes():raise ValueError('Parent rows changed: '+k)
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(schema=SCHEMA,state_dict=model.state_dict(),text_path=str(text_path),technical_probe=False,
  optimizer_state=None,initialization='Fresh fine-tuning initialization, not resumed',parent_sha256=sha(UP/'submission/model.pt')),path)
 write(OUT/'candidate-manifest.json',dict(path=str(path),sha256=sha(path),tensor_sha256=state_hash(model),
  parent_sha256=sha(UP/'submission/model.pt'),text_sha256=sha(text_path),schema=SCHEMA,approximate_text=True,
  optimizer='Absent; initialize fresh reference AdamW',weights_updated=False,old_rows_bit_preserved=True))
