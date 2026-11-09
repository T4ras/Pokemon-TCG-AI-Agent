"""Frozen optimized PPO arithmetic, corrected collection and revised phase stops."""
import collections,copy,gzip,random,time
import numpy as np,torch
import unown_batched_core as batch
from unown_corrected_ppo_common import *
from unown_start_order_actor import actor_class,replace_feature
runtime=batch.runtime;legacy=batch.legacy
for module in [batch,batch.scalar,legacy,runtime,runtime.rt]:module.OUT=OUT;module.PRIVATE=PRIVATE
batch.deterministic_backend()
_load_model=runtime.load_model
def load_model(*a,**kw):
 # The released main sets Torch threads to2 on its first import. Apply our
 # registered backend AFTER that constructor, as well as on checkpoint restore.
 model=_load_model(*a,**kw);batch.deterministic_backend();assert torch.get_num_threads()==1;return model
batch.load_model=load_model;runtime.load_model=load_model
state_hash=batch.state_hash;advantages=batch.advantages
METERS=collections.Counter()
def fresh(device='cpu'):
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 if device=='cuda':torch.cuda.manual_seed_all(SEED)
 model=load_model(PRIVATE/'initialization.pt',device);opt=batch.optimizer(model)
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 if device=='cuda':torch.cuda.manual_seed_all(SEED)
 assert state_hash(model)==INITIAL_TENSOR and not opt.state,'Must be fresh initialization/optimizer'
 return model,opt
def corrected_actor_class(context):
 original=actor_class(runtime,context)
 class CorrectedActor(original):
  def __init__(self,model,*a,**kw):
   self.role='learner' if kw.get('collect',False) else ('candidate' if model is context['candidate_model'] else 'opponent');prior=context['candidate_model'];context['candidate_model']=model
   try:super().__init__(model,*a,**kw)
   finally:context['candidate_model']=prior
   context['actors'].append(self);assert self.candidate
  def act(self,raw):
   before=len(self.records);answer=super().act(raw)
   for row in self.records[before:]:
    # The unchanged inference wrapper sees a corrected COPY of the raw encoder
    # packet. Store that same corrected packet for PPO; never alter old_probs/LP.
    old_lp=row['log_probability'];old_probs=row['old_probs'].tobytes()
    row['packet']=replace_feature(row['packet'],self.order)
    row.update(feature_semantics='starting-order-v1',public_first=self.order.first,physical_seat=row['seat'])
    assert row['packet'].row[row['packet'].layout.offsets['global_f'][0]+31]==(row['seat'] if self.order.first is None else int(row['seat']!=self.order.first))
    assert row['log_probability']==old_lp and row['old_probs'].tobytes()==old_probs
   return answer
 return CorrectedActor
def play(job,model,anchor,collect,limit):
 context=dict(candidate_model=model,actual_starter=None,retain_history=job.get('retain_history',False),handles=[],actors=[],example_count=0)
 def handle(actor):
  role='learner' if actor.role=='learner' else ('candidate' if len(context['actors'])==0 else 'opponent')
  path=PRIVATE/'histories'/(job['id']+'-'+role+'.jsonl.gz');path.parent.mkdir(exist_ok=True);h=gzip.open(path,'wt');context['handles'].append((h,path));return h
 context['history_handle']=handle;cls=corrected_actor_class(context);original=runtime.PilotActor;runtime.PilotActor=cls
 try:row,experience=runtime.play(job,model,anchor,collect=collect,limit=limit)
 finally:
  runtime.PilotActor=original
  for h,p in context['handles']:h.close()
 actors=context['actors'];row['corrected_both_actors']=len(actors)==2 and all(a.candidate for a in actors)
 row['independent_mutable_state']=len(actors)==2 and actors[0].memory is not actors[1].memory and actors[0].order is not actors[1].order
 row['feature_transitions']=[dict(role=a.role,transitions=a.transitions) for a in actors]
 append(OUT/'corrected-match-checks.jsonl',dict(id=job['id'],corrected_both_actors=row['corrected_both_actors'],independent_mutable_state=row['independent_mutable_state']))
 if row['status']=='DONE':assert row['corrected_both_actors'] and row['independent_mutable_state']
 if sum(p.stat().st_size for p in (PRIVATE/'histories').glob('*.gz'))>512*1024**2:raise RuntimeError('512MiB full legal history cap')
 return row,experience
@torch.no_grad()
def phase_metrics(model,rows):
 started=time.monotonic();runtime.freeze_inference(model);points=[];maxlp=0.;maxkl=0.;meaningful=[]
 for _,group in batch.groups(rows):
  logits,value=batch.forward(model,group);lp=logits.log_softmax(-1);p=lp.exp()
  if not torch.isfinite(lp).all() or not torch.isfinite(value).all():raise FloatingPointError('Nonfinite policy/value')
  for j,r in enumerate(group):
   mask=r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0;old=torch.as_tensor(r['old_probs'],device=lp.device)
   if not mask.any() or np.any(r['old_probs'][~mask]) or abs(float(old.sum())-1)>1e-5 or float(p[j,~mask].sum())!=0:raise FloatingPointError('Invalid legal probability/mask')
   if not mask[r['action']]:raise FloatingPointError('Chosen rollout action is masked')
   kl=float((old*(old.clamp_min(1e-30).log()-lp[j])).sum());entropy=float(-(p[j]*lp[j]).sum());ratio=float((lp[j,r['action']]-r['log_probability']).exp())
   maxlp=max(maxlp,abs(float(lp[j,r['action']])-r['log_probability']));maxkl=max(maxkl,kl);points.append([kl,entropy,float(abs(ratio-1)>.2)])
   if int(mask.sum())>1:meaningful.append(entropy)
 batch.clear(model);a=np.asarray(points)
 result=dict(phase_KL=float(a[:,0].mean()),entropy=float(a[:,1].mean()),clip_fraction=float(a[:,2].mean()),meaningful_action_entropy=float(np.mean(meaningful)) if meaningful else None,meaningful_decisions=len(meaningful),decisions=len(rows),max_chosen_logprob_difference=maxlp,max_old_policy_KL=maxkl);METERS['phase_diagnostic_seconds']+=time.monotonic()-started;return result
# The established transaction has ONLY phase guards. Its numerical snapshot,
# rejected diagnostics and exact restore remain unchanged. No initialization gate.
batch.diagnose=phase_metrics
_verified_step=batch.step
def timed_step(*a,**kw):
 t=time.monotonic()
 try:return _verified_step(*a,**kw)
 finally:METERS['optimizer_seconds']+=time.monotonic()-t
batch.step=timed_step
transaction=batch.transaction
class ZeroCohorts:
 def __init__(self):self.consecutive=0
 def record(self,accepted):self.consecutive=self.consecutive+1 if accepted==0 else 0;return self.consecutive>=3
def cohort_jobs(phase):
 rng=np.random.default_rng(SEED+phase);seats=np.array([0]*8+[1]*8);rng.shuffle(seats)
 return [dict(id=f'train-{phase:04d}-{i:02d}',stage='training',phase=phase,learner_seat=int(seat),learner_deck='Hydro',opponent_deck='Hydro',seed=SEED+100000+phase*100+i,retain_history=phase in [0,16] and i<2) for i,seat in enumerate(seats)]
@torch.no_grad()
def reference(rows,initial):
 runtime.freeze_inference(initial)
 for _,group in batch.groups(rows):
  lp=batch.forward(initial,group)[0].log_softmax(-1)
  for row,v in zip(group,lp):row['reference_probs']=v.exp().cpu().numpy();row['reference_greedy']=int(v.argmax())
 batch.clear(initial)
@torch.no_grad()
def fixed_drift(model,rows):
 runtime.freeze_inference(model);kls=[];entropy=[];agree=[];tv=[]
 for _,group in batch.groups(rows):
  lp=batch.forward(model,group)[0].log_softmax(-1);p=lp.exp()
  if not torch.isfinite(lp).all():raise FloatingPointError('Nonfinite fixed diagnostic')
  for r,l,v in zip(group,lp,p):
   q=torch.as_tensor(r['reference_probs'],device=l.device);kls.append(float((q*(q.clamp_min(1e-30).log()-l)).sum()));entropy.append(float(-(v*l).sum()));agree.append(int(v.argmax())==r['reference_greedy']);tv.append(float((v-q).abs().sum()/2))
 batch.clear(model);a=np.asarray(kls)
 return dict(decisions=len(rows),initialization_KL=float(a.mean()),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()),greedy_action_agreement=float(np.mean(agree)),mean_probability_TV=float(np.mean(tv)),entropy=float(np.mean(entropy)),stop_rule='Diagnostic only; no cumulative KL stop/anchor loss')
