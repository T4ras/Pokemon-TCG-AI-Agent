"""Stratified SMC with fixed-cost whole-history guided importance recovery."""
from unown_belief_provenance.common import *
from unown_belief_provenance.state import guided_history
from unown_belief_provenance.collapsed import CollapsedState as State
from unown_belief_provenance.adapter import Compiler
from unown_belief_provenance.baseline import Facts
import random

def allocate(masses,n):
 keys=sorted(k for k,p in masses.items() if p>0);remain=n-4*len(keys)
 if remain<0:raise ValueError('Four-particle list floor exceeds budget')
 total=sum(masses.values());quota={k:remain*masses[k]/total for k in keys};a={k:4+math.floor(quota[k]) for k in keys}
 for k in sorted(keys,key=lambda k:(-(quota[k]-math.floor(quota[k])),k))[:n-sum(a.values())]:a[k]+=1
 return a

def reconstruct(cards,history,rng,trials,budget=None):
 """Fixed-size guided SMC replay, including its incremental evidence factors.

 Resampling is internal to one list and carries its current evidence. This
 avoids exponential rejection of complete initial deals. The resulting
 normalized posterior remains a finite-particle approximation.
 """
 start=time.process_time();steps=0;states=[State(cards,rng) for _ in range(trials)];weights=[1/trials]*trials;failures=collections.Counter();guided=guided_history(history);lz=0.;resamples=0
 for index,op in enumerate(guided):
  kept=[];w=[]
  for state,weight in zip(states,weights):
   if budget is not None:
    if budget['remaining']<=0:raise Depletion(json.dumps(dict(reason='DETERMINISTIC_RECOVERY_STEP_CAP',steps=steps)))
    budget['remaining']-=1
   steps+=1
   try:f=state.apply(op,rng);kept.append(state);w.append(weight*f)
   except IncompatibleAllocation as e:failures[str(e)]+=1
  z=sum(w)
  if not z>0:raise Depletion(json.dumps(dict(trials=trials,accepted=0,at_operation=index,operation=op,failures=failures,CPU_seconds=time.process_time()-start)))
  lz+=math.log(z);states=kept;weights=[v/z for v in w]
  if 1/sum(v*v for v in weights)<trials/2:
   states=[v.clone() for v in rng.choices(states,weights=weights,k=trials)];weights=[1/trials]*trials;resamples+=1
 return dict(states=states,weights=weights,log_evidence=lz,trials=trials,accepted=len(states),CPU_seconds=time.process_time()-start,failures=dict(failures),guided_operations=len(guided),resamples=resamples,proposal_state_steps=steps)

class Observer:
 """Every compatible list keeps its own evidence and unresolved status.

 A failed stratum never vanishes from the aggregate. Other strata continue
 updating; full readout resumes only after justified replay or hard exclusion.
 """
 def __init__(self,library,masses,cards,seed=11,particles=512,recovery_trials=None):
  self.library={k:tuple(v) for k,v in library.items()};self.prior=dict(masses);self.rng=random.Random(seed);self.seed=seed;self.n=particles;self.trials=recovery_trials or (256 if particles<=512 else 1024);self.compiler=Compiler(cards);self.facts=Facts(cards);self.log_mass={k:math.log(v) for k,v in masses.items()};self.base=dict(self.log_mass);self.pop={};self.history=[];self.completed=[];self.game_evidence={};self.unresolved={};self.semantic_failure=None;self.degraded=None;self.recoveries=[];self.timings=[];self.operations=0;self.game=None;self.callbacks=0;self.retry_count=0;self.last_retry=-100;self.budget=dict(remaining=2000000*max(1,particles//512));self.transitions=[]
 def masses(self):
  z=logsumexp(list(self.log_mass.values()));weights={k:math.exp(max(-700,v-z)) for k,v in self.log_mass.items()};total=sum(weights.values());return {k:v/total for k,v in weights.items()}
 def new_game(self,g):
  if self.game is not None:
   previous=len(self.completed)
   for k in self.log_mass:
    if k not in self.unresolved and not self.semantic_failure:self.game_evidence[(previous,k)]=self.log_mass[k]-self.base[k]
   self.completed.append(self.history)
  self.game=g;self.history=[];self.retry_count=0;self.last_retry=-100
  a=allocate(self.masses(),self.n)
  self.pop={k:dict(states=[State(self.library[k],self.rng) for _ in range(n)],weights=[1/n]*n) for k,n in a.items() if k not in self.unresolved};self.base=dict(self.log_mass)
 def prune_composition(self):
  for k in list(self.log_mass):
   if not self.facts.compatible(self.library[k]):self.log_mass.pop(k);self.pop.pop(k,None);self.unresolved.pop(k,None)
  if not self.log_mass:raise OutOfLibrary('No library composition satisfies legal copy facts')
 def resample(self):
  if self.unresolved:return
  q=self.masses();ess=1/sum((q[k]*w)**2 for k,p in self.pop.items() for w in p['weights'])
  if ess>=self.n/2:return
  for k,n in allocate(q,self.n).items():
   p=self.pop[k];self.pop[k]=dict(states=[s.clone() for s in self.rng.choices(p['states'],weights=p['weights'],k=n)],weights=[1/n]*n)
 def recover(self,k,whole=False):
  started=time.perf_counter();base=math.log(self.prior[k]);past=[]
  try:
   for g,hist in enumerate(self.completed):
    if (g,k) not in self.game_evidence:
     r=reconstruct(self.library[k],hist,self.rng,self.trials,self.budget);self.game_evidence[(g,k)]=r['log_evidence'];past.append(dict(game_index=g+1,log_evidence=r['log_evidence']))
    base+=self.game_evidence[(g,k)]
   r=reconstruct(self.library[k],self.history,self.rng,self.trials,self.budget);self.base[k]=base;self.log_mass[k]=base+r['log_evidence'];n=allocate(self.masses(),self.n)[k]
   # Allocation changes are proposal effort only. Limit the current stratum
   # to its old slots until a joint resample can rebuild the full allocation.
   n=min(n,max(4,self.n-sum(len(p['states']) for h,p in self.pop.items() if h!=k)))
   self.pop[k]=dict(states=[s.clone() for s in self.rng.choices(r['states'],weights=r['weights'],k=n)],weights=[1/n]*n);self.unresolved.pop(k,None)
   self.recoveries.append(dict(hypothesis=k,game=self.game,operation=self.operations,success=True,whole_match_evidence=True,completed_game_evidence=past,elapsed_seconds=time.perf_counter()-started,**{a:b for a,b in r.items() if a not in ('states','weights')}));return True
  except Depletion as e:
   self.recoveries.append(dict(hypothesis=k,game=self.game,operation=self.operations,success=False,whole_match_evidence=True,detail=str(e),elapsed_seconds=time.perf_counter()-started));return False
 def status(self):
  if self.semantic_failure:self.degraded=self.semantic_failure
  elif self.unresolved:self.degraded=dict(kind='FINITE_PARTICLE_DEPLETION',hypotheses=sorted(self.unresolved),healthy_hypotheses=sorted(self.pop),recovery_steps_remaining=self.budget['remaining'])
  else:self.degraded=None
  return dict(status='FALLBACK' if self.degraded else 'SUPPORTED',game=self.game,reason=self.degraded)
 def observe(self,raw):
  started=time.perf_counter();self.callbacks+=1;was_degraded=bool(self.degraded)
  try:
   self.facts.observe(raw);g=self.facts.game
   if g!=self.game:self.new_game(g)
   self.prune_composition()
   if self.semantic_failure:return self.status()
   try:compiled=self.compiler.compile(raw)
   except UnsupportedEvent as exc:
    self.semantic_failure=dict(kind='UNSUPPORTED_SEMANTICS',game=g,message=str(exc),callback=self.callbacks);return self.status()
   if compiled['duplicate']:return dict(status='DUPLICATE',game=g)
   for op in compiled['operations']:
    self.history.append(op);self.operations+=1;before=self.masses();current=not bool(self.unresolved)
    for k,p in list(self.pop.items()):
     if k in self.unresolved:continue
     states=[];weights=[]
     for state,w in zip(p['states'],p['weights']):
      try:factor=state.apply(op,self.rng);states.append(state);weights.append(w*factor)
      except IncompatibleAllocation:pass
     z=sum(weights)
     if z>0:
      self.log_mass[k]+=math.log(z);weights=[w/z for w in weights]
      if len(states)<len(p['states']):states=[s.clone() for s in self.rng.choices(states,weights=weights,k=len(p['states']))];weights=[1/len(states)]*len(states)
      self.pop[k]=dict(states=states,weights=weights)
     elif not self.recover(k):
      reason=dict(kind='FINITE_PARTICLE_DEPLETION',hypothesis=k,game=g,operation=self.operations,operation_event=op,pre_event_masses=before,masses_current=current,callback=self.callbacks)
      self.unresolved[k]=reason;self.pop.pop(k,None);self.transitions.append(dict(event='STRATUM_DEPLETED',**reason))
    self.resample()
   # Deterministic, bounded retry schedule. No truth, outcomes, scheduled deck
   # or empirical strength enters it. Recompute evidence from game-start/past
   # cache, never multiply an old posterior by the same history twice.
   if self.unresolved and self.retry_count<4 and self.callbacks-self.last_retry>=16 and self.budget['remaining']>0:
    self.retry_count+=1;self.last_retry=self.callbacks
    for k in sorted(self.unresolved):self.recover(k,True)
    if not self.unresolved:self.resample()
   result=self.status()
   if was_degraded and result['status']=='SUPPORTED':self.transitions.append(dict(event='FULL_READOUT_RESUMED',game=g,callback=self.callbacks,method='all compatible strata have complete-prefix evidence; no prior reset'))
   return result
  finally:self.timings.append(time.perf_counter()-started)
 def predict(self,queries):
  if self.degraded:raise Depletion('Full aggregate unavailable while any compatible stratum is unresolved')
  q=self.masses();known=self.facts.known_hand();out={c:dict(p_unknown_hand=0.,hand_unknown_mean=0.,deck_mean=0.,prize_mean=0.) for c in queries}
  for k,p in self.pop.items():
   for state,w in zip(p['states'],p['weights']):
    h,d,pr=state.counts('H'),state.counts('D'),state.counts('P');w*=q[k]
    for c in queries:
     v=out[c];v['p_unknown_hand']+=w*(h[c]>known[c]);v['hand_unknown_mean']+=w*(h[c]-known[c]);v['deck_mean']+=w*d[c];v['prize_mean']+=w*pr[c]
  return out
