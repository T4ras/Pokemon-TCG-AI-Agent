"""Stratified SMC with fixed-cost whole-history guided importance recovery."""
from unown_belief_native_common import *
from unown_belief_native_state import State,guided_history
from unown_belief_native_adapter import Compiler
from unown_belief_native_baseline import Facts
import random

def allocate(masses,n):
 keys=sorted(k for k,p in masses.items() if p>0);remain=n-4*len(keys)
 if remain<0:raise ValueError('Four-particle list floor exceeds budget')
 total=sum(masses.values());quota={k:remain*masses[k]/total for k in keys};a={k:4+math.floor(quota[k]) for k in keys}
 for k in sorted(keys,key=lambda k:(-(quota[k]-math.floor(quota[k])),k))[:n-sum(a.values())]:a[k]+=1
 return a

def reconstruct(cards,history,rng,trials):
 """Fixed-size guided SMC replay, including its incremental evidence factors.

 Resampling is internal to one list and carries its current evidence. This
 avoids exponential rejection of complete initial deals. The resulting
 normalized posterior remains a finite-particle approximation.
 """
 start=time.process_time();states=[State(cards,rng) for _ in range(trials)];weights=[1/trials]*trials;failures=collections.Counter();guided=guided_history(history);lz=0.;resamples=0
 for index,op in enumerate(guided):
  kept=[];w=[]
  for state,weight in zip(states,weights):
   try:f=state.apply(op,rng);kept.append(state);w.append(weight*f)
   except IncompatibleAllocation as e:failures[str(e)]+=1
  z=sum(w)
  if not z>0:raise Depletion(json.dumps(dict(trials=trials,accepted=0,at_operation=index,operation=op,failures=failures,CPU_seconds=time.process_time()-start)))
  lz+=math.log(z);states=kept;weights=[v/z for v in w]
  if 1/sum(v*v for v in weights)<trials/2:
   states=[v.clone() for v in rng.choices(states,weights=weights,k=trials)];weights=[1/trials]*trials;resamples+=1
 return dict(states=states,weights=weights,log_evidence=lz,trials=trials,accepted=len(states),CPU_seconds=time.process_time()-start,failures=dict(failures),guided_operations=len(guided),resamples=resamples)

class Observer:
 def __init__(self,library,masses,cards,seed=11,particles=512,recovery_trials=None):
  self.library={k:tuple(v) for k,v in library.items()};self.prior=dict(masses);self.rng=random.Random(seed);self.seed=seed;self.n=particles;self.trials=recovery_trials or (256 if particles==512 else 1024);self.compiler=Compiler(cards);self.facts=Facts(cards);self.log_mass={k:math.log(p) for k,p in masses.items() if p>0};self.base=dict(self.log_mass);self.pop={};self.history=[];self.degraded=None;self.recoveries=[];self.timings=[];self.operations=0;self.game=None
 def masses(self):
  z=logsumexp(list(self.log_mass.values()));return {k:math.exp(v-z) for k,v in self.log_mass.items()}
 def fresh(self):
  a=allocate(self.masses(),self.n);self.pop={k:dict(states=[State(self.library[k],self.rng) for _ in range(n)],weights=[1/n]*n) for k,n in a.items()};self.base=dict(self.log_mass);self.history=[]
 def prune_composition(self):
  bad=[k for k in self.log_mass if not self.facts.compatible(self.library[k])]
  for k in bad:self.log_mass.pop(k);self.pop.pop(k,None)
  if not self.log_mass:raise OutOfLibrary('No library composition satisfies legal per-game copy lower bounds')
 def resample(self):
  masses=self.masses();ess=1/sum((masses[k]*w)**2 for k,p in self.pop.items() for w in p['weights'])
  if ess>=self.n/2:return
  for k,n in allocate(masses,self.n).items():
   p=self.pop[k];chosen=self.rng.choices(p['states'],weights=p['weights'],k=n);self.pop[k]=dict(states=[s.clone() for s in chosen],weights=[1/n]*n)
 def observe(self,raw):
  t=time.perf_counter();result={};oldgame=self.compiler.game
  try:
   self.facts.observe(raw)
   if self.degraded:return dict(status='FALLBACK',reason=self.degraded,game=self.facts.game)
   result=self.compiler.compile(raw)
   if result['duplicate']:return dict(status='DUPLICATE',game=self.game)
   self.prune_composition()
   if result['new_game']:
    self.game=result['game']
    if not self.degraded:self.fresh()
   if self.degraded:return dict(status='FALLBACK',reason=self.degraded,game=self.game)
   for op in result['operations']:
    self.history.append(op);self.operations+=1
    for k,p in list(self.pop.items()):
     states=[];weights=[]
     for s,w in zip(p['states'],p['weights']):
      try:f=s.apply(op,self.rng);states.append(s);weights.append(w*f)
      except IncompatibleAllocation:pass
     z=sum(weights)
     if z>0:
      self.log_mass[k]+=math.log(z);weights=[w/z for w in weights]
      if len(states)<len(p['states']):
       states=[v.clone() for v in self.rng.choices(states,weights=weights,k=len(p['states']))];weights=[1/len(states)]*len(states)
      self.pop[k]=dict(states=states,weights=weights)
     else:
      started=time.perf_counter()
      try:
       r=reconstruct(self.library[k],self.history,self.rng,self.trials);self.log_mass[k]=self.base[k]+r['log_evidence'];self.pop[k]=dict(states=[v.clone() for v in self.rng.choices(r['states'],weights=r['weights'],k=len(p['states']))],weights=[1/len(p['states'])]*len(p['states']));self.recoveries.append(dict(hypothesis=k,game=self.game,operation=self.operations,elapsed_seconds=time.perf_counter()-started,**{q:v for q,v in r.items() if q not in ('states','weights')}))
      except Depletion as e:
       self.degraded=dict(kind='FINITE_PARTICLE_DEPLETION',hypothesis=k,game=self.game,operation=self.operations,detail=str(e));self.recoveries.append(dict(self.degraded,elapsed_seconds=time.perf_counter()-started));return dict(status='FALLBACK',reason=self.degraded,game=self.game)
    self.resample()
   return dict(status='SUPPORTED',game=self.game)
  except UnsupportedEvent as e:
   self.degraded=dict(kind='UNSUPPORTED_SEMANTICS',message=str(e),game=self.compiler.game);return dict(status='FALLBACK',reason=self.degraded,game=self.compiler.game)
  finally:self.timings.append(time.perf_counter()-t)
 def predict(self,queries):
  if self.degraded:raise Depletion('Full posterior unavailable; call the explicitly labeled baseline fallback')
  q=self.masses();known=self.facts.known_hand();out={c:dict(p_unknown_hand=0.,hand_unknown_mean=0.,deck_mean=0.,prize_mean=0.) for c in queries}
  for k,p in self.pop.items():
   for s,w in zip(p['states'],p['weights']):
    h,d,pr=s.counts('H'),s.counts('D'),s.counts('P');w*=q[k]
    for c in queries:
     v=out[c];v['p_unknown_hand']+=w*(h[c]>known[c]);v['hand_unknown_mean']+=w*(h[c]-known[c]);v['deck_mean']+=w*d[c];v['prize_mean']+=w*pr[c]
  return out
