"""Bounded exact integration of one anonymous initial-copy assignment bit.

An outer state samples the initial permutation modulo a fixed transposition.
The two orientations are summed exactly with their own complete-history weights.
Subsequent kernel randomness is sampled and coupled, not integrated out.
"""
from .common import *
from .state import State as ExplicitState
import random,copy

class CountedRandom(random.Random):
 def __init__(self,state):
  super().__init__(0);self.setstate(state);self.words=0
 def random(self):self.words+=2;return super().random()
 def getrandbits(self,k):self.words+=(k+31)//32;return super().getrandbits(k)

class CollapsedState:
 integration_CPU=0.;branch_steps=0;instances=0
 def __init__(self,cards,rng):
  a=ExplicitState(cards,rng);counts=collections.Counter(cards);types=sorted(counts,key=lambda c:(-counts[c],c));self.cards=a.cards
  if len(types)<2:self.atoms=[a];self.weights=[1.];return
  x=cards.index(types[0]);y=cards.index(types[1]);ix=a.z['D'].index(x);iy=a.z['D'].index(y)
  # Uniform quotient of the full permutation by this transposition.
  if ix>iy:a.z['D'][ix],a.z['D'][iy]=a.z['D'][iy],a.z['D'][ix];ix,iy=iy,ix
  b=a.clone();b.z['D'][ix],b.z['D'][iy]=b.z['D'][iy],b.z['D'][ix]
  self.atoms=[a,b];self.weights=[.5,.5];type(self).instances+=1
 @classmethod
 def from_atoms(cls,atoms,weights):
  if not 1<=len(atoms)<=2:raise ValueError('Exact component bound2')
  if any(w<0 for w in weights) or sum(weights)<=0:raise ValueError('Invalid component weights')
  r=object.__new__(cls);r.cards=atoms[0].cards;r.atoms=[s.clone() for s in atoms];r.weights=[w/sum(weights) for w in weights];return r
 def clone(self):return type(self).from_atoms(self.atoms,self.weights)
 def check(self):
  assert len(self.atoms)<=2 and abs(sum(self.weights)-1)<1e-10
  for s in self.atoms:s.check()
 def apply(self,op,rng):
  start=time.process_time();base=rng.getstate();furthest=None;atoms=[];weights=[]
  # Same underlying MT word tape for both conditional orientations. Advance
  # the outer RNG past ALL consumed words, including rejected branches.
  try:
   for s,w in zip(self.atoms,self.weights):
    stream=CountedRandom(base)
    try:
     factor=s.apply(op,stream)
     if factor>0:atoms.append(s);weights.append(w*factor)
    except IncompatibleAllocation:pass
    finally:
     type(self).branch_steps+=1
     if furthest is None or stream.words>furthest.words:furthest=stream
   if furthest is not None:rng.setstate(furthest.getstate())
   z=sum(weights)
   if not z>0:raise IncompatibleAllocation('Both exact assignment orientations contradicted')
   self.atoms=atoms;self.weights=[w/z for w in weights];self.check();return z
  finally:type(self).integration_CPU+=time.process_time()-start
 def materialize(self,rng):
  # Sampling the exact conditional mixture retains all serial/order/history.
  return rng.choices(self.atoms,weights=self.weights,k=1)[0].clone()
 def moments(self,zone,cid,known=0):
  values=[s.counts(zone)[cid] for s in self.atoms];mean=sum(w*x for w,x in zip(self.weights,values));second=sum(w*x*x for w,x in zip(self.weights,values))
  return dict(mean=mean,second=second,variance=max(0,second-mean*mean),presence=sum(w*(x>known) for w,x in zip(self.weights,values)))
