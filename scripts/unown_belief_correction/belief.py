"""Incremental legal-only encoding, copied readout math from frozen features.encode."""
from .common import *
from unown_belief_full_training.features import sketch,public,tag,support
from unown_belief_full_training.distributions import allocation,moments,mask_counts
from unown_belief_full_training.baseline import Facts,hard_layout
from unown_belief_full_training.allocation import allocation_one
from unown_belief_bayes_repair.adapter import legal
from unown_belief_bayes_repair.deck import catalogue,deck_weights,mix_allocations
from unown_belief_bayes_repair.cache import bounded
from unown_belief_full_training.models import make_neural
from unown_belief_full_training.fit import predict
import numpy as np,torch,collections

def current_features(f,o,within):
 decks,q,counts,pop=catalogue();S=int(counts.max())+1;g=o['current']['round'];index=0
 zone=[collections.Counter(c for c,d in f.copies.values() if z in d and len(d)==1) for z in ('H','D','P','B')]
 hashes=np.concatenate([sketch(f.lower)]+[sketch(z) for z in zone]+[sketch(within)])
 card_low=[];card_high=[];reference_A=[];uncertain=[];globals_=[];locals_=[];A=[];lower=[];upper=[];deckp=[];ledgers=[];knowns=[];selected=[];games=[];zone_sizes=[];points=[]
 known=f.known_hand();supported=[];pmfs=[];references=[];cache={}
 def readout():
  for d in range(len(decks)):
   if not f.compatible(prior()[decks[d]]['cards']):continue
   try:
    value=allocation(f,prior()[decks[d]]['cards'],q,S,cache)
    reference=allocation_one(f,prior()[decks[d]]['cards'],q)
   except IncompatibleAllocation:continue
   actual=moments(value);expected=np.array([reference[c] for c in q]);np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
   supported.append(d);pmfs.append(value);references.append(expected)
 bounded(readout,10)
 if not supported:raise OutOfLibrary('No compatible per-deck allocation')
 w=deck_weights([f.lower.get(c,0) for c in q],np.isin(np.arange(37),supported),pop,beta=0)
 a=np.einsum('d,dqzs->qzs',w[supported],np.asarray(pmfs));am=mix_allocations(w[supported],np.array(references));lo,hi=support(f,counts,np.asarray(supported),q,known)
 k=np.arange(S);outside=(k[None,None,:]<lo[:,:,None])|(k[None,None,:]>hi[:,:,None])
 if np.any(a[outside]>1e-6):raise InvariantFailure('Baseline mass outside legal count bounds')
 a[outside]=0.;a/=a.sum(-1,keepdims=True)
 local=np.column_stack((np.array([tag(i) for i in range(len(q))]),am/np.array([1,60,60,60,3600,3600,3600]),np.array([known[c] for c in q])/60,lo/60,hi/60,counts.max(0)/60,a.reshape((len(q),-1))))
 fixed=hard_layout(f)[0];card_low.append([known[c]+fixed['D'][c]+fixed['P'][c] for c in q]);card_high.append([int(counts[supported,i].max())-sum(cid==c and not domain.intersection({'H','D','P'}) for cid,domain in f.copies.values()) for i,c in enumerate(q)]);reference_A.append(am);uncertain.append((am[:,0]>=.1)&(am[:,0]<=.9))
 globals_.append(np.concatenate((hashes,public(o),w)));locals_.append(local);A.append(a);lower.append(lo);upper.append(hi);deckp.append(w);ledgers.append([f.lower.get(c,0) for c in q]);knowns.append([known[c] for c in q]);selected.append(index);games.append(g);zone_sizes.append([f.sizes[z] for z in ('H','D','P')]);points.append(dict(index=index,key=[g,o['current']['turn'],o['current']['turnActionCount'],o['current']['yourIndex']],game=g,stage='setup' if o['current']['turn']==0 else 'early' if o['current']['turn']<=4 else 'middle' if o['current']['turn']<=12 else 'late',known_serials=sorted(s for s,(c,d) in f.copies.items() if d=={'H'})))
 return dict(global_X=np.asarray(globals_,np.float32),local_X=np.asarray(locals_,np.float32),A=np.asarray(A,np.float32),lower=np.asarray(lower,np.uint8),upper=np.asarray(upper,np.uint8),deck_A=np.asarray(deckp,np.float32),ledger=np.asarray(ledgers,np.uint8),known=np.asarray(knowns,np.uint8),games=np.asarray(games,np.uint8),zone_sizes=np.asarray(zone_sizes,np.uint8),A_reference=np.asarray(reference_A),uncertain=np.asarray(uncertain),card_total_lower=np.asarray(card_low),card_total_upper=np.asarray(card_high))

def vector_from_predictions(f,deck,p):
 _,queries,counts,_=catalogue();scale=np.maximum(counts.max(0),1);prob=1-p[0,...,0];k=np.arange(p.shape[-1]);mu=p[0]@k;var=np.maximum(0,p[0]@(k*k)-mu*mu);normalized_var=var/scale[:,None]**2;channels=np.stack([prob,mu/scale[:,None],normalized_var],axis=-1).reshape(-1);known=f['known'][0]/scale;size=f['zone_sizes'][0]/np.array([60.,60.,6.]);dp=deck[0];entropy=-np.sum(dp*np.log(np.maximum(dp,1e-30)))/np.log(37);bp=np.clip(prob.astype(np.float64),1e-12,1-1e-12);binary=-(bp*np.log(bp)+(1-bp)*np.log(1-bp)).mean(0)/np.log(2);summary=np.r_[entropy,dp.max(),np.count_nonzero(f['deck_A'][0])/37,binary,normalized_var.mean(0)];out=np.r_[dp,channels,known,size,summary].astype(np.float32)
 if out.shape!=(BELIEF_DIM,) or not np.isfinite(out).all():raise InvariantFailure('Invalid1479dimensionalbeliefvector')
 out.flags.writeable=False;return out

_MODEL=None
def frozen_belief_model():
 global _MODEL
 if _MODEL is None:
  info=read(OUT/'frozen-components.json')['belief_model'];path=R/info['path'];assert sha(path)==info['sha256'];data=torch.load(path,map_location='cpu',weights_only=False);assert data['updates']==14004 and data['job']['seed']==11 and data['job']['arm']=='population';_MODEL=make_neural('MLP',11);_MODEL.load_state_dict(data['model']);_MODEL.eval()
  for p in _MODEL.parameters():p.requires_grad_(False)
 return _MODEL

class OnlineBelief:
 def __init__(self,check=guard):self.check=check;self.reset_match();self.costs=collections.Counter()
 def reset_match(self):
  self.facts=Facts(metadata()['cards']);self.game=None;self.within=collections.Counter();self.current=None;self.cached=None
 def observe(self,raw):
  self.check();start=time.process_time()
  if not raw.get('select'):self.reset_match();return
  o=legal(raw);g=o['current']['round']
  if g!=self.game:self.within=collections.Counter();self.game=g
  changed=bounded(lambda:self.facts.observe(o),30)
  if changed:
   for e in o.get('logs') or []:
    if e.get('cardId') is not None:self.within[e['cardId']]+=1
  self.current=o;self.cached=None;self.costs['observer_CPU_seconds']+=time.process_time()-start;self.costs['callbacks']+=1
 def readout(self):
  self.check()
  if self.current is None:raise InvariantFailure('Beliefreadoutwithoutlegalobservation')
  if self.cached is not None:return self.cached
  start=time.process_time();f=current_features(self.facts,self.current,self.within);self.costs['A_CPU_seconds']+=time.process_time()-start;start=time.process_time();deck,p=predict(frozen_belief_model(),f,'MLP');out=vector_from_predictions(f,deck,p);self.costs['MLP_CPU_seconds']+=time.process_time()-start;self.costs['readouts']+=1;self.cached=out;return out
