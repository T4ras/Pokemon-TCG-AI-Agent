"""Smoothed, tempered naive Bayes over certified match reveal lower bounds."""
from .common import *
import numpy as np

GRID=tuple((alpha,beta) for beta in (.25,.5,1.) for alpha in (120.,30.))
def catalogue():
 p=prior();decks=sorted(p);cards=sorted({c for v in p.values() for c in v['cards']});counts=np.array([[collections.Counter(p[d]['cards'])[c] for c in cards] for d in decks],dtype=float);prevalence=np.array([p[d]['prior'] for d in decks],dtype=float);assert np.all(counts.sum(axis=1)==60) and abs(prevalence.sum()-1)<1e-10
 return decks,cards,counts,prevalence

def fit_theta(revealed,labels,deck_counts,alpha):
 revealed=np.asarray(revealed,dtype=float);labels=np.asarray(labels,dtype=int);assert revealed.shape[0]==len(labels) and np.all(revealed>=0);counts=np.zeros_like(deck_counts,dtype=float)
 for x,y in zip(revealed,labels):
  if np.any(x>deck_counts[y]):raise InvariantFailure('Training reveal ledger exceeds labelled original deck')
  counts[y]+=x
 theta=(counts+float(alpha)*deck_counts/60.)/(counts.sum(axis=1,keepdims=True)+float(alpha));assert np.allclose(theta.sum(axis=1),1)
 return theta

def deck_weights(revealed,compatible,prevalence,theta=None,beta=0.):
 x=np.asarray(revealed,dtype=float);mask=np.asarray(compatible,dtype=bool).copy();prior=np.asarray(prevalence,dtype=float);logw=np.log(prior)
 if beta:
  if theta is None:raise ValueError('Nonzero beta requires fitted theta')
  observed=x>0;mask&=np.all(theta[:,observed]>0,axis=1)
  # No0*log0; unobserved absent card identities contribute no evidence.
  if observed.any():logw=logw+float(beta)*(np.log(np.maximum(theta[:,observed],np.finfo(float).tiny))@x[observed])
 if not mask.any():raise OutOfLibrary('No conditional allocation/deck satisfies legal reveal facts')
 logw=np.where(mask,logw,-np.inf);peak=np.max(logw);w=np.exp(logw-peak);return w/w.sum()

def mix_allocations(weights,allocations):
 """Fields: p_unknown, unknown-H mean,D mean,P mean,H variance,D variance,P variance."""
 w=np.asarray(weights,dtype=float);a=np.asarray(allocations,dtype=float);mean=np.einsum('d,dqf->qf',w,a[:,:,:4]);out=np.empty((a.shape[1],7));out[:,:4]=mean
 for z in range(3):
  mu=a[:,:,z+1];second=a[:,:,z+4]+mu*mu;out[:,z+4]=np.maximum(0,np.einsum('d,dq->q',w,second)-mean[:,z+1]**2)
 return out
