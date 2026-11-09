"""Exact marginal count distributions under unchanged baseline A assumptions."""
from .common import *
from .baseline import hard_layout,choose
import numpy as np,copy

def hypergeom(K,C,n,S):
 out=np.zeros(S)
 if not K:out[0]=1.;return out
 den=choose(K,n)
 if not den:raise IncompatibleAllocation('Impossible zone capacity')
 for k in range(min(C,n,S-1)+1):out[k]=choose(C,k)*choose(K-C,n-k)/den
 return out

def component(counts,where,basic,queries,S,no_basic=False):
 K=sum(counts.values());B=sum(n for c,n in counts.items() if c in basic);h,d,p=where[1],where[0],where[2];out=np.zeros((len(queries),3,S))
 if not K:out[:,:,0]=1.;return out
 Q=choose(K-B,h)/choose(K,h)
 if no_basic and not Q:out[:,:,0]=1.;return out
 for i,c in enumerate(queries):
  C=counts[c]
  if not no_basic:
   for z,n in enumerate((h,d,p)):out[i,z]=hypergeom(K,C,n,S)
  else:
   hand=np.zeros(S);hand[0]=1.
   if c not in basic:hand=hypergeom(K-B,C,h,S)
   out[i,0]=hand
   for z,n in ((1,d),(2,p)):
    for k,v in enumerate(hand):
     if v:out[i,z]+=v*hypergeom(K-h,C-k,n,S)
 return out

def ordinary(facts,cards,queries,S,cache):
 if not facts.order_supported:raise UnsupportedEvent('Conditional order unsupported')
 fixed,groups,parts,known=hard_layout(facts);used=collections.Counter(c for c,d in facts.copies.values());total=collections.Counter(cards)
 if used-total:raise IncompatibleAllocation('Known copies exceed deck')
 rest=total-used;N=sum(rest.values());F=1<<(2*S-2).bit_length();acc=np.zeros((len(queries),3,F//2+1),complex);normalizer=0.;known_basic=any(c in facts.basic and n for c,n in fixed['H'].items());condition=facts.setup_basic
 for weight,slots,alloc in parts:
  if sum(slots)!=N:raise InvariantFailure('Residual count conservation')
  raw=np.ones_like(acc);no=np.ones_like(acc);Q=0. if known_basic else 1.
  for counts,where in [(x[1],w) for x,w in zip(groups,alloc)]+[(rest,slots)]:
   K=sum(counts.values());B=sum(n for c,n in counts.items() if c in facts.basic);h=where[1];key=(tuple(sorted(counts.items())),where,condition is not None)
   if key not in cache:
    fft=np.fft.rfft(component(counts,where,facts.basic,queries,S),n=F,axis=-1)
    nft=np.fft.rfft(component(counts,where,facts.basic,queries,S,True),n=F,axis=-1) if condition is not None else None
    cache[key]=(fft,nft,choose(K-B,h)/choose(K,h) if K else 1.)
   fft,nft,q=cache[key];raw*=fft
   if condition is not None:no*=nft;Q*=q
  if condition is None:Z=1.;result=raw
  elif condition:Z=1-Q;result=raw-Q*no
  else:Z=Q;result=Q*no
  normalizer+=weight*Z;acc+=weight*result
 if normalizer<=0:raise IncompatibleAllocation('Zero Basic/location support')
 raw=np.fft.irfft(acc/normalizer,n=F,axis=-1);out=np.zeros((len(queries),3,S))
 for i,c in enumerate(queries):
  for z,shift in enumerate((fixed['H'][c]-known[c],fixed['D'][c],fixed['P'][c])):
   for k,v in enumerate(raw[i,z]):
    at=k+shift
    if 0<=at<S:out[i,z,at]+=v
    elif abs(v)>1e-8:raise InvariantFailure('Count support overflow')
 if out.min()<-1e-8:raise InvariantFailure('Negative conditional mass')
 out=np.maximum(out,0.);sums=out.sum(axis=-1,keepdims=True)
 if not np.allclose(sums,1,atol=1e-7):raise InvariantFailure('Marginal count mass lost')
 return out/sums

def allocation(facts,cards,queries,S,cache=None):
 cache={} if cache is None else cache
 if not facts.unknown_basic:return ordinary(facts,cards,queries,S,cache)
 used=collections.Counter(c for c,d in facts.copies.values());rest=collections.Counter(cards)-used;types=[c for c in sorted(rest) if c in facts.basic];K=facts.unknown_basic;den=choose(sum(rest[c] for c in types),K);result=np.zeros((len(queries),3,S));nodes=0
 if not den:raise IncompatibleAllocation('Concealed Basic count exceeds list')
 def visit(i,left,picked,w):
  nonlocal nodes,result
  nodes+=1
  if nodes>4096:raise UnsupportedEvent('Concealed Basic4096-node cap')
  if i==len(types):
   if not left:
    f=copy.copy(facts);f.unknown_basic=0;remaining=list((collections.Counter(cards)-collections.Counter(picked)).elements());result+=w/den*ordinary(f,remaining,queries,S,cache)
   return
  c=types[i]
  for n in range(min(rest[c],left)+1):visit(i+1,left-n,picked+[c]*n,w*choose(rest[c],n))
 visit(0,K,[],1);return result

def moments(pmf):
 k=np.arange(pmf.shape[-1]);mu=pmf@k;var=pmf@(k*k)-mu*mu
 return np.column_stack((1-pmf[:,0,0],mu,np.maximum(var,0.)))

def mask_counts(prob,lower,upper):
 k=np.arange(prob.shape[-1]);mask=(k>=np.asarray(lower)[...,None])&(k<=np.asarray(upper)[...,None]);p=np.maximum(prob,0)*mask;den=p.sum(-1,keepdims=True)
 if np.any(den<=0):raise InvariantFailure('No count mass in legal support')
 return p/den
