"""Baseline A conditional allocation, extended with exact marginal second moments."""
from .common import *
from .baseline import hard_layout,choose,cdf_components
import copy

def hg(K,C,n):
 if K<=0:return 0.,0.
 mu=n*C/K;second=mu+(n*(n-1)*C*(C-1)/(K*(K-1)) if K>1 else 0.)
 return mu,second

def component_moments(counts,where,K,cid,basic,no_basic=False):
 if K==0:return [0.]*3,[0.]*3
 C=counts[cid];B=sum(v for c,v in counts.items() if c in basic);h,d,p=where[1],where[0],where[2]
 if not no_basic:
  pairs=[hg(K,C,n) for n in (h,d,p)];return [x[0] for x in pairs],[x[1] for x in pairs]
 mh,hh=(0.,0.) if cid in basic else hg(K-B,C,h);M=K-h;remaining=C-mh;factorial=C*(C-1)-(2*C-1)*mh+hh;means=[mh];seconds=[hh]
 for n in (d,p):
  mu=n*remaining/M if M else 0.;second=mu+(n*(n-1)*factorial/(M*(M-1)) if M>1 else 0.);means.append(mu);seconds.append(second)
 return means,seconds

def ordinary(facts,cards,queries):
 """Same exact mixture as the scalar reference, caching each component marginal.

 Query-independent layout enumeration is unchanged. A component's marginal
 depends only on its card counts and zone sizes, so reuse it across layouts
 and evaluate the fixed query vector together. No probabilities are pruned.
 """
 import numpy as np
 if not facts.order_supported:raise UnsupportedEvent('Baseline abstains on unresolved conditional known order')
 fixed,groups,parts,known=hard_layout(facts);used=collections.Counter(c for c,d in facts.copies.values());total=collections.Counter(cards)
 if used-total:raise IncompatibleAllocation('Public physical copies exceed list composition')
 rest=total-used;N=sum(rest.values());queries=list(queries);size=len(queries)
 known_basic=any(c in facts.basic and n for c,n in fixed['H'].items());condition=facts.setup_basic
 basic=np.array([c in facts.basic for c in queries]);excess=np.array([known[c]-fixed['H'][c] for c in queries],dtype=float)
 shifts=np.array([[-excess[i],fixed['D'][c],fixed['P'][c]] for i,c in enumerate(queries)],dtype=float).reshape((size,3))
 extra=np.flatnonzero(excess);descriptors=[]
 for counts in [x[1] for x in groups]+[rest]:
  descriptors.append((counts,sum(counts.values()),sum(v for c,v in counts.items() if c in facts.basic),np.array([counts[c] for c in queries],dtype=float)))
 cache={}
 def marginal(j,where):
  key=(j,where)
  if key in cache:return cache[key]
  counts,K,B,C=descriptors[j];h,d,p=where[1],where[0],where[2];ns=np.array([h,d,p],dtype=float)
  if not K:
   zero=np.zeros((size,3));answer=(np.ones(size),1.,np.ones(size),zero,zero,zero,zero);cache[key]=answer;return answer
  mean=C[:,None]*ns[None,:]/K
  second=mean+(C*(C-1))[:,None]*(ns*(ns-1))[None,:]/(K*(K-1)) if K>1 else mean.copy()
  variance=second-mean*mean;den=choose(K,h)
  none=np.array([choose(K-int(c),h)/den for c in C])
  Q=choose(K-B,h)/den
  both=np.array([choose(K-B-(0 if is_basic else int(c)),h)/den for c,is_basic in zip(C,basic)]) if condition is not None else None
  no=novar=None
  if condition is not None:
   # Impossible no-Basic events have zero weight; avoid undefined moments.
   no=np.zeros((size,3));novar=np.zeros((size,3))
   if Q:
    mh=np.where(basic,0.,h*C/(K-B)) if K>B else np.zeros(size)
    hh=mh+np.where(basic,0.,h*(h-1)*C*(C-1)/((K-B)*(K-B-1))) if K-B>1 else mh.copy()
    M=K-h;remaining=C-mh;factorial=C*(C-1)-(2*C-1)*mh+hh
    no[:,0]=mh;no[:,1:]=remaining[:,None]*ns[None,1:]/M if M else 0.
    ss=no.copy();ss[:,0]=hh
    if M>1:ss[:,1:]+=factorial[:,None]*(ns[1:]*(ns[1:]-1))[None,:]/(M*(M-1))
    novar=ss-no*no
  answer=(none,Q,both,mean,variance,no,novar);cache[key]=answer;return answer
 normalizer=0.;numprob=np.zeros(size);mu_sum=np.zeros((size,3));second_sum=np.zeros((size,3))
 for weight,slots,alloc in parts:
  if sum(slots)!=N:raise InvariantFailure('Baseline residual physical-copy conservation')
  raw=np.zeros((size,3));rawvar=np.zeros((size,3));no=np.zeros((size,3));novar=np.zeros((size,3));none=np.ones(size);both=np.ones(size);Q=0. if known_basic else 1.
  locations=list(alloc)+[slots]
  for j,where in enumerate(locations):
   cn,cq,cb,cm,cv,cz,czv=marginal(j,where);none*=cn;raw+=cm;rawvar+=cv
   if condition is not None:Q*=cq;both*=cb;no+=cz;novar+=czv
  if known_basic:both.fill(0.)
  if len(extra):
   components=[(counts,where,K) for (counts,K,B,C),where in zip(descriptors,locations)]
   for i in extra:
    none[i]=cdf_components(components,queries[i],int(excess[i]),facts.basic)
    if condition is not None:both[i]=Q*cdf_components(components,queries[i],int(excess[i]),facts.basic,True)
  rawsecond=np.maximum(0.,rawvar)+raw*raw
  if condition is None:Z=1.;presence=1-none;m=raw;s=rawsecond
  else:
   nosecond=np.maximum(0.,novar)+no*no
   if condition:Z=1-Q;presence=Z-none+both;m=raw-Q*no;s=rawsecond-Q*nosecond
   else:Z=Q;presence=Z-both;m=Q*no;s=Q*nosecond
  normalizer+=weight*Z;numprob+=weight*presence;mu_sum+=weight*(m+shifts*Z);second_sum+=weight*(s+2*shifts*m+shifts*shifts*Z)
 if normalizer<=0:raise IncompatibleAllocation('Opening Basic/location predicates have zero baseline support')
 means=mu_sum/normalizer;variance=np.maximum(0.,second_sum/normalizer-means*means)
 matrix=np.column_stack((numprob/normalizer,means,variance));return {c:matrix[i].tolist() for i,c in enumerate(queries)}

def allocation_one(facts,cards,queries):
 if not facts.unknown_basic:return ordinary(facts,cards,queries)
 used=collections.Counter(c for c,d in facts.copies.values());rest=collections.Counter(cards)-used;types=[c for c in sorted(rest) if c in facts.basic];K=facts.unknown_basic;den=choose(sum(rest[c] for c in types),K)
 if not den:raise IncompatibleAllocation('Concealed Basic slots exceed list Basics')
 results=[];nodes=0
 def visit(i,left,picked,w):
  nonlocal nodes
  nodes+=1
  if nodes>4096:raise UnsupportedEvent('Concealed-Basic baseline enumeration exceeds4096-node cap')
  if i==len(types):
   if left==0:
    f=copy.copy(facts);f.unknown_basic=0;remaining=list((collections.Counter(cards)-collections.Counter(picked)).elements());results.append((w/den,ordinary(f,remaining,queries)))
   return
  c=types[i]
  for n in range(min(rest[c],left)+1):visit(i+1,left-n,picked+[c]*n,w*choose(rest[c],n))
 visit(0,K,[],1)
 if not results:raise IncompatibleAllocation('No Basic-slot allocation')
 out={}
 for c in queries:
  mu=[sum(w*a[c][j] for w,a in results) for j in range(4)];variance=[max(0,sum(w*(a[c][z+4]+a[c][z+1]**2) for w,a in results)-mu[z+1]**2) for z in range(3)];out[c]=mu+variance
 return out
