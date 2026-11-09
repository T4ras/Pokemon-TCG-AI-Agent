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
 if not facts.order_supported:raise UnsupportedEvent('Baseline abstains on unresolved conditional known order')
 fixed,groups,parts,known=hard_layout(facts);used=collections.Counter(c for c,d in facts.copies.values());total=collections.Counter(cards)
 if used-total:raise IncompatibleAllocation('Public physical copies exceed list composition')
 rest=total-used;N=sum(rest.values());out={};known_basic=any(c in facts.basic and n for c,n in fixed['H'].items())
 for cid in queries:
  numprob=0.;mu_sum=[0.]*3;second_sum=[0.]*3;normalizer=0.
  for weight,slots,alloc in parts:
   if sum(slots)!=N:raise InvariantFailure('Baseline residual physical-copy conservation')
   components=[(counts,where,sum(counts.values())) for (_,counts),where in zip(groups,alloc)]+[(rest,slots,N)];raw=[0.]*3;rawvar=[0.]*3;no=[0.]*3;novar=[0.]*3;none=both=1.;Q=0. if known_basic else 1.
   for counts,where,K in components:
    if not K:continue
    C=counts[cid];B=sum(v for c,v in counts.items() if c in facts.basic);h=where[1];den=choose(K,h);none*=choose(K-C,h)/den;Q*=choose(K-B,h)/den;both*=choose(K-B-(0 if cid in facts.basic else C),h)/den
    m,s=component_moments(counts,where,K,cid,facts.basic);mn,sn=component_moments(counts,where,K,cid,facts.basic,True)
    for z in range(3):raw[z]+=m[z];rawvar[z]+=s[z]-m[z]*m[z];no[z]+=mn[z];novar[z]+=sn[z]-mn[z]*mn[z]
   if known_basic:both=0.
   excess=known[cid]-fixed['H'][cid]
   if excess:none=cdf_components(components,cid,excess,facts.basic);both=Q*cdf_components(components,cid,excess,facts.basic,True)
   rawsecond=[max(0,rawvar[z])+raw[z]*raw[z] for z in range(3)];nosecond=[max(0,novar[z])+no[z]*no[z] for z in range(3)]
   if facts.setup_basic is None:Z=1.;presence=1-none;m=raw;s=rawsecond
   elif facts.setup_basic:Z=1-Q;presence=Z-none+both;m=[raw[z]-Q*no[z] for z in range(3)];s=[rawsecond[z]-Q*nosecond[z] for z in range(3)]
   else:Z=Q;presence=Z-both;m=[Q*v for v in no];s=[Q*v for v in nosecond]
   shifts=[-excess,fixed['D'][cid],fixed['P'][cid]];normalizer+=weight*Z;numprob+=weight*presence
   for z,shift in enumerate(shifts):mu_sum[z]+=weight*(m[z]+shift*Z);second_sum[z]+=weight*(s[z]+2*shift*m[z]+shift*shift*Z)
  if normalizer<=0:raise IncompatibleAllocation('Opening Basic/location predicates have zero baseline support')
  means=[v/normalizer for v in mu_sum];variances=[max(0,second_sum[z]/normalizer-means[z]*means[z]) for z in range(3)];out[cid]=[numprob/normalizer]+means+variances
 return out

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
