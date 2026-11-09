"""Certified legal facts and exact exchangeable conditional-allocation baselines."""
from unown_belief_rb.common import *
from unown_belief_rb.adapter import legal,signature,visible
from .locations import zone,destination_position
import itertools

class Facts:
 def __init__(self,cards):
  self.basic={c['cardId'] for c in cards if c.get('basic')};self.lower={};self.game=None;self.seen=set();self.copies={};self.sizes={'D':60,'H':0,'P':0,'L':0};self.top=[];self.bottom=[];self.order_supported=True;self.setup_basic=None;self.effect=None;self.crispin_pending=False;self.unknown_basic=0;self.constraints=[];self.begun=False;self._layout_cache=None;self.bottom_blocks=[]
 def note(self,c,s,z):
  self._layout_cache=None
  if s in self.copies and self.copies[s][0]!=c:raise UnsupportedEvent('Identity-transform in certified facts')
  self.copies[s]=(c,frozenset([z]));count=collections.Counter(c for c,d in self.copies.values())
  for c,n in count.items():self.lower[c]=max(self.lower.get(c,0),n)
 def protected_bottom(self):
  blocks=[b for b in self.bottom_blocks if b['size']>0];total=sum(b['size'] for b in blocks)
  active=None if self.sizes['D']>total else (blocks[0] if blocks else None)
  return set().union(*(b['sids'] for b in blocks if b is not active)) if blocks else set()
 def bound_transfer(self,src,dst,sid=None,basic_only=False):
  movable={s for s,(c,d) in self.copies.items() if src in d and (not basic_only or c in self.basic) and not(src=='D' and s in self.protected_bottom())}
  updated=[]
  for members,zone,lo,hi in self.constraints:
   if sid is not None:
    delta=(int(dst==zone)-int(src==zone)) if sid in members else 0;lo+=delta;hi+=delta
   elif members & movable:
    if src==zone and dst!=zone:lo-=1
    elif src!=zone and dst==zone:hi+=1
   updated.append((members,zone,max(0,lo),min(len(members),hi)))
  self.constraints=updated
  if sid is None and src!=dst:
   exact=frozenset(s for s in movable if self.copies[s][1]=={src})
   if len(exact)>1:self.constraints.append((exact,src,len(exact)-1,len(exact)))
 def simplify_bounds(self):
  merged={}
  for members,zone,lo,hi in self.constraints:
   lower=sum(self.copies[s][1]=={zone} for s in members);upper=sum(zone in self.copies[s][1] for s in members)
   if hi<lower or lo>upper:raise InvariantFailure('Joint physical-copy fact contradicted')
   if lo<=lower and hi>=upper:continue
   key=(members,zone);old=merged.get(key,(0,len(members)));merged[key]=(max(lo,old[0]),min(hi,old[1]))
  self.constraints=[(m,z,lo,hi) for (m,z),(lo,hi) in merged.items()];self._layout_cache=None
 def transfer(self,src,dst,c=None,s=None,position=None,basic_only=False):
  if s in self.copies and len(self.copies[s][1])==1:
   actual=next(iter(self.copies[s][1]))
   if actual=='R' and src=='H':src='R'
   if actual==dst and src!=dst:self.note(c,s,dst);return
  active=None
  if src=='D' and position=='draw_top':
   total=sum(b['size'] for b in self.bottom_blocks)
   if self.bottom_blocks and self.sizes['D']<=total:
    active=self.bottom_blocks[0]
    if active['size']==1 and len(active['sids'])==1:s=next(iter(active['sids']));c=self.copies[s][0]
  if src=='D' and position!='draw_top' and any(b['sids'] for b in self.bottom_blocks) and s not in set().union(*(b['sids'] for b in self.bottom_blocks)):self.order_supported=False
  if s is not None:
   for block in self.bottom_blocks:
    if s in block['sids']:
     block['sids'].remove(s)
     if src=='D' and active is not block:block['size']-=1
  elif src=='D' and position!='draw_top' and any(b['sids'] for b in self.bottom_blocks):self.order_supported=False
  protected=self.protected_bottom() if src=='D' and position=='draw_top' else set()
  self.bound_transfer(src,dst,s,basic_only)
  if src=='D':
   if s in self.top:self.top.remove(s)
   elif s in self.bottom:self.bottom.remove(s)
   elif c is None and self.top:
    # Unknown top draw preserves known identity when the top is certified.
    if position=='draw_top':s=self.top.pop(0);c=self.copies[s][0]
    else:self.order_supported=False
  if c is not None:self.note(c,s,dst)
  else:
   for sid,(cid,domain) in list(self.copies.items()):
    if src in domain and sid not in protected and (not basic_only or cid in self.basic):self.copies[sid]=(cid,domain|{dst})
  if dst=='D' and position in ('top','bottom'):
   if s is None:self.order_supported=False
   elif position=='top':self.top.insert(0,s)
   else:self.bottom.append(s)
  if src in self.sizes:self.sizes[src]-=1
  if dst in self.sizes:self.sizes[dst]+=1
  if any(n<0 for n in self.sizes.values()):raise InvariantFailure('Certified-fact movement has negative source count')
  if src in self.sizes and self.sizes[src]==0:
   for sid,(cid,domain) in list(self.copies.items()):
    remaining=domain-{src}
    if not remaining:raise InvariantFailure('A known copy disappeared when its source emptied')
    self.copies[sid]=(cid,frozenset(remaining))
  if active is not None:
   active['size']-=1
   if active['size']==0:
    for sid in active['sids']:
     cid,d=self.copies[sid]
     if 'D' in d and len(d)>1:self.copies[sid]=(cid,frozenset(d-{'D'}))
  self.bottom_blocks=[b for b in self.bottom_blocks if b['size']>0]
  self._layout_cache=None
  if src=='H':self.setup_basic=None
 def observe(self,raw):
  o=legal(raw);cur=o['current'];g=cur.get('round',1)
  if g!=self.game:
   self.game=g;self.seen=set();self.copies={};self.top=[];self.bottom=[];self.order_supported=True;self.setup_basic=None;self.effect=None;self.crispin_pending=False;self.unknown_basic=0;self.constraints=[];self.begun=False;self._layout_cache=None;self.bottom_blocks=[];self.sizes={'D':60,'H':0,'P':0,'L':0}
  sig=signature(o)
  if sig in self.seen:return False
  self.seen.add(sig);opponent=1-cur['yourIndex'];logs=o.get('logs') or [];i=0
  while i<len(logs):
   e=logs[i];i+=1;t=e['type'];cid=e.get('cardId');sid=e.get('serial')
   if t in (10,15) and cid is not None:self.effect=cid
   if t==2:self.begun=True
   if e.get('playerIndex')!=opponent:continue
   if t==0:self.crispin_pending=False;self.top=[];self.bottom=[];self.bottom_blocks=[]
   elif t==1:self.setup_basic=e['hasBasicPokemon']
   elif t in (4,5):self.transfer('D','H',cid,sid,'draw_top')
   elif t in (6,7):
    src,dst=zone(e['fromArea'],True),zone(e['toArea'])
    if e['toArea']==14 and t==7:
     n=1
     while i<len(logs) and logs[i].get('type')==7 and logs[i].get('playerIndex')==opponent and logs[i].get('fromArea')==e['fromArea'] and logs[i].get('toArea')==14:n+=1;i+=1
     known={s for s,(c,d) in self.copies.items() if d=={src}}
     if n!=self.sizes.get(src) and known:self.order_supported=False
     for _ in range(n):self.transfer(src,'D')
     self.bottom_blocks.append(dict(sids=set(known) if self.sizes.get(src)==0 else set(),size=n))
     continue
    if t==7 and src=='H' and dst=='D':
     n=1
     while i<len(logs) and logs[i].get('type')==7 and logs[i].get('playerIndex')==opponent and logs[i].get('fromArea')==2 and logs[i].get('toArea')==1:n+=1;i+=1
     # Certify an all-hand return from a contiguous block only when the
     # observed previous count and intervening logs establish its size.
     if n==self.sizes.get('H',-1):
      for _ in range(n):self.transfer('H','D')
      for s,(c,d) in list(self.copies.items()):self.copies[s]=(c,frozenset((d-{'H'})|({'D'} if 'H' in d else set())))
     else:
      for _ in range(n):self.transfer(src,dst)
    else:
     position=destination_position(e['toArea'])
     if dst=='D' and t==7:
      # Unknown source copy/order is not made into a known prefix. Native
      # explicit shuffles clear this latent top return before next query.
      position=None
      if not any(x.get('type')==0 and x.get('playerIndex')==opponent for x in logs[i:]) and any(d=={src} for c,d in self.copies.values()):self.order_supported=False
     self.transfer(src,dst,cid,sid,position,basic_only=(not self.begun and src=='H' and dst in ('A','B')))
     if e['toArea']==14 and sid is not None:self.bottom_blocks.append(dict(sids={sid},size=1))
   elif t==10:
    self.crispin_pending=cid==1198
    self.transfer('H','R',cid,sid)
   elif t in (11,12):
    domain=self.copies.get(sid,(None,{'H'}))[1];src='D' if t==11 and self.crispin_pending else next(iter(domain)) if len(domain)==1 else 'H'
    if t==11 and self.crispin_pending:self.crispin_pending=False
    self.transfer(src,'B',cid,sid);self.setup_basic=None
   elif t==9:raise UnsupportedEvent('Identity-transform in certified facts')
  p=cur['players'][opponent];observed=dict(D=p['deckCount'],H=p['handCount'],P=len(p.get('prize') or []))
  if any(self.sizes[z]!=n for z,n in observed.items()):raise InvariantFailure('Certified-fact native movement/count mismatch')
  self.sizes.update(observed)
  public=visible(o,opponent)
  for c in public:self.note(c['id'],c['serial'],c['zone'])
  # All absent public cards must have an explicit legal movement above;
  # public snapshots do not reveal unobserved hand/deck identities.
  looking=cur.get('looking')
  self.sizes['L']=len(looking) if looking is not None and all(x is None or x.get('playerIndex')==opponent for x in looking) else 0
  for c in p.get('prize') or []:
   if c:self.note(c['id'],c['serial'],'P')
  for c in (o.get('select') or {}).get('deck') or []:
   if c.get('playerIndex')==opponent:self.note(c['id'],c['serial'],'D')
  for c in looking or []:
   if c and c.get('playerIndex')==opponent:self.note(c['id'],c['serial'],'L')
  for sid,(cid,domain) in list(self.copies.items()):
   d=frozenset(z for z in domain if z not in self.sizes or self.sizes[z]>0)
   if not d:raise InvariantFailure('Certified physical copy has no possible location')
   self.copies[sid]=(cid,d)
  self.unknown_basic=sum(x is None for zone in ('active','bench') for x in p.get(zone) or [])
  self.simplify_bounds();self.last=o;return True
 def known_hand(self):return hard_layout(self)[3]
 def compatible(self,cards):return not(collections.Counter(self.lower)-collections.Counter(cards))

def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0

def assignments(groups,capacity,cap=4096,constraints=()):
 """Uniform physical slot assignments conditioned on ALL joint legal facts."""
 zones=('D','H','P','L');done=[];visited=0
 def valid_partial(alloc,j):
  for ci,(members,zone,lo,hi) in enumerate(constraints):
   zi=zones.index(zone);count=sum(where[zi] for (key,counts),where in zip(groups,alloc) if ci in key[1]);possible=sum(sum(counts.values()) for key,counts in groups[j:] if ci in key[1] and zone in key[0])
   if count>hi or count+possible<lo:return False
  return True
 def visit(j,slots,alloc,w):
  nonlocal visited
  visited+=1
  if visited>cap:raise UnsupportedEvent('Exact baseline joint-location enumeration exceeds4096-node cap')
  if not valid_partial(alloc,j):return
  if j==len(groups):done.append((w,slots,tuple(alloc)));return
  key,counts=groups[j];domain=key[0];K=sum(counts.values());allowed=[i for i,z in enumerate(zones) if z in domain]
  def split(k,left,ns,way):
   if k==len(allowed):
    if left==0:visit(j+1,tuple(a-b for a,b in zip(slots,ns)),alloc+[tuple(ns)],w*way)
    return
   at=allowed[k]
   for n in range(min(left,slots[at])+1):ns[at]=n;split(k+1,left-n,ns,way*choose(slots[at],n))
   ns[at]=0
  split(0,K,[0]*4,1)
 visit(0,tuple(capacity[z] for z in zones),[],1);total=sum(w for w,_,_ in done)
 if not total:raise IncompatibleAllocation('No simple allocation satisfies all certified joint facts')
 return [(w/total,s,a) for w,s,a in done]

def hard_layout(facts):
 if facts._layout_cache is not None:return facts._layout_cache
 zones=('D','H','P','L');fixed={z:collections.Counter() for z in ('D','H','P','L','A','B','R')};uncertain={};fixed_serial={}
 for sid,(cid,domain) in facts.copies.items():
  if len(domain)==1:zone=next(iter(domain));fixed[zone][cid]+=1;fixed_serial[sid]=zone
  else:
   if not domain<=set(zones):raise UnsupportedEvent('Ambiguous concealed/public serial domain requires a Basic-slot allocation model')
   uncertain[sid]=(cid,domain)
 constraints=[]
 for members,zone,lo,hi in facts.constraints:
  if zone not in zones:continue
  already=sum(fixed_serial.get(s)==zone for s in members);remaining=frozenset(s for s in members if s in uncertain);lo=max(0,lo-already);hi=min(len(remaining),hi-already)
  if lo>hi:raise InvariantFailure('Fixed serial locations contradict joint bounds')
  if remaining and (lo>0 or hi<len(remaining)):constraints.append((remaining,zone,lo,hi))
 grouped={}
 for sid,(cid,domain) in uncertain.items():
  key=(tuple(sorted(domain)),tuple(i for i,(members,z,lo,hi) in enumerate(constraints) if sid in members));grouped.setdefault(key,collections.Counter())[cid]+=1
 groups=sorted(grouped.items());capacity={z:facts.sizes[z]-sum(fixed[z].values()) for z in zones}
 if min(capacity.values())<0:raise InvariantFailure('Certified known counts exceed public zone count')
 parts=assignments(groups,capacity,constraints=constraints);known=collections.Counter(fixed['H'])
 for c in {c for c,d in uncertain.values()}:
  minimum=min(sum(max(0,where[1]-(sum(counts.values())-counts[c])) for (_,counts),where in zip(groups,alloc)) for w,slots,alloc in parts);known[c]+=minimum
 facts._layout_cache=(fixed,groups,parts,known);return facts._layout_cache

def cdf_components(components,cid,limit,basic,exclude_basic=False):
 dist=[1.]+[0.]*max(0,limit)
 for counts,where,K in components:
  if not K:continue
  h=where[1];C=counts[cid];B=sum(n for c,n in counts.items() if c in basic)
  if exclude_basic:
   if cid in basic:C=0
   K-=B
  den=choose(K,h)
  if not den:return 0.
  probabilities=[choose(C,k)*choose(K-C,h-k)/den for k in range(min(C,h,limit)+1)];nxt=[0.]*(limit+1)
  for a,p in enumerate(dist):
   for b,q in enumerate(probabilities):
    if a+b<=limit:nxt[a+b]+=p*q
  dist=nxt
 return sum(dist)

def ordinary(facts,cards,queries):
 if not facts.order_supported:raise UnsupportedEvent('Baseline abstains on unresolved conditional known order')
 total=collections.Counter(cards);fixed,groups,parts,known=hard_layout(facts);used=collections.Counter(c for c,d in facts.copies.values())
 if used-total:raise IncompatibleAllocation('Public physical copies exceed list composition')
 remainder=total-used
 N=sum(remainder.values());out={};known_basic=any(c in facts.basic and n for c,n in fixed['H'].items())
 for cid in queries:
  prob=mh=md=mp=normalizer=0.
  for weight,slots,alloc in parts:
   if sum(slots)!=N:raise InvariantFailure('Baseline residual physical-copy conservation')
   kh=kd=kp=0.;none=both=1.;no_basic=0. if known_basic else 1.;nh=nd=npz=0.
   components=[(counts,where,sum(counts.values())) for (_,counts),where in zip(groups,alloc)]+[(remainder,slots,N)]
   for counts,where,K in components:
    if K==0:continue
    C=counts[cid];B=sum(n for c,n in counts.items() if c in facts.basic);h,d,p=where[1],where[0],where[2];den=choose(K,h)
    kh+=h*C/K;kd+=d*C/K;kp+=p*C/K
    none*=choose(K-C,h)/den;no_basic*=choose(K-B,h)/den;both*=choose(K-B-(0 if cid in facts.basic else C),h)/den
    hc=0. if cid in facts.basic else h*C/(K-B) if K>B else 0.
    nh+=hc;nd+=(C-hc)*d/(K-h) if K>h else 0.;npz+=(C-hc)*p/(K-h) if K>h else 0.
   if known_basic:both=0.
   excess=known[cid]-fixed['H'][cid]
   if excess:
    none=cdf_components(components,cid,excess,facts.basic);both=no_basic*cdf_components(components,cid,excess,facts.basic,True)
   condition=facts.setup_basic
   if condition is None:Z=1.;presence=1-none;hmean,dmean,pmean=kh,kd,kp
   elif condition:
    Z=1-no_basic;presence=Z-none+both;hmean,dmean,pmean=kh-no_basic*nh,kd-no_basic*nd,kp-no_basic*npz
   else:Z=no_basic;presence=Z-both;hmean,dmean,pmean=no_basic*nh,no_basic*nd,no_basic*npz
   normalizer+=weight*Z;prob+=weight*presence;mh+=weight*(hmean-Z*excess);md+=weight*(Z*fixed['D'][cid]+dmean);mp+=weight*(Z*fixed['P'][cid]+pmean)
  if normalizer<=0:raise IncompatibleAllocation('Opening Basic/location predicates have zero baseline support')
  out[cid]=dict(p_unknown_hand=prob/normalizer,hand_unknown_mean=mh/normalizer,deck_mean=md/normalizer,prize_mean=mp/normalizer)
 return out


def baseline_one(facts,cards,queries):
 import copy
 if facts.unknown_basic:
  # Simple current-allocation model: concealed setup board slots contain
  # uniformly allocated Basic copies. Marginalize their multivariate counts.
  used=collections.Counter(c for c,d in facts.copies.values());rest=collections.Counter(cards)-used;types=[c for c in sorted(rest) if c in facts.basic];K=facts.unknown_basic;B=sum(rest[c] for c in types);den=choose(B,K)
  if not den:raise IncompatibleAllocation('Concealed Basic slots exceed list Basics')
  results=[];nodes=0
  def visit(i,left,picked,w):
   nonlocal nodes
   nodes+=1
   if nodes>4096:raise UnsupportedEvent('Concealed-Basic baseline enumeration exceeds4096-node cap')
   if i==len(types):
    if left==0:
     remaining=list((collections.Counter(cards)-collections.Counter(picked)).elements());f=copy.copy(facts);f.unknown_basic=0;results.append((w/den,baseline_one(f,remaining,queries)))
    return
   c=types[i]
   for n in range(min(rest[c],left)+1):visit(i+1,left-n,picked+[c]*n,w*choose(rest[c],n))
  visit(0,K,[],1)
  if not results:raise IncompatibleAllocation('No Basic-slot allocation')
  return {c:{k:sum(w*r[c][k] for w,r in results) for k in results[0][1][c]} for c in queries}
 return ordinary(facts,cards,queries)

def baseline(facts,library,masses,queries):
 kept={};results={};rejected={}
 for k,p in masses.items():
  if p<=0 or not facts.compatible(library[k]):continue
  try:results[k]=baseline_one(facts,library[k],queries);kept[k]=p
  except IncompatibleAllocation as e:rejected[k]=str(e)
 if not kept:raise OutOfLibrary('No baseline allocation satisfies legal facts')
 z=sum(kept.values());return {c:{field:sum(kept[k]/z*results[k][c][field] for k in kept) for field in results[next(iter(kept))][c]} for c in queries}
