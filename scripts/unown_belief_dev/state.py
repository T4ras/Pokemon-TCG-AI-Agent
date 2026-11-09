"""Serial-aware location states and fully adapted chance/availability proposals."""
from unown_belief_dev.common import *
import random,copy
ZONES=('D','H','P','L','A','B','R')
class State:
 def __init__(self,cards,rng):
  self.cards=tuple(cards);self.z={k:[] for k in ZONES};self.z['D']=list(range(len(cards)));rng.shuffle(self.z['D']);self.serial={};self.order_pins=set();self.proposal_pins=set();self.plan_remaining=0
 def clone(self):
  s=object.__new__(State);s.cards=self.cards;s.z={k:list(v) for k,v in self.z.items()};s.serial=dict(self.serial);s.order_pins=set(self.order_pins);s.proposal_pins=set(self.proposal_pins);s.plan_remaining=self.plan_remaining;return s
 @property
 def pinned(self):return self.order_pins|self.proposal_pins
 def check(self):
  tokens=[t for z in self.z.values() for t in z]
  if sorted(tokens)!=list(range(len(self.cards))) or len(set(self.serial.values()))!=len(self.serial):raise InvariantFailure('Physical-copy conservation/serial uniqueness')
 def location(self,t):return next((z for z,v in self.z.items() if t in v),None)
 def matches(self,t,cid=None,sid=None):
  if cid is not None and self.cards[t]!=cid:return False
  if sid is None:return True
  if sid in self.serial:return t==self.serial[sid]
  return t not in self.serial.values()
 def bind(self,t,cid,sid):
  if cid is not None and self.cards[t]!=cid:raise IncompatibleAllocation('Revealed identity disagrees with latent copy')
  if sid is not None:
   if sid in self.serial and self.serial[sid]!=t:raise IncompatibleAllocation('Physical serial bound to another copy')
   if t in self.serial.values() and self.serial.get(sid)!=t:raise IncompatibleAllocation('One copy received two physical serials')
   self.serial[sid]=t
 def exchange_unknown(self,zone,rng):
  if zone!='D':return
  indices=[i for i,t in enumerate(self.z[zone]) if t not in self.pinned];tokens=[self.z[zone][i] for i in indices];rng.shuffle(tokens)
  for i,t in zip(indices,tokens):self.z[zone][i]=t
 def pick(self,src,op,rng):
  pool=self.z[src];cid=op.get('cid');sid=op.get('sid');kind=op.get('kind','available')
  if not pool:raise IncompatibleAllocation('Empty '+src+' source')
  if kind=='top':
   self.exchange_unknown(src,rng);t=pool[0]
   if cid is not None and t not in self.pinned and src=='D':
    unknown=[x for x in pool if x not in self.pinned];allowed=[x for x in unknown if self.matches(x,cid,sid)]
    if not allowed:raise IncompatibleAllocation('Observed chance draw unavailable')
    chosen=rng.choice(allowed);j=pool.index(chosen);pool[0],pool[j]=pool[j],pool[0];return chosen,len(allowed)/len(unknown)
   if not self.matches(t,cid,sid):raise IncompatibleAllocation('Pinned top/card constraint')
   return t,1.
  if kind=='index':
   i=op['index']
   if not 0<=i<len(pool) or not self.matches(pool[i],cid,sid):raise IncompatibleAllocation('Known location/index constraint')
   return pool[i],1.
  allowed=[t for t in pool if self.matches(t,cid,sid) and (not op.get('basic_only') or self.cards[t] in op['basic_only'])]
  if not allowed:raise IncompatibleAllocation('Positive source availability contradicted')
  # Identity selected strategically contributes availability only. A declared
  # random uniform-index reveal contributes its exact chance probability.
  denom=[t for t in pool if not op.get('basic_only') or self.cards[t] in op['basic_only']]
  factor=len(allowed)/len(denom) if kind=='guided_selection' else (len(allowed)/len(pool) if kind=='random_index' and cid is not None else 1.)
  return rng.choice(allowed),factor
 def apply(self,op,rng):
  kind=op['op'];factor=1.
  if kind=='move':
   src=op['src'];dst=op['dst'];sid=op.get('sid')
   if sid in self.serial:
    actual=self.location(self.serial[sid])
    if actual=='R' and op.get('resolving_source'):src='R'
    if actual in ('A','B') and src in ('A','B'):src=actual
    if actual!=src:raise IncompatibleAllocation('Known physical copy is not in claimed source '+src)
   t,factor=self.pick(src,op,rng);self.bind(t,op.get('cid'),sid);self.z[src].remove(t)
   position=op.get('position','append')
   if position=='top':self.z[dst].insert(0,t);self.order_pins.add(t)
   elif position=='bottom':self.z[dst].append(t);self.order_pins.add(t)
   else:self.z[dst].append(t)
   if src=='D' and dst!='D':
    self.order_pins.discard(t);self.proposal_pins.discard(t)
    if op.get('kind')=='top' and self.plan_remaining:
     self.plan_remaining-=1
     if not self.plan_remaining:self.proposal_pins.clear()
  elif kind=='shuffle':rng.shuffle(self.z['D']);self.order_pins.clear();self.proposal_pins.clear();self.plan_remaining=0
  elif kind=='return_all':
   src=op['src'];self.z['D'].extend(self.z[src]);self.z[src]=[]
  elif kind=='basic':
   found=any(self.cards[t] in op['basic'] for t in self.z['H'])
   if found!=op['value']:raise IncompatibleAllocation('Opening Basic evidence')
  elif kind=='known':
   zone=op['zone'];cid=op['cid'];sid=op['sid']
   if sid in self.serial:
    t=self.serial[sid]
    if self.location(t)!=zone or self.cards[t]!=cid:raise IncompatibleAllocation('Known location/identity evidence')
   else:t,_=self.pick(zone,dict(kind='available',cid=cid,sid=sid),rng);self.bind(t,cid,sid)
  elif kind=='snapshot':
   sizes=op['sizes']
   for zone,n in sizes.items():
    if len(self.z[zone])!=n:raise IncompatibleAllocation('Location count '+zone+f' {len(self.z[zone])}!={n}')
   # Delayed setup flips and board switches are public location facts, not
   # draws. Existing public/limbo tokens may be repartitioned without a loss.
   known_public=set()
   for card in op['public']:
    cid,sid,zone=card['id'],card['serial'],card['zone']
    if sid in self.serial:
     t=self.serial[sid];loc=self.location(t)
     if self.cards[t]!=cid:raise UnsupportedEvent('Native identity-changing copy requires a declared transform kernel')
     if loc not in ('A','B','R'):raise IncompatibleAllocation('Visible card occupies hidden source '+str(loc))
    else:
     eligible=[t for q in (zone,'R') for t in self.z[q] if self.matches(t,cid,sid)]
     if not eligible:raise IncompatibleAllocation('Delayed/public reveal unavailable')
     t=rng.choice(eligible);self.bind(t,cid,sid);loc=self.location(t)
    if loc!=zone:self.z[loc].remove(t);self.z[zone].append(t)
    known_public.add(t)
   for i,card in enumerate(op.get('prizes',[])):
    if card:
     t=self.z['P'][i];self.bind(t,card['id'],card['serial'])
   for card in op.get('deck_known',[]):self.apply(dict(op='known',zone='D',cid=card['id'],sid=card['serial']),rng)
   for i,card in enumerate(op.get('looking',[])):
    if card:
     if i>=len(self.z['L']):raise IncompatibleAllocation('Known looking index missing')
     self.bind(self.z['L'][i],card['id'],card['serial'])
  elif kind=='plan_draws':
   from unown_belief_dev.proposal import plan
   factor=plan(self,op,rng)
  elif kind=='bridge_draw':
   # Condition the complete chance block on a multiset of future source
   # requirements. Enumerate category counts, never initial deck deals.
   from functools import lru_cache
   n=op['n'];self.exchange_unknown('D',rng);deck=self.z['D']
   if n>len(deck):raise IncompatibleAllocation('Bridge draws exceed deck')
   dst=op.get('dst','H');req=collections.Counter({int(k):v for k,v in op['required'].items()});hand=self.counts(dst);fixed=collections.Counter(self.cards[t] for t in deck[:n] if t in self.pinned)
   pool=[t for t in deck if t not in self.pinned];top=[i for i,t in enumerate(deck[:n]) if t not in self.pinned];M,q=len(pool),len(top)
   needs={c:max(0,k-hand[c]-fixed[c]) for c,k in req.items()};needs={c:k for c,k in needs.items() if k}
   cats=sorted(needs);groups=[[t for t in pool if self.cards[t]==c] for c in cats];background=[t for t in pool if self.cards[t] not in needs]
   @lru_cache(None)
   def count(j,left):
    if j==len(cats):return math.comb(len(background),left) if 0<=left<=len(background) else 0
    C=len(groups[j]);return sum(math.comb(C,k)*count(j+1,left-k) for k in range(needs[cats[j]],min(C,left)+1))
   valid=count(0,q);total=math.comb(M,q)
   if not valid:raise IncompatibleAllocation('Bridge cannot satisfy future availability')
   factor=valid/total;picked=[];left=q
   for j,c in enumerate(cats):
    C=len(groups[j]);ks=list(range(needs[c],min(C,left)+1));weights=[math.comb(C,k)*count(j+1,left-k) for k in ks];k=rng.choices(ks,weights=weights,k=1)[0];picked.extend(rng.sample(groups[j],k));left-=k
   picked.extend(rng.sample(background,left));rng.shuffle(picked);rest=[t for t in pool if t not in picked];rng.shuffle(rest)
   # Capture destination indices before writing: a duplicate value during a
   # partial in-place permutation must not change which slots are pinned.
   tail=[i for i,t in enumerate(deck) if i>=n and t not in self.pinned]
   for i,t in zip(top,picked):deck[i]=t
   for i,t in zip(tail,rest):deck[i]=t
   for _ in range(n):
    t=self.z['D'].pop(0);self.z[dst].append(t);self.order_pins.discard(t);self.proposal_pins.discard(t)
  else:raise UnsupportedEvent('Unknown production operation '+kind)
  self.check();return factor
 def counts(self,zone):return collections.Counter(self.cards[t] for t in self.z[zone])

def guided_history(history):
 """Use only evidence already present in the requested replay prefix."""
 history=[dict(x) for x in history]
 # A hidden setup active has a uniform-Basic target kernel. Its first public
 # flip can be proposed directly with count(identity)/count(Basics), rather
 # than treating a guessed concealed identity as hard prior knowledge.
 for i,e in enumerate(history):
  if e['op']=='move' and e.get('src')=='H' and e.get('dst')=='A' and e.get('basic_only') and e.get('cid') is None:
   for snap in history[i+1:]:
    if snap['op']=='snapshot':
     roots=[c for c in snap['public'] if c['zone']=='A']
     if roots:
      c=roots[0];e.update(kind='guided_selection',cid=c['id'],sid=c['serial']);break
    if snap['op']=='move' and snap.get('dst')=='A':break
 out=[]
 for i,e in enumerate(history):
  start=e['op']=='move' and e.get('src')=='D' and e.get('dst') in ('H','L') and e.get('kind')=='top' and e.get('cid') is None
  target=e.get('dst');previous=history[i-1] if i else {}
  continued=previous.get('op')=='move' and previous.get('src')=='D' and previous.get('dst')==target and previous.get('kind')=='top' and previous.get('cid') is None
  if start and not continued:
   positions=[];cursor=0;unique={};anonymous=collections.Counter();reserve={}
   for offset,x in enumerate(history[i:]):
    op=x['op']
    if op in ('shuffle','return_all'):break
    if op=='move':
     src,dst=x.get('src'),x.get('dst')
     if dst=='D':
      if src==target:
       # A failed opening exposes its entire returned hand. Condition the
       # opening chance block jointly on all those physical-copy reveals.
       for ret in history[i+offset:]:
        if ret.get('op')!='move' or ret.get('src')!=target or ret.get('dst')!='D':break
        if ret.get('cid') is not None:
         if ret.get('sid') is None:anonymous[ret['cid']]+=1
         else:unique[ret['sid']]=ret['cid']
      break
     if src=='D':
      if x.get('kind')!='top':
       if x.get('cid') is not None:reserve[x.get('sid',str(x['cid']))]=x['cid']
       break
      if dst==target:positions.append(cursor)
      cursor+=1
     if dst==target and not(src=='D' and x.get('kind')=='top'):break
     if src==target and x.get('cid') is not None:
      if x.get('sid') is None:anonymous[x['cid']]+=1
      else:unique[x['sid']]=x['cid']
   required=collections.Counter(unique.values())+anonymous
   if positions and (required or reserve):out.append(dict(op='plan_draws',dst=target,positions=positions,required=dict(required),deck_after=cursor,reserve=dict(collections.Counter(reserve.values()))))
  out.append(e)
 return out
