"""Exact finite-population proposal for joint future chance-location evidence."""
from unown_belief_provenance.common import *
from functools import lru_cache

def multinomial(n,a,b):return math.comb(n,a)*math.comb(n-a,b) if 0<=a<=n and 0<=b<=n-a else 0

def plan(state,op,rng):
 D=list(state.z['D']);positions=set(op['positions']);cursor=op.get('deck_after',len(D));required=collections.Counter({int(k):v for k,v in op.get('required',{}).items()});reserve=collections.Counter({int(k):v for k,v in op.get('reserve',{}).items()});dst=op.get('dst','H')
 if any(i>=len(D) for i in positions) or cursor>len(D):raise IncompatibleAllocation('Future proposal exceeds deck')
 slot_groups=[[i for i in sorted(positions) if D[i] not in state.pinned],[i for i in range(cursor,len(D)) if i not in positions and D[i] not in state.pinned]]
 used=set(slot_groups[0]+slot_groups[1]);slot_groups.append([i for i,t in enumerate(D) if t not in state.pinned and i not in used])
 fixedH=collections.Counter(state.cards[D[i]] for i in positions if D[i] in state.pinned);fixedD=collections.Counter(state.cards[D[i]] for i in range(cursor,len(D)) if D[i] in state.pinned and i not in positions)
 H=state.counts(dst);needH={c:max(0,v-H[c]-fixedH[c]) for c,v in required.items()};needD={c:max(0,v-fixedD[c]) for c,v in reserve.items()};cats=sorted(set(needH)|set(needD));pool=[t for t in D if t not in state.pinned];groups=[[t for t in pool if state.cards[t]==c] for c in cats];background=[t for t in pool if state.cards[t] not in cats];qH,qD,qO=map(len,slot_groups);M=len(pool)
 @lru_cache(None)
 def count(j,a,b,o):
  if min(a,b,o)<0:return 0
  if j==len(cats):return multinomial(len(background),a,b) if a+b+o==len(background) else 0
  C=len(groups[j]);c=cats[j];z=0
  for h in range(needH.get(c,0),min(C,a)+1):
   for d in range(needD.get(c,0),min(C-h,b)+1):
    other=C-h-d
    if other<=o:z+=multinomial(C,h,d)*count(j+1,a-h,b-d,o-other)
  return z
 total=multinomial(M,qH,qD);valid=count(0,qH,qD,qO)
 if not valid:raise IncompatibleAllocation('Joint chance-location constraints have zero support in this allocation')
 chosen=[[],[],[]];a,b,o=qH,qD,qO
 for j,c in enumerate(cats):
  C=len(groups[j]);options=[];weights=[]
  for h in range(needH.get(c,0),min(C,a)+1):
   for d in range(needD.get(c,0),min(C-h,b)+1):
    other=C-h-d
    if other<=o:
     w=multinomial(C,h,d)*count(j+1,a-h,b-d,o-other)
     if w:options.append((h,d,other));weights.append(w)
  h,d,other=rng.choices(options,weights=weights,k=1)[0];tokens=list(groups[j]);rng.shuffle(tokens);chosen[0]+=tokens[:h];chosen[1]+=tokens[h:h+d];chosen[2]+=tokens[h+d:];a-=h;b-=d;o-=other
 tokens=list(background);rng.shuffle(tokens);chosen[0]+=tokens[:a];chosen[1]+=tokens[a:a+b];chosen[2]+=tokens[a+b:]
 for slots,tokens in zip(slot_groups,chosen):
  rng.shuffle(tokens)
  for i,t in zip(slots,tokens):D[i]=t
 state.z['D']=D;state.proposal_pins.update(D);state.plan_remaining=op.get('deck_after',max(positions)+1);state.check();return valid/total
