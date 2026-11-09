"""Synthetic subset of tests/test_unown_belief_rb.py; no private fixtures or native matches."""
import itertools,random,math,collections,copy

import pytest

from unown_belief_rb.common import *

from unown_belief_rb.state import State

from unown_belief_rb.collapsed import CollapsedState,CountedRandom

from unown_belief_rb.filter import Observer as PF

from unown_belief_rb.candidate import Observer as RB

from unown_belief_rb.baseline import Facts,baseline_one

def move(src,dst,**kwargs):return dict(dict(op='move',src=src,dst=dst,kind='available'),**kwargs)

def draw(dst='H'):return move('D',dst,kind='index',index=0)

def states(cards):
 types=sorted(collections.Counter(cards),key=lambda c:(-cards.count(c),c));x=cards.index(types[0]);y=cards.index(types[1])
 for perm in itertools.permutations(range(len(cards))):
  if perm.index(x)>perm.index(y):continue
  a=State(cards,random.Random(11));a.z['D']=list(perm);b=a.clone();ix=b.z['D'].index(x);iy=b.z['D'].index(y);b.z['D'][ix],b.z['D'][iy]=b.z['D'][iy],b.z['D'][ix];yield CollapsedState.from_atoms([a,b],[.5,.5])

def exact(cards,ops,cid=1):
 z=mean=presence=deck=prize=0.;n=0
 for s in states(cards):
  n+=1;w=1.;rng=random.Random(11)
  try:
   for op in ops:w*=s.apply(op,rng)
  except IncompatibleAllocation:continue
  z+=w;m=s.moments('H',cid);mean+=w*m['mean'];presence+=w*m['presence'];deck+=w*s.moments('D',cid)['mean'];prize+=w*s.moments('P',cid)['mean']
 return dict(Z=z/n,hand=mean/z,p=presence/z,deck=deck/z,prize=prize/z)

def test_exact_hand_deck_prize_random_allocation():
 r=exact([1,1,2,3],[draw(),draw('P')]);assert r==pytest.approx(dict(Z=1,hand=.5,p=.5,deck=1,prize=.5))

def test_available_play_preserves_history_beyond_baseline_A():
 r=exact([1,1,2,2],[draw(),draw(),move('H','B',cid=1,sid=99)]);assert r['Z']==pytest.approx(5/6);assert r['p']==pytest.approx(.2)
 # Baseline A's current uniform remaining allocation would be1/3. The
 # historical availability constraint is not discarded by the candidate.
 assert r['p']!=pytest.approx(1/3)

def test_search_is_availability_not_random_card_likelihood():
 r=exact([1,2,2,2],[move('D','H',cid=1,sid=99)]);assert r['Z']==1 and r['hand']==1
 r=exact([1,2,2,2],[move('D','H',kind='top',cid=1,sid=99)]);assert r['Z']==pytest.approx(.25)

def test_deck_hypothesis_evidence_exact():
 op=[draw(),draw(),move('H','B',cid=1,sid=99)];a=exact([1,1,2,2],op);b=exact([1,2,2,2],op);assert b['Z']/(a['Z']+b['Z'])==pytest.approx(.375)

def test_opening_basic_predicate_exact():
 r=exact([1,2,2,2],[draw(),draw(),dict(op='basic',basic=[1],value=True)]);assert r['Z']==pytest.approx(.5);assert r['hand']==1
 r=exact([1,2,2,2],[draw(),draw(),dict(op='basic',basic=[1],value=False)]);assert r['Z']==pytest.approx(.5);assert r['hand']==0

def test_weight_by_conditional_availability_then_move():
 a=State([1,2],random.Random(11));a.z={z:[] for z in a.z};a.z['H']=[0];a.z['D']=[1];b=a.clone();b.z['H']=[1];b.z['D']=[0];r=CollapsedState.from_atoms([a,b],[.3,.7]);assert r.apply(move('H','B',cid=1,sid=20),random.Random(11))==pytest.approx(.3);assert len(r.atoms)==1 and r.atoms[0].counts('B')[1]==1

def test_known_top_bottom_and_partial_return():
 for seed in (11,29,47):
  a=State([1,2,3,4],random.Random(seed));a.apply(move('D','L',cid=1,sid=99),random.Random(seed));a.apply(move('L','D',cid=1,sid=99,position='bottom'),random.Random(seed));r=CollapsedState.from_atoms([a],[1.]);
  for _ in range(3):r.apply(move('D','H',kind='top'),random.Random(seed));assert r.moments('H',1)['mean']==0
  r.apply(move('D','H',kind='top'),random.Random(seed));assert r.moments('H',1)['mean']==1

def test_pinned_order_exact_identity_likelihood():
 a=State([1,2],random.Random(11));a.z['D']=[0,1];a.order_pins={0,1};b=a.clone();b.z['D']=[1,0];r=CollapsedState.from_atoms([a,b],[.5,.5]);assert r.apply(move('D','H',kind='top',cid=1,sid=42),random.Random(11))==pytest.approx(.5);assert r.moments('H',1)['mean']==1

def test_conditional_materialization_retains_serials_and_order():
 a=State([1,2],random.Random(11));a.apply(move('D','H',cid=1,sid=99),random.Random(1));r=CollapsedState.from_atoms([a],[1]);s=r.materialize(random.Random(29));assert s.serial==a.serial and s.z==a.z;assert s.apply(move('H','D',cid=1,sid=99,position='top'),random.Random(1))==1;assert s.z['D'][0]==s.serial[99]

def test_materialization_distribution_and_total_variance():
 a=State([1,2],random.Random(11));a.z['H']=[0];a.z['D']=[1];b=a.clone();b.z['H']=[1];b.z['D']=[0];r=CollapsedState.from_atoms([a,b],[.3,.7]);assert r.moments('H',1)['variance']==pytest.approx(.21);rng=random.Random(47);samples=[r.materialize(rng).counts('H')[1] for _ in range(10000)];assert sum(samples)/len(samples)==pytest.approx(.3,abs=.02)

def test_counted_random_matches_exact_underlying_MT_tape():
 base=random.Random(47);r=CountedRandom(base.getstate());assert [r.random(),r.getrandbits(5),r.getrandbits(65)]==[base.random(),base.getrandbits(5),base.getrandbits(65)];assert r.getstate()==base.getstate() and r.words==6
