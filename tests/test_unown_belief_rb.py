"""Exact physical-permutation references; no native or held-out access."""
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

def test_bo3_evidence_not_reset_or_double_counted():
 for cls in (PF,RB):
  ob=cls({'a':[1,1,2,2],'b':[1,2,2,2]},{'a':.5,'b':.5},metadata()['cards'],11,512,recovery_trials=1024);ob.game=2;ob.completed=[[draw(),draw(),move('H','B',cid=1,sid=99)]];ob.unresolved={'a':{},'b':{}}
  assert ob.recover('a') and ob.recover('b');assert ob.masses()['b']==pytest.approx(.375,abs=.04);old=ob.masses();assert ob.recover('a') and ob.recover('b');assert ob.masses()==pytest.approx(old)

def test_saved_16_shared_facts_and_compiler_after_narrow_fix():
 import gzip
 from unown_belief_rb.adapter import Compiler
 for row in read(OUT/'development-inputs.json'):
  f=Facts(metadata()['cards']);c=Compiler(metadata()['cards']);h=json.load(gzip.open(R/row['corpus']['legal_path'],'rt'))
  for e in h:
   f.observe(e['observation']);c.compile(e['observation']);f.known_hand()
   lower=copy.deepcopy(f.lower);copies=copy.deepcopy(f.copies);assert c.compile(e['observation'])['duplicate'];assert not f.observe(e['observation']);assert f.lower==lower and f.copies==copies


def test_candidate256_keeps_registered_recovery_budget():
 ob=RB({'a':[1,2]},{'a':1},metadata()['cards'],11,256);assert ob.trials==256 and ob.budget['remaining']==2000000

def test_within_and_between_component_variance_are_both_retained():
 a=State([1,2],random.Random(11));a.z['H']=[0];a.z['D']=[1];b=a.clone();b.z['H']=[1];b.z['D']=[0]
 mixed=CollapsedState.from_atoms([a,b],[.3,.7]);certain=CollapsedState.from_atoms([a],[1]);ob=RB({'a':[1,2]},{'a':1},metadata()['cards'],11,512);ob.pop={'a':dict(states=[mixed,certain],weights=[.5,.5])};result=ob.predict([1])[1]
 assert result['hand_total_variance']==pytest.approx(.65*.35);assert result['within_component_variance']==pytest.approx(.5*.21);assert result['between_component_variance']==pytest.approx(.1225)

def test_process_cpu_parser_uses_correct_proc_fields():
 from unown_belief_rb.run import process_CPU
 fields=['S','99','123']+['0']*8+['70','30']+['0']*6+['999']
 assert process_CPU('1 (name with spaces) '+' '.join(fields))==(123,100,'999')

def test_partial_bottom_block_keeps_rest_and_order_constraints_exact():
 # Native Close shuffles just two returned Looking cards; marginalize the
 # two possible returned-block orders while preserving the earlier prefix.
 a=State([1,2,3,4],random.Random(11));a.z={z:[] for z in a.z};a.z['D']=[2,3,0,1];a.order_pins={0,1};a.serial={11:0,22:1};b=a.clone();b.z['D']=[2,3,1,0];r=CollapsedState.from_atoms([a,b],[.5,.5]);rng=random.Random(29)
 for _ in range(2):r.apply(move('D','H',kind='top'),rng);assert r.moments('H',1)['mean']==0 and r.moments('H',2)['mean']==0
 assert r.apply(move('D','H',kind='top',cid=1,sid=11),rng)==pytest.approx(.5);assert r.moments('D',2)['mean']==1

def test_observer_duplicate_and_actual_game_reset_preserve_prior_evidence():
 cards=metadata()['cards'];o=dict(current=dict(round=1,yourIndex=0,turn=0,turnActionCount=1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=60,handCount=0,hand=None,prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=dict(context=41,minCount=1,maxCount=1,option=[{},{}]),logs=[])
 ob=RB({'a':[1]*60,'b':[1]*59+[2]},{'a':.4,'b':.6},cards,11,256);ob.observe(o);before=(ob.masses(),ob.rng.getstate(),ob.operations);ob.observe(o);assert (ob.masses(),ob.rng.getstate(),ob.operations)==before
 new=copy.deepcopy(o);new['current']['round']=2;ob.observe(new);assert ob.masses()==pytest.approx(before[0]);assert len(ob.completed)==1 and ob.facts.known_hand()=={}

def test_complete_replay_serialization_and_separate_scoring_on_synthetic_input(monkeypatch,tmp_path):
 import gzip
 import unown_belief_rb.replay as replay
 from unown_belief_rb.common import write,sha
 monkeypatch.setattr(replay,'OUT',tmp_path);monkeypatch.setattr(replay,'R',tmp_path);monkeypatch.setattr(replay,'PRIVATE',tmp_path/'private');monkeypatch.setattr(replay,'check',lambda:None);monkeypatch.setattr(replay,'prior',lambda:{'a':dict(cards=[1]*60,prior=1.)})
 o=dict(current=dict(round=1,yourIndex=0,turn=0,turnActionCount=1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=60,handCount=0,hand=None,prize=[],active=[],bench=[],discard=[])],stadium=[],looking=None),select=dict(context=41,minCount=1,maxCount=1,option=[{},{}]),logs=[])
 h=tmp_path/'legal.json.gz';t=tmp_path/'labels.json.gz'
 with gzip.open(h,'wt') as f:json.dump([dict(observation=o,action=[0])],f)
 with gzip.open(t,'wt') as f:json.dump([dict(key=[1,0,1,0],hand={},hand_serials=[],deck={'1':60},prize={})],f)
 evidence=dict(legal_path=h.name,legal_sha256=sha(h),truth_path=t.name,truth_sha256=sha(t),selected_decisions=1,selection={'0':{'game':1}});write(tmp_path/'development-inputs.json',[dict(id='synthetic',corpus=evidence)]);write(tmp_path/'query-set.json',dict(card_ids=[1,2],categories={}))
 for method,n in (('A',None),('PF',512),('RB',256)):
  taskid='A-synthetic' if method=='A' else method+str(n)+'-synthetic';result=replay.run_one(dict(task_id=taskid,id='synthetic',method=method,particles=n,seed=11));assert result['status']=='COMPLETE',result['analysis_error'];assert result['scored']==1 and result['supported']==1
