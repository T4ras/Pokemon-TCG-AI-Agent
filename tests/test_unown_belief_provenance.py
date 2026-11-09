"""Replacement/provenance repairs: synthetic and retained legal histories only."""
import copy,random,gzip,json
from pathlib import Path
import pytest
from unown_belief_provenance.common import *
from unown_belief_provenance.provenance import replacement,validate_identity,UnknownProvenance
from unown_belief_provenance.state import State
from unown_belief_provenance.adapter import Compiler
from unown_belief_provenance.baseline import Facts,baseline

def obs(active=1,serial=11,discard=2,discard_serial=22,deck=58):
 return dict(current=dict(round=1,yourIndex=0,turn=2,turnActionCount=1,players=[dict(deckCount=60,handCount=0,hand=[],prize=[],active=[],bench=[],discard=[]),dict(deckCount=deck,handCount=0,hand=None,prize=[],active=[dict(id=active,serial=serial,playerIndex=1)],bench=[],discard=[dict(id=discard,serial=discard_serial,playerIndex=1)])],stadium=[],looking=None),select=dict(context=0,minCount=1,maxCount=1,option=[{},{}]),logs=[])

def test_trash_replacement_preserves_original_tokens_and_copy_counts():
 for effect in (1090,1415):
  c=Compiler(metadata()['cards']);f=Facts(metadata()['cards']);c.game=f.game=1;c.sizes=f.sizes=dict(D=58,H=0,P=0,L=0);c.note(1,11,'A');c.note(2,22,'B');f.note(1,11,'A');f.note(2,22,'B');c.effect=f.effect=effect
  o=obs(active=2,serial=22,discard=1,discard_serial=11);o['logs']=[dict(type=9,playerIndex=1,cardIdBefore=1,serialBefore=11,cardIdAfter=2,serialAfter=22)]
  compiled=c.compile(o);f.observe(o);assert f.copies[11]==(1,frozenset({'B'})) and f.copies[22]==(2,frozenset({'A'}));assert f.lower=={1:1,2:1};assert f.sizes==dict(D=58,H=0,P=0,L=0)
  s=State([1,2]+[3]*58,random.Random(11));s.z['D']=list(range(2,60));s.z['A']=[0];s.z['B']=[1];s.serial={11:0,22:1}
  for op in compiled['operations']:s.apply(op,random.Random(11))
  s.check();assert s.serial=={11:0,22:1};assert s.location(0)=='B' and s.location(1)=='A';assert s.counts('D')[3]==58
  assert c.compile(o)['duplicate'];assert not f.observe(o)

def test_deck_replacement_preserves_known_top_before_explicit_shuffle():
 e=dict(cardIdBefore=1,serialBefore=11,cardIdAfter=2,serialAfter=22);ops=replacement(e,{11:(1,'B')},106);s=State([1,2,3],random.Random(11));s.z={z:[] for z in s.z};s.z['B']=[0];s.z['D']=[1,2];s.serial={11:0}
 for op in ops:s.apply(op,random.Random(11))
 assert s.serial=={11:0,22:1} and s.z['D'][0]==0 and 0 in s.order_pins and s.location(1)=='B';s.check();s.apply(dict(op='shuffle'),random.Random(29));assert not s.order_pins

@pytest.mark.parametrize('effect',[None,999999])
def test_unknown_replacement_is_not_guessed(effect):
 with pytest.raises(UnknownProvenance):replacement(dict(cardIdBefore=1,serialBefore=11,cardIdAfter=2,serialAfter=22),{11:(1,'A')},effect)

def test_outside_library_card_does_not_poison_deck_lower_bounds():
 c=Compiler(metadata()['cards']);f=Facts(metadata()['cards']);before=(dict(c.lower),dict(f.lower));external=999999
 with pytest.raises(UnknownProvenance):c.note(external,77,'H')
 with pytest.raises(UnknownProvenance):f.note(external,77,'H')
 assert (c.lower,f.lower)==before and 77 not in c.known and 77 not in f.copies

def test_identity_change_is_not_conflated_with_replacing_two_cards():
 with pytest.raises(UnknownProvenance):replacement(dict(cardIdBefore=1,serialBefore=11,cardIdAfter=2,serialAfter=11),{11:(1,'A')},1415)
 with pytest.raises(UnknownProvenance):validate_identity(2,11,{11:(1,'A')},{1,2})

def test_replacement_conflicting_source_does_not_mutate_known_facts():
 known={11:(1,'A'),22:(2,'H')};saved=copy.deepcopy(known)
 with pytest.raises(UnknownProvenance):replacement(dict(cardIdBefore=1,serialBefore=11,cardIdAfter=2,serialAfter=22),known,1415)
 assert known==saved

def test_existing_16_legal_histories_keep_shared_facts_and_compilation():
 inputs=read(R/'experiments/astra-unown-belief-rao-blackwellization-001/development-inputs.json')
 for row in inputs:
  c=Compiler(metadata()['cards']);f=Facts(metadata()['cards']);hist=json.load(gzip.open(R/row['corpus']['legal_path'],'rt'))
  for e in hist:c.compile(e['observation']);f.observe(e['observation']);f.known_hand()
