"""CPU-only synthetic compatibility checks; never start a native game."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from unown_actor import check_supported,CompatibilityError
from unown_common import normalize

def obs(n=63,forced=False):
 return dict(current=dict(players=[dict(hand=[dict(id=96)])]),select=dict(minCount=n if forced else 1,maxCount=n if forced else 1,option=[dict(type=0,number=i) for i in range(n)]),logs=[])

def test_original_limits_accept_all_scored_slots():check_supported(obs(),{96},{120})
def test_forced_large_selection_is_not_truncated():check_supported(obs(64,True),{96},{120})
def test_nonforced_overflow_stops_instead_of_dropping_choices():
 with pytest.raises(CompatibilityError,match='option overflow'):check_supported(obs(64),{96},{120})
def test_new_card_does_not_alias_reserved_embedding():
 x=obs();x['current']['players'][0]['hand'][0]['id']=1283
 with pytest.raises(CompatibilityError,match='Unsupported card'):check_supported(x,{96},{120})
def test_changed_attack_id_is_not_assumed_identical():
 x=obs();x['select']['option'][0]['attackId']=1408
 with pytest.raises(CompatibilityError,match='Unsupported attack'):check_supported(x,{96},{120})
def test_only_surrounding_whitespace_is_normalized():
 assert normalize(dict(name=' Wild Growth',text='text'))==dict(name='Wild Growth',text='text')
 assert normalize(dict(damage=50,name='Tera'))!=normalize(dict(damage=50,name='Orichalcum Fang'))
