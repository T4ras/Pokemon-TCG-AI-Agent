"""Synthetic checks of actual original policy, likelihood and visibility boundaries."""
import copy
import pytest
import torch
from ptcg_agent.learning.model import PolicyNetwork,D
from ptcg_agent.learning.observations import DeckKnowledge,encode
def observation():
    p=dict(active=[],bench=[],hand=[],discard=[],prize=[],handCount=0,deckCount=60)
    enemy=copy.deepcopy(p);enemy["hand"]=None
    return dict(current=dict(players=[p,enemy],yourIndex=0,round=1,turn=0,turnActionCount=1,firstPlayer=-1,stadium=[]),select=dict(type=9,context=41,minCount=1,maxCount=3,option=[dict(type=0,number=i) for i in range(4)]),logs=[])
def test_visible_encoder_excludes_hidden_fields_and_match_score():
    a=observation();b=copy.deepcopy(a);b["search_begin_input"]="hidden";b["visualize"]={"opponent_deck":[999]*60};b["current"]["players"][1]["win"]=2
    assert encode(a,[1]*60,DeckKnowledge())==encode(b,[1]*60,DeckKnowledge())
def test_revealed_copies_deduplicate_and_maximize_across_games():
    k=DeckKnowledge();o=observation();o["current"]["players"][1]["discard"]=[dict(id=3,serial=61),dict(id=3,serial=62)]
    assert k.update(o);assert k.lower[3]==2;assert not k.update(o);assert k.lower[3]==2
    o["current"]["round"]=2;o["current"]["players"][1]["discard"]=[dict(id=3,serial=88),dict(id=3,serial=89)]
    assert k.update(o);assert k.lower[3]==2;assert DeckKnowledge().lower[3]==0
def test_transformer_gru_joint_likelihood_includes_stop():
    torch.manual_seed(11);torch.set_num_threads(1)
    model=PolicyNetwork([[0.]*8 for _ in range(8)],8,dropout=0).eval()
    assert len(model.transformer.layers)==4 and model.game_gru.hidden_size==192 and model.pick_gru.hidden_size==192
    row=encode(observation(),[1]*60,DeckKnowledge())
    with torch.no_grad():
        stats,_=model.decision([row],torch.zeros(1,D),actions=[[2,0]]);refs=stats[-1]
        assert torch.allclose(stats[1][0],refs[0][0,2]+refs[1][0,0]+refs[2][0,4])
    with pytest.raises(ValueError):model.decision([row],torch.zeros(1,D),actions=[[2,2]])
