import copy, json
from pathlib import Path
import pytest
import torch
from ptcg_agent.engine import load_sdk, sample_deck
from ptcg_agent.learning.observations import DeckKnowledge, encode
from ptcg_agent.learning.model import PolicyNetwork, metadata_tables, D
from ptcg_agent.learning.live import NeuralAgent, play_bo3, balanced_schedule
from ptcg_agent.learning.training import save_checkpoint, resume_check, ppo_update

torch.set_num_threads(2)

def opening():
    players=[dict(active=[],bench=[],hand=[],discard=[],prize=[],handCount=0,deckCount=60),dict(active=[],bench=[],hand=None,discard=[],prize=[],handCount=0,deckCount=60)]
    return dict(current=dict(players=players,yourIndex=0,round=1,turn=0,turnActionCount=1,firstPlayer=-1,stadium=[]),select=dict(type=9,context=41,minCount=1,maxCount=1,option=[dict(type=1),dict(type=2)]),logs=[])

def network():
    api,_=load_sdk();table,attacks=metadata_tables(api);return PolicyNetwork(table,attacks).eval()

def test_encoder_ignores_hidden_and_match_score():
    a=opening();b=copy.deepcopy(a);b['search_begin_input']='private opaque bytes';b['visualize']={'opponent_deck':[723]*60};b['current']['players'][1]['win']=2;b['current']['round']=3
    assert encode(a,sample_deck(),DeckKnowledge())==encode(b,sample_deck(),DeckKnowledge())

def test_revealed_deck_counts_do_not_accumulate_across_games():
    k=DeckKnowledge();obs=opening();obs['current']['players'][1]['discard']=[dict(id=3,serial=61),dict(id=3,serial=62)]
    assert k.update(obs);assert k.lower[3]==2
    assert not k.update(obs);assert k.lower[3]==2
    obs['current']['round']=2;obs['current']['players'][1]['discard']=[dict(id=3,serial=88),dict(id=3,serial=89)]
    assert k.update(obs);assert k.lower[3]==2
    obs['current']['players'][1]['discard'].append(dict(id=3,serial=90));k.update(obs);assert k.lower[3]==3

def test_joint_likelihood_multi_select_and_stop():
    model=network();obs=opening();obs['select'].update(minCount=1,maxCount=3,option=[dict(type=0,number=i) for i in range(4)])
    row=encode(obs,sample_deck(),DeckKnowledge());hidden=torch.zeros(1,D)
    with torch.no_grad():
        stats,_=model.decision([row],hidden,actions=[[2,0]])
        refs=stats[-1];expected=refs[0][0,2]+refs[1][0,0]+refs[2][0,4]
        assert torch.allclose(stats[1][0],expected)
        for _ in range(12):
            action=model.decision([row],hidden)[0][0][0]
            assert 1<=len(action)<=3 and len(set(action))==len(action)
    with pytest.raises(ValueError):model.decision([row],hidden,actions=[[2,2]])

def test_resume_restores_next_update(tmp_path):
    model=network();row=encode(opening(),sample_deck(),DeckKnowledge());row['action']=[0]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    stats,_=model.decision([row],torch.zeros(1,D),actions=[[0]]);(-stats[1].mean()).backward();optimizer.step()
    path=tmp_path/'checkpoint.pt';save_checkpoint(path,model,optimizer,dict(label='test'))
    result=resume_check(model,path,[row]);assert result['next_optimizer_update_equal']

def test_native_bo3_reuses_engine_balances_chooser_and_ppo(tmp_path):
    model=network();reference=copy.deepcopy(model);opponent=copy.deepcopy(model)
    before=[p.detach().clone() for p in opponent.parameters()]
    matches=[];games=[]
    for i,seat in enumerate(balanced_schedule(2,123)):
        learner=NeuralAgent(model,sample_deck(),reference=reference,collect=True)
        other=NeuralAgent(opponent,sample_deck())
        agents=[learner,other] if seat==0 else [other,learner]
        summary,rollouts=play_bo3(agents,f'test-{i}',tmp_path,max_seconds=60)
        assert summary['status']=='DONE',summary
        assert summary['games'][0]['chooser']==0
        assert all(g['chooser']==(g['round']-1)%2 for g in summary['games'])
        assert summary['games'][0]['first_player'] in (0,1)
        games.extend(rollouts[seat]);matches.append(summary)
    assert sum(m['games'][0]['chooser']==s for m,s in zip(matches,balanced_schedule(2,123)))==1
    # Small complete-game PPO smoke; no optimizer may run in an active BO3.
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5)
    metrics=ppo_update(model,optimizer,games[:1],epochs=1,minibatch=1024)
    assert metrics[0]['decisions']>0
    assert all(torch.equal(a,b) for a,b in zip(before,opponent.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
