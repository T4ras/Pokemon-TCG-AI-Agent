import pytest
from ptcg_agent.learning.full_bc import cosine_lr

def test_cosine_boundaries_and_monotonicity():
    kwargs=dict(peak=.003,floor=.00003,warmup=193,total=6948)
    values=[cosine_lr(**kwargs,update=i) for i in range(6948)]
    assert values[0]==pytest.approx(.003/193)
    assert values[192]==pytest.approx(.003)
    assert values[193]==pytest.approx(.003)
    assert values[-1]==pytest.approx(.00003)
    assert all(a>=b for a,b in zip(values[193:],values[194:]))

def test_budget_splice_and_resume_have_no_lr_jump():
    original=dict(peak=.003,floor=.00003,warmup=193,total=6948)
    anchor=dict(update=579,lr=cosine_lr(**original,update=579))
    adjusted=dict(original,total=6176,anchor=anchor)
    assert cosine_lr(**adjusted,update=579)==pytest.approx(anchor['lr'])
    assert cosine_lr(**adjusted,update=6175)==pytest.approx(.00003)
    import json
    restored=json.loads(json.dumps(adjusted))
    assert cosine_lr(**adjusted,update=2000)==cosine_lr(**restored,update=2000)


def test_full_trial_checkpoint_continuation_preserves_schedule(tmp_path):
    import json,torch
    from ptcg_agent.engine import load_sdk
    from ptcg_agent.learning.model import PolicyNetwork,metadata_tables
    from ptcg_agent.learning.lr_search import atomic_save,new_state,make_scaler
    from ptcg_agent.learning.full_bc import FullTrial
    from test_lr_search import tiny_data
    torch.set_num_threads(2);torch.manual_seed(17)
    sdk,_=load_sdk();table,attacks=metadata_tables(sdk);network=PolicyNetwork(table,attacks)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.003,weight_decay=.0001)
    initial=tmp_path/'initial.pt';atomic_save(initial,network,optimizer,make_scaler(torch.device('cpu')),new_state(17))
    train,val=tiny_data();config=dict(total_epochs=6,learning_rate=.003,min_lr=.00003,warmup_epochs=1,batch_sequences=4,accumulation=1,chunks_per_teacher=1,weight_decay=.0001,clip_grad_norm=1.,validation_interval_updates=100,periodic_checkpoint_updates=100,save_every_epochs=6,max_epochs=6)
    def make(name,start):return FullTrial(table,attacks,config,tmp_path/name,torch.device('cpu'),train,val,start)
    whole=make('whole',initial);whole.train('whole',.003,warmup_epochs=1,finish_epoch=6)
    partial=make('partial',initial);partial.train('partial',.003,warmup_epochs=1,finish_epoch=6,max_updates=3)
    resumed=make('resumed',tmp_path/'partial/last.pt');resumed.train('resumed',.003,warmup_epochs=1,finish_epoch=6)
    assert all(torch.equal(a,b) for a,b in zip(whole.network.parameters(),resumed.network.parameters()))
    assert whole.state['cosine']==resumed.state['cosine']
    assert resumed.state['updates']==6
