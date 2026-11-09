import torch
from ptcg_agent.learning.model import PolicyNetwork,D

def test_cross_attention_stays_float32_inside_half_autocast():
    torch.set_num_threads(2);torch.manual_seed(1)
    network=PolicyNetwork([[0.]*13]*2,2,dropout=0.)
    # Inject a large but finite encoder residual; real failure reached this
    # attention output range after epoch ten.
    class LargeEncoder(torch.nn.Module):
        def forward(self,x,**kwargs):return torch.ones_like(x)*100000.
    network.transformer=LargeEncoder()
    with torch.no_grad():
        network.option_cross.in_proj_weight.zero_();network.option_cross.in_proj_bias.zero_()
        network.option_cross.in_proj_weight[2*D:3*D].copy_(torch.eye(D))
        network.option_cross.out_proj.weight.copy_(torch.eye(D));network.option_cross.out_proj.bias.zero_()
    from test_lr_search import tiny_data
    from ptcg_agent.learning.model import collate
    train,_=tiny_data();row=train[0][0]
    # The synthetic row uses SDK card IDs, so remap for this minimal network.
    row=dict(row,ids=[0]*len(row['ids']),option_ids=[0]*len(row['option_ids']),targets=[0]*len(row['targets']),attack_ids=[0]*len(row['attack_ids']))
    b=collate([row],'cpu');outputs=[]
    network.option_cross.register_forward_hook(lambda m,i,o:outputs.append(o[0]))
    with torch.autocast('cpu',dtype=torch.float16):global_,options=network.encode_batch(b)
    assert outputs[0].dtype==torch.float32
    assert torch.isfinite(options).all()
    assert outputs[0].abs().max()>torch.finfo(torch.float16).max
    options.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


def test_numerical_failure_retains_actual_microbatch_and_rng(tmp_path,monkeypatch):
    from test_lr_search import setup_trial
    import ptcg_agent.learning.lr_search as search
    trial=setup_trial(tmp_path)('failure-capture')
    trial.config['capture_numerical_failure']=True
    def fail(*args,**kwargs):raise FloatingPointError('Injected forward overflow')
    monkeypatch.setattr(search,'sequence_loss',fail)
    result=trial.train('failure-capture',.0003,max_updates=1)
    assert result['status']=='FAILED_NUMERICAL'
    record=torch.load(tmp_path/'failure-capture/failed-microbatch.pt',weights_only=False)
    assert record['update']==0 and record['chunk_offset']==0
    assert record['batch'] and record['torch_rng'].dtype==torch.uint8
    assert trial.state['updates']==0
