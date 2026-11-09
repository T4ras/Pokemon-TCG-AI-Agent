import copy,torch
from ptcg_agent.learning.model import PolicyNetwork,D
from ptcg_agent.learning.ppo_batched import loss_serial,loss_batched,optimizer_windows

def test_loss_and_gradients_match_with_padding_stop_burn_and_resets():
    torch.set_num_threads(2);torch.manual_seed(17)
    net=PolicyNetwork([[0.]*13],1).eval();batch=[]
    for k,options in enumerate([1,3,0,2]):
        hidden=torch.randn(1,D)*.1;start_hidden=hidden.clone();rows=[]
        for i in range(2+k):
            action=[] if not options else [0] if i%2 else list(range(min(2,options)))
            row=dict(ids=[0]*(1+k),kinds=[0]*(1+k),features=[[0.]*32]*(1+k),option_ids=[0]*options,targets=[0]*options,attack_ids=[0]*options,option_features=[[0.]*32]*options,minimum=0,maximum=min(2,options),reset=i==0)
            before=hidden.clone()
            with torch.no_grad():
                stats,hidden=net.decision([row],hidden,actions=[action])
            row.update(action=action,hidden=before,old_logp=float(stats[1]),old_value=float(stats[3]),reference=stats[-1],advantage=(i-1)*.2,**{"return":.4})
            rows.append(row)
        batch.append((rows,min(k,2),len(rows)-min(k,2)))
    clone=copy.deepcopy(net)
    a,n,ka=loss_serial(net,batch,0.,1.);b,m,kb=loss_batched(clone,batch,0.,1.)
    assert n==m
    torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
    a.backward();b.backward()
    for p,q in zip(net.parameters(),clone.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=2e-3)

def test_optimizer_boundaries_do_not_depend_on_computational_batch():
    chunks=[([],0,n) for n in [32]*31+[17,32,4]]
    windows=optimizer_windows(chunks)
    assert [sum(c[2] for c in w) for w in windows]==[1041,4]
