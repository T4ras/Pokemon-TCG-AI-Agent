"""Vectorized recurrent PPO with unchanged optimizer windows."""
import random, time
import torch
from torch import nn
from .model import D, collate
from .training import chunks

def optimizer_windows(data,target=1024):
    windows=[];window=[];count=0
    for chunk in data:
        window.append(chunk);count+=chunk[2]
        if count>=target:windows.append(window);window=[];count=0
    if window:windows.append(window)
    return windows

def loss_serial(network,batch,mean,std,clip=.2,entropy_coefficient=.005,bc_kl_coefficient=.01,value_coefficient=.5):
    loss=next(network.parameters()).sum()*0;kl=loss.detach();count=0
    for seq,burn,body in batch:
        hidden=seq[0]["hidden"].to(next(network.parameters()).device)
        for i,row in enumerate(seq):
            stats,hidden=network.decision([row],hidden,actions=[row["action"]],reference=row["reference"])
            if i<burn:hidden=hidden.detach();continue
            _,logp,entropy,value,ref_kl,_=stats
            ratio=(logp-row["old_logp"]).exp();adv=(row["advantage"]-mean)/std
            surrogate=torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
            loss=loss+(-surrogate+value_coefficient*(value-row["return"]).square()-entropy_coefficient*entropy+bc_kl_coefficient*ref_kl).sum()
            kl=kl+((ratio-1)-(logp-row["old_logp"])).detach().sum();count+=1
    return loss,count,kl

def padded_references(rows,option_width,device):
    depth=max((len(r["reference"] or []) for r in rows),default=0);result=[]
    for step in range(depth):
        padded=torch.full((len(rows),option_width+1),float("-inf"),device=device)
        for i,row in enumerate(rows):
            refs=row["reference"] or [];n=len(row["option_ids"])
            if step<len(refs):
                ref=refs[step].reshape(-1).to(device)
                if len(ref)!=n+1:raise ValueError("Reference option width mismatch")
                padded[i,:n]=ref[:n];padded[i,option_width]=ref[n]
            else:
                # This row has already finished decoding, or has maximum zero.
                padded[i,option_width]=0.
        result.append(padded)
    return result

def loss_batched(network,batch,mean,std,clip=.2,diagnostics=None,entropy_coefficient=.005,bc_kl_coefficient=.01,value_coefficient=.5,components=None):
    device=next(network.parameters()).device;B=len(batch);T=max(len(c[0]) for c in batch)
    rows=[batch[i][0][t] if t<len(batch[i][0]) else batch[i][0][-1] for t in range(T) for i in range(B)]
    b=collate(rows,device);global_,options=network.encode_batch(b)
    global_=global_.reshape(T,B,D);options=options.reshape(T,B,options.shape[1],D)
    hidden=torch.cat([seq[0]["hidden"] for seq,_,_ in batch]).to(device)
    loss=global_.sum()*0;kl=loss.detach();count=0
    for t in range(T):
        state={k:v[t*B:(t+1)*B] for k,v in b.items()};current=rows[t*B:(t+1)*B]
        hidden=network.recurrent(global_[t],hidden,state["reset"])
        trained=torch.tensor([t>=burn and t<len(seq) for seq,burn,_ in batch],device=device)
        actions=[row["action"] for row in current]
        refs=padded_references(current,options.shape[2],device)
        _,logp,entropy,value,ref_kl,_=network.decode(options[t],hidden,state,actions=actions,reference=refs)
        old=torch.tensor([r["old_logp"] for r in current],device=device)
        adv=torch.tensor([(r["advantage"]-mean)/std for r in current],device=device)
        returns=torch.tensor([r["return"] for r in current],device=device)
        log_ratio=torch.where(trained,logp-old,torch.zeros_like(logp))
        ratio=log_ratio.exp()
        surrogate=torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
        terms=-surrogate+value_coefficient*(value-returns).square()-entropy_coefficient*entropy+bc_kl_coefficient*ref_kl
        if components is not None:
            for name,part in {"policy":-surrogate,"value":value_coefficient*(value-returns).square()}.items():
                components[name]=components.get(name,0)+(part*trained).sum()
        if not torch.isfinite(terms[trained]).all():raise FloatingPointError("Nonfinite batched PPO loss")
        if diagnostics is not None:
            for name,value_ in {"action_entropy":entropy,"policy_loss":-surrogate,"value_loss":value_coefficient*(value-returns).square(),"value_mse":(value-returns).square(),"bc_reference_kl":ref_kl,"clip_fraction":((ratio-1).abs()>.2).float()}.items():
                diagnostics[name]=diagnostics.get(name,0)+(value_*trained).detach().sum()
        if diagnostics is not None:
            meaningful=torch.tensor([bool(row.get("meaningful",True)) for row in current],device=device)&trained
            diagnostics["meaningful_entropy_sum"]=diagnostics.get("meaningful_entropy_sum",0)+(entropy*meaningful).detach().sum()
            diagnostics["meaningful_count"]=diagnostics.get("meaningful_count",0)+meaningful.detach().sum()
        loss=loss+(terms*trained).sum()
        kl=kl+(((ratio-1)-log_ratio)*trained).detach().sum();count+=int(trained.sum())
        # Same truncation as serial: detach each sequence throughout its burn-in.
        burn_mask=torch.tensor([t<burn for _,burn,_ in batch],device=device)[:,None]
        hidden=torch.where(burn_mask,hidden.detach(),hidden)
    return loss,count,kl

def update(network,optimizer,games,batch_sequences,epochs=2,seed=42,callback=None,target_kl=None,deadline=None,allow_oom_fallback=False,collect_diagnostics=False,entropy_coefficient=.005,bc_kl_coefficient=.01,value_coefficient=.5,component_gradients=False):
    """Identical optimizer windows; optional production KL/deadline safeguards."""
    import gc, math
    coefficients=dict(entropy_coefficient=entropy_coefficient,bc_kl_coefficient=bc_kl_coefficient,value_coefficient=value_coefficient)
    if any(not math.isfinite(v) or v<0 for v in coefficients.values()):raise ValueError("Invalid loss coefficient")
    kwargs={k:v for k,v in coefficients.items() if v!={"entropy_coefficient":.005,"bc_kl_coefficient":.01,"value_coefficient":.5}[k]}
    network.eval();flat=[r for game in games for r in game]
    adv=torch.tensor([r["advantage"] for r in flat]);mean=float(adv.mean());std=max(float(adv.std(unbiased=False)),1e-6)
    data=chunks(games);metrics=[];rng=random.Random(seed);effective_batch=batch_sequences
    for epoch in range(epochs):
        ordered=list(data);rng.shuffle(ordered);windows=optimizer_windows(ordered)
        trained=0;evaluated=0;kl_sum=0.;loss_sum=0.;updates=0;reason=None;fallbacks=[];epoch_diagnostics={}
        for wi,window in enumerate(windows):
            if deadline is not None and time.monotonic()>=deadline:reason="deadline";break
            denom=sum(c[2] for c in window)
            while True:
                optimizer.zero_grad(set_to_none=True);window_kl=0.;window_loss=0.;window_count=0;window_diagnostics={}
                try:
                    for start in range(0,len(window),effective_batch or 1):
                        group=window[start:start+(effective_batch or 1)]
                        if collect_diagnostics and effective_batch:
                            parts={} if component_gradients and epoch==0 and wi==0 and start==0 else None
                            loss,n,kl=loss_batched(network,group,mean,std,diagnostics=window_diagnostics,components=parts,**kwargs)
                            if parts:
                                shared=[p for name,p in network.named_parameters() if name.startswith(("card.","static_projection.","kind.","features.","transformer.","game_gru."))]
                                pg=torch.autograd.grad(parts["policy"]/denom,shared,retain_graph=True,allow_unused=True)
                                vg=torch.autograd.grad(parts["value"]/denom,shared,retain_graph=True,allow_unused=True)
                                pn=sum((g.square().sum() for g in pg if g is not None),torch.tensor(0.,device=loss.device))
                                vn=sum((g.square().sum() for g in vg if g is not None),torch.tensor(0.,device=loss.device))
                                dot=sum((a.mul(b).sum() for a,b in zip(pg,vg) if a is not None and b is not None),torch.tensor(0.,device=loss.device))
                                if callback:callback(dict(epoch=epoch,window=wi,event="shared_gradients",policy_norm=float(pn.sqrt()),value_norm=float(vn.sqrt()),cosine=float(dot/(pn*vn).sqrt().clamp_min(1e-12))))
                                del pg,vg,parts
                        else:loss,n,kl=(loss_batched if effective_batch else loss_serial)(network,group,mean,std,**kwargs)
                        (loss/denom).backward();window_count+=n;window_kl+=float(kl);window_loss+=float(loss.detach())
                        del loss,kl
                    break
                except torch.cuda.OutOfMemoryError:
                    if not allow_oom_fallback or effective_batch<=4:raise
                    # No optimizer step has occurred. Discard all partial window
                    # gradients and recompute the identical window at smaller B.
                    optimizer.zero_grad(set_to_none=True)
                    if 'loss' in locals():del loss
                    if 'kl' in locals():del kl
                    gc.collect();torch.cuda.empty_cache();effective_batch//=2
                    fallbacks.append(dict(window=wi,batch_sequences=effective_batch))
                    if callback:callback(dict(epoch=epoch,window=wi,event="oom_fallback",batch_sequences=effective_batch))
            evaluated+=window_count;kl_sum+=window_kl;loss_sum+=window_loss
            for name,value_ in window_diagnostics.items():epoch_diagnostics[name]=epoch_diagnostics.get(name,0)+float(value_)
            if target_kl is not None and kl_sum/max(evaluated,1)>target_kl:
                optimizer.zero_grad(set_to_none=True);reason="old_policy_kl";break
            norm=nn.utils.clip_grad_norm_(network.parameters(),.5,error_if_nonfinite=True);optimizer.step();updates+=1;trained+=window_count
            if callback:callback(dict(epoch=epoch,window=wi,decisions=trained,gradient_norm=float(norm),approx_old_policy_kl=kl_sum/max(evaluated,1),batch_sequences=effective_batch))
        metrics.append(dict(epoch=epoch,decisions=trained,evaluated_decisions=evaluated,optimizer_updates=updates,mean_loss=loss_sum/max(evaluated,1),approx_old_policy_kl=kl_sum/max(evaluated,1),stopped=reason is not None,stop_reason=reason,batch_sequences=effective_batch,oom_fallbacks=fallbacks,diagnostics={k:v/max(evaluated,1) for k,v in epoch_diagnostics.items()}))
        if reason:break
    return metrics
