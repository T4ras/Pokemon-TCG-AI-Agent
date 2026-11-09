"""Bounded BC and PPO pilots with atomic resumable checkpoints."""
import copy, json, random, time
from pathlib import Path
import torch
from torch import nn
from .model import D, collate

def save_checkpoint(path,network,optimizer,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(observation_schema=getattr(network,"observation_schema","visible-v1"),model=network.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,metadata=metadata,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temp=path.with_suffix(".tmp");torch.save(payload,temp);temp.replace(path)

def load_checkpoint(path,network,optimizer=None,restore_rng=False):
    checkpoint=torch.load(path,map_location=next(network.parameters()).device,weights_only=False)
    network.load_state_dict(checkpoint["model"])
    if optimizer is not None:optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        random.setstate(checkpoint["python_rng"]);torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint["cuda_rng"]])
    return checkpoint["metadata"]

def load_sequences(processed,split,teacher=None):
    result=[]
    manifest=json.loads((Path(processed)/"manifest.json").read_text())
    for name in manifest["files"]:
        data=json.loads((Path(processed)/name).read_text())
        if data["split"]==split and (teacher is None or data["teacher"]==teacher):result.append(data["rows"])
    return result

def chunks(sequences,length=32,burn=8):
    result=[]
    for sequence in sequences:
        # Separate games so the chunk starts never import prior-game GRU state.
        games=[]
        for row in sequence:
            if not games or row["reset"]:games.append([])
            games[-1].append(row)
        for game in games:
            for begin in range(0,len(game),length):
                prefix=game[max(0,begin-burn):begin]
                body=game[begin:begin+length]
                result.append((prefix+body,len(prefix),len(body)))
    return result

def sequence_loss(network,batch):
    device=next(network.parameters()).device
    B=len(batch);T=max(len(c[0]) for c in batch)
    # Flatten in time-major order for one batched state transformer call.
    rows=[batch[b][0][t] if t<len(batch[b][0]) else batch[b][0][-1] for t in range(T) for b in range(B)]
    tensors=collate(rows,device);global_,options=network.encode_batch(tensors)
    global_=global_.reshape(T,B,D);options=options.reshape(T,B,options.shape[1],D)
    hidden=torch.zeros(B,D,device=device);nll=global_.sum()*0;value_loss=nll;count=0
    for t in range(T):
        b={k:v[t*B:(t+1)*B] for k,v in tensors.items()}
        hidden=network.recurrent(global_[t],hidden,b["reset"])
        trained=torch.tensor([t>=burn and t<len(seq) and seq[t].get("trainable",True) for seq,burn,_ in batch],device=device)
        actions=[rows[t*B+i]["action"] for i in range(B)]
        _,logp,_,value,_,_=network.decode(options[t],hidden,b,actions)
        targets=torch.tensor([rows[t*B+i]["outcome"] for i in range(B)],device=device,dtype=torch.float32)
        nll=nll-(logp*trained).sum();value_loss=value_loss+((value-targets).square()*trained).sum();count+=int(trained.sum())
        # Burn-in is a truncated recurrent approximation, not an exact full prefix.
        cutoff=torch.tensor([t+1==burn for _,burn,_ in batch],device=device)[:,None]
        hidden=torch.where(cutoff,hidden.detach(),hidden)
    return (nll+0.1*value_loss)/max(count,1),nll.detach(),count

@torch.no_grad()
def evaluate_nll(network,sequences,batch_sequences=8):
    network.eval();data=chunks(sequences);total=0.;count=0
    for start in range(0,len(data),batch_sequences):
        _,n,c=sequence_loss(network,data[start:start+batch_sequences]);total+=float(n);count+=c
    return dict(action_nll=total/max(1,count),decisions=count)

def train_bc(network,train,validation,output,label,epochs=3,learning_rate=3e-4,deadline=None,batch_sequences=8,accumulation=4):
    output=Path(output);optimizer=torch.optim.AdamW(network.parameters(),lr=learning_rate,weight_decay=1e-4)
    data=chunks(train);updates=0;history=[];best=float("inf");interrupted=False
    # Equal deck exposure in the shared model, independent of demonstration volume.
    groups={}
    for c in data:groups.setdefault(c[0][0]["teacher"],[]).append(c)
    for epoch in range(epochs):
        epoch_data=[];target=max(map(len,groups.values()))
        for group in groups.values():
            epoch_data.extend(random.choices(group,k=target) if len(group)<target else group)
        random.shuffle(epoch_data);network.train();optimizer.zero_grad();pending=0;seen=0;loss_sum=0.
        for start in range(0,len(epoch_data),batch_sequences):
            if deadline is not None and time.monotonic()>deadline:interrupted=True;break
            batch=epoch_data[start:start+batch_sequences]
            loss,_,count=sequence_loss(network,batch)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite BC loss")
            (loss/accumulation).backward();pending+=1;seen+=count;loss_sum+=float(loss.detach())
            if pending==accumulation:
                nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();optimizer.zero_grad();pending=0;updates+=1
        if pending:
            # Normalize the final partial accumulation to its actual batch count.
            for parameter in network.parameters():
                if parameter.grad is not None:parameter.grad.mul_(accumulation/pending)
            nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();optimizer.zero_grad();updates+=1
        metrics=evaluate_nll(network,validation)
        record=dict(stage=label,epoch=epoch,updates=updates,trained_decisions=seen,train_loss_sum=loss_sum,validation=metrics,interrupted=interrupted)
        history.append(record);print(json.dumps(record),flush=True)
        with (output/"training.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        save_checkpoint(output/f"{label}-last.pt",network,optimizer,dict(label=label,epoch=epoch,updates=updates,history=history))
        if metrics["action_nll"]<best:
            best=metrics["action_nll"];save_checkpoint(output/f"{label}.pt",network,optimizer,dict(label=label,epoch=epoch,updates=updates,history=history))
        if interrupted:break
    load_checkpoint(output/f"{label}.pt",network);network.eval()
    return history

def resume_check(network,checkpoint,rows):
    optimizer=torch.optim.AdamW(network.parameters(),lr=1e-4)
    metadata=load_checkpoint(checkpoint,network,optimizer)
    network.eval();hidden=torch.zeros(len(rows),D,device=next(network.parameters()).device)
    actions=[r["action"] for r in rows]
    with torch.no_grad():before=network.decision(rows,hidden,actions)[0][1].clone()
    clone=copy.deepcopy(network);clone_optimizer=torch.optim.AdamW(clone.parameters(),lr=1e-4)
    load_checkpoint(checkpoint,clone,clone_optimizer,restore_rng=True)
    with torch.no_grad():after=clone.decision(rows,hidden,actions)[0][1]
    if not torch.equal(before,after):raise AssertionError("Resume changed policy likelihoods")
    if len(optimizer.state)!=len(clone_optimizer.state):raise AssertionError("Optimizer state not restored")
    # The next optimizer update must agree as well, including moments and step count.
    for model,opt in ((network,optimizer),(clone,clone_optimizer)):
        opt.zero_grad();stats,_=model.decision(rows,hidden,actions);loss=-stats[1].mean();loss.backward();opt.step()
    for a,b in zip(network.parameters(),clone.parameters()):
        if not torch.equal(a,b):raise AssertionError("Resume changed the next optimizer update")
    load_checkpoint(checkpoint,network)
    return dict(policy_likelihood_equal=True,next_optimizer_update_equal=True,optimizer_state_entries=len(optimizer.state),metadata=metadata["label"])

def ppo_update(network,optimizer,games,epochs=2,clip=0.2,minibatch=1024,deadline=None,progress_callback=None):
    """Fresh recurrent game sequences; behavior hidden state initializes each chunk."""
    network.eval();data=chunks(games);flat=[row for game in games for row in game]
    adv=torch.tensor([r["advantage"] for r in flat]);mean=float(adv.mean());std=max(float(adv.std(unbiased=False)),1e-6)
    metrics=[]
    for epoch in range(epochs):
        random.shuffle(data);optimizer.zero_grad();pending=0;trained=0;epoch_kl=0.;stop=False
        for seq,burn,body in data:
            if deadline is not None and time.monotonic()>deadline:stop=True;break
            device=next(network.parameters()).device
            hidden=seq[0]["hidden"].to(device)
            loss=hidden.sum()*0;count=0
            for i,row in enumerate(seq):
                stats,hidden=network.decision([row],hidden,actions=[row["action"]],reference=row["reference"])
                _,logp,entropy,value,ref_kl,_=stats
                if i<burn:hidden=hidden.detach();continue
                old=torch.tensor(row["old_logp"],device=device);ratio=(logp-old).exp();a=(row["advantage"]-mean)/std
                if not torch.isfinite(ratio).all():raise FloatingPointError("Nonfinite PPO probability ratio")
                surrogate=torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a)
                loss=loss+(-surrogate+0.5*(value-row["return"]).square()-0.005*entropy+0.01*ref_kl).mean();count+=1
                epoch_kl+=float(((ratio-1)-(logp-old)).detach().item())
            if count:
                (loss/minibatch).backward();pending+=count;trained+=count
            if pending>=minibatch:
                norm=nn.utils.clip_grad_norm_(network.parameters(),0.5,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad();pending=0
                if progress_callback:progress_callback(dict(epoch=epoch,decisions=trained,gradient_norm=float(norm),approx_old_policy_kl=epoch_kl/max(trained,1)))
                if epoch_kl/max(trained,1)>0.01:stop=True;break
        if pending:
            for parameter in network.parameters():
                if parameter.grad is not None:parameter.grad.mul_(minibatch/pending)
            nn.utils.clip_grad_norm_(network.parameters(),0.5,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad()
        metrics.append(dict(epoch=epoch,decisions=trained,approx_old_policy_kl=epoch_kl/max(trained,1),stopped=stop))
        if stop:break
    return metrics
