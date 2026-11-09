"""Controlled CUDA 4/8/16 sequence throughput and parity benchmark."""
import copy,gc,hashlib,json,random,statistics,time
from pathlib import Path
import torch
from .model import PolicyNetwork
from .training import chunks
from .ppo_batched import loss_serial,loss_batched,update

def run(inputs,output,config):
    inputs,output=Path(inputs),Path(output);output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("Kaggle GPU required")
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(config["seed"]);start=time.perf_counter()
    state=torch.load(inputs/"epoch-34.pt",map_location="cpu",weights_only=True)["model"]
    corpus=torch.load(inputs/"rollouts.pt",map_location="cpu",weights_only=False)
    games=corpus["games"];flat=[r for g in games for r in g];data=chunks(games)
    adv=torch.tensor([r["advantage"] for r in flat]);mean=float(adv.mean());std=max(float(adv.std(unbiased=False)),1e-6)
    def model():
        net=PolicyNetwork(state["static_cards"].tolist(),state["attack.weight"].shape[0]).cuda().eval();net.load_state_dict(state);return net
    def optimizer(net):return torch.optim.AdamW([{"params":[p for n,p in net.named_parameters() if not n.startswith("value.")],"lr":3e-5},{"params":list(net.value.parameters()),"lr":1e-4}],weight_decay=1e-4)
    report=dict(status="RUNNING",config=config,gpu=torch.cuda.get_device_name(0),total_gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,corpus_decisions=len(flat),corpus_sequences=len(data),corpus_sha256=hashlib.sha256((inputs/"rollouts.pt").read_bytes()).hexdigest(),parity=[],trials=[])
    def emit(kind,**kw):
        e=dict(kind=kind,elapsed_seconds=time.perf_counter()-start,**kw)
        print(json.dumps(e),flush=True)
        with (output/"events.jsonl").open("a") as f:f.write(json.dumps(e)+"\n")
    def persist():
        report["wall_seconds"]=time.perf_counter()-start
        (output/"report.json").write_text(json.dumps(report,indent=2))
    sample=random.Random(config["seed"]).sample(data,min(16,len(data)));denom=sum(c[2] for c in sample)
    net=model();net.zero_grad();loss,n,_=loss_serial(net,sample,mean,std);(loss/denom).backward()
    base_loss=float(loss.detach()/denom);base_grad=torch.cat([p.grad.detach().flatten().cpu() for p in net.parameters() if p.grad is not None])
    del net,loss;gc.collect();torch.cuda.empty_cache()
    for size in config["sizes"]:
        net=model();net.zero_grad();total=0.
        for j in range(0,len(sample),size):
            loss,n,_=loss_batched(net,sample[j:j+size],mean,std);(loss/denom).backward();total+=float(loss.detach())/denom
        grad=torch.cat([p.grad.detach().flatten().cpu() for p in net.parameters() if p.grad is not None]);difference=grad-base_grad
        record=dict(batch_sequences=size,mean_loss=total,serial_mean_loss=base_loss,loss_abs_difference=abs(total-base_loss),gradient_relative_l2=float(difference.norm()/base_grad.norm().clamp_min(1e-12)),gradient_max_abs=float(difference.abs().max()))
        record["passed"]=record["loss_abs_difference"]<.002 and record["gradient_relative_l2"]<.005 and record["gradient_max_abs"]<.002
        report["parity"].append(record);emit("parity",**record);persist()
        del net,loss,grad,difference;gc.collect();torch.cuda.empty_cache()
        if not record["passed"]:raise AssertionError("CUDA serial/batch numerical parity failed")
    # Same full dataset, shuffle seed, optimizer windows and epochs in every trial.
    plan=[(0,0)]+[(size,repeat) for repeat in range(config["repeats"]) for size in config["sizes"]]
    random.Random(config["seed"]+1).shuffle(plan)
    for size,repeat in plan:
        emit("trial_started",batch_sequences=size,repeat=repeat)
        warm=model();opt=optimizer(warm);torch.cuda.synchronize();t=time.perf_counter()
        group=sample[:size or 1];loss,n,_=(loss_batched if size else loss_serial)(warm,group,mean,std);(loss/n).backward();opt.step();torch.cuda.synchronize();warm_seconds=time.perf_counter()-t
        del warm,opt,loss;gc.collect();torch.cuda.empty_cache()
        net=model();opt=optimizer(net);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        metrics=update(net,opt,games,size,epochs=config["ppo_epochs"],seed=config["seed"],callback=lambda e:emit("update_progress",batch_sequences=size,repeat=repeat,**e) if e["window"]%4==0 else None)
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        trained=sum(m["decisions"] for m in metrics)
        record=dict(batch_sequences=size,repeat=repeat,seconds=seconds,warmup_seconds=warm_seconds,trained_decisions=trained,decisions_per_second=trained/seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),metrics=metrics,parameters_finite=all(torch.isfinite(p).all() for p in net.parameters()))
        if not record["parameters_finite"]:raise FloatingPointError("Nonfinite benchmark parameters")
        report["trials"].append(record);emit("trial_complete",**record);persist()
        del net,opt;gc.collect();torch.cuda.empty_cache()
    baseline=next(t for t in report["trials"] if t["batch_sequences"]==0);summary=[]
    for size in config["sizes"]:
        trials=[t for t in report["trials"] if t["batch_sequences"]==size];seconds=statistics.median(t["seconds"] for t in trials)
        summary.append(dict(batch_sequences=size,median_seconds=seconds,min_seconds=min(t["seconds"] for t in trials),max_seconds=max(t["seconds"] for t in trials),decisions_per_second=2*len(flat)/seconds,speedup_vs_serial=baseline["seconds"]/seconds,peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),peak_reserved_bytes=max(t["peak_reserved_bytes"] for t in trials)))
    report["summary"]=summary;report["status"]="COMPLETE";persist();emit("complete",summary=summary,serial_seconds=baseline["seconds"])
    return report
