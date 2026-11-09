"""Matched BC representation pilot with fixed final endpoints and recoverable arms."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from .model import PolicyNetwork, initialize_semantic, schema_from_state
from .lr_search import Trial, atomic_save, make_scaler, new_state, epoch_order
from .training import load_sequences, chunks, resume_check

def digest(network):
    h=hashlib.sha256()
    for name,tensor in sorted(network.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(inputs, output, config, require_gpu=True, smoke=False):
    inputs,output=Path(inputs),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError("Fresh pilot output required; retain prior attempts")
    started=time.monotonic();torch.set_num_threads(2)
    if require_gpu and not torch.cuda.is_available():raise RuntimeError("GPU required")
    device=torch.device("cuda" if require_gpu else "cpu")
    state=torch.load(inputs/"initial.pt",map_location="cpu",weights_only=True)["model"]
    if schema_from_state(state)!="visible-v1":raise ValueError("BC34 legacy initialization required")
    legacy=PolicyNetwork(state["static_cards"].tolist(),state["attack.weight"].shape[0])
    legacy.load_state_dict(state)
    torch.manual_seed(config["seed"]);random.seed(config["seed"])
    semantic=initialize_semantic(legacy)
    if any(not torch.equal(t,semantic.state_dict()[name]) for name,t in legacy.state_dict().items()):
        raise AssertionError("Shared initial tensors differ")
    initial={"control":legacy,"semantic":semantic}
    shared=digest(legacy)
    report=dict(status="RUNNING",config=config,smoke=smoke,device=str(device),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0) if require_gpu else None,
        allocated_gpu_count=torch.cuda.device_count(),shared_initial_tensor_sha256=shared,arms={})
    def persist():
        report["wall_seconds"]=time.monotonic()-started
        p=output/"report.json";tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
    persist()
    schedules={}
    for arm in ("control","semantic"):
        arm_started=time.monotonic();folder=output/arm;folder.mkdir()
        net=initial.pop(arm).to(device)
        arm_config=dict(config,observation_schema=net.observation_schema)
        processed=inputs/arm
        train=load_sequences(processed,"train",2)
        val=load_sequences(processed,"validation",2)
        if smoke:
            train=[train[0][:8]];val=[val[0][:8]]
            arm_config.update(batch_sequences=1,accumulation=1)
        optimizer=torch.optim.AdamW(net.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        random.seed(config["seed"]);torch.manual_seed(config["seed"])
        if require_gpu:torch.cuda.manual_seed_all(config["seed"])
        initial_path=folder/"initial.pt"
        atomic_save(initial_path,net,optimizer,make_scaler(device),new_state(config["seed"]))
        trial=Trial(state["static_cards"].tolist(),state["attack.weight"].shape[0],arm_config,folder,device,train,{2:val},initial_path)
        del net,optimizer
        ids={id(c):i for i,c in enumerate(trial.data)}
        count=1 if smoke else config["epochs"]
        orders=[[ids[id(c)] for c in epoch_order(trial.data,config["seed"],epoch)] for epoch in range(count)]
        schedules[arm]=hashlib.sha256(json.dumps(orders).encode()).hexdigest()
        if arm=="semantic" and schedules["semantic"]!=schedules["control"]:
            raise AssertionError("Matched chunk schedules differ")
        deadline=min(started+config["runtime_seconds"],arm_started+config["arm_seconds_cap"])
        result=trial.train(arm,config["learning_rate"],warmup_epochs=0,finish_epoch=count,deadline=deadline,max_updates=1 if smoke else None)
        if result["status"]=="COMPLETE":
            final=folder/"final.pt"
            atomic_save(final,trial.network,trial.optimizer,trial.scaler,trial.state)
            model_only=folder/"model.pt";torch.save({"model":trial.network.state_dict()},model_only)
            result.update(final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
                          model_sha256=hashlib.sha256(model_only.read_bytes()).hexdigest(),
                          trained_decisions=trial.state["decisions"],
                          tensor_sha256=digest(trial.network),schedule_sha256=schedules[arm])
            if smoke:
                result["recovery"]=resume_check(trial.network,final,train[0][:1])
        report["arms"][arm]=result;persist()
        del trial,train,val
        if require_gpu:torch.cuda.empty_cache()
        if result["status"]!="COMPLETE":
            report["status"]="INCOMPLETE";persist();return report
    a,b=(report["arms"][k] for k in ("control","semantic"))
    if (a["updates"],a["trained_decisions"],a["schedule_sha256"])!=(b["updates"],b["trained_decisions"],b["schedule_sha256"]):
        raise AssertionError("Unequal training budgets")
    report.update(status="COMPLETE",matched_schedules=True,matched_updates=a["updates"],
                  matched_decisions_per_arm=a["trained_decisions"])
    persist();print(json.dumps(report),flush=True);return report
