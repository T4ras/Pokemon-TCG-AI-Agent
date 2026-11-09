"""Six-hour two-objective recurrent PPO with bounded durable recovery."""
import copy,datetime,gc,gzip,hashlib,json,os,pickle,random,resource,time
from pathlib import Path
import torch
from .model import PolicyNetwork
from .rl_timing import TimedAgent,apply_returns,schedule
from .live import play_bo3
from .training import save_checkpoint
from .ppo_batched import update

def should_start_round(remaining,previous,initial=420):
    estimate=max(previous[-3:]) if previous else initial
    return remaining>=estimate*1.03+20

def atomic_save(path,payload):
    path=Path(path);temp=path.with_suffix(".tmp");torch.save(payload,temp);temp.replace(path)

def paired_checkpoint(path,learners,optimizers,history,history_ids,completed_rounds,config):
    atomic_save(path,dict(models={k:n.state_dict() for k,n in learners.items()},optimizers={k:o.state_dict() for k,o in optimizers.items()},history={k:[m.state_dict() for m in bank] for k,bank in history.items()},history_ids=history_ids,completed_rounds=completed_rounds,config=config,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

def training_outcomes(matches):
    result={}
    for key in ("game","match"):
        counters={name:dict(win=0,loss=0,draw=0) for name in ("bo3","first_game","all_games")};attempts=0;failures=0
        for m in matches:
            if key not in m["collecting_seats"]:continue
            attempts+=1;seat=m["collecting_seats"][key]
            def label(r):return "draw" if r==2 else "win" if r==seat else "loss"
            if m["status"]=="DONE":counters["bo3"][label(m["result"])]+=1
            else:failures+=1
            if m["games"]:counters["first_game"][label(m["games"][0]["result"])]+=1
            for game in m["games"]:counters["all_games"][label(game["result"])]+=1
        result[key]=dict(attempts=attempts,failures=failures,**counters)
        for name in ("bo3","first_game"):
            counts=counters[name];n=sum(counts.values());result[key][name+"_win_rate"]=counts["win"]/n if n else None
    return result

def run(inputs,output,config,require_gpu=True,script_started=None):
    inputs,output=Path(inputs),Path(output);output.mkdir(parents=True,exist_ok=True)
    initialization_started=time.perf_counter();started=script_started if script_started is not None else initialization_started
    target_deadline=started+config["runtime_seconds"];finish_deadline=target_deadline+config["finish_margin_seconds"]
    if require_gpu and not torch.cuda.is_available():raise RuntimeError("Kaggle GPU required")
    device="cuda" if require_gpu else "cpu"
    torch.set_num_threads(2);random.seed(config["seed"]);torch.manual_seed(config["seed"])
    manifest=json.loads((inputs/"checkpoints.json").read_text());decks=manifest["decks"]
    fixed={}
    for epoch in (30,34):
        source=inputs/f"epoch-{epoch}.pt"
        if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest["sha256"][str(epoch)]:raise ValueError("Checkpoint hash mismatch")
        state=torch.load(source,map_location=device,weights_only=True)["model"]
        model=PolicyNetwork(state["static_cards"].cpu().tolist(),state["attack.weight"].shape[0]).to(device)
        model.load_state_dict(state);model.eval();model.requires_grad_(False);fixed[epoch]=model
    learners={k:copy.deepcopy(fixed[34]).requires_grad_(True) for k in ("game","match")}
    optimizers={k:torch.optim.AdamW([
        {"params":[p for n,p in m.named_parameters() if not n.startswith("value.")],"lr":3e-5},
        {"params":list(m.value.parameters()),"lr":1e-4}],weight_decay=1e-4) for k,m in learners.items()}
    history={k:[copy.deepcopy(fixed[34])] for k in learners};history_ids={k:[0] for k in learners}
    report=dict(status="RUNNING",config=config,hardware=dict(device=device,gpu=torch.cuda.get_device_name(0) if require_gpu else None,gpu_count=torch.cuda.device_count(),cpu_count=os.cpu_count(),torch=torch.__version__),initialization_seconds=time.perf_counter()-initialization_started,rounds=[],native_rng_seeded=False,critic_match_score_input=False,critic_calibration=False)
    def emit(kind,**values):
        event=dict(kind=kind,elapsed_seconds=time.perf_counter()-started,**values)
        with (output/"events.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
        print(json.dumps(event),flush=True)
    def persist():
        report["wall_seconds"]=time.perf_counter()-started
        tmp=output/"report.tmp";tmp.write_text(json.dumps(report,indent=2));tmp.replace(output/"report.json")
    report.update(completed_rounds=0,total_completed_matches=0,benchmarks_scheduled=False,training_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),estimated_target_end_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=max(0,target_deadline-time.perf_counter()))).isoformat(),checkpoints=[])
    paired_checkpoint(output/"paired-latest.pt",learners,optimizers,history,history_ids,0,config)
    emit("initialized",**report["hardware"],target_end_utc=report["estimated_target_end_utc"],budget_seconds=config["runtime_seconds"]);persist()
    for ri in range(config["rounds"]):
        remaining=target_deadline-time.perf_counter()
        if not should_start_round(remaining,[r["round_wall_seconds"] for r in report["rounds"] if "round_wall_seconds" in r],config["round_seconds_initial_estimate"]):
            emit("budget_stop",remaining_seconds=remaining,completed_rounds=report["completed_rounds"]);break
        if time.perf_counter()>finish_deadline:raise TimeoutError("Finishing margin exhausted")
        rt=time.perf_counter();jobs=schedule(ri,config["seed"],history)
        for job in jobs:
            if job["category"]=="history":job["historical_round"]=history_ids[job["lineage"]][job["version"]]
        buffers={k:[] for k in learners};timings=dict(observation_encoding=0.,policy_inference=0.,reference_inference=0.,reward_calculation=0.)
        decisions={k:0 for k in learners};appearances={k:0 for k in learners};chooser={k:0 for k in learners};matches=[]
        ct=time.perf_counter()
        for j,job in enumerate(jobs):
            if job["category"]=="cross":
                a=TimedAgent(learners["game"],decks[2],fixed[34],True);b=TimedAgent(learners["match"],decks[2],fixed[34],True)
                agents=[a,b] if job["seat"]==0 else [b,a];collect={"game":job["seat"],"match":1-job["seat"]}
            else:
                key=job["learner"];a=TimedAgent(learners[key],decks[2],fixed[34],True)
                if job["category"]=="fixed":b=TimedAgent(fixed[job["checkpoint"]],decks[job["deck"]],temperature=job["temperature"])
                else:b=TimedAgent(history[job["lineage"]][job["version"]],decks[2])
                agents=[a,b] if job["seat"]==0 else [b,a];collect={key:job["seat"]}
            if time.perf_counter()>=finish_deadline:raise TimeoutError("In-progress cohort exceeded finishing allowance; partial records retained")
            summary,rollouts=play_bo3(agents,f"r{ri+1}-{j+1}",output,max_seconds=min(config["match_seconds"],max(1,finish_deadline-time.perf_counter())),metadata=dict(round_index=ri+1,schedule=job,collecting_seats=collect,policy_version=ri))
            matches.append(summary)
            for agent in agents:
                for metric in ("observation_encoding","policy_inference","reference_inference"):timings[metric]+=agent.timing[metric]
            if summary["status"]!="DONE":
                report["status"]="FAILED_COLLECTION";report["failure"]=summary;report["partial_training_outcomes"]=training_outcomes(matches);persist();raise RuntimeError("Failed match retained; refusing imbalanced PPO update")
            if summary["games"][0]["chooser"]!=0:raise AssertionError("Native initial chooser mismatch")
            t=time.perf_counter()
            for key,seat in collect.items():
                appearances[key]+=1;chooser[key]+=int(seat==0)
                games=apply_returns(rollouts[seat],summary["result"],seat,key)
                buffers[key].extend(games);decisions[key]+=sum(map(len,games))
            timings["reward_calculation"]+=time.perf_counter()-t
            if sum(decisions.values())>config["rollout_cap_decisions"]:raise RuntimeError("Registered rollout cap exceeded")
            if (j+1)%8==0:emit("collection_progress",round=ri+1,matches=j+1,planned=len(jobs),collection_seconds=time.perf_counter()-ct,decisions=decisions.copy())
        timings["collection_total"]=time.perf_counter()-ct
        timings["engine_and_collection_overhead"]=timings["collection_total"]-sum(timings[k] for k in ("observation_encoding","policy_inference","reference_inference","reward_calculation"))
        for key in learners:
            if appearances[key]!=36 or chooser[key]!=18:raise AssertionError("Learner chooser balance violation")
        record=dict(round=ri+1,matches=len(matches),failures=0,learner_appearances=appearances,chooser_counts=chooser,learner_decisions=decisions,timings_seconds=timings,ppo={},training_outcomes=training_outcomes(matches),status="COLLECTED")
        t=time.perf_counter()
        # One bounded rollout buffer survives a numerical failure during update.
        atomic_save(output/"active-rollouts.pt",dict(round=ri+1,buffers=buffers,schedule=jobs,config=config,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if require_gpu else []))
        record["rollout_serialized_bytes"]=(output/"active-rollouts.pt").stat().st_size
        if record["rollout_serialized_bytes"]>config["rollout_buffer_max_bytes"]:raise RuntimeError("Rollout storage cap exceeded")
        raw=output/"matches.jsonl";shard=output/f"matches-round-{ri+1:03d}.jsonl.gz"
        with raw.open("rb") as source,gzip.open(shard,"wb",compresslevel=3) as target:target.write(source.read())
        raw.unlink();timings["rollout_recovery_and_log_shard"]=time.perf_counter()-t
        report["rounds"].append(record);persist();emit("collection_complete",**record)
        for key in ("game","match"):
            emit("update_started",round=ri+1,learner=key);t=time.perf_counter()
            metrics=update(learners[key],optimizers[key],buffers[key],config["batch_sequences"],epochs=2,seed=config["seed"]+ri*2+(key=="match"),target_kl=config["old_policy_kl_stop"],deadline=finish_deadline,allow_oom_fallback=config["allow_oom_fallback"],collect_diagnostics=True,callback=lambda m,k=key:emit("update_progress",round=ri+1,learner=k,**m))
            if require_gpu:torch.cuda.synchronize()
            timings[f"ppo_{key}"]=time.perf_counter()-t;record["ppo"][key]=metrics
            if not all(torch.isfinite(p).all() for p in learners[key].parameters()):raise FloatingPointError("Nonfinite updated parameter")
            emit("update_complete",round=ri+1,learner=key,seconds=timings[f"ppo_{key}"],metrics=metrics);persist()
        t=time.perf_counter()
        for key in learners:
            save_checkpoint(output/f"{key}-round-{ri+1}.pt",learners[key],optimizers[key],dict(round=ri+1,objective=key,deck=decks[2],config=config))
            item=output/f"{key}-round-{ri+1}.pt"
            report["checkpoints"].append(dict(file=item.name,objective=key,round=ri+1,sha256=hashlib.sha256(item.read_bytes()).hexdigest(),bytes=item.stat().st_size))
            historical=copy.deepcopy(learners[key]).eval().requires_grad_(False);history[key].append(historical);history[key]=history[key][-4:]
            history_ids[key].append(ri+1);history_ids[key]=history_ids[key][-4:]
        paired_checkpoint(output/"paired-latest.pt",learners,optimizers,history,history_ids,ri+1,config)
        (output/"active-rollouts.pt").unlink()
        if require_gpu:torch.cuda.synchronize()
        timings["checkpoint_and_history"]=time.perf_counter()-t
        checkpoint_bytes=sum(p.stat().st_size for p in output.glob("*.pt"))
        if checkpoint_bytes>config["max_checkpoint_bytes"]:raise RuntimeError("Checkpoint storage cap exceeded")
        record["round_wall_seconds"]=time.perf_counter()-rt
        record.update(status="UPDATED",host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if require_gpu else 0,checkpoint_bytes=checkpoint_bytes)
        report["completed_rounds"]=ri+1;report["total_completed_matches"]+=64
        for key in optimizers:optimizers[key].zero_grad(set_to_none=True)
        del buffers,rollouts,agents,a,b,games;gc.collect()
        if require_gpu:torch.cuda.empty_cache()
        emit("round_complete",**record);persist()
    report["status"]="COMPLETE";report["completion_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["final_checkpoints"]={k:f"{k}-round-{report['completed_rounds']}.pt" for k in learners}
    report["paired_recovery_sha256"]=hashlib.sha256((output/"paired-latest.pt").read_bytes()).hexdigest()
    persist();emit("complete",wall_seconds=report["wall_seconds"],completed_rounds=report["completed_rounds"],matches=report["total_completed_matches"],benchmarks_scheduled=False)
    return report

