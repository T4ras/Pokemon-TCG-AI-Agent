"""One fixed mirror/diverse PPO comparison, with no automatic continuation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy,gc,gzip,json,multiprocessing,os,random,time,traceback
import torch
from .rl_stage1 import load_model,tensor_sha,write_json,sha
from .training import save_checkpoint,load_checkpoint
from .ppo_batched import update,loss_batched
from .economical_collector import collect_matches
from .anchor_checks import diagnose_anchor,require_anchor_equivalence
_WORKER=None
def initialize_worker(runtime_path,output):
    global _WORKER
    torch.set_num_threads(1)
    reg=json.loads(Path(runtime_path).read_text())
    bc=load_model(reg["bc_anchor"]["path"]).requires_grad_(False)
    opponents={p["id"]:bc for p in reg["panel"][:4]}
    opponents.update({p["id"]:load_model(p["path"]).requires_grad_(False) for p in reg["history"]})
    _WORKER=(reg,load_model(reg["source"]["path"]).requires_grad_(False),opponents,Path(output))
def collect_worker(payload):
    arm,cohort,state,jobs,deadline=payload
    reg,network,opponents,output=_WORKER
    network.load_state_dict(state)
    return collect_matches(network,opponents,reg,jobs,prefix=f"{arm}-r{cohort}-p{os.getpid()}",
        output=output/f"worker-{os.getpid()}",deadline=deadline,live_reference=True)

def live_reference_summary(values,games):
    if any(row.get("reference") is None for game in games for row in game):
        raise AssertionError("Missing live CPU anchor reference")
    return dict(seconds=0.,device="cpu",mode="live",included_in_collection_seconds=True,
        cpu_worker_reference_seconds=sum(v["timing"].get("reference_inference",0.) for v in values),
        decisions=sum(map(len,games)),full_game_prefixes=True,forced_observations_retained=True,
        gpu_anchor_forward_calls=0)

def optimizer_for(network,config):
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in network.named_parameters() if not n.startswith("value.")],lr=3e-5),
        dict(params=list(network.value.parameters()),lr=1e-4)],weight_decay=1e-4)
    load_checkpoint(config["source"]["path"],network,optimizer,restore_rng=False)
    return optimizer

def _run(runtime_path,output,config,script_started):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    finish=script_started+6600
    contract=config.get("anchor_runtime")
    if contract is None or contract.get("reference_device")!="cpu" or contract.get("reference_mode")!="live" or contract.get("torch_version")!=torch.__version__.split("+")[0] or contract.get("atol")!=5e-4 or contract.get("rtol")!=1e-5:
        raise RuntimeError("Missing or mismatched frozen anchor runtime; register a new comparison before training")
    if not torch.cuda.is_available():raise RuntimeError("CUDA session required")
    torch.set_num_threads(1)
    reg=json.loads(Path(runtime_path).read_text())
    for p in [reg["source"],reg["bc_anchor"]]+reg["history"]:
        if sha(p["path"])!=p["sha256"]:raise ValueError("Frozen input changed")
    original=load_model(reg["source"]["path"],"cuda")
    if original.observation_schema!="visible-v1":raise ValueError("visible-v1 required")
    bc=load_model(reg["bc_anchor"]["path"],"cpu").requires_grad_(False)
    initial=tensor_sha(original);anchor=tensor_sha(bc)
    networks={k:copy.deepcopy(original).requires_grad_(True) for k in ("control","diverse")}
    optimizers={k:optimizer_for(n,reg) for k,n in networks.items()}
    assert all(tensor_sha(n)==initial for n in networks.values())
    report=dict(status="RUNNING",hardware=dict(gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count(),
        cpu_count=os.cpu_count(),torch=torch.__version__),config=config,initial_tensor_sha256=initial,bc_anchor_tensor_sha256=anchor,
        bootstrap_seconds=time.monotonic()-script_started,arms={k:dict(completed_bo3=0,cohorts=[]) for k in networks},checkpoints=[])
    def persist():report["script_seconds"]=time.monotonic()-script_started;write_json(output/"report.json",report)
    def emit(kind,**detail):
        with (output/"events.jsonl").open("a") as f:f.write(json.dumps(dict(kind=kind,seconds=time.monotonic()-script_started,**detail))+"\n")
        print(json.dumps(dict(kind=kind,seconds=time.monotonic()-script_started,**detail)),flush=True)
    persist()
    # Anchor stays on the original CPU live path; no GPU anchor calculation.
    sample=torch.load(config["gpu_parity_rows"],weights_only=False)
    gate=diagnose_anchor(bc,sample,output/"anchor-preflight",deadline=min(finish-300,time.monotonic()+120),production="cpu-live")
    report["anchor_reference_gate"]=gate
    persist()
    require_anchor_equivalence(gate)
    # One discarded optimizer window checks finite losses/backward/recovery state.
    probe=copy.deepcopy(original).requires_grad_(True);opt=optimizer_for(probe,reg)
    seq=copy.deepcopy(sample[0][:32])
    loss,n,kl=loss_batched(probe,[(seq,0,len(seq))],0.,1.,entropy_coefficient=.001)
    (loss/n).backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),.5,error_if_nonfinite=True);opt.step()
    assert all(torch.isfinite(p).all() for p in probe.parameters())
    del probe,opt,loss,sample;gc.collect();torch.cuda.empty_cache()
    ctx=multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=initialize_worker,initargs=(str(runtime_path),str(output/"collection"))) as pool:
            state={k:v.detach().cpu() for k,v in original.state_dict().items()}
            t=time.monotonic()
            futures=[pool.submit(collect_worker,("preflight",0,state,config["gpu_preflight_jobs"][i::2],finish-300)) for i in range(2)]
            values=[f.result() for f in futures]
            elapsed=time.monotonic()-t
            games=[g for v in values for g in v["games"]]
            reference=live_reference_summary(values,games)
            # Exact target already frozen; this gate can only stop, never lower/raise it.
            collection_upper=max(v["wall_seconds"] for v in values)/4*1.5
            forecast=2*config["bo3_per_arm"]*collection_upper+config["cohorts_per_arm"]*2*180+300
            report["gpu_collection_gate"]=dict(complete_bo3=4,wall_seconds=elapsed,worker_seconds=[v["wall_seconds"] for v in values],
                reference=reference,remaining_seconds=finish-time.monotonic(),forecast_seconds=forecast)
            persist()
            if forecast>finish-time.monotonic():
                report["status"]="STOPPED_GPU_FEASIBILITY";persist();return report
            del values,games
            for arm,cohort in config["training_order"]:
                if finish-time.monotonic()<config["cohort_admission_seconds"]:
                    report["status"]="INCOMPLETE_BUDGET";persist();return report
                random.seed(config["seed"]);torch.manual_seed(config["seed"])
                network=networks[arm];optimizer=optimizers[arm];started=time.monotonic()
                state={k:v.detach().cpu() for k,v in network.state_dict().items()}
                before=tensor_sha(network);jobs=config["training_schedules"][f"{arm}-{cohort}"]
                futures=[pool.submit(collect_worker,(arm,cohort,state,jobs[i::2],finish-240)) for i in range(2)]
                values=[f.result() for f in futures]
                collection=time.monotonic()-started
                outcomes=[r for v in values for r in v["matches"]];games=[g for v in values for g in v["games"]]
                if len(outcomes)!=64 or sum(r["learner_seat"]==0 for r in outcomes)!=32:raise AssertionError("Cohort balance")
                reference=live_reference_summary(values,games)
                record=dict(cohort=cohort,complete_bo3=64,decisions=sum(map(len,games)),collection_seconds=collection,reference=reference,
                    opponent_counts=dict(Counter(r["opponent"] for r in outcomes)),policy_version_before=before,
                    meaningful_entropy=sum(r["old_entropy"] for g in games for r in g if r["meaningful"])/max(1,sum(r["meaningful"] for g in games for r in g)))
                active=output/f"{arm}-active.pt"
                torch.save(dict(model=network.state_dict(),optimizer=optimizer.state_dict(),games=games,record=record,
                    arm=arm,cohort=cohort,config=config,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),active)
                if active.stat().st_size>2*1024**3:raise RuntimeError("Rollout storage cap")
                with gzip.open(output/f"{arm}-cohort-{cohort:02d}-matches.jsonl.gz","wt") as f:
                    for r in outcomes:f.write(json.dumps(r)+"\n")
                t=time.monotonic()
                metrics=update(network,optimizer,games,16,epochs=2,seed=config["seed"]+cohort,entropy_coefficient=.001,
                    bc_kl_coefficient=.01,value_coefficient=.5,target_kl=.01,deadline=finish-180,
                    allow_oom_fallback=True,collect_diagnostics=True,component_gradients=False)
                torch.cuda.synchronize();record.update(ppo=metrics,update_seconds=time.monotonic()-t)
                if any(m["stop_reason"]=="deadline" for m in metrics):raise TimeoutError("Partial PPO; recoverable buffer retained")
                if not all(torch.isfinite(p).all() for p in network.parameters()):raise FloatingPointError("Nonfinite weights")
                file=output/f"{arm}-cohort-{cohort:02d}.pt"
                save_checkpoint(file,network,optimizer,dict(arm=arm,cohort=cohort,completed_bo3=cohort*64,config=config,record=record,deck=reg["deck"]))
                report["checkpoints"].append(dict(arm=arm,cohort=cohort,file=file.name,sha256=sha(file),tensor_sha256=tensor_sha(network),bytes=file.stat().st_size))
                report["arms"][arm]["completed_bo3"]+=64;report["arms"][arm]["cohorts"].append(record)
                record["cohort_wall_seconds"]=time.monotonic()-started
                active.unlink();optimizer.zero_grad(set_to_none=True);del values,games,outcomes;gc.collect();torch.cuda.empty_cache()
                persist();emit("cohort_complete",arm=arm,**record)
        for arm,n in networks.items():
            if report["arms"][arm]["completed_bo3"]!=config["bo3_per_arm"]:raise AssertionError("Unequal/incomplete endpoints")
            file=output/f"{arm}-final.pt"
            save_checkpoint(file,n,optimizers[arm],dict(arm=arm,completed_bo3=config["bo3_per_arm"],config=config,deck=reg["deck"]))
            report["arms"][arm].update(final_file=file.name,final_sha256=sha(file),tensor_sha256=tensor_sha(n))
        if tensor_sha(bc)!=anchor:raise AssertionError("BC anchor changed")
        report["status"]="COMPLETE";persist();return report
    except Exception as exc:
        report.update(status="FAILED",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc());persist();raise


def run(runtime_path,output,config,script_started):
    """Persist every failure, including initialization and equivalence preflight."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    try:
        return _run(runtime_path,output,config,script_started)
    except Exception as exc:
        path=output/"report.json"
        report=json.loads(path.read_text()) if path.exists() else dict(arms={},checkpoints=[])
        report.update(status="FAILED",error=type(exc).__name__+": "+str(exc),
            traceback=traceback.format_exc(),script_seconds=time.monotonic()-script_started,
            runtime=dict(torch=str(torch.__version__),expected_anchor_runtime=config.get("anchor_runtime")))
        gate=output/"anchor-preflight/anchor-checks.json"
        if gate.exists():report["anchor_reference_gate"]=json.loads(gate.read_text())
        write_json(path,report)
        raise
