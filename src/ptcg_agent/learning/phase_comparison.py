"""Small sequential phase runner: CPU live collection and immutable update inputs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import datetime,hashlib,json,multiprocessing,os,random,time,traceback
import torch
from .rl_stage1 import load_model,tensor_sha,write_json,sha
from .training import save_checkpoint,load_checkpoint
from .diverse_rl import optimizer_for
from .economical_collector import collect_matches
from .ppo_batched import update
from .model import collate

ROOT=Path("experiments/astra-stage-d-completed-001")
PRIVATE=Path("data/private/astra-stage-d-completed-001")
def read(path):return json.loads(Path(path).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(value):
    """Content identity independent of torch archive names/storage object sharing."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(str((str(t.dtype),list(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"{")
            for k in sorted(x,key=lambda k:(str(type(k)),str(k))):visit(k);visit(x[k])
            h.update(b"}")
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
            h.update(b"]")
        else:h.update((str(type(x))+":"+repr(x)+";").encode())
    visit(value);return h.hexdigest()
def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");torch.save(value,tmp);tmp.replace(path)
def intent(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:json.dump(value,f,indent=2)
def masks_digest(games):
    masks=[]
    flat=[r for g in games for r in g]
    for i in range(0,len(flat),32):
        b=collate(flat[i:i+32],"cpu")
        masks.append({k:v for k,v in b.items() if "mask" in k or k in ("minimum","maximum","reset")})
    return digest(masks)
_WORKER=None
def initialize_worker(reg,checkpoint,output,barrier):
    global _WORKER
    start=time.perf_counter();torch.set_num_threads(1)
    bc=load_model(reg["bc_anchor"]["path"]).requires_grad_(False)
    opponents={p["id"]:bc for p in reg["panel"][:4]}
    opponents.update({p["id"]:load_model(p["path"]).requires_grad_(False) for p in reg["history"]})
    learner=load_model(checkpoint).requires_grad_(False)
    before={k:tensor_sha(n) for k,n in dict(opponents,learner=learner).items()}
    initialized=time.perf_counter()-start
    barrier.wait(timeout=180)
    _WORKER=(reg,learner,opponents,Path(output),before,initialized)

def collect_worker(payload):
    jobs,number,prefix,deadline=payload
    reg,learner,opponents,output,before,initialized=_WORKER
    started=time.perf_counter()
    result=collect_matches(learner,opponents,reg,jobs,prefix=f"{prefix}-{number}",
        output=output/f"worker-{number}",deadline=deadline,live_reference=True)
    ended=time.perf_counter()
    assert before=={k:tensor_sha(n) for k,n in dict(opponents,learner=learner).items()}
    offset=0
    for match in result["matches"]:
        count=len(match["games"])
        match["learner_decisions"]=sum(map(len,result["games"][offset:offset+count]))
        offset+=count
    assert offset==len(result["games"])
    result["policy_identity"]=before
    result["python_rng"]=random.getstate();result["torch_rng"]=torch.get_rng_state()
    path=output/f"worker-{number}.pt";t=time.perf_counter();atomic_save(path,result)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,
        initialized_seconds=initialized,start=started,end=ended,
        serialization_seconds=time.perf_counter()-t,timing=result["timing"],matches=len(result["matches"]),
        decisions=sum(map(len,result["games"])),policy_identity=before)

def collect(reg,checkpoint,jobs,output,prefix,deadline=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    intent(output/"collection-intent.json",dict(at_utc=now(),checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),jobs=jobs,prefix=prefix))
    workers=reg["preflight"]["workers"];ctx=multiprocessing.get_context("spawn")
    barrier=ctx.Barrier(workers);started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize_worker,
        initargs=(reg,str(checkpoint),str(output),barrier)) as pool:
        futures=[pool.submit(collect_worker,(jobs[i::workers],i,prefix,deadline)) for i in range(workers)]
        files=[f.result() for f in futures]
    write_json(output/"worker-files.json",dict(files=files,pool_start=started,pool_end=time.perf_counter()))
    merge_started=time.perf_counter()
    values=[]
    for f in files:
        assert sha(f["path"])==f["sha256"]
        values.append(torch.load(f["path"],weights_only=False))
    games=[g for v in values for g in v["games"]];matches=[r for v in values for r in v["matches"]]
    assert len(matches)==len(jobs)
    assert len({r["match_id"] for r in matches})==len(jobs)
    assert all(r["status"]=="DONE" and r["checkpoint_fixed_per_match"] for r in matches)
    assert all(g and g[0]["reset"] for g in games)
    assert all(r["reference"] is not None for g in games for r in g)
    for g in games:
        for r in g:
            assert torch.isfinite(torch.tensor([r["old_logp"],r["old_value"],r["old_entropy"]])).all()
            assert all(ref.numel()==len(r["option_ids"])+1 for ref in r["reference"])
            # reset is consumed by network.recurrent; saved pre-reset hidden need not be zero.
    report=dict(workers=files,wall_seconds=time.perf_counter()-started,
        startup_seconds=min(f["start"] for f in files)-started,
        steady_collection_seconds=max(f["end"] for f in files)-min(f["start"] for f in files),
        worker_serialization_seconds=sum(f["serialization_seconds"] for f in files),
        merge_seconds=time.perf_counter()-merge_started,
        complete_bo3=len(matches),decisions=sum(map(len,games)),
        policy_version=tensor_sha(load_model(checkpoint)),rollout_digest=digest(games),
        legal_masks_digest=masks_digest(games),forced_observations=sum(not r["meaningful"] for g in games for r in g),
        failures=0,live_anchor=True,reference_seconds=sum(f["timing"]["reference_inference"] for f in files),
        opponent_counts=dict(Counter(r["opponent"] for r in matches)))
    write_json(output/"collection-complete.json",report)
    return games,matches,report

def run_update(network,optimizer,games,seed,deadline,events):
    t=time.perf_counter();cpu=time.process_time()
    def callback(event):
        with Path(events).open("a") as f:f.write(json.dumps(dict(seconds=time.perf_counter()-t,**event))+"\n")
    metrics=update(network,optimizer,games,16,epochs=2,seed=seed,target_kl=.01,
        entropy_coefficient=.001,bc_kl_coefficient=.01,value_coefficient=.5,
        collect_diagnostics=True,deadline=deadline,callback=callback)
    result=dict(wall_seconds=time.perf_counter()-t,process_cpu_seconds=time.process_time()-cpu,metrics=metrics,
        evaluated_decisions=sum(m["evaluated_decisions"] for m in metrics),
        trained_decisions=sum(m["decisions"] for m in metrics),optimizer_updates=sum(m["optimizer_updates"] for m in metrics))
    if any(m["stop_reason"]=="deadline" for m in metrics):
        write_json(Path(events).with_suffix(".failure.json"),result)
        raise TimeoutError("PPO deadline; immutable pre-update input retained")
    assert all(torch.isfinite(v).all() for v in network.parameters())
    return result

def make_bundle(network,optimizer,games,metadata,path):
    bundle=dict(observation_schema=network.observation_schema,model=network.state_dict(),
        optimizer=optimizer.state_dict(),games=games,metadata=metadata,
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=[])
    content=digest(bundle);masks=masks_digest(games);t=time.perf_counter();atomic_save(path,bundle)
    saved=torch.load(path,weights_only=False)
    assert digest(saved)==content and masks_digest(saved["games"])==masks
    assert Path(path).stat().st_size<=2*1024**3
    return bundle,dict(path=str(path),sha256=sha(path),content_digest=content,legal_masks_digest=masks,
        bytes=Path(path).stat().st_size,save_verify_seconds=time.perf_counter()-t,
        model_digest=digest(bundle["model"]),optimizer_digest=digest(bundle["optimizer"]),
        rng_digest=digest((bundle["python_rng"],bundle["torch_rng"])),rollout_digest=digest(games))

def restore_bundle(path,reg):
    bundle=torch.load(path,weights_only=False)
    network=load_model(reg["source"]["path"]).requires_grad_(True)
    optimizer=optimizer_for(network,reg)
    network.load_state_dict(bundle["model"]);optimizer.load_state_dict(bundle["optimizer"])
    random.setstate(bundle["python_rng"]);torch.set_rng_state(bundle["torch_rng"])
    return network,optimizer,bundle

def benchmark():
    reg=read(ROOT/"preflight-registration.json")
    output=PRIVATE/"benchmark-attempt2";output.mkdir(parents=True,exist_ok=True)
    intent(ROOT/"preflight-attempt2-intent.json",dict(at_utc=now(),matches=16,previous_matches=16,maximum_new_BO3=32))
    t=time.perf_counter()
    try:
        for spec in [reg["source"],reg["bc_anchor"]]+reg["history"]:assert sha(spec["path"])==spec["sha256"]
        games,matches,collection=collect(reg,reg["source"]["path"],reg["preflight"]["jobs"],output/"collection","benchmark")
        torch.set_num_threads(reg["preflight"]["update_threads"])
        network=load_model(reg["source"]["path"]).requires_grad_(True);optimizer=optimizer_for(network,reg)
        random.seed(reg["scientific_seed"]);torch.manual_seed(reg["scientific_seed"])
        bundle,bundle_manifest=make_bundle(network,optimizer,games,dict(benchmark_only=True,never_train=True),output/"update-input.pt")
        write_json(ROOT/"benchmark-collection.json",dict(collection=collection,bundle=bundle_manifest))
        baseline=run_update(network,optimizer,games,reg["scientific_seed"]+1,
            time.monotonic()+reg["preflight"]["max_probe_update_seconds"],output/"baseline-events.jsonl")
        assert baseline["evaluated_decisions"]>=1024 and baseline["optimizer_updates"]>=1
        baseline_state=dict(model=network.state_dict(),optimizer=optimizer.state_dict(),
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),metrics=baseline["metrics"])
        atomic_save(output/"baseline-result.pt",baseline_state)
        write_json(ROOT/"benchmark-update.json",baseline)
        del network,optimizer,bundle
        network,optimizer,resumed=restore_bundle(output/"update-input.pt",reg)
        assert digest(resumed)==bundle_manifest["content_digest"]
        repeat=run_update(network,optimizer,resumed["games"],reg["scientific_seed"]+1,
            time.monotonic()+reg["preflight"]["max_probe_update_seconds"],output/"resumed-events.jsonl")
        resumed_state=dict(model=network.state_dict(),optimizer=optimizer.state_dict(),
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),metrics=repeat["metrics"])
        baseline_state=torch.load(output/"baseline-result.pt",weights_only=False)
        equal={k:digest(baseline_state[k])==digest(resumed_state[k]) for k in baseline_state}
        equivalence=dict(status="PASS" if all(equal.values()) else "FAIL",exact_equal=equal,
            tolerances=dict(model_atol=0,optimizer_atol=0),backend=str(torch.__version__),
            update_threads=torch.get_num_threads(),baseline=baseline,resumed=repeat,bundle=bundle_manifest,
            scope="Collection/update boundary: full registered update uninterrupted versus restored immutable input; no mid-window replay.")
        write_json(ROOT/"resume-equivalence.json",equivalence)
        assert equivalence["status"]=="PASS"
        # Poststratify real per-opponent wall cost; phase worker imbalance covered by 1.5x.
        cells={}
        for opponent in {r["opponent"] for r in matches}:
            rows=[r for r in matches if r["opponent"]==opponent]
            cells[opponent]=dict(matches=len(rows),seconds=sum(r["wall_seconds"] for r in rows)/len(rows))
        # Each worker has its own complete decision count. Derive per-opponent exact counts from compact game decisions.
        for opponent in cells:
            rows=[r for r in matches if r["opponent"]==opponent]
            cells[opponent]["learner_decisions"]=sum(r["learner_decisions"] for r in rows)/len(rows)
        probs={p["id"]:p["probability"] for p in reg["history"]}
        probs.update({"BC-Sinistcha-Banette":.2,"BC-Hydrapple-Meganium":.2,"BC-OceanMix":.1,"BC-Mega-Excadrill":.1})
        sec_per_eval=baseline["wall_seconds"]/baseline["evaluated_decisions"]
        # Charge two complete epochs on inflated decisions, regardless of observed KL stop.
        arms={}
        overhead=collection["wall_seconds"]-collection["steady_collection_seconds"]+bundle_manifest["save_verify_seconds"]
        for arm in ("control","diverse"):
            law={"BC-OceanMix":1.} if arm=="control" else probs
            per_match=sum(cells[p]["seconds"]*w for p,w in law.items())
            per_decisions=sum(cells[p]["learner_decisions"]*w for p,w in law.items())
            collection_cost=256*per_match/reg["preflight"]["workers"]*1.5+4*overhead*1.5
            update_cost=256*per_decisions*2*sec_per_eval*1.5
            arms[arm]=dict(collection_seconds=collection_cost,update_seconds=update_cost,
                total_seconds=collection_cost+update_cost,mean_BO3_serial_seconds=per_match,mean_learner_decisions=per_decisions)
        forecast=sum(a["total_seconds"] for a in arms.values())
        projection=dict(status="CPU_FITS" if forecast<=21600 else "CPU_EXCEEDS",at_utc=now(),
            cells=cells,arms=arms,total_training_seconds=forecast,training_limit_seconds=21600,
            safety_factor=1.5,seconds_per_evaluated_PPO_decision=sec_per_eval,
            full_two_epochs_charged=True,benchmark_seconds=time.perf_counter()-t,GPU_seconds=0,
            torch=str(torch.__version__),collection=collection,update=baseline)
        write_json(ROOT/"feasibility.json",projection)
        print(json.dumps(projection),flush=True)
    except Exception as exc:
        write_json(ROOT/"preflight-attempt2-failure.json",dict(at_utc=now(),error=repr(exc),traceback=traceback.format_exc(),elapsed=time.perf_counter()-t))
        raise


def verify_live(reg,games,output):
    """No new deals: teacher-force saved complete prefixes, retaining every forced row."""
    from .anchor_checks import diagnose_anchor,require_anchor_equivalence
    torch.set_num_threads(1)
    learner=load_model(reg["source"]["path"]).requires_grad_(False)
    bc=load_model(reg["bc_anchor"]["path"]).requires_grad_(False)
    from copy import deepcopy
    prefixes=deepcopy([g[:64] for g in games[:2]])
    # Live collector stores decoder references as1xN; diagnostic API uses1D.
    # Normalize the diagnostic copy only, preserving the production rollout verbatim.
    for game in prefixes:
        for row in game:row["reference"]=[ref.reshape(-1) for ref in row["reference"]]
    report=diagnose_anchor(bc,prefixes,Path(output)/"anchor-normalized",production="cpu-live")
    require_anchor_equivalence(report)
    checked=0;worst=0.
    for game in prefixes:
        hidden=torch.zeros_like(game[0]["hidden"])
        with torch.no_grad():
            for row in game:
                assert row["reset"] or torch.equal(hidden,row["hidden"])
                stats,hidden=learner.decision([row],hidden,actions=[row["action"]])
                error=abs(float(stats[1])-row["old_logp"]);worst=max(worst,error)
                assert error<=5e-4+1e-5*abs(row["old_logp"])
                assert abs(float(stats[3])-row["old_value"])<=5e-4+1e-5*abs(row["old_value"])
                checked+=1
    result=dict(status="PASS",checked_decisions=checked,maximum_logp_error=worst,
        actor_hidden_equal=True,CPU_live_anchor_gate=report,
        unchanged_tolerances=dict(atol=5e-4,rtol=1e-5),new_BO3=0)
    write_json(ROOT/"execution-policy-checks.json",result)
    return result

def freeze_cpu():
    reg=read(ROOT/"preflight-registration.json");feasibility=read(ROOT/"feasibility.json")
    assert feasibility["status"]=="CPU_FITS" and feasibility["total_training_seconds"]<=21600
    assert read(ROOT/"resume-equivalence.json")["status"]=="PASS"
    # Only persisted benchmark files are used in this no-new-BO3 path check.
    files=read(ROOT/"benchmark-collection.json")["collection"]["workers"]
    games=[g for record in files for g in torch.load(record["path"],weights_only=False)["games"]]
    verify_live(reg,games,PRIVATE/"benchmark-attempt2"/"execution-checks")
    old=read(Path("experiments/astra-stage-d-diverse-rl-001/training-config.json"))
    paths=["src/ptcg_agent/learning/phase_comparison.py","src/ptcg_agent/learning/economical_collector.py",
        "src/ptcg_agent/learning/ppo_batched.py","src/ptcg_agent/learning/training.py",
        "src/ptcg_agent/learning/model.py","src/ptcg_agent/learning/live.py",
        "src/ptcg_agent/learning/anchor_checks.py","scripts/astra_stage_d_complete.py"]
    original=read(Path("experiments/astra-stage-d-diverse-rl-001/registration.json"))
    paths=sorted(set(paths)|set(original["code_sha256"])|
        {"scripts/astra_stage_d_complete_evaluate.py","scripts/astra_stage_d_complete_report.py",
         "scripts/astra_stage_c_evaluate.py","tests/test_phase_comparison.py"})
    registration=dict(reg,at_utc=now(),execution_path="entirely local CPU; zero GPU jobs",
        torch=str(torch.__version__),CPU_workers=4,CPU_actor_threads=1,CPU_update_threads=4,
        bo3_per_arm=256,cohorts_per_arm=4,training_order=old["training_order"],
        training_schedules=old["training_schedules"],GPU_jobs=0,GPU_startup_seconds=0,
        GPU_session_cap_seconds=7200,local_training_elapsed_cap_seconds=21600,
        measured_training_upper_seconds=feasibility["total_training_seconds"],
        code_sha256={p:sha(p) for p in paths},
        recovery="Exclusive collection/update intents; immutable phase inputs; atomic checkpoint then commit. Completed operations verify and return. Unknown interrupted updates fail closed; never auto-replay.",
        optimizer_rng="Load complete predecessor optimizer; retain original registered per-phase seed reset; local PPO shuffle seed scientific_seed+phase.",
        collection_improvement="Four CPU worker processes with live CPU anchor; worker-local rollout serialization instead of large IPC objects.",
        diagnostic_BO3=dict(registered_benchmark=32,actual_total=34,extra_unit_integration=2,
            cap=32,overrun=2,evidence_sha256=sha(ROOT/"preflight-cap-deviation.json")),
        historical_evidence_sha256={p:sha(p) for p in (
            "reports/astra-stage-d-diverse-rl.md","reports/astra-stage-d-anchor-repair.md")},
        preflight_registration_sha256=sha(ROOT/"preflight-registration.json"),
        feasibility_sha256=sha(ROOT/"feasibility.json"),resume_equivalence_sha256=sha(ROOT/"resume-equivalence.json"))
    intent(ROOT/"registration.json",registration)
    oldroot=Path("experiments/astra-stage-d-diverse-rl-001")
    for name in ["evaluation-schedule.json","evaluation-analysis-plan.json","evaluation-analysis-clarification.json"]:
        (ROOT/name).write_bytes((oldroot/name).read_bytes())
    print("Frozen CPU execution: 512 training BO3, four phases per arm, zero GPU sessions.",flush=True)

def committed_phase(directory,bundle_manifest,registration):
    """Recover a fully exported result without executing its update again."""
    directory=Path(directory);commit=directory/"commit.json";checkpoint=directory/"checkpoint.pt"
    if commit.exists():
        record=read(commit)
        assert sha(checkpoint)==record["checkpoint_sha256"]
        assert record["input_sha256"]==bundle_manifest["sha256"]
        return record
    if (directory/"update-intent.json").exists():
        if not checkpoint.exists():raise RuntimeError("Unresolved update intent: no automatic duplicate update")
        state=torch.load(checkpoint,weights_only=False);record=state["metadata"]["phase_commit"]
        assert record["input_sha256"]==bundle_manifest["sha256"]
        record.update(checkpoint_sha256=sha(checkpoint),recovered_atomic_export=True)
        intent(commit,record);return record
    return None

def execute_phase(reg,arm,phase,predecessor,jobs,finish):
    directory=PRIVATE/f"{arm}-phase-{phase:02d}";directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/"input-manifest.json"
    if manifest_path.exists():
        manifest=read(manifest_path)
        assert sha(manifest["path"])==manifest["sha256"]
        existing=committed_phase(directory,manifest,reg)
        if existing:return existing
        bundle=torch.load(manifest["path"],weights_only=False)
        assert bundle["metadata"]["predecessor_sha256"]==sha(predecessor)
    else:
        if (directory/"collection-intent.json").exists() or (directory/"collection"/"collection-intent.json").exists():
            raise RuntimeError("Interrupted collection; retain evidence, no replacement cohort")
        games,matches,collection=collect(reg,predecessor,jobs,directory/"collection",
            f"{arm}-phase-{phase}",finish-300)
        assert len(matches)==64 and sum(r["learner_seat"]==0 for r in matches)==32
        torch.set_num_threads(4)
        network=load_model(predecessor).requires_grad_(True);optimizer=optimizer_for(network,reg)
        load_checkpoint(predecessor,network,optimizer,restore_rng=True)
        random.seed(reg["scientific_seed"]);torch.manual_seed(reg["scientific_seed"])
        assert collection["policy_version"]==tensor_sha(network)
        metadata=dict(arm=arm,phase=phase,completed_bo3_before=(phase-1)*64,
            predecessor=str(predecessor),predecessor_sha256=sha(predecessor),
            registration_sha256=sha(ROOT/"registration.json"),collection=collection,matches=matches,
            policy_before=tensor_sha(network),meaningful_entropy=
            sum(r["old_entropy"] for g in games for r in g if r["meaningful"])/max(1,sum(r["meaningful"] for g in games for r in g)))
        _,manifest=make_bundle(network,optimizer,games,metadata,directory/"update-input.pt")
        write_json(manifest_path,manifest)
        del network,optimizer,games,matches,_
    assert sha(manifest["path"])==manifest["sha256"]
    torch.set_num_threads(4)
    network,optimizer,bundle=restore_bundle(manifest["path"],reg)
    assert digest(bundle)==manifest["content_digest"] and masks_digest(bundle["games"])==manifest["legal_masks_digest"]
    assert bundle["metadata"]["registration_sha256"]==sha(ROOT/"registration.json")
    intent(directory/"update-intent.json",dict(at_utc=now(),arm=arm,phase=phase,
        input_sha256=manifest["sha256"],backend=str(torch.__version__)))
    # Input is durable before the first optimizer step.
    result=run_update(network,optimizer,bundle["games"],reg["scientific_seed"]+phase,
        finish-120,directory/"update-events.jsonl")
    record=dict(arm=arm,phase=phase,completed_bo3=phase*64,input_sha256=manifest["sha256"],
        input_manifest=manifest,collection=bundle["metadata"]["collection"],
        meaningful_entropy=bundle["metadata"]["meaningful_entropy"],
        predecessor_sha256=sha(predecessor),policy_before=bundle["metadata"]["policy_before"],
        update=result,checkpoint=str(directory/"checkpoint.pt"),policy_after=tensor_sha(network),
        at_utc=now(),failures=0)
    metadata=dict(arm=arm,cohort=phase,completed_bo3=phase*64,phase_commit=record,
        deck=reg["deck"],registration_sha256=sha(ROOT/"registration.json"))
    save_checkpoint(directory/"checkpoint.pt",network,optimizer,metadata)
    # Verify full exported state, optimizer and RNG before accepting next collection.
    exported=torch.load(directory/"checkpoint.pt",weights_only=False)
    assert digest(exported["model"])==digest(network.state_dict())
    assert digest(exported["optimizer"])==digest(optimizer.state_dict())
    assert torch.equal(exported["torch_rng"],torch.get_rng_state()) and exported["python_rng"]==random.getstate()
    record["checkpoint_sha256"]=sha(directory/"checkpoint.pt")
    record["export_verification"]=dict(model=True,optimizer=True,python_rng=True,torch_rng=True)
    intent(directory/"commit.json",record)
    return record

def train():
    reg=read(ROOT/"registration.json")
    assert reg["execution_path"]=="entirely local CPU; zero GPU jobs"
    assert str(torch.__version__)==reg["torch"]
    for p,h in reg["code_sha256"].items():assert sha(p)==h
    for p in [reg["source"],reg["bc_anchor"]]+reg["history"]:assert sha(p["path"])==p["sha256"]
    if (ROOT/"training-intent.json").exists():
        previous=read(ROOT/"training-intent.json")
        assert previous["registration_sha256"]==sha(ROOT/"registration.json")
        elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(previous["at_utc"])).total_seconds()
        start=time.monotonic()-elapsed;finish=start+21600
    else:
        start=time.monotonic();finish=start+21600
        intent(ROOT/"training-intent.json",dict(at_utc=now(),monotonic_start=start,
            deadline=finish,registration_sha256=sha(ROOT/"registration.json"),GPU_jobs=0))
    report=dict(status="RUNNING",at_utc=now(),arms={a:dict(completed_bo3=0,cohorts=[]) for a in ("control","diverse")},
        checkpoints=[],GPU_session_seconds=0,CPU_training_seconds=0)
    predecessors={a:reg["source"]["path"] for a in ("control","diverse")}
    try:
        for arm,phase in reg["training_order"]:
            assert report["arms"][arm]["completed_bo3"]==(phase-1)*64
            if time.monotonic()+120>=finish:raise TimeoutError("Six-hour CPU training cap")
            record=execute_phase(reg,arm,phase,predecessors[arm],reg["training_schedules"][f"{arm}-{phase}"],finish)
            predecessors[arm]=record["checkpoint"]
            report["arms"][arm]["completed_bo3"]=phase*64
            report["arms"][arm]["cohorts"].append(record)
            report["checkpoints"].append(dict(arm=arm,cohort=phase,path=record["checkpoint"],sha256=record["checkpoint_sha256"]))
            report["CPU_training_seconds"]=time.monotonic()-start
            write_json(ROOT/"training-report.json",report)
            # Compact public recovery/cost evidence. Rollout tensors stay private.
            write_json(ROOT/f"{arm}-phase-{phase:02d}.json",record)
            print(json.dumps(dict(event="phase_complete",arm=arm,phase=phase,
                complete_bo3=phase*64,collection_seconds=record["collection"]["wall_seconds"],
                update_seconds=record["update"]["wall_seconds"],elapsed=time.monotonic()-start)),flush=True)
        for arm in predecessors:
            assert report["arms"][arm]["completed_bo3"]==256 and len(report["arms"][arm]["cohorts"])==4
            report["arms"][arm].update(final_file=predecessors[arm],final_sha256=sha(predecessors[arm]),
                tensor_sha256=tensor_sha(load_model(predecessors[arm])))
        report.update(status="COMPLETE",finished_at_utc=now(),CPU_training_seconds=time.monotonic()-start)
        write_json(ROOT/"training-report.json",report)
        write_json(ROOT/"training-completion.json",dict(status="COMPLETE",report=str(ROOT/"training-report.json"),CPU_seconds=report["CPU_training_seconds"],GPU_seconds=0))
        return report
    except Exception as exc:
        report.update(status="INCOMPLETE",error=repr(exc),traceback=traceback.format_exc(),CPU_training_seconds=time.monotonic()-start)
        write_json(ROOT/"training-report.json",report)
        raise
