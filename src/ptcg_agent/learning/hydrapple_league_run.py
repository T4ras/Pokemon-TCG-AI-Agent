"""Sequential frozen-pool Hydrapple training; one endpoint per registered arm."""
from pathlib import Path
import datetime as dt,json,random,time,copy,os
import torch
from . import hydrapple_league as h
from .rl_stage1 import load_model,tensor_sha
from .training import save_checkpoint,load_checkpoint
def ensure_snapshot(record,out,phase,arm):
    """Finish an interrupted export from the atomic checkpoint; never repeat PPO."""
    if phase%4:return
    out=Path(out);snapshot=out.parent/"snapshots"/f"snapshot-{phase//4:02d}.pt";snapshot.parent.mkdir(exist_ok=True)
    checkpoint=Path(record["checkpoint"])
    assert h.sha(checkpoint)==record["checkpoint_sha256"]
    expected=dict(creation_index=phase//4,eligible_from_phase=phase+1,source_checkpoint_sha256=record["checkpoint_sha256"],arm=arm,phase=phase)
    if not snapshot.exists():
        original=torch.load(checkpoint,map_location="cpu",weights_only=False)
        payload=dict(model={k:v.detach().cpu().clone() for k,v in original["model"].items()},metadata=expected)
        tmp=snapshot.with_suffix(".tmp");torch.save(payload,tmp);tmp.replace(snapshot)
    payload=torch.load(snapshot,map_location="cpu",weights_only=False)
    assert payload["metadata"]==expected
    assert h.digest(payload["model"])==h.digest(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    item=dict(path=str(snapshot),sha256=h.sha(snapshot),creation_index=phase//4,eligible_from_phase=phase+1)
    supplement=out/"snapshot-commit.json"
    if supplement.exists():assert h.read(supplement)==item
    else:h.exclusive(supplement,item)
    record["snapshot"]=item
def execute_phase(reg,arm,phase,predecessor,snapshots,out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);commit=out/"commit.json"
    if commit.exists():
        record=h.read(commit);assert h.sha(record["checkpoint"])==record["checkpoint_sha256"]
        assert h.sha(predecessor)==record["predecessor_sha256"];ensure_snapshot(record,out,phase,arm);return record
    if (out/"update-intent.json").exists():
        checkpoint=out/"checkpoint.pt"
        if not checkpoint.exists():raise RuntimeError("Unknown interrupted update; no duplicate update")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False);record=payload["metadata"]["phase_commit"]
        assert record["predecessor_sha256"]==h.sha(predecessor)
        record.update(checkpoint_sha256=h.sha(checkpoint),recovered_atomic_export=True);h.exclusive(commit,record);ensure_snapshot(record,out,phase,arm);return record
    pool=h.pool_for_phase(reg["initial"],snapshots,arm)
    for item in pool:assert (item["creation"]*4)<phase
    pool_record=dict(at_utc=h.iso(),arm=arm,phase=phase,completed_BO3_before=(phase-1)*64,policies=pool)
    pool_file=out/"opponent-pool.json"
    if pool_file.exists():
        existing_pool=h.read(pool_file);assert existing_pool["policies"]==pool and existing_pool["arm"]==arm and existing_pool["phase"]==phase
    else:h.exclusive(pool_file,pool_record)
    jobs=h.phase_jobs(phase,pool)
    # Resolve only dynamic historical opponent IDs; categories/seeds/chooser are fixed pretraining.
    expected=reg["base_training_schedules"][str(phase)]
    assert [{k:j[k] for k in ("pair","category","learner_seat","policy_seed")} for j in jobs]==expected
    h.write(out/"schedule.json",jobs)
    input_path=out/"update-input.pt.gz";input_manifest=out/"input-manifest.json"
    if input_manifest.exists():
        manifest=h.read(input_manifest);assert h.sha(input_path)==manifest["sha256"]
        bundle=h.load_gzip(input_path)
        assert bundle["metadata"]["predecessor_sha256"]==h.sha(predecessor)
        assert bundle["metadata"]["pool_sha256"]==h.sha(pool_file)
    else:
        if (out/"collection"/"collection-intent.json").exists():raise RuntimeError("Interrupted cohort, no automatic replacement")
        games,matches,worker_rng,collection=h.collect(reg,predecessor,pool,jobs,out/"collection",f"{arm}-phase{phase}",deadline)
        assert len(matches)==64 and sum(m["learner_seat"]==0 for m in matches)==32
        net=load_model(predecessor).to("cuda").requires_grad_(True);opt=h.fresh_optimizer(net)
        load_checkpoint(predecessor,net,opt,restore_rng=True)
        assert tensor_sha(net)==collection["policy_identity"]
        # Identical phase-level PPO RNG rule in both arms, independent native deals.
        random.seed(h.SEED+phase);torch.manual_seed(h.SEED+phase)
        metadata=dict(arm=arm,phase=phase,policy_before=collection["policy_identity"],predecessor_sha256=h.sha(predecessor),pool_sha256=h.sha(pool_file),collection=collection,matches=matches,meaningful_action_entropy=sum(r["old_entropy"] for g in games for r in g if r["meaningful"])/max(1,sum(r["meaningful"] for g in games for r in g)),registration_digest=reg["registration_digest"])
        manifest=h.bundle_input(net,opt,games,worker_rng,metadata,input_path);h.write(input_manifest,manifest)
        del net,opt,games,matches,worker_rng;torch.cuda.empty_cache()
        bundle=h.load_gzip(input_path)
    assert bundle["metadata"]["arm"]==arm and bundle["metadata"]["phase"]==phase
    assert bundle["metadata"]["registration_digest"]==reg["registration_digest"]
    assert h.digest(bundle)==manifest["content_digest"]
    assert h.masks_digest(bundle["games"])==manifest["legal_masks_digest"]
    net=load_model(reg["initial"]).to("cuda").requires_grad_(True);net.load_state_dict(bundle["model"])
    opt=h.fresh_optimizer(net);opt.load_state_dict(bundle["optimizer"])
    random.setstate(bundle["python_rng"]);torch.set_rng_state(bundle["torch_rng"]);torch.cuda.set_rng_state_all(bundle["cuda_rng"])
    assert tensor_sha(net)==bundle["metadata"]["policy_before"]
    h.exclusive(out/"update-intent.json",dict(at_utc=h.iso(),arm=arm,phase=phase,input_sha256=manifest["sha256"],policy_identity=tensor_sha(net),pool_sha256=h.sha(pool_file)))
    updated=h.run_update(net,opt,bundle["games"],h.SEED+phase,deadline,out/"update-events.jsonl")
    record=dict(arm=arm,phase=phase,completed_BO3=phase*64,predecessor_sha256=h.sha(predecessor),checkpoint=str(out/"checkpoint.pt"),input_sha256=manifest["sha256"],policy_before=bundle["metadata"]["policy_before"],policy_after=tensor_sha(net),collection=bundle["metadata"]["collection"],meaningful_action_entropy=bundle["metadata"]["meaningful_action_entropy"],update=updated,pool_manifest_sha256=h.sha(pool_file),at_utc=h.iso(),failures=0)
    save_checkpoint(record["checkpoint"],net,opt,dict(deck=reg["deck"],arm=arm,phase=phase,completed_BO3=phase*64,registration_digest=reg["registration_digest"],phase_commit=record))
    saved=torch.load(record["checkpoint"],map_location="cpu",weights_only=False)
    assert h.digest(saved["model"])==h.digest(net.state_dict())
    assert h.digest(saved["optimizer"])==h.digest(opt.state_dict())
    assert saved["python_rng"]==random.getstate() and torch.equal(saved["torch_rng"],torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(saved["cuda_rng"],torch.cuda.get_rng_state_all()))
    record.update(checkpoint_sha256=h.sha(record["checkpoint"]),export_model_optimizer_RNG_verified=True)
    h.exclusive(commit,record)
    ensure_snapshot(record,out,phase,arm)
    del net,opt,bundle;torch.cuda.empty_cache();return record
def train(reg,out,arm,script_started_utc):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert arm in ("control","league") and reg["target_BO3_per_arm"] in (4096,2048)
    deadline=(dt.datetime.fromisoformat(script_started_utc)+dt.timedelta(seconds=reg["script_seconds"]-180)).isoformat()
    h.exclusive(out/"training-intent.json",dict(at_utc=h.iso(),arm=arm,target=reg["target_BO3_per_arm"],deadline_utc=deadline,registration_digest=reg["registration_digest"]))
    predecessor=reg["initial"];snapshots={};records=[]
    report=dict(status="RUNNING",arm=arm,target_BO3=reg["target_BO3_per_arm"],completed_BO3=0,phases=[],started_at_utc=script_started_utc,deadline_utc=deadline)
    try:
        for phase in range(1,reg["target_BO3_per_arm"]//64+1):
            if h.remaining(deadline)<reg["next_phase_admission_seconds"]:raise TimeoutError("Insufficient registered session budget for another full phase")
            record=execute_phase(reg,arm,phase,predecessor,snapshots,out/f"phase-{phase:03d}",deadline)
            predecessor=record["checkpoint"]
            if phase%4==0:
                snap=h.read(out/f"phase-{phase:03d}"/"snapshot-commit.json");snapshots[snap["creation_index"]]=snap["path"];record["snapshot"]=snap
            assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file())<=reg["storage_cap_bytes_per_arm"],"Registered recovery storage cap"
            records.append(record);report.update(completed_BO3=phase*64,phases=records,latest_checkpoint=predecessor)
            h.write(out/"report.json",report)
            print(json.dumps(dict(event="phase_complete",arm=arm,phase=phase,completed_BO3=phase*64,UTC_remaining_seconds=h.remaining(deadline),collection_seconds=record["collection"]["UTC_seconds"],update_seconds=record["update"]["UTC_seconds"],optimizer_updates=record["update"]["optimizer_updates"],meaningful_entropy=record["meaningful_action_entropy"])),flush=True)
        assert report["completed_BO3"]==reg["target_BO3_per_arm"]
        report.update(status="COMPLETE",finished_at_utc=h.iso(),final_checkpoint=predecessor,final_sha256=h.sha(predecessor),final_tensor_sha256=tensor_sha(load_model(predecessor)),actual_optimizer_updates=sum(x["update"]["optimizer_updates"] for x in records))
        h.write(out/"report.json",report);return report
    except Exception as error:
        import traceback
        report.update(status="INCOMPLETE",failed_at_utc=h.iso(),error=repr(error),traceback=traceback.format_exc());h.write(out/"report.json",report);raise
