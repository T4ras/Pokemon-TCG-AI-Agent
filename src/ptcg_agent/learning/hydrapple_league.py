"""Controlled Hydrapple historical-self-play: fixed recipe, CPU-live anchor."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy,datetime as dt,gzip,hashlib,json,multiprocessing as mp,os,random,resource,time,traceback
import torch
from .rl_stage1 import load_model,tensor_sha
from .training import save_checkpoint
from .economical_collector import EconomicalAgent
from .live import play_bo3
from .rl_timing import apply_returns
from .ppo_batched import update
from .phase_comparison import digest,masks_digest
SEED=904115
CATEGORIES=["BC-Sinistcha-Banette","BC-Hydrapple-Meganium","BC-OceanMix","BC-Mega-Excadrill","heuristic-v2-Abomasnow","self-slot"]
PROBS=[.15,.15,.10,.15,.05,.40]
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,indent=2));q.replace(p)
def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def append(p,v):
    with Path(p).open("a") as f:f.write(json.dumps(v)+"\n");f.flush()
def remaining(deadline):return (dt.datetime.fromisoformat(deadline)-now()).total_seconds()
def fresh_optimizer(net):
    return torch.optim.AdamW([dict(params=[p for n,p in net.named_parameters() if not n.startswith("value.")],lr=3e-5),dict(params=list(net.value.parameters()),lr=1e-4)],weight_decay=1e-4)
def save_gzip(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp")
    with gzip.open(q,"wb",compresslevel=1) as f:torch.save(value,f)
    q.replace(p)
def load_gzip(p):
    with gzip.open(p,"rb") as f:return torch.load(f,map_location="cpu",weights_only=False)
def keep_older(indices,maximum=8):
    """Evenly spaced creation positions, round-half-up; include both ends."""
    a=sorted(set(indices))
    if len(a)<=maximum:return a
    return [a[(2*i*(len(a)-1)+(maximum-1))//(2*(maximum-1))] for i in range(maximum)]
def pool_for_phase(initial,snapshots,arm):
    if arm=="control" or not snapshots:return [dict(id="initial",path=str(initial),creation=0,probability=.4,sha256=sha(initial))]
    latest=max(snapshots)
    older=keep_older([0]+[i for i in snapshots if i<latest])
    result=[dict(id=f"snapshot-{latest}",creation=latest,path=str(snapshots[latest]),probability=.2,sha256=sha(snapshots[latest]))]
    for i in older:
        p=initial if i==0 else snapshots[i]
        result.append(dict(id="initial" if i==0 else f"snapshot-{i}",creation=i,path=str(p),probability=.2/len(older),sha256=sha(p)))
    assert len(result)<=9 and any(p["creation"]==0 for p in result)
    return result
def phase_jobs(phase,pool,count=64,seed=SEED,diagnostic=False):
    assert count%2==0
    rng=random.Random(seed+phase);jobs=[]
    for pair in range(count//2):
        # Fixed category coverage only in disposable diagnostic cohorts.
        category=CATEGORIES[pair] if diagnostic and pair<len(CATEGORIES) else rng.choices(CATEGORIES,PROBS)[0]
        chosen=None
        if category=="self-slot":
            x=rng.random()*.4;cum=0.
            for p in pool:
                cum+=p["probability"]
                if x<cum:chosen=p["id"];break
            assert chosen is not None
        for seat in (0,1):
            jobs.append(dict(pair=pair,category=category,opponent=chosen if chosen else category,learner_seat=seat,policy_seed=(seed+phase)*10000+2*pair+seat))
    rng.shuffle(jobs);assert sum(j["learner_seat"]==0 for j in jobs)==count//2
    return jobs
_WORKER=None
def initialize(reg,checkpoint,pool,output,deadline):
    global _WORKER
    t=now();cpu=time.process_time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert str(torch.__version__).split("+")[0]=="2.11.0","Frozen CPU-live anchor runtime"
    bc=load_model(reg["initial"]).requires_grad_(False);learner=load_model(checkpoint).requires_grad_(False)
    nets={p["id"]:(bc if p["creation"]==0 else load_model(p["path"]).requires_grad_(False)) for p in pool}
    for p in pool:assert sha(p["path"])==p["sha256"]
    from ..engine import load_sdk
    api,_=load_sdk();cards={c.cardId:c for c in api.all_card_data()};attacks={a.attackId:a for a in api.all_attack()}
    identity={k:tensor_sha(n) for k,n in dict(nets,anchor=bc,learner=learner).items()}
    _WORKER=(reg,learner,bc,nets,Path(output),deadline,cards,attacks,identity,t,cpu)
def collect_worker(payload):
    jobs,number,prefix=payload
    reg,learner,bc,nets,out,deadline,cards,attacks,before,began,cpu=_WORKER
    out.mkdir(parents=True,exist_ok=True);games=[];matches=[];times=Counter();ready=now()
    from ..policies import make_policy
    class Guarded(EconomicalAgent):
        def __call__(self,obs):
            if remaining(deadline)<=0:raise TimeoutError("UTC collection deadline")
            return super().__call__(obs)
    for i,j in enumerate(jobs):
        if remaining(deadline)<=0:raise TimeoutError("UTC collection deadline")
        random.seed(j["policy_seed"]);torch.manual_seed(j["policy_seed"])
        own=Guarded(learner,reg["deck"],reference=bc)
        if j["category"]=="self-slot":spec=dict(deck=reg["deck"],kind="neural");opponent=nets[j["opponent"]]
        else:spec=next(p for p in reg["panel"] if p["id"]==j["category"]);opponent=bc
        other=make_policy("heuristic-v2",spec["deck"],j["policy_seed"]+1,cards,attacks) if spec["kind"]!="neural" else Guarded(opponent,spec["deck"],False)
        agents=[own,other] if j["learner_seat"]==0 else [other,own]
        append(out/f"attempts-{number}.jsonl",dict(at_utc=iso(),job=j,index=i))
        t=now()
        (out/f"failures-{number}").mkdir(parents=True,exist_ok=True)
        row,rollouts=play_bo3(agents,f"{prefix}-w{number}-{i:03d}",out/f"failures-{number}",max_seconds=300,metadata=dict(j,policy_identity=before["learner"],opponent_policy_identity=tensor_sha(opponent) if spec["kind"]=="neural" else "heuristic-v2",checkpoint_fixed_per_match=True))
        row.update(started_at_utc=t.isoformat(),finished_at_utc=iso(),UTC_seconds=(now()-t).total_seconds())
        append(out/f"matches-{number}.jsonl",row)
        if row["status"]!="DONE":raise RuntimeError("Failed native collection; no replacement")
        assert all(g["chooser"]==(g["round"]-1)%2 for g in row["games"])
        gs=apply_returns(rollouts[j["learner_seat"]],row["result"],j["learner_seat"],"game",lam=1.)
        row["learner_decisions"]=sum(map(len,gs));games.extend(gs);matches.append(row)
        times.update(own.timing)
        if hasattr(other,"timing"):times.update(other.timing)
    assert before=={k:tensor_sha(n) for k,n in dict(nets,anchor=bc,learner=learner).items()}
    state=dict(games=games,matches=matches,policy_identity=before,python_rng=random.getstate(),torch_rng=torch.get_rng_state())
    p=out/f"worker-{number}.pt.gz";t=now();save_gzip(p,state)
    return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,started_at_utc=began.isoformat(),ready_at_utc=ready.isoformat(),collection_UTC_seconds=(t-ready).total_seconds(),serialization_UTC_seconds=(now()-t).total_seconds(),process_CPU_seconds=time.process_time()-cpu,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,timing=dict(times),policy_identity=before)
def collect(reg,checkpoint,pool,jobs,out,prefix,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    exclusive(out/"collection-intent.json",dict(at_utc=iso(),policy_sha256=sha(checkpoint),pool=pool,jobs=jobs,deadline_utc=deadline))
    t=now();workers=reg["workers"];ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,initargs=(reg,str(checkpoint),pool,str(out),deadline)) as executor:
        fs=[executor.submit(collect_worker,(jobs[i::workers],i,prefix)) for i in range(workers)]
        files=[f.result() for f in fs]
    games=[];matches=[];rng=[]
    for f in files:
        assert sha(f["path"])==f["sha256"];v=load_gzip(f["path"])
        games.extend(v["games"]);matches.extend(v["matches"]);rng.append(dict(python=v["python_rng"],torch=v["torch_rng"]))
    assert len(matches)==len(jobs) and len({r["match_id"] for r in matches})==len(jobs)
    assert reg.get("diagnostic_remainder",False) or sum(r["learner_seat"]==0 for r in matches)==len(jobs)//2
    assert all(g[0]["reset"] for g in games) and all(row["reference"] is not None for g in games for row in g)
    report=dict(at_utc=iso(),UTC_seconds=(now()-t).total_seconds(),workers=files,completed_BO3=len(matches),decisions=sum(map(len,games)),forced=sum(not r["meaningful"] for g in games for r in g),policy_identity=files[0]["policy_identity"]["learner"],opponent_counts=dict(Counter(r["category"] for r in matches)),opponent_versions=dict(Counter(r["opponent"] for r in matches)),legal_masks_digest=masks_digest(games),rollout_digest=digest(games),process_CPU_seconds=sum(f["process_CPU_seconds"] for f in files))
    write(out/"collection-complete.json",report)
    return games,matches,rng,report
def run_update(net,opt,games,seed,deadline,events):
    t=now();cpu=time.process_time()
    def callback(event):
        append(events,dict(at_utc=iso(),**event))
        if remaining(deadline)<=0:raise TimeoutError("UTC PPO deadline")
    metrics=update(net,opt,games,16,epochs=2,seed=seed,callback=callback,target_kl=.01,deadline=time.monotonic()+remaining(deadline),allow_oom_fallback=False,collect_diagnostics=True,entropy_coefficient=.001,bc_kl_coefficient=.01,value_coefficient=.5)
    if next(net.parameters()).device.type=="cuda":torch.cuda.synchronize()
    result=dict(UTC_seconds=(now()-t).total_seconds(),process_CPU_seconds=time.process_time()-cpu,metrics=metrics,optimizer_updates=sum(m["optimizer_updates"] for m in metrics),evaluated_decisions=sum(m["evaluated_decisions"] for m in metrics),trained_decisions=sum(m["decisions"] for m in metrics))
    write(Path(events).with_suffix(".result.json"),result)
    assert all(torch.isfinite(p).all() for p in net.parameters())
    if any(m["stop_reason"]=="deadline" for m in metrics):raise TimeoutError("Incomplete PPO deadline; no continuation")
    return result
def bundle_input(net,opt,games,worker_rng,metadata,path):
    value=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),games=games,worker_rng=worker_rng,metadata=metadata,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    content=digest(value);t=now();save_gzip(path,value);again=load_gzip(path)
    assert digest(again)==content
    return dict(path=str(path),sha256=sha(path),content_digest=content,legal_masks_digest=masks_digest(games),bytes=Path(path).stat().st_size,UTC_seconds=(now()-t).total_seconds())
def verify_prefix(reg,games,out):
    """Exactly the repaired CPU-live path/tolerances, no GPU anchor calls."""
    from .anchor_checks import diagnose_anchor,require_anchor_equivalence
    from .model import D
    bc=load_model(reg["initial"]).requires_grad_(False);worst=0.;count=0
    sample=copy.deepcopy([g[:64] for g in games[:2]])
    for g in sample:
        for r in g:r["reference"]=[p.reshape(-1) for p in r["reference"]]
    report=diagnose_anchor(bc,sample,Path(out)/"anchor",production="cpu-live");require_anchor_equivalence(report)
    for g in sample:
        hidden=torch.zeros(1,D)
        for r in g:
            with torch.no_grad():stats,hidden=bc.decision([r],hidden,actions=[r["action"]])
            error=abs(float(stats[1])-r["old_logp"]);worst=max(worst,error);count+=1
            assert error<=5e-4+1e-5*abs(r["old_logp"])
            assert abs(float(stats[3])-r["old_value"])<=5e-4+1e-5*abs(r["old_value"])
    result=dict(status="PASS",anchor=report,maximum_actor_logp_error=worst,decisions=count,new_matches=0)
    write(Path(out)/"prefix-check.json",result);return result
def probe(reg,out,count,device,jobs_override=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    exclusive(out/"probe-intent.json",dict(at_utc=iso(),native_BO3=count,device=device,disposable=True))
    began=now();deadline=(began+dt.timedelta(seconds=reg["preflight_seconds"]-300)).isoformat()
    pool=pool_for_phase(reg["initial"],{},"control");jobs=jobs_override if jobs_override is not None else phase_jobs(0,pool,count=count,seed=SEED+500,diagnostic=True)
    write(out/"diagnostic-schedule.json",jobs)
    games,matches,worker_rng,collection=collect(reg,reg["initial"],pool,jobs,out/"collection","probe",deadline)
    torch.set_num_threads(1);checks=verify_prefix(reg,games,out)
    net=load_model(reg["initial"]).to(device).requires_grad_(True);opt=fresh_optimizer(net)
    random.seed(SEED);torch.manual_seed(SEED)
    manifest=bundle_input(net,opt,games,worker_rng,dict(disposable=True,never_training=True,collection=collection),out/"probe-input.pt.gz")
    before=load_gzip(out/"probe-input.pt.gz")
    first=run_update(net,opt,games,SEED+1,deadline,out/"uninterrupted-events.jsonl")
    target=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()))
    save_gzip(out/"probe-result.pt.gz",target)
    net.load_state_dict(before["model"]);opt=fresh_optimizer(net);opt.load_state_dict(before["optimizer"])
    random.setstate(before["python_rng"]);torch.set_rng_state(before["torch_rng"])
    if device=="cuda":torch.cuda.set_rng_state_all(before["cuda_rng"])
    second=run_update(net,opt,before["games"],SEED+1,deadline,out/"resumed-events.jsonl")
    del target
    target=load_gzip(out/"probe-result.pt.gz")
    maximum=max((v.detach().cpu()-target["model"][k]).abs().max().item() for k,v in net.state_dict().items())
    assert maximum<=1e-6
    assert digest(opt.state_dict())==digest(target["optimizer"])
    assert digest(before["games"])==collection["rollout_digest"] and masks_digest(before["games"])==collection["legal_masks_digest"]
    result=dict(status="PASS",at_utc=iso(),device=device,native_BO3=count,collection=collection,serialization=manifest,prefix_checks=checks,first_update=first,resumed_update=second,resume_model_max_error=maximum,resume_optimizer_exact=True,disposable_not_training=True,total_UTC_seconds=(now()-began).total_seconds(),hardware=dict(cpu_count=os.cpu_count(),torch=str(torch.__version__),gpu=torch.cuda.get_device_name() if device=="cuda" else None))
    write(out/"probe-report.json",result);print(json.dumps(dict(event="preflight_complete",**result)),flush=True);return result
