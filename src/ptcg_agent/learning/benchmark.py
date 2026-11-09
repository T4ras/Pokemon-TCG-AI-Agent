"""Fresh chooser-stratified CPU benchmarks for one selected shared checkpoint."""
import argparse, concurrent.futures, hashlib, json, math, os, random, statistics, time
from collections import Counter
from pathlib import Path
import torch
from .model import PolicyNetwork,metadata_tables
from .training import load_checkpoint,load_sequences
from .lr_search import evaluate_teachers
from .live import NeuralAgent,play_bo3,balanced_schedule,rate_interval
from ..engine import load_sdk
from ..policies import make_policy

def outcome(result,seat):return "draw" if result==2 else "win" if result==seat else "loss"

def stratified_interval(groups):
    strata=[rate_interval(groups[str(i)]) for i in (0,1)]
    if any(s["n"]==0 for s in strata):return dict(estimate=None,interval_95=None,strata=strata)
    p=sum(s["win_rate"] for s in strata)/2
    lower=math.sqrt(sum((s["win_rate"]-s["wilson_95"][0])**2/4 for s in strata))
    upper=math.sqrt(sum((s["wilson_95"][1]-s["win_rate"])**2/4 for s in strata))
    return dict(estimate=p,interval_95=[max(0.,p-lower),min(1.,p+upper)],strata=strata,method="Equal-chooser-weight Wilson-MOVER; draws are nonwins")

def summarize(rows):
    attempted=len(rows);bo3=Counter();first=Counter();bo3_strata={"0":Counter(),"1":Counter()};first_strata={"0":Counter(),"1":Counter()};starts={"first":Counter(),"second":Counter(),"unknown":Counter()};all_games=Counter();mismatches=0
    for r in rows:
        seat=r["learner_seat"];stratum=str(int(seat==0))
        if r.get("games"):
            g=r["games"][0];observed=g["chooser"]==seat
            if observed!=(seat==0):mismatches+=1
            o=outcome(g["result"],seat);first[o]+=1;first_strata[stratum][o]+=1
            label="unknown" if g["first_player"] not in (0,1) else "first" if g["first_player"]==seat else "second"
            starts[label][o]+=1
        if r["status"]=="DONE":
            o=outcome(r["result"],seat);bo3[o]+=1;bo3_strata[stratum][o]+=1
        else:bo3["failure"]+=1
        for g in r.get("games",[]):all_games[outcome(g["result"],seat)]+=1
    return dict(attempted=attempted,completed_bo3=attempted-bo3["failure"],completed_first_games=sum(first.values()),failures=bo3["failure"],chooser_protocol_violations=mismatches,bo3=dict(bo3),first_game=dict(first),first_game_interval=stratified_interval(first_strata),bo3_interval=stratified_interval(bo3_strata),actual_first_game_order={k:dict(v) for k,v in starts.items()},all_games=dict(all_games),seconds=sum(r["wall_seconds"] for r in rows))

def worker(job):
    root=Path(job["results"]);output=Path(job["output"])/f"worker-{job['worker']}";output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(job["threads"]);torch.manual_seed(job["seed"]);random.seed(job["seed"])
    manifest=json.loads((root/"agents.json").read_text());api,_=load_sdk();table,attacks=metadata_tables(api)
    cards=api.all_card_data();attack_objects=api.all_attack();network=PolicyNetwork(table,attacks).eval();load_checkpoint(root/"selected.pt",network)
    before=hashlib.sha256((root/"selected.pt").read_bytes()).hexdigest()
    retained=0;dropped=0;rows=[]
    for item in job["jobs"]:
        teacher,opponent,index,seat=item;entry=manifest["agents"][teacher]
        torch.manual_seed(job["seed"]+teacher*100000+index*17+['random','heuristic-v1','heuristic-v2'].index(opponent))
        agent=NeuralAgent(network,entry["deck"]);other=make_policy(opponent,entry["deck"],job["seed"]+index,cards,attack_objects)
        agents=[agent,other] if seat==0 else [other,agent]
        match_id=f"bench-{teacher}-{opponent}-{index}";trace=[]
        def record(obs,action):
            nonlocal retained,dropped
            if len(trace)<1500:trace.append(dict(round=obs["current"]["round"],actor=obs["current"]["yourIndex"],observation={k:v for k,v in obs.items() if k in ('current','select')},action=action))
            else:dropped+=1
        callback=record if match_id in job["diagnostic_ids"] else None
        (output/'active.json').write_text(json.dumps(dict(match_id=match_id,started_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())))
        summary,_=play_bo3(agents,match_id,output,max_seconds=180,metadata=dict(teacher=teacher,opponent=opponent,learner_seat=seat,checkpoint_sha256=before,deck_sha256=hashlib.sha256(json.dumps(entry["deck"]).encode()).hexdigest(),inference_temperature=1.,scheduler_seed=job["seed"],possible_interruption_retry=match_id in job.get("interrupted_ids",[])),trace_callback=callback)
        rows.append(summary)
        (output/"active.json").unlink()
        if trace:
            payload=json.dumps(dict(summary=summary,trace=trace),separators=(',',':'))
            if retained+len(payload.encode())<=job["diagnostic_cap"]:
                folder=output/"sampled-visible-traces";folder.mkdir(exist_ok=True);(folder/f"{match_id}.json").write_text(payload);retained+=len(payload.encode())
            else:dropped+=1
        if len(rows)%25==0:print(json.dumps(dict(worker=job["worker"],completed=len(rows),planned=len(job["jobs"]),seconds=sum(r["wall_seconds"] for r in rows))),flush=True)
    if hashlib.sha256((root/"selected.pt").read_bytes()).hexdigest()!=before:raise AssertionError("Checkpoint changed")
    (output/"completion.json").write_text(json.dumps(dict(matches=len(rows),diagnostic_bytes=retained,dropped_details=dropped),indent=2))
    return rows

def run(results,output,processed=None,count=400,workers=4,seed=904112,threads=1):
    results=Path(results);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((results/"selected.pt").read_bytes()).hexdigest()
    manifest=json.loads((results/"agents.json").read_text())
    if len(manifest["agents"])!=4 or any(a["sha256"]!=digest for a in manifest["agents"]):raise ValueError("Agent/checkpoint provenance mismatch")
    jobs=[]
    for teacher in range(4):
        for opponent in ['random','heuristic-v1','heuristic-v2']:
            seats=balanced_schedule(count,seed+teacher*7+['random','heuristic-v1','heuristic-v2'].index(opponent))
            jobs.extend((teacher,opponent,j,seat) for j,seat in enumerate(seats))
    rng=random.Random(seed);rng.shuffle(jobs);sample=rng.sample(jobs,min(100,len(jobs)))
    diagnostic_ids=[f"bench-{t}-{o}-{j}" for t,o,j,s in sample]
    registration=dict(status="REGISTERED",checkpoint_sha256=digest,matches=len(jobs),matches_per_pairing=count,chooser_balance="Half per pairing, shuffled",seed=seed,workers=workers,threads=threads,native_rng_seeded=False,interval_method="Equal chooser-weight Wilson-MOVER, per matchup, pointwise 95%",draws="Nonwins with separate counts",sampled_visible_traces=len(sample),diagnostic_cap_bytes=2*1024**3,test_set_used_for_selection=False)
    (output/"registration.json").write_text(json.dumps(registration,indent=2));start=time.monotonic()
    work=[dict(results=str(results.resolve()),output=str(output.resolve()),worker=i,jobs=jobs[i::workers],threads=threads,seed=seed,diagnostic_ids=diagnostic_ids,diagnostic_cap=2*1024**3//workers) for i in range(workers)]
    if workers==1:rows=worker(work[0])
    else:
        import multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:rows=[r for shard in pool.map(worker,work) for r in shard]
    metrics=[]
    for teacher in range(4):
        for opponent in ['random','heuristic-v1','heuristic-v2']:
            group=[r for r in rows if r['teacher']==teacher and r['opponent']==opponent]
            metrics.append(dict(teacher=teacher,source_team=manifest['agents'][teacher]['source_team'],opponent=opponent,**summarize(group)))
    report=dict(status='COMPLETE',matches=len(rows),failures=sum(r['status']!='DONE' for r in rows),wall_seconds=time.monotonic()-start,matchups=metrics,checkpoint_sha256=digest)
    if processed:
        torch.set_num_threads(threads);api,_=load_sdk();table,attacks=metadata_tables(api);network=PolicyNetwork(table,attacks).eval();load_checkpoint(results/'selected.pt',network)
        report['heldout_test']=evaluate_teachers(network,{i:load_sequences(processed,'test',i) for i in range(4)},8)
    if any(r['chooser_protocol_violations'] for r in metrics):report['status']='FAILED_PROTOCOL'
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--processed');p.add_argument('--count',type=int,default=400);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=904112)
    a=p.parse_args();run(a.results,a.output,a.processed,a.count,a.workers,a.seed)
if __name__=='__main__':main()
