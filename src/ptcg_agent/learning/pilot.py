"""Kaggle GPU entry point: exactly four frozen BC demonstrators and one learner."""
import argparse, copy, json, os, random, time
from pathlib import Path
import torch
from .model import PolicyNetwork, metadata_tables
from .replays import prepare
from .live import NeuralAgent, play_bo3, balanced_schedule, summarize_matchups
from .training import train_bc, load_sequences, save_checkpoint, load_checkpoint, resume_check, ppo_update
from ..engine import load_sdk
from ..policies import RandomPolicy

def run(source,output,config,require_gpu=True):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    if config["demonstrators"]!=4 or config["learners"]!=1:raise ValueError("This registered pilot requires N=4 K=1")
    if config["runtime_seconds"]>3000:raise ValueError("Exploratory pilot runtime cap is 3000 seconds")
    started=time.monotonic();hard_deadline=started+config["runtime_seconds"]
    if require_gpu and not torch.cuda.is_available():raise RuntimeError("Kaggle GPU required; refusing CPU training")
    torch.set_num_threads(min(2,os.cpu_count() or 1));random.seed(config["seed"]);torch.manual_seed(config["seed"])
    device="cuda" if require_gpu else "cpu"
    processed=output/"processed";data=prepare(source,processed)
    api,_=load_sdk();table,attack_count=metadata_tables(api)
    shared=PolicyNetwork(table,attack_count).to(device)
    report=dict(status="RUNNING",config=config,device=device,torch_version=torch.__version__,gpu_count=torch.cuda.device_count() if require_gpu else 0,gpu=torch.cuda.get_device_name(0) if require_gpu else None,parameters=sum(p.numel() for p in shared.parameters()),data_counts=data["counts"],data_failures=data["failures"],source_teachers=[{k:v for k,v in t.items() if k!="episode_ids"} for t in data["teachers"]],native_rng_seeded=False)
    def persist():
        report["wall_seconds"]=time.monotonic()-started
        (output/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ("source_teachers","config")}),flush=True)
    persist()
    train=load_sequences(processed,"train");validation=load_sequences(processed,"validation")
    report["shared_bc"]=train_bc(shared,train,validation,output,"shared-bc",epochs=config["bc_epochs"],deadline=min(hard_deadline-1200,time.monotonic()+config["bc_seconds"]),batch_sequences=config["batch_sequences"],accumulation=config["accumulation"])
    opponents=[];report["specialists"]=[]
    for teacher in range(4):
        model=copy.deepcopy(shared)
        history=train_bc(model,load_sequences(processed,"train",teacher),load_sequences(processed,"validation",teacher),output,f"demonstrator-{teacher}",epochs=1,learning_rate=1e-4,deadline=min(hard_deadline-900,time.monotonic()+config["specialist_seconds"]),batch_sequences=config["batch_sequences"],accumulation=config["accumulation"])
        opponents.append(NeuralAgent(model,data["teachers"][teacher]["deck"]))
        report["specialists"].append(history)
    learner=copy.deepcopy(opponents[0].network);reference=copy.deepcopy(learner);learner_deck=opponents[0].deck
    for model in [reference]+[o.network for o in opponents]:
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    report["resume_verification"]=resume_check(learner,output/"demonstrator-0.pt",train[0][:2])
    learner.eval();optimizer=torch.optim.AdamW(learner.parameters(),lr=3e-5)
    save_checkpoint(output/"learner-bc.pt",learner,optimizer,dict(label="learner-bc",deck=learner_deck,cohort=0))
    frozen_hashes=[{k:v.detach().clone() for k,v in o.network.state_dict().items()} for o in opponents]
    def evaluate(network,label):
        results=[]
        for teacher,opponent in enumerate(opponents):
            matches=[];seats=balanced_schedule(config["evaluation_bo3_per_opponent"],config["seed"]+teacher)
            for j,seat in enumerate(seats):
                if time.monotonic()>hard_deadline-90:raise TimeoutError("Evaluation deadline")
                candidate=NeuralAgent(network,learner_deck)
                agents=[candidate,opponent] if seat==0 else [opponent,candidate]
                summary,_=play_bo3(agents,f"{label}-{teacher}-{j}",output,max_seconds=config["match_seconds"],metadata=dict(stage=label,opponent=teacher,learner_seat=seat))
                matches.append(summary)
            results.append(dict(opponent=teacher,**summarize_matchups(matches,seats)))
        return results
    # Qualification is only a technical screen, not a claim of teacher-level strength.
    qualification=[]
    for teacher,opponent in enumerate(opponents):
        matches=[];seats=balanced_schedule(2,config["seed"]+100+teacher)
        for j,seat in enumerate(seats):
            random_agent=RandomPolicy(opponent.deck,config["seed"]+j)
            agents=[opponent,random_agent] if seat==0 else [random_agent,opponent]
            summary,_=play_bo3(agents,f"clone-smoke-{teacher}-{j}",output,max_seconds=config["match_seconds"],metadata=dict(stage="clone-smoke",demonstrator=teacher,demonstrator_seat=seat,opponent="random"));matches.append(summary)
        qualification.append(dict(teacher=teacher,**summarize_matchups(matches,seats)))
    report["clone_gameplay_smoke"]=qualification
    report["bc_validation_matchups"]=evaluate(reference,"bc-evaluation")
    persist()
    historical=[];report["ppo_cohorts"]=[]
    for cohort in range(config["ppo_cohorts"]):
        if time.monotonic()>hard_deadline-config["minimum_ppo_seconds"]:break
        current=copy.deepcopy(learner).eval()
        for parameter in current.parameters():parameter.requires_grad_(False)
        # Every opponent category and specific BC matchup has paired chooser seats.
        jobs=[]
        repeats=3 if not historical else 2
        for teacher in range(4):
            for _ in range(repeats):jobs.extend([(f"bc-{teacher}",opponents[teacher],0),(f"bc-{teacher}",opponents[teacher],1)])
        for _ in range(4):
            jobs.extend([("current",NeuralAgent(current,learner_deck),0),("current",NeuralAgent(current,learner_deck),1)])
        if historical:
            for _ in range(4):
                past=random.choice(historical)
                jobs.extend([("history",NeuralAgent(past,learner_deck),0),("history",NeuralAgent(past,learner_deck),1)])
        random.shuffle(jobs);games=[];collected=[];incomplete=False
        for j,(category,opponent,seat) in enumerate(jobs):
            if time.monotonic()>hard_deadline-config["minimum_ppo_seconds"]:
                incomplete=True;break
            agent=NeuralAgent(learner,learner_deck,reference=reference,collect=True)
            agents=[agent,opponent] if seat==0 else [opponent,agent]
            summary,rollouts=play_bo3(agents,f"ppo-{cohort}-{j}",output,max_seconds=config["match_seconds"],metadata=dict(stage="ppo",opponent_category=category,learner_seat=seat,learner_version=cohort))
            collected.append(dict(category=category,learner_seat=seat,summary=summary))
            games.extend(rollouts[seat])
            if sum(map(len,games))>config["rollout_cap_decisions"]:raise RuntimeError("Rollout cap exceeded")
        if incomplete:
            # Retain diagnostics, but no partial/imbalanced cohort optimizer update.
            report["ppo_cohorts"].append(dict(cohort=cohort,status="INCOMPLETE",matches=len(collected)));break
        decisions=sum(map(len,games))
        if not decisions:raise RuntimeError("No reliable on-policy rollouts")
        metrics=ppo_update(learner,optimizer,games,epochs=2,deadline=hard_deadline-180)
        record=dict(cohort=cohort,status="UPDATED",matches=len(collected),decisions=decisions,optimizer=metrics,categories={k:sum(r["category"]==k for r in collected) for k in sorted({r["category"] for r in collected})})
        report["ppo_cohorts"].append(record)
        save_checkpoint(output/"learner-ppo.pt",learner,optimizer,dict(label="learner-ppo",deck=learner_deck,cohort=cohort+1,config=config,history=record))
        historical.append(copy.deepcopy(current));historical=historical[-4:]
        persist()
    if not any(c["status"]=="UPDATED" for c in report["ppo_cohorts"]):raise RuntimeError("Pilot failed to complete a PPO update")
    report["ppo_validation_matchups"]=evaluate(learner,"ppo-evaluation")
    for before,opponent in zip(frozen_hashes,opponents):
        for key,tensor in opponent.network.state_dict().items():
            if not torch.equal(before[key],tensor):raise AssertionError("Frozen opponent mutated")
    report["frozen_opponents_verified"]=True
    import hashlib
    entries=[]
    for teacher,t in enumerate(data["teachers"]):
        checkpoint=output/f"demonstrator-{teacher}.pt"
        entries.append(dict(name=f"demonstrator-{teacher}",role="frozen-bc-opponent",source_submission_id=t["submission_id"],source_team=t["name"],deck=t["deck"],checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),sampling_temperature=1.0))
    checkpoint=output/"learner-ppo.pt"
    entries.append(dict(name="learner-0",role="ppo-learner",deck=learner_deck,checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),sampling_temperature=1.0))
    (output/"agents.json").write_text(json.dumps(dict(agents=entries,memory="GRU resets per game; revealed deck lower bounds persist per BO3"),indent=2))
    report["status"]="COMPLETE";report["strength_claim"]="Technical pilot only; too few matches for promotion or leaderboard-strength claims"
    persist()
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",required=True);parser.add_argument("--output",required=True);parser.add_argument("--config",required=True);parser.add_argument("--allow-cpu-smoke",action="store_true")
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    try:run(args.source,args.output,config,require_gpu=not args.allow_cpu_smoke)
    except Exception as error:
        path=Path(args.output)/"run-failure.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(error=f"{type(error).__name__}: {error}"),indent=2));raise

if __name__=="__main__":main()
