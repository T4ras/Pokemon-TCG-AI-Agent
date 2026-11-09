"""Registered two-objective, three-round Kaggle timing experiment."""
import argparse, copy, hashlib, json, os, random, time
from pathlib import Path
import torch
from .model import PolicyNetwork, D
from .observations import encode
from .live import NeuralAgent, play_bo3
from .training import ppo_update, save_checkpoint
from ..policies import validate_action

def apply_returns(games, result, seat, objective, lam=0.99):
    if objective not in ("game", "match"):raise ValueError("Unknown reward objective")
    if objective == "game":
        sequences=[(rows, rows[-1]["game_reward"]) for rows in games if rows]
    else:
        rows=[row for game in games for row in game]
        sequences=[(rows, 0 if result==2 else 1 if result==seat else -1)] if rows else []
    for rows,reward in sequences:
        next_value=0.;advantage=0.
        for i in range(len(rows)-1,-1,-1):
            r=float(reward) if i==len(rows)-1 else 0.
            delta=r+next_value-rows[i]["old_value"]
            advantage=delta+lam*advantage
            rows[i].update(reward=r, advantage=advantage, **{"return":advantage+rows[i]["old_value"]})
            next_value=rows[i]["old_value"]
    return games

class TimedAgent(NeuralAgent):
    def __init__(self,network,deck,reference=None,collect=False,temperature=1.):
        self.temperature=temperature
        self.timing=dict(observation_encoding=0., policy_inference=0., reference_inference=0., decisions=0)
        super().__init__(network,deck,reference,collect)
    def __call__(self,obs):
        t=time.perf_counter();row=(encode(obs,self.deck,self.knowledge)
             if getattr(self.network,"observation_schema","visible-v1")=="visible-v1"
             else encode(obs,self.deck,self.knowledge,schema=self.network.observation_schema))
        self.timing["observation_encoding"]+=time.perf_counter()-t
        before=self.hidden.detach().clone();t=time.perf_counter()
        with torch.no_grad():
            stats,self.hidden=self.network.decision([row],self.hidden,temperature=self.temperature)
            action,logp,entropy,value,_,_=stats
            action=action[0];old_logp=float(logp.item());old_value=float(value.item())
        self.timing["policy_inference"]+=time.perf_counter()-t
        refs=None
        if self.reference is not None:
            t=time.perf_counter()
            with torch.no_grad():
                refstats,self.ref_hidden=self.reference.decision([row],self.ref_hidden,actions=[action])
                refs=[r.cpu() for r in refstats[-1]]
            self.timing["reference_inference"]+=time.perf_counter()-t
        validate_action(obs["select"],action)
        if self.collect:
            row.update(action=action,old_logp=old_logp,old_value=old_value,hidden=before.cpu(),reference=refs)
            self.game_rows.append(row)
        self.timing["decisions"]+=1
        return action
    def finish_game(self,result,seat):
        rows=self.game_rows;self.game_rows=[]
        if rows:rows[-1]["game_reward"]=0 if result==2 else 1 if result==seat else -1
        return rows

def schedule(round_index,seed,history):
    rng=random.Random(seed+round_index);jobs=[]
    # Each round has 12 chooser and 12 non-chooser fixed matches per learner.
    # Every specific opponent reverses chooser assignment in the next round.
    for learner in ("game","match"):
        for checkpoint in (30,34):
            for deck in range(4):
                for ti,temperature in enumerate((0.1,0.5,1.)):
                    index=((checkpoint==34)*4+deck)*3+ti
                    jobs.append(dict(category="fixed",learner=learner,checkpoint=checkpoint,deck=deck,temperature=temperature,seat=(index+round_index)%2))
    for seat in [0]*4+[1]*4:jobs.append(dict(category="cross",seat=seat))
    for learner in ("game","match"):
        for lineage in ("game","match"):
            version=rng.randrange(len(history[lineage]))
            for seat in (0,1):jobs.append(dict(category="history",learner=learner,lineage=lineage,version=version,seat=seat))
    rng.shuffle(jobs);return jobs

def run(inputs,output,config,require_gpu=True):
    inputs,output=Path(inputs),Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
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
    history={k:[copy.deepcopy(fixed[34])] for k in learners}
    report=dict(status="RUNNING",config=config,hardware=dict(device=device,gpu=torch.cuda.get_device_name(0) if require_gpu else None,gpu_count=torch.cuda.device_count(),cpu_count=os.cpu_count(),torch=torch.__version__),initialization_seconds=time.perf_counter()-started,rounds=[],native_rng_seeded=False,critic_match_score_input=False,critic_calibration=False)
    def emit(kind,**values):
        event=dict(kind=kind,elapsed_seconds=time.perf_counter()-started,**values)
        with (output/"events.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
        print(json.dumps(event),flush=True)
    def persist():
        report["wall_seconds"]=time.perf_counter()-started
        tmp=output/"report.tmp";tmp.write_text(json.dumps(report,indent=2));tmp.replace(output/"report.json")
    emit("initialized",**report["hardware"]);persist()
    for ri in range(config["rounds"]):
        rt=time.perf_counter();jobs=schedule(ri,config["seed"],history)
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
            summary,rollouts=play_bo3(agents,f"r{ri+1}-{j+1}",output,max_seconds=config["match_seconds"],metadata=dict(round_index=ri+1,schedule=job,collecting_seats=collect,policy_version=ri))
            matches.append(summary)
            for agent in agents:
                for metric in ("observation_encoding","policy_inference","reference_inference"):timings[metric]+=agent.timing[metric]
            if summary["status"]!="DONE":
                report["status"]="FAILED_COLLECTION";report["failure"]=summary;persist();raise RuntimeError("Failed match retained; refusing imbalanced PPO update")
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
        record=dict(round=ri+1,matches=len(matches),failures=0,learner_appearances=appearances,chooser_counts=chooser,learner_decisions=decisions,timings_seconds=timings,ppo={})
        report["rounds"].append(record);persist();emit("collection_complete",**record)
        for key in ("game","match"):
            emit("update_started",round=ri+1,learner=key);t=time.perf_counter()
            metrics=ppo_update(learners[key],optimizers[key],buffers[key],epochs=2,minibatch=1024,progress_callback=lambda m,k=key:emit("update_progress",round=ri+1,learner=k,**m))
            if require_gpu:torch.cuda.synchronize()
            timings[f"ppo_{key}"]=time.perf_counter()-t;record["ppo"][key]=metrics
            if not all(torch.isfinite(p).all() for p in learners[key].parameters()):raise FloatingPointError("Nonfinite updated parameter")
            emit("update_complete",round=ri+1,learner=key,seconds=timings[f"ppo_{key}"],metrics=metrics);persist()
        t=time.perf_counter()
        for key in learners:
            save_checkpoint(output/f"{key}-round-{ri+1}.pt",learners[key],optimizers[key],dict(round=ri+1,objective=key,deck=decks[2],config=config))
            historical=copy.deepcopy(learners[key]).eval().requires_grad_(False);history[key].append(historical);history[key]=history[key][-4:]
        if require_gpu:torch.cuda.synchronize()
        timings["checkpoint_and_history"]=time.perf_counter()-t
        record["round_wall_seconds"]=time.perf_counter()-rt
        del buffers
        emit("round_complete",**record);persist()
    report["status"]="COMPLETE";persist();emit("complete",wall_seconds=report["wall_seconds"])
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",required=True);p.add_argument("--output",required=True);p.add_argument("--config",required=True);p.add_argument("--cpu-smoke",action="store_true");a=p.parse_args()
    run(a.inputs,a.output,json.loads(Path(a.config).read_text()),not a.cpu_smoke)
if __name__=="__main__":main()
