"""CPU actors and deferred full-game BC anchor evaluation; unchanged PPO rows."""
from copy import deepcopy
from collections import Counter
import random,time
import torch
from torch.nn.utils.rnn import pad_sequence
from .model import D,collate
from .live import NeuralAgent,play_bo3
from .observations import encode
from .rl_timing import apply_returns
from ..policies import validate_action

def meaningful(row):
    n=len(row["option_ids"])
    return row["maximum"]>0 and not row["minimum"]==row["maximum"]==n

class EconomicalAgent(NeuralAgent):
    def __init__(self,network,deck,collect=True,reference=None):
        self.timing=Counter()
        if reference is not None and (next(reference.parameters()).device.type!="cpu" or next(network.parameters()).device.type!="cpu"):
            raise ValueError("Live anchor and acting policy must both use CPU")
        super().__init__(network,deck,reference=reference,collect=collect)
    def __call__(self,obs):
        t=time.perf_counter();row=encode(obs,self.deck,self.knowledge)
        self.timing["observation_encoding"]+=time.perf_counter()-t
        before=self.hidden.detach().clone();t=time.perf_counter()
        with torch.no_grad():
            stats,self.hidden=self.network.decision([row],self.hidden,temperature=1.)
        action,logp,entropy,value,_,_=stats;action=action[0]
        self.timing["policy_inference"]+=time.perf_counter()-t
        validate_action(obs["select"],action)
        refs=None
        if self.reference is not None:
            t=time.perf_counter()
            with torch.no_grad():
                refstats,self.ref_hidden=self.reference.decision([row],self.ref_hidden,actions=[action])
            refs=[r.cpu() for r in refstats[-1]]
            self.timing["reference_inference"]+=time.perf_counter()-t
        if self.collect:
            row.update(action=action,old_logp=float(logp),old_value=float(value),
                old_entropy=float(entropy),meaningful=meaningful(row),hidden=before.cpu(),reference=refs)
            self.game_rows.append(row)
        self.timing["decisions"]+=1
        return action
    def finish_game(self,result,seat):
        rows=self.game_rows;self.game_rows=[]
        if rows:rows[-1]["game_reward"]=0 if result==2 else 1 if result==seat else -1
        return rows

@torch.no_grad()
def deferred_reference(network,games,encode_batch_size=64,game_batch_size=128):
    """Preserve complete game prefixes, including forced observations."""
    network.eval();device=next(network.parameters()).device;started=time.perf_counter()
    total=0
    for start in range(0,len(games),game_batch_size):
        group=games[start:start+game_batch_size]
        flat=[row for game in group for row in game];encoded=[]
        for j in range(0,len(flat),encode_batch_size):
            batch=flat[j:j+encode_batch_size];b=collate(batch,device)
            globals_,opts=network.encode_batch(b)
            encoded.extend((globals_[i],opts[i,:len(row["option_ids"])]) for i,row in enumerate(batch))
        positions=[];offset=0
        for game in group:positions.append(offset);offset+=len(game)
        hidden=torch.zeros(len(group),D,device=device)
        for t in range(max(map(len,group),default=0)):
            active=[i for i,g in enumerate(group) if t<len(g)]
            current=[group[i][t] for i in active]
            global_=torch.stack([encoded[positions[i]+t][0] for i in active])
            options=pad_sequence([encoded[positions[i]+t][1] for i in active],batch_first=True)
            width=options.shape[1]
            b=dict(reset=torch.tensor([r["reset"] for r in current],device=device),
                minimum=torch.tensor([r["minimum"] for r in current],device=device),
                maximum=torch.tensor([r["maximum"] for r in current],device=device),
                option_mask=torch.arange(width,device=device)[None,:]>=torch.tensor([len(r["option_ids"]) for r in current],device=device)[:,None])
            h=network.recurrent(global_,hidden[active],b["reset"]);hidden[active]=h
            stats=network.decode(options,h,b,actions=[r["action"] for r in current])
            for k,row in enumerate(current):
                n=len(row["option_ids"]);depth=len(row["action"])+int(len(row["action"])<row["maximum"])
                row["reference"]=[torch.cat((dist[k,:n],dist[k,width:width+1])).cpu().clone() for dist in stats[-1][:depth]]
                row["anchor_logp"]=float(stats[1][k]);total+=1
        del encoded
    if device.type=="cuda":torch.cuda.synchronize()
    return dict(seconds=time.perf_counter()-started,decisions=total,full_game_prefixes=True,forced_observations_retained=True)

def opponent_schedule(reg,count,seed,diverse):
    if count%2:raise ValueError("Even chooser cohort required")
    rng=random.Random(seed);jobs=[]
    for pair in range(count//2):
        if not diverse:opponent="BC-OceanMix"
        else:
            u=rng.random()
            if u<.4:opponent=rng.choice(reg["history"])["id"]
            elif u<.8:opponent=rng.choice(reg["panel"][:4])["id"]
            else:opponent=rng.choice(reg["panel"][:2])["id"]
        jobs.extend(dict(opponent=opponent,learner_seat=seat,policy_seed=seed*10000+pair*2+seat) for seat in (0,1))
    rng.shuffle(jobs);return jobs

def collect_matches(learner,opponents,reg,jobs,prefix="collect",output=None,legacy=False,deadline=None,live_reference=False):
    from .rl_timing import TimedAgent
    from pathlib import Path
    if output is not None:Path(output).mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();games=[];rows=[];timing=Counter()
    for index,job in enumerate(jobs):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError("Collection deadline")
        random.seed(job["policy_seed"]);torch.manual_seed(job["policy_seed"])
        spec=next((p for p in reg["history"]+reg["panel"] if p["id"]==job["opponent"]),None)
        if spec is None:raise ValueError("Unregistered opponent")
        a=TimedAgent(learner,reg["deck"],opponents["BC-OceanMix"],True) if legacy else EconomicalAgent(learner,reg["deck"],reference=opponents["BC-OceanMix"] if live_reference else None)
        b=EconomicalAgent(opponents[job["opponent"]],spec["deck"],False)
        agents=[a,b] if job["learner_seat"]==0 else [b,a]
        summary,rollouts=play_bo3(agents,f"{prefix}-{index:03d}",output,max_seconds=min(300,max(1,deadline-time.monotonic())) if deadline is not None else 300,
            metadata=dict(**job,checkpoint_fixed_per_match=True))
        rows.append(summary)
        if summary["status"]!="DONE" or summary["games"][0]["chooser"]!=0:
            raise RuntimeError("Failed collection; no replacement match")
        game_rows=apply_returns(rollouts[job["learner_seat"]],summary["result"],job["learner_seat"],"game",lam=1.)
        games.extend(game_rows)
        for agent in (a,b):timing.update(agent.timing)
        if sum(map(len,games))>100000:raise RuntimeError("Rollout decision cap")
    return dict(games=games,matches=rows,timing=dict(timing),wall_seconds=time.perf_counter()-started)
