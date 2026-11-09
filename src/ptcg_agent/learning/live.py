"""Persistent native BO3 runner with bounded, player-visible learning buffers."""
import json, time, math
from collections import Counter
from pathlib import Path
import torch
from .model import D
from .observations import DeckKnowledge, encode
from ..engine import load_sdk, validate_deck
from ..policies import validate_action

class NeuralAgent:
    def __init__(self, network, deck, reference=None, collect=False):
        if reference is not None and getattr(reference,"observation_schema","visible-v1") != getattr(network,"observation_schema","visible-v1"):
            raise ValueError("Policy/reference observation schema mismatch")
        self.network,self.deck,self.reference,self.collect=network,deck,reference,collect
        self.reset_match()
    def reset_match(self):
        device=next(self.network.parameters()).device
        self.knowledge=DeckKnowledge();self.hidden=torch.zeros(1,D,device=device)
        self.ref_hidden=torch.zeros_like(self.hidden);self.game_rows=[]
    def __call__(self,obs):
        row=(encode(obs,self.deck,self.knowledge)
             if getattr(self.network,"observation_schema","visible-v1")=="visible-v1"
             else encode(obs,self.deck,self.knowledge,schema=self.network.observation_schema))
        before=self.hidden.detach().clone()
        with torch.no_grad():
            stats,self.hidden=self.network.decision([row],self.hidden)
            action,logp,entropy,value,_,_=stats
            refs=None
            if self.reference is not None:
                refstats,self.ref_hidden=self.reference.decision([row],self.ref_hidden,actions=action)
                refs=[r.cpu() for r in refstats[-1]]
        action=action[0];validate_action(obs["select"],action)
        if self.collect:
            row.update(action=action,old_logp=float(logp.item()),old_value=float(value.item()),hidden=before.cpu(),reference=refs)
            self.game_rows.append(row)
        return action
    def finish_game(self,result,seat):
        rows=self.game_rows;self.game_rows=[]
        reward=0 if result==2 else (1 if result==seat else -1)
        next_value=0.;advantage=0.
        for i in range(len(rows)-1,-1,-1):
            r=reward if i==len(rows)-1 else 0.
            delta=r+next_value-rows[i]["old_value"]
            advantage=delta+0.95*advantage
            rows[i]["advantage"]=advantage;rows[i]["return"]=advantage+rows[i]["old_value"]
            next_value=rows[i]["old_value"]
        return rows

def play_bo3(agents,match_id,output=None,max_decisions=10000,max_seconds=180,metadata=None,trace_callback=None):
    """Match initial chooser is seat 0; subsequent chooser alternates officially."""
    api,game=load_sdk();cards={c.cardId:c for c in api.all_card_data()}
    for a in agents:
        validate_deck(a.deck,cards)
        if hasattr(a,"reset_match"):a.reset_match()
    wins=[0,0];games=[];rollouts=[[],[]];started=time.monotonic();decision_count=0
    try:
        for round_id in range(1,4):
            obs,start=game.battle_start(agents[0].deck,agents[1].deck,reverse_player=round_id%2==0)
            if not obs or start.errorPlayer>=0:raise ValueError("Native deck validation failed")
            chooser=obs["current"]["yourIndex"];choice=None;first=-1;game_start=decision_count
            try:
                while obs["current"]["result"]<0:
                    obs["current"]["round"]=round_id
                    actor=obs["current"]["yourIndex"]
                    if obs["current"]["firstPlayer"]>=0:first=obs["current"]["firstPlayer"]
                    action=agents[actor](obs)
                    if trace_callback is not None:trace_callback(obs,action)
                    if obs["select"]["context"]==41:choice=obs["select"]["option"][action[0]]["type"]==1
                    decision_count+=1
                    if decision_count>max_decisions or time.monotonic()-started>max_seconds:raise TimeoutError("Match resource cap")
                    obs=game.battle_select(action)
                result=obs["current"]["result"]
                if result in (0,1):wins[result]+=1
                games.append(dict(round=round_id,result=result,chooser=chooser,chooser_chose_first=choice,first_player=first,decisions=decision_count-game_start,finish_reason="native_terminal_result",completion_status="DONE"))
                for seat,a in enumerate(agents):
                    if hasattr(a,"finish_game"):
                        rows=a.finish_game(result,seat)
                        if rows:rollouts[seat].append(rows)
            finally:
                game.battle_finish()
            remaining=3-round_id
            if wins[0]>wins[1]+remaining or wins[1]>wins[0]+remaining:break
        result=0 if wins[0]>wins[1] else 1 if wins[1]>wins[0] else 2
        summary=dict(match_id=match_id,result=result,games=games,decisions=decision_count,wall_seconds=time.monotonic()-started,status="DONE")
    except Exception as error:
        summary=dict(match_id=match_id,status="FAILED",error=f"{type(error).__name__}: {error}",games=games,decisions=decision_count,wall_seconds=time.monotonic()-started)
        if output:
            folder=Path(output)/"failures";folder.mkdir(exist_ok=True,parents=True)
            signature=type(error).__name__
            existing=list(folder.glob(signature+"-*.json"))
            if len(existing)<10:
                # Bounded diagnostic observation; contains only the actor's legal view.
                (folder/f"{signature}-{match_id}.json").write_text(json.dumps(dict(summary=summary,observation={k:v for k,v in (obs or {}).items() if k in ("current","select","logs")}),default=str))
        rollouts=[[],[]]
    if metadata:
        if any(key in summary for key in metadata):raise ValueError("Match metadata cannot override measured results")
        summary.update(metadata)
    if output:
        with (Path(output)/"matches.jsonl").open("a") as f:f.write(json.dumps(summary)+"\n")
    return summary,rollouts

def balanced_schedule(count,seed):
    import random
    if count%2:raise ValueError("Chooser-balanced match count must be even")
    seats=[0]*(count//2)+[1]*(count//2);random.Random(seed).shuffle(seats);return seats

def rate_interval(counts):
    n=sum(counts.get(k,0) for k in ("win","loss","draw"))
    if not n:return dict(n=0,win_rate=None,wilson_95=None)
    p=counts.get("win",0)/n;z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n,win_rate=p,wilson_95=[center-half,center+half],draws_count_as_nonwins=True)

def summarize_matchups(matches,learner_seats):
    series=Counter();first=Counter();choosers=Counter();starts=Counter();all_games=Counter()
    for m,seat in zip(matches,learner_seats):
        if m["status"]!="DONE":series["failure"]+=1;continue
        def result(r):return "draw" if r==2 else "win" if r==seat else "loss"
        series[result(m["result"])]+=1;first[result(m["games"][0]["result"])]+=1
        choosers[str(m["games"][0]["chooser"]==seat)]+=1
        for g in m["games"]:
            starts[str(g["first_player"]==seat)]+=1;all_games[result(g["result"])]+=1
    return dict(bo3=dict(series),bo3_interval=rate_interval(series),first_game=dict(first),first_game_interval=rate_interval(first),initial_chooser=dict(choosers),actual_first_player_games=dict(starts),all_games=dict(all_games))
