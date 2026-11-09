"""Bounded visible-v1 tactical wrapper. Only complete SDK terminal certificates override G.

A certificate is conditional on the tested legal hidden-card hypotheses and
the supported deterministic within-turn effects. It is not a full-game value
estimate or an exhaustive proof over every possible unseen deck.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
import torch
from .observations import encode
from .decoder_matrix import EvaluationAgent
from ..policies import validate_action

@dataclass(frozen=True)
class SearchConfig:
    enabled: bool = True
    hypotheses: int = 3
    max_setup_actions: int = 2
    max_candidates: int = 8
    max_followup_candidates: int = 4
    max_nodes: int = 128
    decision_seconds: float = .20
    match_seconds: float = 20.
    max_path_decisions: int = 12
    max_terminal_plans: int = 1
    discovery_fraction: float = .65
    def __post_init__(self):
        if self.hypotheses < 2 or not 0 < self.discovery_fraction < 1:
            raise ValueError("At least two worlds and a proper discovery allocation required")
        if not 0 <= self.max_setup_actions <= 2:
            raise ValueError("At most two setup actions")
        if min(self.max_candidates,self.max_followup_candidates,self.max_path_decisions,self.max_terminal_plans) < 1:
            raise ValueError("Positive branching limits required")
        if min(self.max_nodes,self.decision_seconds,self.match_seconds) < 0:
            raise ValueError("Negative budget")

class Exhausted(Exception): pass
class NoHypothesis(Exception): pass

def sparse(observation, round_id):
    """SDK result -> the historical encoder's sparse option format."""
    obs=asdict(observation) if not isinstance(observation,dict) else deepcopy(observation)
    if obs.get("current") is not None:obs["current"]["round"]=round_id
    if obs.get("select"):
        obs["select"]["option"]=[{k:v for k,v in o.items() if v is not None} for o in obs["select"]["option"]]
    # Native search exposes the sampled prize identities. They are hypotheses,
    # never legal network input. Known-prize roots are conservatively unsupported.
    if obs.get("current"):
        for player in obs["current"]["players"]:
            player["prize"]=[None for _ in player["prize"]]
    obs["logs"]=[{k:v for k,v in log.items() if v is not None} for log in obs.get("logs",[])]
    return obs

def public_key(api,obs):
    obj=asdict(api.to_observation_class(obs))
    return hashlib.sha256(json.dumps({k:obj[k] for k in ("current","select")},sort_keys=True,separators=(",",":")).encode()).hexdigest()

def visible_cards(obs,owner):
    seen={}
    def add(card,default):
        if not card:return
        if card.get("playerIndex",default)==owner and card.get("serial") is not None:
            seen[card["serial"]]=card["id"]
        for key in ("energyCards","tools","preEvolution"):
            for child in card.get(key,[]) or []:add(child,default)
    ps=obs["current"]["players"][owner]
    for zone in ("active","bench","hand","discard","prize"):
        for card in ps.get(zone,[]) or []:add(card,owner)
    for zone in ("stadium","looking"):
        for card in obs["current"].get(zone,[]) or []:add(card,None)
    for key in ("effect","contextCard"):add(obs["select"].get(key),None)
    return Counter(seen.values())

def worlds(obs,deck,knowledge,bank,count,seed_key,cards):
    """Closed, frozen generic prior bank; selection uses public counts only."""
    me=obs["current"]["yourIndex"];ps=obs["current"]["players"];own,enemy=ps[me],ps[1-me]
    if enemy.get("hand") or any(p is None for p in enemy.get("active",[])):
        raise NoHypothesis("hidden-view-contract")
    if any(p for player in ps for p in player["prize"]):raise NoHypothesis("known-prize-root-unsupported")
    if obs["select"].get("deck") is not None or obs["current"].get("looking"):
        raise NoHypothesis("ambiguous-looking-zone")
    vis=visible_cards(obs,me);full=Counter(deck)
    if vis-full:raise NoHypothesis("own-visible-overcount")
    remaining=full-vis
    unknown_prizes=sum(p is None for p in own["prize"])
    if sum(remaining.values())!=own["deckCount"]+unknown_prizes:
        raise NoHypothesis("own-hidden-accounting")
    enemy_vis=visible_cards(obs,1-me)
    lower=knowledge.lower|enemy_vis
    from ..engine import validate_deck
    compatible=[]
    # Constrained public completions, not lookup of the replay's true opponent.
    # Every frozen template supplies a generic ordering and a basic-energy filler.
    for template in bank:
        proposed=list(lower.elements())
        energy=next((cid for cid in template if cards[cid].cardType==5),6)
        try:validate_deck(proposed+[energy]*(60-len(proposed)),cards)
        except ValueError:continue
        for cid in template:
            if len(proposed)>=60:break
            trial=proposed+[cid]
            try:validate_deck(trial+[energy]*(60-len(trial)),cards)
            except ValueError:continue
            proposed=trial
        proposed += [energy]*(60-len(proposed))
        validate_deck(proposed,cards)
        if sorted(proposed) not in compatible:compatible.append(sorted(proposed))
    if not compatible:raise NoHypothesis("no-legal-public-completion")
    # Seed excludes opaque state, outcome, opponent identity and private replay metadata.
    visible=json.dumps(dict(public_key=seed_key,lower=sorted(lower.items())),sort_keys=True)
    rng=random.Random(int(hashlib.sha256(visible.encode()).hexdigest()[:16],16))
    compatible=list(compatible);rng.shuffle(compatible);result=[]
    for i in range(count):
        own_pool=list(remaining.elements());rng.shuffle(own_pool)
        own_prizes=[p["id"] if p else own_pool.pop() for p in own["prize"]]
        pool=list((Counter(compatible[i%len(compatible)])-enemy_vis).elements());rng.shuffle(pool)
        prizes=[p["id"] if p else pool.pop() for p in enemy["prize"]]
        hand=[pool.pop() for _ in range(enemy["handCount"])]
        if len(pool)!=enemy["deckCount"]:raise NoHypothesis("opponent-hidden-accounting")
        result.append(dict(your_deck=own_pool,your_prize=own_prizes,opponent_deck=pool,
                           opponent_prize=prizes,opponent_hand=hand,opponent_active=[],manual_coin=True))
    return result

class Budget:
    def __init__(self,config,remaining,clock=time.perf_counter):
        self.clock=clock;self.started=clock();self.deadline=self.started+min(config.decision_seconds,max(0,remaining))
        self.limit=config.max_nodes;self.nodes=0
    def check(self):
        if self.nodes>=self.limit:raise Exhausted("node-budget")
        if self.clock()>=self.deadline:raise Exhausted("time-budget")
    def call(self,method,*args,**kwargs):
        self.check();self.nodes+=1
        result=method(*args,**kwargs)
        if self.clock()>=self.deadline:raise Exhausted("time-budget")
        return result

class TacticalSearch:
    def __init__(self,network,deck,bank,api,config=SearchConfig()):
        if network.observation_schema!="visible-v1":raise ValueError("Frozen visible-v1 G is required")
        self.network,self.deck,self.bank,self.api,self.config=network,list(deck),deepcopy(bank),api,config
        self.cards={c.cardId:c for c in api.all_card_data()}
        self.attacks={a.attackId:a for a in api.all_attack()}
        self.reset_match()
    def reset_match(self):
        self.spent=0.;self.pending=[];self.round=None;self.counts=Counter();self.nodes=0
        self.max_decision_seconds=0.;self.max_search_seconds=0.;self.last_event=None
    def source(self,obs,option):
        me=obs["current"]["yourIndex"];owner=option.get("playerIndex")
        owner=me if owner is None else owner
        area=option.get("area",2 if option["type"]==7 else None);idx=option.get("index")
        if owner not in (0,1) or not isinstance(idx,int) or idx<0:return None
        zones={2:"hand",3:"discard",4:"active",5:"bench",6:"prize"}
        cards=obs["current"]["players"][owner].get(zones.get(area),[]) or []
        return cards[idx] if idx<len(cards) else None
    def supported_main(self,obs,action):
        if len(action)!=1:return False
        o=obs["select"]["option"][action[0]];typ=o["type"]
        if typ==13:return o.get("attackId") in range(976,984)
        if typ in (8,9,12):return True
        if typ==7:
            cid=(self.source(obs,o) or {}).get("id")
            return cid in (1123,1141,1182,673,675,676,677)
        return False
    def candidates(self,obs,base,setup):
        s=obs["select"];options=s["option"];out=[list(base)]
        def add(a):
            if a not in out:out.append(a)
        if s["minCount"]==s["maxCount"]==len(options):
            return [list(range(len(options)))]
        if s["context"]==0:
            for i,o in enumerate(options):
                if o["type"]==13:add([i])
            if setup<self.config.max_setup_actions:
                groups={}
                for i,o in enumerate(options):
                    if o["type"] in (7,8,9,12) and self.supported_main(obs,[i]):
                        groups.setdefault(o["type"],[]).append([i])
                # Round-robin categories prevents many attachments crowding out gust/evolution.
                for j in range(max(map(len,groups.values()),default=0)):
                    for typ in (9,7,8,12):
                        if j<len(groups.get(typ,[])):add(groups[typ][j])
            return out[:self.config.max_candidates]
        if s["minCount"]==0 and s["context"] in (21,22):
            add([])
            return out[:self.config.max_followup_candidates]
        if options and all(o.get("area")==6 and o.get("playerIndex")==obs["current"]["yourIndex"] for o in options):
            add(list(range(s["minCount"])))
            return out[:self.config.max_followup_candidates]
        if s["type"]==9 and s["context"] in (43,44):
            for i in range(len(options)):add([i])
        elif s["minCount"]==s["maxCount"]==1 and s["context"] in (3,4,18,19,34,35,37):
            def rank(i):
                card=self.source(obs,options[i]) or {};meta=self.cards.get(card.get("id"))
                prizes=3 if meta and meta.megaEx else 2 if meta and meta.ex else 1
                return (-prizes,card.get("hp",10000),i)
            for i in sorted(range(len(options)),key=rank):add([i])
        elif s["type"] in (2,4) and s["context"] in (26,30):
            # Same G energy-payment choice only; never invent subset legality.
            pass
        else:return []
        return out[:self.config.max_followup_candidates]
    @staticmethod
    def stochastic(obs):
        return obs["select"]["context"]==46 or any(p.get(k,False) for p in obs["current"]["players"] for k in ("confused","asleep","burned"))
    def eligible(self,obs):
        s=obs["select"];state=obs["current"];me=state["yourIndex"];enemy=state["players"][1-me]
        if s["context"]!=0:return False
        if s["minCount"]!=1 or s["maxCount"]!=1:return False
        field=[p for z in ("active","bench") for p in enemy.get(z,[]) if p]
        if not field or not state["players"][me]["active"]:return False
        # A supported single-active-target attack cannot take more than three baseline prizes.
        return len(state["players"][me]["prize"])<=3 or len(field)==1
    def predict(self,obs,hidden,knowledge):
        knowledge=deepcopy(knowledge);row=encode(obs,self.deck,knowledge)
        with torch.no_grad():stats,next_hidden=self.network.decision([row],hidden.detach().clone(),greedy=True)
        return stats[0][0],next_hidden,knowledge
    def _discover(self,obs,base,hidden,knowledge,world,budget):
        plans=[];me=obs["current"]["yourIndex"];turn=obs["current"]["turn"];round_id=obs["current"].get("round",1)
        cutoff=budget.started+(budget.deadline-budget.started)*self.config.discovery_fraction
        def dfs(node,h,k,path,setup,attack_used=False,processed=False):
            if len(plans)>=self.config.max_terminal_plans:return
            budget.check()
            current=sparse(node.observation,round_id)
            if current["current"]["result"]>=0:
                if current["current"]["result"]==me and attack_used:
                    plans.append(dict(steps=path,result=me,evidence="deterministic-supported-SDK-terminal-in-anchor"))
                return
            if current["current"]["yourIndex"]!=me or current["current"]["turn"]!=turn:return
            if self.stochastic(current):return
            if len(path)>=self.config.max_path_decisions:return
            if budget.clock()>=cutoff:raise Exhausted("discovery-budget")
            original=base
            if not processed:original,h,k=self.predict(current,h,k)
            budget.check()
            choices=self.candidates(current,original,setup)
            if current["select"]["context"]==0:
                choices.sort(key=lambda a:not (len(a)==1 and current["select"]["option"][a[0]]["type"]==13))
            for action in choices:
                if len(plans)>=self.config.max_terminal_plans:return
                s=current["select"];o=s["option"][action[0]] if len(action)==1 else None
                is_main=s["context"]==0
                supported=not is_main or self.supported_main(current,action)
                attack=bool(o and o["type"]==13)
                next_setup=setup+int(is_main and o and o["type"] in (7,8,9,12))
                if is_main and not attack and next_setup>self.config.max_setup_actions:continue
                validate_action(s,action)
                child=None
                try:
                    child=budget.call(self.api.search_step,node.searchId,action)
                    step=dict(key=public_key(self.api,current),action=list(action),context=s["context"],
                              option_types=[s["option"][i]["type"] for i in action])
                    if supported:
                        dfs(child,h.detach().clone(),deepcopy(k),path+[step],next_setup,attack_used or attack)
                finally:
                    if child is not None and child.searchId!=node.searchId:self.api.search_release(child.searchId)
        try:
            root=budget.call(self.api.search_begin,self.api.to_observation_class(obs),**world)
            if public_key(self.api,sparse(root.observation,round_id))!=public_key(self.api,obs):
                raise ValueError("root-reconstruction-mismatch")
            try:dfs(root,hidden.detach().clone(),deepcopy(knowledge),[],0,processed=True)
            except Exhausted as exc:
                if str(exc)!="discovery-budget" or not plans:raise
        finally:self.api.search_end()
        return plans
    def _verify(self,obs,plan,world,budget,hidden,knowledge):
        round_id=obs["current"].get("round",1);me=obs["current"]["yourIndex"];node=None
        h=hidden.detach().clone();k=deepcopy(knowledge)
        try:
            node=budget.call(self.api.search_begin,self.api.to_observation_class(obs),**world)
            for j,step in enumerate(plan["steps"]):
                current=sparse(node.observation,round_id)
                if current["current"]["result"]>=0 or current["current"]["yourIndex"]!=me or self.stochastic(current):
                    return False
                if public_key(self.api,current)!=step["key"]:return False
                if j:h_action,h,k=self.predict(current,h,k)
                validate_action(current["select"],step["action"])
                parent=node;node=budget.call(self.api.search_step,node.searchId,step["action"])
                if node.searchId!=parent.searchId:self.api.search_release(parent.searchId)
            budget.check()
            return node.observation.current.result==me
        finally:self.api.search_end()
    def choose(self,obs,base,hidden,knowledge):
        started=time.perf_counter();event=dict(base=list(base),action=list(base),override=False,nodes=0,seconds=0.,reason="disabled")
        round_id=obs["current"].get("round",1)
        if round_id!=self.round:self.pending=[];self.round=round_id
        if not self.config.enabled:self.last_event=event;return list(base),event
        key=public_key(self.api,obs)
        if self.pending:
            step=self.pending.pop(0)
            if key==step["key"]:
                action=step["action"];validate_action(obs["select"],action)
                event.update(action=action,override=action!=base,reason="certified-terminal-continuation",
                             certificate=self.certificate,context=obs["select"]["context"])
                self._finish(event,started);return action,event
            self.pending=[];event["reason"]="plan-diverged"
            self._finish(event,started);return list(base),event
        if not self.eligible(obs):
            event["reason"]="not-terminal-window";self._finish(event,started);return list(base),event
        if self.stochastic(obs):
            event["reason"]="stochastic-root";self._finish(event,started);return list(base),event
        if self.spent>=self.config.match_seconds:
            event["reason"]="match-budget";self._finish(event,started);return list(base),event
        budget=Budget(self.config,self.config.match_seconds-self.spent)
        try:
            hypotheses=worlds(obs,self.deck,knowledge,self.bank,self.config.hypotheses,key,self.cards)
            plans=self._discover(obs,base,hidden,knowledge,hypotheses[0],budget)
            event["reason"]="no-certified-win"
            for plan in plans:
                confirmations=1
                for world in hypotheses[1:]:
                    if not self._verify(obs,plan,world,budget,hidden,knowledge):break
                    confirmations+=1
                if confirmations!=self.config.hypotheses:continue
                budget.check()
                action=plan["steps"][0]["action"]
                self.pending=deepcopy(plan["steps"][1:])
                self.certificate=dict(worlds=confirmations,manual_coin=True,coin_requests_rejected=True,
                    evidence="SDK-terminal-in-all-tested-observation-consistent-worlds",
                    not_exhaustive_hidden_world_proof=True,steps=deepcopy(plan["steps"]))
                event.update(action=action,override=action!=base,reason="certified-terminal-plan",certificate=self.certificate)
                break
        except NoHypothesis as exc:event["reason"]=str(exc)
        except Exhausted as exc:event["reason"]=str(exc);self.pending=[]
        except Exception as exc:
            event.update(reason="search-failure",error_type=type(exc).__name__,error=str(exc)[:160]);self.pending=[]
        finally:
            event["nodes"]=budget.nodes
            elapsed=time.perf_counter()-budget.started;self.spent+=elapsed;self.max_search_seconds=max(self.max_search_seconds,elapsed)
            event["seconds"]=elapsed
        self._finish(event,started);return list(event["action"]),event
    def _finish(self,event,started):
        event["total_wrapper_seconds"]=time.perf_counter()-started
        self.max_decision_seconds=max(self.max_decision_seconds,event["total_wrapper_seconds"])
        self.nodes+=event["nodes"];self.counts[event["reason"]]+=1
        self.counts["overrides"]+=int(event["override"]);self.last_event=event
    def summary(self):
        return dict(search_seconds=self.spent,nodes=self.nodes,counts=dict(self.counts),
                    max_wrapper_seconds=self.max_decision_seconds,max_search_seconds=self.max_search_seconds)

class SearchAgent(EvaluationAgent):
    def __init__(self,network,deck,bank,api,config=SearchConfig(),event_callback=None):
        self.search=TacticalSearch(network,deck,bank,api,config);self.event_callback=event_callback
        super().__init__(network,deck,"argmax")
    def reset_match(self):
        super().reset_match()
        if hasattr(self,"search"):self.search.reset_match()
    def __call__(self,obs):
        started=time.perf_counter();action=super().__call__(obs)
        result,event=self.search.choose(obs,action,self.hidden,self.knowledge)
        event["agent_seconds"]=time.perf_counter()-started
        if self.event_callback:self.event_callback(obs,event)
        return result
