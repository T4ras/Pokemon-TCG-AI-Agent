"""Allowlisted player-visible encoding shared by replay training and live play."""
from collections import Counter
from copy import deepcopy

WIDTH = 32
SCHEMA = "visible-v1"

def vector(values):
    if len(values) > WIDTH: raise ValueError("Feature width exceeded")
    return [float(v or 0) for v in values] + [0.] * (WIDTH-len(values))

def card_children(card):
    if not isinstance(card, dict): return
    yield card
    for key in ("energyCards", "tools", "preEvolution"):
        for child in card.get(key, []) or []: yield from card_children(child)

class DeckKnowledge:
    """Distinct visible serials within a game; max lower bounds across games."""
    def __init__(self): self.round = None; self.seen = {}; self.lower = Counter()
    def update(self, obs):
        state = obs["current"]; round_id = state.get("round", 1)
        reset = round_id != self.round
        if reset: self.round = round_id; self.seen = {}
        enemy = 1-state["yourIndex"]
        def observe(card):
            if card and card.get("serial") is not None and card.get("id", 0):
                self.seen[card["serial"]] = card["id"]
        ps = state["players"][enemy]
        for zone in ("active", "bench", "discard", "hand", "prize"):
            for c in ps.get(zone, []) or []:
                for child in card_children(c):
                    if child.get("playerIndex", enemy) == enemy: observe(child)
        for c in state.get("stadium", []) or []:
            if c and c.get("playerIndex") == enemy: observe(c)
        for c in state.get("looking", []) or []:
            if c and c.get("playerIndex") == enemy: observe(c)
        for log in obs.get("logs", []):
            if log.get("playerIndex") != enemy: continue
            # Target ownership can differ; record only the actor's identified cards.
            for cid, serial in (("cardId", "serial"), ("cardIdBefore", "serialBefore"), ("cardIdAfter", "serialAfter"), ("cardIdActive", "serialActive"), ("cardIdBench", "serialBench")):
                if cid in log and serial in log: observe(dict(id=log[cid], serial=log[serial]))
        counts = Counter(self.seen.values())
        for cid, n in counts.items(): self.lower[cid] = max(n, self.lower[cid])
        return reset

def _legacy_resolve(obs, option, target=False):
    state, sel = obs["current"], obs["select"]
    owner = option.get("playerIndex", state["yourIndex"])
    if target: owner = state["yourIndex"]
    area = option.get("inPlayArea" if target else "area", 2)
    index = option.get("inPlayIndex" if target else "index", 0)
    ps = state["players"][owner]
    zones = {1:sel.get("deck"), 2:ps.get("hand"), 3:ps.get("discard"), 4:ps.get("active"), 5:ps.get("bench"), 6:ps.get("prize"), 7:state.get("stadium"), 12:state.get("looking")}
    cards = zones.get(area) or []
    if index is None or not 0 <= index < len(cards): return None
    card = cards[index]
    if card and not target:
        if option["type"] == 4:
            cards = card.get("tools", []); index = option.get("toolIndex", 0)
            card = cards[index] if 0 <= index < len(cards) else None
        elif option["type"] == 5:
            cards = card.get("energyCards", []); index = option.get("energyIndex", 0)
            card = cards[index] if 0 <= index < len(cards) else None
    return card

def _encode_v1(obs, deck, knowledge):
    """Ignore unknown fields, opaque search bytes, match score and replay visuals."""
    reset = knowledge.update(obs)
    s, state = obs["select"], obs["current"]; me = state["yourIndex"]
    ids, kinds, features = [], [], []
    def token(cid, kind, values):
        ids.append(int(cid or 0)); kinds.append(kind); features.append(vector(values))
    globals_ = [state.get("turn",0)/100, state.get("turnActionCount",0)/100, state.get("firstPlayer",-1)==me, state.get("firstPlayer",-1)<0]
    globals_ += [state.get(k, False) for k in ("supporterPlayed","stadiumPlayed","energyAttached","retreated")]
    globals_ += [s["type"]/16, s["context"]/64, s["minCount"]/60, s["maxCount"]/60, s.get("remainDamageCounter",0)/30, s.get("remainEnergyCost",0)/10]
    for owner in (me, 1-me):
        ps = state["players"][owner]
        globals_ += [ps.get("deckCount",0)/60, ps.get("handCount",0)/60, len(ps.get("prize",[]))/6, ps.get("benchMax",5)/8]
        globals_ += [ps.get(k,False) for k in ("poisoned","burned","asleep","paralyzed","confused")]
    token(0, 0, globals_)
    for cid, count in sorted(Counter(deck).items()): token(cid, 1, [count/60])
    for cid, count in sorted(knowledge.lower.items()): token(cid, 2, [count/60])
    def add_card(card, kind, position, depth=0):
        if not card: token(0,kind,[position/60,depth,1]); return
        token(card.get("id",0),kind,[position/60,depth,0,card.get("hp",0)/400,card.get("maxHp",0)/400,card.get("appearThisTurn",False),len(card.get("energies",[]))/10])
        for child_kind, key in enumerate(("energyCards","tools","preEvolution"), 20):
            for j, child in enumerate(card.get(key,[]) or []): add_card(child,child_kind+(0 if kind<10 else 4),j,depth+1)
    for side, owner in enumerate((me,1-me)):
        ps=state["players"][owner]
        for zone_id, zone in enumerate(("active","bench","hand","discard","prize"),3):
            for i,c in enumerate(ps.get(zone,[]) or []): add_card(c,zone_id+side*7,i)
    for kind, values in ((17,state.get("stadium")),(18,state.get("looking")),(19,s.get("deck"))):
        for i,c in enumerate(values or []): add_card(c,kind,i)
    for key in ("contextCard","effect"):
        if s.get(key): add_card(s[key],29,0)
    option_ids, targets, attack_ids, option_features = [], [], [], []
    for i,o in enumerate(s["option"]):
        source = _legacy_resolve(obs,o); target = _legacy_resolve(obs,o,True) if "inPlayArea" in o else None
        option_ids.append(o.get("cardId") or (source or {}).get("id",0))
        targets.append((target or {}).get("id",0)); attack_ids.append(o.get("attackId",0))
        option_features.append(vector([o["type"]/16,o.get("area",0)/12,o.get("index",0)/60,(o.get("playerIndex",me)!=me),o.get("inPlayArea",0)/12,o.get("inPlayIndex",0)/8,o.get("number",0)/60,o.get("count",0)/10,o.get("energyIndex",0)/10,o.get("toolIndex",0)/10,o.get("specialConditionType",0)/5,s["context"]/64,(source or {}).get("hp",0)/400,(target or {}).get("hp",0)/400,i/max(1,len(s["option"])-1)]))
    return dict(ids=ids,kinds=kinds,features=features,option_ids=option_ids,targets=targets,attack_ids=attack_ids,option_features=option_features,minimum=s["minCount"],maximum=s["maxCount"],reset=reset)

# Historical callers/checkpoints stay on visible-v1 until explicitly migrated.
SEMANTIC_SCHEMA = "visible-v2"

def resolve(obs, option, target=False):
    from .semantics import resolve as semantic_resolve
    return semantic_resolve(obs, option, target)

def encode(obs, deck, knowledge, schema=SCHEMA, mechanics=None):
    if schema == SCHEMA:
        return _encode_v1(obs, deck, knowledge)
    if schema == SEMANTIC_SCHEMA:
        from .semantics import encode as semantic_encode
        return semantic_encode(obs, deck, knowledge, mechanics)
    raise ValueError(f"Unsupported observation schema: {schema}")
