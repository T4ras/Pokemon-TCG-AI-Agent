"""Visible-v2 entity links and SDK mechanics; no hidden-world or search inputs.

Location tuples identify tokens within a single observation. Raw card serials
are used only to match visible SKILL options, never as numeric model features.
Printed attack damage and baseline KO prize stakes are not predicted outcomes.
"""
from collections import Counter
from functools import lru_cache

SCHEMA = "visible-v2"
ENERGY_TYPES = 12
MECHANIC_FIELDS = (
    "attack_known", "printed_damage", "cost_units",
    *(f"cost_{i}" for i in range(ENERGY_TYPES)),
    "effect_text_present", "effects_supported", "self_damage",
    "needs_lunatone", "ignores_weakness_resistance", "discard_acceleration",
    "next_turn_attack_lock", "own_prizes", "opponent_prizes",
    "defender_baseline_prizes", "capped_baseline_prizes", "prizes_finish_if_ko",
    "lone_opponent", "source_energy_units", "defender_max_hp",
    "weakness_matches", "resistance_matches", "evolution_known",
    "evolution_compatible", "own_deck_count", "lunatone_on_bench",
)
MECHANIC_WIDTH = len(MECHANIC_FIELDS)


class Mechanics:
    def __init__(self, cards, attacks):
        self.cards = {c.cardId: c for c in cards}
        self.attacks = {a.attackId: a for a in attacks}

    @classmethod
    @lru_cache(maxsize=1)
    def from_sdk(cls):
        from ..engine import load_sdk
        api, _ = load_sdk()
        return cls(api.all_card_data(), api.all_attack())

    def features(self, obs, option, source, target):
        s = obs["current"]; me = s["yourIndex"]
        own, enemy = s["players"][me], s["players"][1-me]
        values = dict(own_prizes=len(own.get("prize") or [])/6,
                      opponent_prizes=len(enemy.get("prize") or [])/6,
                      own_deck_count=own.get("deckCount", 0)/60)
        source_meta = self.cards.get((source or {}).get("id"))
        target_meta = self.cards.get((target or {}).get("id"))
        if option["type"] == 9 and source_meta and target_meta:
            values.update(evolution_known=1,
                          evolution_compatible=source_meta.evolvesFrom == target_meta.name)
        attack = self.attacks.get(option.get("attackId"))
        if option["type"] == 13 and attack:
            values.update(attack_known=1, printed_damage=attack.damage/400,
                          cost_units=len(attack.energies)/10,
                          effect_text_present=bool(attack.text),
                          source_energy_units=len((source or {}).get("energies") or [])/10)
            costs = Counter(attack.energies)
            values.update({f"cost_{i}": costs[i]/10 for i in range(ENERGY_TYPES)})
            # Narrow, named SDK effects. Other text remains explicitly unsupported.
            # Matching name AND text prevents silently applying a stale numeric ID.
            text = attack.text
            if not text:
                values["effects_supported"] = 1
            elif attack.name == "Wild Press" and text == "This Pokémon also does 70 damage to itself.":
                values.update(effects_supported=1, self_damage=70/400)
            elif attack.name == "Cosmic Beam" and "Lunatone on your Bench" in text and "Weakness or Resistance" in text:
                values.update(effects_supported=1, needs_lunatone=1,
                              ignores_weakness_resistance=1,
                              lunatone_on_bench=any(
                                  self.cards.get(p.get("id")) and self.cards[p["id"]].name == "Lunatone"
                                  for p in own.get("bench", []) if p))
            elif attack.name == "Aura Jab" and text == "Attach up to 3 Basic {F} Energy cards from your discard pile to your Benched Pokémon in any way you like.":
                values.update(effects_supported=1, discard_acceleration=3/10)
            elif attack.name in ("Accelerating Stab", "Mega Brave") and text == f"During your next turn, this Pokémon can’t use {attack.name}.":
                values.update(effects_supported=1, next_turn_attack_lock=1)
            if target_meta:
                prizes = 3 if target_meta.megaEx else 2 if target_meta.ex else 1
                remaining = len(own.get("prize") or [])
                values.update(defender_baseline_prizes=prizes/3,
                              capped_baseline_prizes=min(prizes, remaining)/3,
                              prizes_finish_if_ko=0 < remaining <= prizes,
                              lone_opponent=sum(bool(p) for z in ("active", "bench") for p in enemy.get(z, [])) == 1,
                              defender_max_hp=(target or {}).get("maxHp", 0)/400,
                              weakness_matches=source_meta is not None and target_meta.weakness == source_meta.energyType,
                              resistance_matches=source_meta is not None and target_meta.resistance == source_meta.energyType)
        return [float(values.get(name, 0)) for name in MECHANIC_FIELDS]


def _at(obs, owner, area, index):
    if owner not in (0, 1) or not isinstance(index, int) or index < 0:
        return None
    s = obs["current"]; ps = s["players"][owner]
    cards = {1: obs["select"].get("deck"), 2: ps.get("hand"),
             3: ps.get("discard"), 4: ps.get("active"), 5: ps.get("bench"),
             6: ps.get("prize"), 7: s.get("stadium"), 12: s.get("looking")}.get(area) or []
    return cards[index] if index < len(cards) else None


def location(obs, option, target=False):
    """SDK OptionType semantics; absent addresses never mean hand index zero."""
    me = obs["current"]["yourIndex"]; typ = option["type"]
    owner = option.get("playerIndex")
    owner = me if owner is None else owner
    if target:
        if typ in (8, 9):
            area, index = option.get("inPlayArea"), option.get("inPlayIndex")
            return (me, area, index) if area in (4, 5) and index is not None else None
        if typ == 13 and obs["select"]["context"] in (0, 35):
            return (1-me, 4, 0)
        return None
    if typ in (3, 4, 5, 6, 8, 9, 10, 11):
        area, index = option.get("area"), option.get("index")
        if area is None or index is None:
            return None
        loc = (owner, area, index)
        if typ == 4:
            return loc + ("tools", option.get("toolIndex"))
        if typ in (5, 6):
            return loc + ("energyCards", option.get("energyIndex"))
        return loc
    if typ == 7:
        index = option.get("index")
        return (me, 2, index) if index is not None else None
    if typ == 12 or (typ == 13 and obs["select"]["context"] in (0, 35)):
        return (me, 4, 0)
    return None


def resolve(obs, option, target=False):
    loc = location(obs, option, target)
    if loc is None:
        return None
    card = _at(obs, *loc[:3])
    if len(loc) == 5:
        children = (card or {}).get(loc[3]) or []
        i = loc[4]
        return children[i] if isinstance(i, int) and 0 <= i < len(children) else None
    return card


def encode(obs, deck, knowledge, mechanics=None):
    from .observations import vector
    if obs["current"]["players"][1-obs["current"]["yourIndex"]].get("hand"):
        raise ValueError("Opponent hand must be hidden in a player-visible observation")
    mechanics = mechanics or Mechanics.from_sdk()
    reset = knowledge.update(obs)
    state, sel = obs["current"], obs["select"]; me = state["yourIndex"]
    ids, kinds, features, parents = [], [], [], []
    addresses, serials = {}, {}
    def token(cid, kind, values, parent=-1):
        ids.append(int(cid or 0)); kinds.append(kind)
        features.append(vector(values)); parents.append(parent)
        return len(ids)-1
    globals_ = [state.get("turn", 0)/100, state.get("turnActionCount", 0)/100,
                state.get("firstPlayer", -1) == me, state.get("firstPlayer", -1) < 0]
    globals_ += [state.get(k, False) for k in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated")]
    globals_ += [sel["type"]/16, sel["context"]/64, sel["minCount"]/60, sel["maxCount"]/60,
                 (sel.get("remainDamageCounter") or 0)/30, (sel.get("remainEnergyCost") or 0)/10]
    for owner in (me, 1-me):
        ps = state["players"][owner]
        globals_ += [ps.get("deckCount", 0)/60, ps.get("handCount", 0)/60,
                     len(ps.get("prize") or [])/6, ps.get("benchMax", 5)/8]
        globals_ += [ps.get(k, False) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")]
    token(0, 0, globals_)
    for cid, count in sorted(Counter(deck).items()): token(cid, 1, [count/60])
    for cid, count in sorted(knowledge.lower.items()): token(cid, 2, [count/60])
    def add(card, kind, position, address, parent=-1, depth=0):
        energy = Counter((card or {}).get("energies") or [])
        values = [position/60, depth, not bool(card), (card or {}).get("hp", 0)/400,
                  (card or {}).get("maxHp", 0)/400, (card or {}).get("appearThisTurn", False),
                  sum(energy.values())/10] + [energy[i]/10 for i in range(ENERGY_TYPES)]
        idx = token((card or {}).get("id", 0), kind, values, parent)
        addresses[address] = idx
        if card and card.get("serial") is not None:
            serials.setdefault(card["serial"], idx)
        for j, key in enumerate(("energyCards", "tools", "preEvolution"), 20):
            for n, child in enumerate((card or {}).get(key) or []):
                add(child, j+(0 if kind < 10 else 4), n, address+(key, n), idx, depth+1)
    for side, owner in enumerate((me, 1-me)):
        ps = state["players"][owner]
        for kind, (area, zone) in enumerate(((4, "active"), (5, "bench"), (2, "hand"), (3, "discard"), (6, "prize")), 3):
            for i, card in enumerate(ps.get(zone) or []):
                add(card, kind+side*7, i, (owner, area, i))
    for kind, area, cards in ((17, 7, state.get("stadium")), (18, 12, state.get("looking")), (19, 1, sel.get("deck"))):
        for i, card in enumerate(cards or []):
            add(card, kind, i, (me, area, i))
            # Shared zones are independent of the option's owner.
            addresses[(1-me, area, i)] = addresses[(me, area, i)]
    for key in ("contextCard", "effect"):
        if sel.get(key): add(sel[key], 29, 0, (key,))
    option_ids, targets, attack_ids, option_features = [], [], [], []
    sources, target_links, attack_features = [], [], []
    for i, option in enumerate(sel["option"]):
        source = resolve(obs, option); target = resolve(obs, option, True)
        src = addresses.get(location(obs, option), -1) if source else -1
        dst = addresses.get(location(obs, option, True), -1) if target else -1
        if option["type"] == 15 and option.get("cardId") and option.get("serial") in serials:
            src = serials[option["serial"]]
            source = {"id": ids[src]}
        option_ids.append((source or {}).get("id") or option.get("cardId") or 0)
        targets.append((target or {}).get("id", 0)); attack_ids.append(option.get("attackId") or 0)
        sources.append(src); target_links.append(dst)
        def n(key): return option.get(key) or 0
        option_features.append(vector([
            option["type"]/16, n("area")/12, n("index")/60,
            option.get("playerIndex") is not None and option["playerIndex"] != me,
            n("inPlayArea")/12, n("inPlayIndex")/8, n("number")/60, n("count")/10,
            n("energyIndex")/10, n("toolIndex")/10, n("specialConditionType")/5,
            sel["context"]/64, (source or {}).get("hp", 0)/400,
            (target or {}).get("hp", 0)/400, i/max(1, len(sel["option"])-1)]))
        attack_features.append(mechanics.features(obs, option, source, target))
    return dict(schema=SCHEMA, ids=ids, kinds=kinds, features=features, parents=parents,
                option_ids=option_ids, targets=targets, attack_ids=attack_ids,
                option_features=option_features, sources=sources, target_links=target_links,
                mechanics=attack_features, minimum=sel["minCount"], maximum=sel["maxCount"], reset=reset)
