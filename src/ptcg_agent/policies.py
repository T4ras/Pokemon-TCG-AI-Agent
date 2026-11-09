import random

def validate_action(selection, action):
    if not isinstance(action, list) or any(type(i) is not int for i in action):
        raise ValueError("Action must be list[int]")
    if not selection["minCount"] <= len(action) <= selection["maxCount"]:
        raise ValueError("Selection count outside bounds")
    if len(set(action)) != len(action) or any(not 0 <= i < len(selection["option"]) for i in action):
        raise ValueError("Duplicate or out-of-range index")

class RandomPolicy:
    def __init__(self, deck, seed):
        self.deck = deck
        self.rng = random.Random(seed)

    def __call__(self, obs, configuration=None):
        s = obs.get("select")
        if s is None:
            return list(self.deck)
        n = self.rng.randint(s["minCount"], s["maxCount"])
        action = self.rng.sample(range(len(s["option"])), n)
        validate_action(s, action)
        return action


class HeuristicPolicy(RandomPolicy):
    """Fixed v1 engine, or v2 with only attack scoring changed.

    Sample-deck plan: start Snover, evolve to Mega Abomasnow, charge active to
    three Water, build a backup, use Waitress for acceleration and Lillie for draw.
    Never accesses replay visualizer or opaque search input.
    """
    def __init__(self, deck, seed, cards, attacks, version=1):
        super().__init__(deck, seed)
        self.cards, self.attacks, self.version = cards, attacks, version
        self.unsupported = set()

    def resolve(self, obs, option, target=False):
        state, s = obs['current'], obs['select']
        area = option.get('inPlayArea' if target else 'area', 2)
        index = option.get('inPlayIndex' if target else 'index', 0)
        owner = option.get('playerIndex', state['yourIndex'])
        if target:
            owner = state['yourIndex']
        ps = state['players'][owner]
        values = {1:s.get('deck'), 2:ps.get('hand'), 3:ps['discard'],
                  4:ps['active'], 5:ps['bench'], 6:ps['prize'],
                  7:state['stadium'], 12:state.get('looking')}.get(area)
        return values[index] if values and index is not None and 0 <= index < len(values) else None

    def card_value(self, card, obs):
        if not card:
            return 0  # Face-down prizes carry no identity information.
        cid = card['id']
        ps = obs['current']['players'][obs['current']['yourIndex']]
        field = ps['active'] + ps['bench']
        if cid == 723:
            return 100 if any(p and p['id'] == 722 for p in field) else 45
        if cid == 722:
            return 85 if sum(bool(p) and p['id'] in (722,723) for p in field) < 2 else 30
        if cid == 1235:
            return 75
        if cid == 1227:
            return 65
        if cid == 721:
            return 50
        if cid == 3:
            return 40
        return 25

    def field_value(self, card, obs):
        if not card:
            return 0
        cid = card['id']
        energy = len(card.get('energies', []))
        return {723:110, 722:70, 721:40}.get(cid,20) + min(energy,3)*25 + card.get('hp',0)/20

    def attack_damage(self, obs, option):
        attack = self.attacks[option['attackId']]
        damage = attack.damage
        if self.version == 1:
            return damage
        ps = obs['current']['players'][obs['current']['yourIndex']]
        cid = option['attackId']
        if cid == 1042:  # Kyogre Riptide: public own discard is sufficient.
            damage = 20*sum(c['id'] == 3 for c in ps['discard'])
        elif cid == 1046:  # Hammer-lanche expectation; unknown prizes stay unknown.
            known = list(ps['hand'] or []) + ps['discard']
            for p in ps['active'] + ps['bench']:
                if p:
                    known += p['energyCards']
            known += [c for c in ps['prize'] if c]
            water_left = max(0,self.deck.count(3)-sum(c['id']==3 for c in known))
            unseen = ps['deckCount']+sum(c is None for c in ps['prize'])
            damage = min(6,ps['deckCount'])*100*min(1,water_left/max(1,unseen))
            if ps['deckCount'] <= 6:
                damage -= 500  # Avoid likely next-turn deck-out unless it wins now.
        enemy = obs['current']['players'][1-obs['current']['yourIndex']]['active']
        if enemy and enemy[0]:
            defender = enemy[0]
            meta = self.cards[defender['id']]
            active = ps['active'][0]
            if active and meta.weakness == self.cards[active['id']].energyType:
                damage *= 2
            # Only fixed damage gets a deterministic KO bonus; expectation is uncertain.
            if cid != 1046 and damage >= defender['hp']:
                damage += 500
        return damage

    def score(self, obs, o):
        s, state = obs['select'], obs['current']
        context, typ = s['context'], o['type']
        ps = state['players'][state['yourIndex']]
        if typ == 0:  # Number selection: draw without exhausting deck.
            n = o['number']
            if context == 38:
                return n if n < ps['deckCount'] else -1000
            return n
        if typ in (1,2):
            want_yes = context != 42  # Keep a legal opening hand; avoid optional redraw.
            return 10 if (typ == 1) == want_yes else 0
        if typ == 13:
            return 40 + self.attack_damage(obs,o)/10
        if typ == 14:
            return -100
        if typ == 9:
            return 120
        if typ == 8:
            target = self.resolve(obs,o,True)
            if not target:
                return 0
            n = len(target.get('energies',[]))
            return (115 if o['inPlayArea']==4 else 80)+self.field_value(target,obs)/20-n*20 if n<3 else -20
        if typ == 7:
            card = self.resolve(obs,o)
            cid = card['id'] if card else 0
            if cid == 722:
                return 100 if len(ps['bench']) < 3 else 10
            if cid == 721:
                return 60 if len(ps['bench']) < 2 else 5
            if cid == 1145:
                return 95 if not any(c['id']==723 for c in ps['hand']) else -10
            if cid == 1235:
                return 90 if any(p and len(p['energies'])<3 for p in ps['active']+ps['bench']) else -10
            if cid == 1227:
                return 85 if len(ps['hand'])<6 and ps['deckCount']>8 else -10
            if cid == 1205:
                return 75 if not any(c['id']==723 for c in ps['hand']) else -10
            return 30
        if typ in (10,11,12):
            return -50  # No native abilities in this deck; avoid spending energy retreating.
        card = self.resolve(obs,o)
        if context in (1,3,4,21):
            value = self.field_value(card,obs)
            if context == 21 and card:
                value -= len(card.get('energies',[]))*30
            owner = o.get('playerIndex',state['yourIndex'])
            return value if owner == state['yourIndex'] else -value
        if context in (8,9,10,26,27,29,30):
            return -self.card_value(card,obs)
        if context in (2,5,6,7,11,18,19,22,24,25,28,31,32,33,34,37):
            return self.card_value(card,obs)
        if context in (13,14,15):
            return -card.get('hp',0) if card else 0
        if context in (16,17):
            return card.get('maxHp',0)-card.get('hp',0) if card else 0
        self.unsupported.add(context)
        return 0

    def __call__(self, obs, configuration=None):
        s = obs.get('select')
        if s is None:
            return list(self.deck)
        order = sorted(range(len(s['option'])), key=lambda i:(-self.score(obs,s['option'][i]),i))
        n = s['maxCount']
        # Unknown contexts have a minimal legal fallback rather than guessing card effects.
        if s['context'] in self.unsupported:
            n = s['minCount']
        action = order[:n]
        validate_action(s,action)
        return action

def make_policy(name, deck, seed, cards, attacks):
    if name == 'random':
        return RandomPolicy(deck,seed)
    if name in ('heuristic-v1','heuristic-v2'):
        return HeuristicPolicy(deck,seed,cards,attacks,version=int(name[-1]))
    raise ValueError(f'Unknown policy: {name}')
