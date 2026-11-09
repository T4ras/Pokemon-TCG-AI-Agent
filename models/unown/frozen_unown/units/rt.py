"""Reusable declarations extracted from unown_v2_runtime.py; dependencies injected per run."""

class CapacityRequired(RuntimeError):pass


class UnresolvedIdentity(RuntimeError):pass


@functools.lru_cache(maxsize=1)
def mapping():
 return read(PREVIOUS/'identity-mapping.json')


def tables():
 """Keep every old buffer row; append current mechanics at semantic model rows."""
 parent,_,_,_,_=reference();sd=parent.SA.policy.state_dict()
 cov=read(PREVIOUS/'identity-coverage.json');mp=mapping()
 arrays={}
 for key,kind in [('card_static','cards'),('card_attacks','cards'),('card_skills','cards'),('att_static','attacks')]:
  old=sd[key].numpy();a=np.zeros((cov[kind]['proposed_rows'],*old.shape[1:]),dtype=old.dtype);a[:len(old)]=old;arrays[key]=a
 current=read(PREVIOUS_PRIVATE/'current-tables.json')
 for a in current['attacks']:
  row=mp['attacks'][str(a['attackId'])]
  if row<len(sd['att_static']):continue
  f=arrays['att_static'][row];f[0]=a['damage']/300.
  for e in a['energies']:
   if not 0<=e<12:raise UnresolvedIdentity('Unknown energy type')
   f[1+e]+=1/4.
  f[13]=len(a['energies'])/5.;f[14]=float(bool(a['text']))
 for c in current['cards']:
  row=mp['cards'][str(c['cardId'])]
  if row<len(sd['card_static']):continue
  f=arrays['card_static'][row];f[c['cardType']]=1.;f[7+c['energyType']]=1.
  f[19:27]=[c[k] for k in ['basic','stage1','stage2','ex','megaEx','tera','aceSpec']]+[c['evolvesFrom'] is not None]
  f[27:30]=[c['hp']/300.,c['retreatCost']/5.,(3 if c['megaEx'] else 2 if c['ex'] else 1)/3.]
  f[30+(c['weakness'] if c['weakness'] is not None else 12)]=1
  f[43+(c['resistance'] if c['resistance'] is not None else 12)]=1
  if len(c['attacks'])>4 or len(c['skills'])>2:raise CapacityRequired('Static attack/skill schema requires expansion')
  f[56:58]=[len(c['attacks'])/4.,len(c['skills'])/2.]
  arrays['card_attacks'][row,:len(c['attacks'])]=[mp['attacks'][str(i)] for i in c['attacks']]
  ids=mp['skills_by_card'][str(c['cardId'])];arrays['card_skills'][row,:len(ids)]=ids
 return arrays


def make_model(text_path=None,technical_probe=False):
 parent,_,_,_,_=reference();net=copy.deepcopy(parent.SA.policy)
 if hasattr(net,'_frozen_tables'):del net._frozen_tables
 cov=read(PREVIOUS/'identity-coverage.json')
 for name,kind in [('card_id_emb','cards'),('attack_id_emb','attacks'),('skill_id_emb','skills')]:
  old=getattr(net,name).weight
  new=torch.nn.Embedding(cov[kind]['proposed_rows'],old.shape[1])
  with torch.no_grad():new.weight.copy_(torch.from_numpy(extend_rows(old.detach().numpy(),new.num_embeddings,name+'.weight')))
  setattr(net,name,new)
 old=net.aux_hand
 head=torch.nn.Linear(old.in_features,cov['cards']['proposed_rows'])
 with torch.no_grad():
  head.weight.copy_(torch.from_numpy(extend_rows(old.weight.detach().numpy(),head.out_features,'aux_hand.weight')))
  head.bias.copy_(torch.from_numpy(extend_rows(old.bias.detach().numpy(),head.out_features,'aux_hand.bias')))
 net.aux_hand=head
 for k,a in tables().items():setattr(net,k,torch.from_numpy(a))
 adapted_text=np.load(text_path) if text_path else None
 for name,kind,width in [('card_text','cards',128),('att_text','attacks',64),('skill_text','skills',64)]:
  original=getattr(net,name).numpy();n=cov[kind]['proposed_rows']
  if adapted_text is not None:
   a=adapted_text[kind].astype(np.float32)
   if a.shape!=(n,width) or a[:len(original)].tobytes()!=original.tobytes():raise ValueError('Released text rows changed')
  elif technical_probe:
   # Never exported as an adapted candidate. Only tests the independent interface.
   a=extend_rows(original,n,'NONSEMANTIC-TECHNICAL-PROBE-'+name)
   a[len(original):]/=np.linalg.norm(a[len(original):],axis=1,keepdims=True).clip(1e-9)
  else:raise UnresolvedIdentity('No accepted new-text mapping; candidate export prohibited')
  setattr(net,name,torch.from_numpy(a))
 net.__class__=type('CurrentPoolUnown',(DynamicMixin,type(net)),{})
 net.technical_probe=technical_probe;net.critic_mode='blind';net.eval()
 return net


@functools.lru_cache(maxsize=32)
def functions(layout):
 _,model,encoder,buffers,_=reference()
 ns=dict(model.__dict__);cov=read(PREVIOUS/'identity-coverage.json')
 ns.update(OFFSETS=layout.offsets,MAX_TOKENS=layout.tokens,MAX_OPTIONS=layout.options,TOK_INT=3+layout.tools,
  N_CARDS=cov['cards']['proposed_rows'],N_ATTACKS=cov['attacks']['proposed_rows'],N_SKILLS=cov['skills']['proposed_rows'],
  ORACLE_HAND_SLOTS=layout.oracle[0],ORACLE_SPLIT_SLOTS=layout.oracle[1],ORACLE_PRIZE_SLOTS=layout.oracle[2],N_ORACLE_TOKENS=sum(layout.oracle))
 ns['_take']=types.FunctionType(model._take.__code__,ns)
 source=textwrap.dedent(inspect.getsource(model.PTCGTransformer._core))
 needle='    x = self.token_proj(torch.cat(['
 assert source.count(needle)==1
 source=source.replace(needle,'    for extra_tool in range(5, TOK_INT):\n        et = tok_i[:, :, extra_tool]\n        tool_vec = tool_vec + card_vec[et] * (et > 0).unsqueeze(-1)\n'+needle)
 exec(compile(source,'<Unown-v2-dynamic-core>','exec'),ns)
 oracle=types.FunctionType(model.PTCGTransformer._oracle_value.__code__,ns)
 return ns['_core'],oracle


class DynamicMixin:
 def core(self,packet):
  row=packet.tensor(next(self.parameters()).device)
  return functions(packet.layout)[0](self,row),row
 def policy(self,packet):return self.core(packet)[0]['logits']
 def train_packet(self,packet):
  c,row=self.core(packet);rg=torch.cat([c['r'],c['g']],-1)
  return dict(logits=c['logits'],blind_v=self.value_mlp(rg),oracle_v=functions(packet.layout)[1](self,row,c),
   hand_pred=torch.nn.functional.softplus(self.aux_hand(rg)),prize_logits=self.aux_prize(c['h'][:,1:]).squeeze(-1),core=c)


@functools.lru_cache(maxsize=1)
def env_tables():
 current=read(PREVIOUS_PRIVATE/'current-tables.json');nc=max(c['cardId'] for c in current['cards'])+1;na=max(a['attackId'] for a in current['attacks'])+1
 t=dict(atk_damage=np.zeros(na,np.float32),atk_cost=np.zeros((na,12),np.int8),energy_type=np.zeros(nc,np.int8),
  weakness=np.full(nc,-1,np.int8),resistance=np.full(nc,-1,np.int8),retreat=np.zeros(nc,np.int8),top_attacks=np.zeros((nc,4),np.int64))
 for a in current['attacks']:
  t['atk_damage'][a['attackId']]=a['damage']
  for e in a['energies']:t['atk_cost'][a['attackId'],e]+=1
 for c in current['cards']:
  i=c['cardId'];t['energy_type'][i]=c['energyType'];t['retreat'][i]=c['retreatCost']
  for k in ['weakness','resistance']:t[k][i]=-1 if c[k] is None else c[k]
  ids=sorted(c['attacks'],key=lambda a:-t['atk_damage'][a]);t['top_attacks'][i,:len(ids)]=ids
 return t


def actor_view(obs):
 """No opaque simulator state, search serialization, or private opponent hand."""
 out={k:copy.deepcopy(obs[k]) for k in ['current','select','logs'] if k in obs}
 if out.get('current'):
  cur=out['current'];opp=1-cur['yourIndex'];cur['players'][opp]['hand']=None
  for k in list(cur):
   if k not in {'turn','turnActionCount','yourIndex','firstPlayer','supporterPlayed','stadiumPlayed','energyAttached','retreated','result','stadium','looking','players','round'}:cur.pop(k)
  for p in cur['players']:
   for k in list(p):
    if k not in {'active','bench','benchMax','deckCount','discard','prize','handCount','hand','poisoned','burned','asleep','paralyzed','confused'}:p.pop(k)
 return out


@functools.lru_cache(maxsize=32)
def compiled_encoder(layout):
 _,_,module,_,_=reference();ns=dict(module.__dict__)
 ns.update(OFFSETS=layout.offsets,MAX_TOKENS=layout.tokens,MAX_OPTIONS=layout.options,STOP=layout.stop,TOK_INT=3+layout.tools,CapacityRequired=CapacityRequired,_T=env_tables())
 for name in ['_slice','_tables','_deficit','_attack_feats','_threat']:
  original=getattr(module,name);ns[name]=types.FunctionType(original.__code__,ns)
 src=inspect.getsource(module.encode)
 src=src.replace('if nt >= MAX_TOKENS:\n            return None','if nt >= MAX_TOKENS:\n            raise CapacityRequired("token capacity")')
 src=src.replace('p["tools"][:2]','p["tools"]')
 src=src.replace('n = min(len(options), MAX_OPTIONS - 1)','n = len(options)\n    assert n < MAX_OPTIONS')
 # The parent representation is retained for safe cases; attached-card choices
 # need their official item index and identity, not the old host-only key.
 src=src.replace('if key not in classes:', 'if ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY):\n            key = key + (o.get("toolIndex"), o.get("energyIndex"), o.get("serial"))\n        if key not in classes:')
 exec(compile(src,'<Unown-v2-dynamic-encoder>','exec'),ns)
 return ns['encode']


def encode(obs,picked,stop_allowed,forced_run,tracker=None,strict_current=True):
 obs=actor_view(obs);sel=obs['select'];cur=obs['current'];mp=mapping()
 # Raw official IDs are validated before any model-side clamp can occur.
 def validate(v):
  if isinstance(v,dict):
   for k,x in v.items():
    kind='attacks' if k=='attackId' else 'cards' if k=='id' or k.startswith('cardId') else None
    if kind and isinstance(x,int) and x>0 and str(x) not in mp[kind]:raise UnresolvedIdentity(f'{kind}:{x}')
    if isinstance(x,(dict,list)):validate(x)
  elif isinstance(v,list):
   for x in v:validate(x)
 validate(obs)
 tools=max([2]+[len(p.get('tools',[])) for pl in cur['players'] for p in (pl['active']+pl['bench']) if p])
 layout=Layout(options=max(64,len(sel['option'])+1),tools=tools)
 while True:
  row=np.zeros(layout.size,np.float32)
  try:
   compiled_encoder(layout)(row,obs,picked,stop_allowed,forced_run,tracker)
   nti=row[slice(*layout.offsets['tok_int'])].reshape(layout.tokens,3+tools)
   if int((nti[:,1]>0).sum())+sum(o['type'] in [4,5,6] for o in sel['option'])>layout.tokens:raise CapacityRequired('Attachment capacity')
   break
  except CapacityRequired:
   layout=Layout(layout.tokens*2,layout.options,layout.tools)
   if layout.size*4>16*1024*1024:raise CapacityRequired('Explicit16MiB per-decision resource guard, nothing truncated')
 # Append attachment tokens only for actual attached-card references; preserve
 # host fields and give every distinct tool/energy card a real pointer.
 opt=row[slice(*layout.offsets['opt_int'])].reshape(layout.options,6)
 of=row[slice(*layout.offsets['opt_float'])].reshape(layout.options,8)
 ti=row[slice(*layout.offsets['tok_int'])].reshape(layout.tokens,3+tools)
 tf=row[slice(*layout.offsets['tok_float'])].reshape(layout.tokens,56)
 mask=row[slice(*layout.offsets['opt_mask'])];n=int((ti[:,1]>0).sum());attached={}
 for j,o in enumerate(sel['option']):
  if o['type'] not in [4,5,6]:continue
  who=o.get('playerIndex');who=cur['yourIndex'] if who is None else who
  area=o.get('area');idx=o.get('index');field='tools' if o['type']==4 else 'energyCards';item=o.get('toolIndex') if field=='tools' else o.get('energyIndex')
  if area not in [4,5] or item is None:raise UnresolvedIdentity('Attached-card option lacks official host/item reference')
  host=cur['players'][who]['active' if area==4 else 'bench'][idx]
  cards=host[field]
  if not 0<=item<len(cards):raise UnresolvedIdentity('Attached-card index out of bounds')
  key=(who,area,idx,field,item);hptr=int(opt[j,1])
  if key not in attached:
   if n>=layout.tokens:raise AssertionError('Attachment reservation violated')
   attached[key]=n;ti[n,:3]=[cards[item]['id'],14 if field=='tools' else 15,1 if who==cur['yourIndex'] else 2]
   tf[n,0]=.25;tf[n,27]=float(j not in picked);tf[n,28]=float(j in picked);n+=1
  opt[j,1]=attached[key]+1;opt[j,2]=hptr;opt[j,3]=cards[item]['id'];of[j,5]=item/5.
 # Map semantic identities after all current-metadata derived features.
 def map_array(values,kind):
  return np.array([mp[kind].get(str(int(v)),0) if int(v)==0 else mp[kind][str(int(v))] for v in values],dtype=np.float32)
 ti[:,0]=map_array(ti[:,0],'cards')
 for k in range(3,3+tools):ti[:,k]=map_array(ti[:,k],'cards')
 opt[:,3]=map_array(opt[:,3],'cards');opt[:,4]=map_array(opt[:,4],'attacks')
 di=row[slice(*layout.offsets['dec_int'])]
 for k in [2,3,6,7]:di[k]=mp['cards'].get(str(int(di[k])),0) if di[k]==0 else mp['cards'][str(int(di[k]))]
 for k in [4,5]:di[k]=mp['attacks'].get(str(int(di[k])),0) if di[k]==0 else mp['attacks'][str(int(di[k]))]
 assert np.isfinite(row).all()
 return Packet(row,layout)


@functools.lru_cache(maxsize=1)
def current_effect_tracker():
 """Retain released effect logic, load current metadata and audited records."""
 import re,unicodedata
 from types import SimpleNamespace
 reference()
 from ptcg import effects as original
 current=read(PREVIOUS_PRIVATE/'current-tables.json');old=read(PARENT/'original-tables.json')
 records=read(UP/'submission/data/lingering_effects.json')['records']
 def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s or '').replace('’',"'").lower()).strip()
 oldatt={a['attackId']:a for a in old['attacks']};newatt={a['attackId']:a for a in current['attacks']}
 oldcards={c['cardId']:c for c in old['cards']}
 bytext={}
 for r in records:
  if r['source_kind']!='attack':continue
  c=oldcards[r['card_id']]
  for aid in c['attacks']:
   if norm(oldatt[aid]['name'])==norm(r['source_name']):bytext.setdefault(norm(oldatt[aid]['text']),[]).append(r)
 added=[];unresolved=[];cov=read(PREVIOUS/'identity-coverage.json');changed=set(cov['cards']['new_ids']+cov['cards']['effective_changed_ids'])
 templates=[(r"during your next turn, this pok.mon can't attack\.?",'CANT_ATTACK','SELF','MY_NEXT_TURN'),
  (r"this pok.mon can't attack during your next turn\.?",'CANT_ATTACK','SELF','MY_NEXT_TURN')]
 for c in current['cards']:
  if c['cardId'] not in changed:continue
  for aid in c['attacks']:
   a=newatt[aid];text=norm(a['text']);copies=bytext.get(text,[]) if text else []
   if copies:
    unique=set()
    for r in copies:
     key=(r['clause_text'],r['effect_class'],r['condition'],r['target'],r['duration'])
     if key in unique:continue
     unique.add(key);row=dict(r,card_id=c['cardId'],card_name=c['name'],source_name=a['name'],notes='v2 exact entire attack-text equivalence to released record');added.append(row)
   else:
    matched=False
    for pattern,cls,target,duration in templates:
     if re.fullmatch(pattern,text):
      added.append(dict(card_id=c['cardId'],card_name=c['name'],source_kind='attack',source_name=a['name'],clause_text=a['text'],effect_class=cls,target=target,duration=duration,trigger_timing='IMMEDIATE',magnitude=None,condition='NONE',notes='v2 exact full-clause pattern'));matched=True;break
    if not matched and re.search(r'next turn|until|cannot|can.t|as long as',text):unresolved.append(dict(card_id=c['cardId'],attack_id=aid,text=a['text'],status='engine enforces mechanics; temporal feature unsupported, never treated as hard legality'))
  for s in c['skills']:
   if re.search(r'next turn|until|cannot|can.t|as long as',norm(s['text'])):unresolved.append(dict(card_id=c['cardId'],skill=s['name'],text=s['text'],status='ability trigger/continuous engine mechanics; no generic ability-used log, no fabricated tracker flag'))
 retained=[r for r in records if r['card_id'] not in changed]
 path=PRIVATE/'current-lingering-effects.json';write(path,dict(schema_version=1,records=retained+added))
 def ns(v):return SimpleNamespace(**{k:[SimpleNamespace(**x) if isinstance(x,dict) else x for x in val] if isinstance(val,list) else val for k,val in v.items()})
 import enum
 class Kind(enum.IntEnum):POKEMON=0;SUPPORTER=3
 namespace=dict(original.__dict__);namespace['__name__']='unown_v2_current_effects';namespace['_tables']=None;namespace['TABLE_STATS']={}
 src=inspect.getsource(original._build_tables).replace('from cg.api import all_card_data, all_attack, CardType','all_card_data, all_attack, CardType = CURRENT_METADATA')
 namespace['CURRENT_METADATA']=(lambda:[ns(c) for c in current['cards']],lambda:[ns(a) for a in current['attacks']],Kind)
 namespace['_JSON']=str(path);exec(src,namespace)
 # Rebind every effect method so all calls use this isolated current table.
 attrs={k:types.FunctionType(v.__code__,namespace,k,v.__defaults__,v.__closure__) if isinstance(v,types.FunctionType) else v for k,v in original.EffectTracker.__dict__.items() if k not in ['__dict__','__weakref__']}
 cls=type('CurrentEffectTracker',(),attrs);namespace['EffectTracker']=cls
 namespace['_build_tables']()
 write(OUT/'effect-coverage.json',dict(added_records=len(added),released_records_preserved=len(retained),changed_identity_records_removed=len(records)-len(retained),
  unresolved_temporal_features=unresolved,table_stats=namespace['TABLE_STATS'],
  distinction='Official SDK supplies legal actions, attachments, resources, HP, statuses and outcomes; tracker supplies only observed temporal/revealed information. Unsupported features are not legality overrides.'))
 return cls


class MatchTracker:
 def __init__(self,deck):
  self.deck=list(deck);self.reset_match()
 def reset_match(self):
  self.lower_bounds=Counter();self.provenance={};self.game=None;self.game_seen={};self.tracker=None
 def reset_game(self,game):
  self.game=game;self.game_seen={};self.tracker=reference()[4](self.deck);self.tracker.effects=current_effect_tracker()()
 def update(self,obs):
  obs=actor_view(obs);game=obs['current'].get('round',1)
  if self.game!=game:self.reset_game(game)
  self.tracker.update(obs);cur=obs['current'];op=1-cur['yourIndex'];cards=[]
  pl=cur['players'][op]
  for pk in pl['active']+pl['bench']:
   if pk:cards+=[pk]+pk.get('tools',[])+pk.get('energyCards',[])+pk.get('preEvolution',[])
  cards += [c for c in pl['discard']+pl['prize'] if c]
  for field in ['opp_known_hand','opp_known_deck']:
   cards += [dict(serial=s,id=i) for s,i in getattr(self.tracker,field).items()]
  for c in cards:
   if c.get('serial') is not None and c.get('id'):
    self.game_seen[c['serial']]=c['id']
    self.provenance.setdefault(c['id'],dict(game=game,source='actor-visible physical card'))
  for cid,count in Counter(self.game_seen.values()).items():self.lower_bounds[cid]=max(self.lower_bounds[cid],count)
 def encoder_tracker(self):
  # Lower bounds surviving a game reset reveal composition, never current hand.
  t=self.tracker.clone();visible=t.opp_visible()+Counter(t.opp_known_hand.values())+Counter(t.opp_known_deck.values())
  remainder=self.lower_bounds-visible
  for i,cid in enumerate(remainder.elements()):t.opp_known_deck[-1-i]=cid
  return t


class Actor:
 def __init__(self,model,deck,sample=False,seed=SEED,collect=False):
  self.model=model;self.deck=list(deck);self.sample=sample;self.generator=torch.Generator().manual_seed(seed)
  self.collect=collect;self.records=[];self.memory=MatchTracker(deck);self.forced_run=0;self.last_game=None;self.calls=0;self.forced=0
 def reset_match(self):
  self.memory.reset_match();self.records=[];self.forced_run=0;self.last_game=None
 def act(self,raw):
  if raw.get('select') is None:self.reset_match();return list(self.deck)
  obs=actor_view(raw);self.memory.update(obs);game=self.memory.game
  if self.last_game!=game:self.forced_run=0;self.last_game=game
  self.calls+=1;sel=obs['select'];n=len(sel['option'])
  if sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1:
   self.forced_run+=1;self.forced+=1;return list(range(n))
  picked=[]
  while True:
   stop_ok=len(picked)>=sel['minCount'] and (sel['maxCount']>1 or sel['minCount']==0)
   packet=encode(obs,picked,stop_ok,self.forced_run,self.memory.encoder_tracker())
   with torch.no_grad():
    c,_=self.model.core(packet);logits=c['logits'][0];prob=torch.softmax(logits,-1)
    action=int(torch.multinomial(prob,1,generator=self.generator)) if self.sample else int(logits.argmax())
    lp=float(torch.log_softmax(logits,-1)[action]);v=float(self.model.value_mlp(torch.cat([c['r'],c['g']],-1))) if self.collect else 0.
   if self.collect:self.records.append(dict(packet=packet,action=action,log_probability=lp,value=v,game=game,seat=obs['current']['yourIndex']))
   if action==packet.layout.stop:
    if not stop_ok:raise AssertionError('Illegal STOP')
    break
   if not 0<=action<n or action in picked:raise AssertionError('Illegal/repeated pick')
   picked.append(action)
   if len(picked)>=sel['maxCount']:break
  self.forced_run=0
  if not sel['minCount']<=len(picked)<=sel['maxCount']:raise AssertionError('Invalid selection count')
  return picked

