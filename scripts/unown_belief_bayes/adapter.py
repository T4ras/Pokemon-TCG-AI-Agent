"""Legal-only native event compiler. No scheduled list or truth input exists."""
from unown_belief_bayes.common import *
import copy
from .provenance import replacement,validate_identity
from .locations import AREA,zone,destination_position

def legal(raw):
 o=copy.deepcopy({k:raw[k] for k in ('current','select','logs') if k in raw})
 cur=o.get('current') or {};me=cur.get('yourIndex',0)
 for i,p in enumerate(cur.get('players',[])):
  if i!=me:p['hand']=None
  p.pop('win',None)
 for k in ('result','draw'):cur.pop(k,None)
 return o

def signature(o):return hashlib.sha256(json.dumps(legal(o),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def visible(o,opponent):
 out={}
 def visit(x,zone):
  if isinstance(x,list):
   for q in x:visit(q,zone)
  elif isinstance(x,dict):
   if 'id' in x and 'serial' in x and x.get('playerIndex',opponent)==opponent:out[x['serial']]=dict(id=x['id'],serial=x['serial'],zone=zone)
   for k in ('energyCards','tools','preEvolution'):visit(x.get(k,[]),'B')
 cur=o['current'];p=cur['players'][opponent]
 visit(p.get('active'),'A');visit(p.get('bench'),'B');visit(p.get('discard'),'B');visit(cur.get('stadium'),'B')
 # Opponent-owned attachments may legally appear on the other player's board.
 other=cur['players'][1-opponent]
 for q in ('active','bench'):
  for pk in other.get(q) or []:
   if pk:
    for k in ('energyCards','tools','preEvolution'):visit(pk.get(k,[]),'B')
 return list(out.values())

class Compiler:
 def __init__(self,cards):
  self.allowed_cards={c for v in prior().values() for c in v['cards']};self.profiles=read(OUT/'native-effect-profiles.json')['cards'];self.meta={c['cardId']:c for c in cards};self.basic=[k for k,c in self.meta.items() if c.get('basic')];self.game=None;self.seen_obs=set();self.effect=None;self.crispin_pending=False;self.metang_looking=False;self.deck_evolve_pending=False;self.known={};self.sizes={'D':60,'H':0,'P':0,'L':0};self.events=[];self.coverage=collections.Counter();self.lower={};self.game_seen={};self.begun=False;self.observations=0;self.look_remaining=0
 def reset_game(self,g):
  self.game=g;self.seen_obs=set();self.effect=None;self.crispin_pending=False;self.metang_looking=False;self.deck_evolve_pending=False;self.known={};self.sizes={'D':60,'H':0,'P':0,'L':0};self.events=[];self.game_seen={};self.begun=False;self.look_remaining=0
 def text(self,cid):
  c=self.meta.get(cid,{})
  return ' '.join(s.get('text','') for s in c.get('skills',[])).lower()
 def note(self,cid,sid,zone):
  if sid is None:return
  validate_identity(cid,sid,self.known,self.allowed_cards)
  if sid in self.game_seen and self.game_seen[sid]!=cid:raise UnsupportedEvent('Identity-transform native event needs explicit per-card mapping')
  self.game_seen[sid]=cid;self.known[sid]=(cid,zone)
  counts=collections.Counter(self.game_seen.values())
  for c,n in counts.items():self.lower[c]=max(self.lower.get(c,0),n)
 def emit(self,e,out):
  if e['op']=='move':
   src,dst=e['src'],e['dst'];sid=e.get('sid')
   if sid in self.known:
    loc=self.known[sid][1]
    if loc=='R' and e.get('resolving_source'):e=dict(e,src='R');src='R'
    elif loc==dst and src!=dst:
     # Native ATTACH/EVOLVE can repeat a preceding explicit MOVE, with the
     # same physical serial. This is a location assertion, not a second card.
     out.append(dict(op='known',zone=dst,cid=e['cid'],sid=sid));return
   if src in self.sizes:self.sizes[src]-=1
   if dst in self.sizes:self.sizes[dst]+=1
   if self.sizes['L']==0:self.metang_looking=False
   if min(self.sizes.values())<0:raise InvariantFailure('Native compiler negative source size '+str(e))
   if e.get('cid') is not None:self.note(e['cid'],sid,dst)
   else:
    # A concealed choice can move a previously known copy. Keep only facts
    # true for every permitted choice, rather than guessing its identity.
    for s,(c,z) in list(self.known.items()):
     if z==src:self.known[s]=(c,src+'|'+dst)
  elif e['op']=='return_all':
   src=e['src'];self.sizes['D']+=self.sizes[src];self.sizes[src]=0
   for s,(c,z) in list(self.known.items()):
    if z==src:self.known[s]=(c,'D')
  out.append(e)
 def compile(self,raw):
  o=legal(raw);cur=o['current'];g=cur.get('round',1);new=g!=self.game
  if new:self.reset_game(g)
  sig=signature(o)
  if sig in self.seen_obs:return dict(new_game=False,duplicate=True,operations=[],legal=o)
  self.seen_obs.add(sig);self.observations+=1;opponent=1-cur['yourIndex'];logs=o.get('logs') or [];out=[];i=0
  while i<len(logs):
   e=logs[i];i+=1;t=e['type'];owner=e.get('playerIndex');self.coverage['log_type_'+str(t)]+=1
   if t in (10,15) and e.get('cardId') is not None:self.effect=e['cardId']
   if t==2:self.begun=True
   if owner!=opponent:continue
   cid=e.get('cardId');sid=e.get('serial')
   if t==0:self.crispin_pending=False;self.metang_looking=False;self.deck_evolve_pending=False;self.emit(dict(op='shuffle'),out)
   elif t==15:
    self.deck_evolve_pending=cid==344 and e.get('attackId')==478
   elif t==1:self.emit(dict(op='basic',basic=self.basic,value=e['hasBasicPokemon']),out)
   elif t in (4,5):self.emit(dict(op='move',src='D',dst='H',kind='top',**(dict(cid=cid,sid=sid) if t==4 else {})),out)
   elif t in (6,7):
    src,dst=zone(e['fromArea'],True),zone(e['toArea']);revealed=t==6;kw=dict(cid=cid,sid=sid) if revealed else {};kind='available';position=destination_position(e['toArea']);text=self.text(self.effect)
    if not revealed and src=='H' and dst=='D' and e['toArea']==1:
     count=1
     while i<len(logs) and logs[i].get('type')==7 and logs[i].get('playerIndex')==opponent and logs[i].get('fromArea')==2 and logs[i].get('toArea')==1:count+=1;i+=1
     if count==self.sizes['H']:
      self.emit(dict(op='return_all',src='H'),out);self.coverage['complete_hand_return']+=1;continue
     # A partial strategically chosen concealed return has no universal
     # chance kernel. Recognize only explicit top/bottom effect wording.
     if 'bottom' in text:position='bottom'
     elif 'top of' in text:position='top'
     else:raise UnsupportedEvent('Partial concealed hand-to-deck return lacks selection/order semantics: '+str(dict(effect=self.effect,text=text,count=count,hand=self.sizes['H'])))
     for _ in range(count):self.emit(dict(op='move',src=src,dst=dst,kind='available',position=position),out)
     continue
    if src=='D' and dst=='P' and not self.begun:kind='top'
    elif src=='D' and dst=='L':
     n=1;j=i
     while j<len(logs) and logs[j].get('type') in (6,7) and logs[j].get('playerIndex')==opponent and logs[j].get('fromArea')==1 and logs[j].get('toArea')==12:n+=1;j+=1
     # These three native CardImpl chains use LookDeck (top), not
     # LookDeckBottom. Match the complete block, avoiding a stale PLAY.
     profiles={86:4,120:2,1094:7,1122:7};known={c['id'] for c in visible(o,opponent)}|{x.get('cardId') for x in logs[:i] if x.get('playerIndex')==opponent}
     if n==4 and 86 in known:self.metang_looking=True
     if self.look_remaining:kind='top';self.look_remaining-=1
     elif any(c in known and min(count,self.sizes['D'])==n for c,count in profiles.items()):kind='top';self.look_remaining=n-1
     elif ('look at the top' in text or 'reveal the top' in text) and str(n) in text:kind='top'
     else:raise UnsupportedEvent('Deck-to-looking effect not resolved by verified profile: '+str(dict(last_effect=self.effect,block=n)))
    elif src=='L' and dst=='D':
     # Destination semantics come from native pushCardRef: Deck is top;
     # DeckBottom is bottom. Hidden all-looking returns use an explicit
     # uniform latent permutation (native Close shuffles the target list).
     position=destination_position(e['toArea']);kind='available'
    elif src=='P':kind='available' # selected prize index: uniform latent index, no action likelihood
    elif src=='D' and not revealed:
     known={c['id'] for c in visible(o,opponent)}|{x.get('cardId') for x in logs[:i] if x.get('playerIndex')==opponent}
     if dst=='H' and any(self.profiles.get(str(c),{}).get('unrevealed_deck_selection') for c in known) and any(x.get('type')==0 and x.get('playerIndex')==opponent for x in logs[i:]):kind='available'
     else:raise UnsupportedEvent('Concealed selection not covered by verified native effect profile '+str(self.effect))
    elif dst=='D':position=destination_position(e['toArea'])
    if not revealed and src=='H' and dst in ('A','B') and not self.begun:kw['basic_only']=self.basic
    self.emit(dict(op='move',src=src,dst=dst,kind=kind,position=position,resolving_source=True,**kw),out)
   elif t==10:
    self.crispin_pending=cid==1198
    self.emit(dict(op='move',src='H',dst='R',kind='available',cid=cid,sid=sid),out)
   elif t in (11,12):
    src=('D' if t==11 and self.crispin_pending or t==12 and self.deck_evolve_pending else 'L' if t==11 and self.metang_looking and self.sizes['L']>0 else self.known.get(sid,(None,'H'))[1])
    if t==12:self.deck_evolve_pending=False
    if t==11 and self.crispin_pending:self.crispin_pending=False
    if '|' in src:src='H'
    self.emit(dict(op='move',src=src,dst='B',kind='available',cid=cid,sid=sid),out)
   elif t==9:
    ops=replacement(e,self.known,self.effect)
    for op in ops:self.emit(op,out)
   elif t==13:
    # DEVOLVE is followed by explicit MOVE events; target identities are
    # public assertions resolved by the final snapshot, not extra removals.
    self.coverage['devolve_public_assertion']+=1
   elif t in (2,3,8,14,15,16,17,18,19,20,21,22,23):pass
   else:raise UnsupportedEvent('Unrecognized native LogType '+str(t))
  p=cur['players'][opponent];sizes=dict(D=p['deckCount'],H=p['handCount'],P=len(p.get('prize') or []))
  # Looking belongs to the current selecting player. Opponent looking can
  # remain hidden between actor callbacks; only compare its explicit size
  # when native legal current actually provides that area's contents.
  looking=cur.get('looking')
  if looking is not None and all(x is None or x.get('playerIndex')==opponent for x in looking):sizes['L']=len(looking)
  for z,n in sizes.items():
   if self.sizes[z]!=n:raise InvariantFailure('Native log/snapshot count mismatch '+str(dict(game=g,zone=z,compiled=self.sizes[z],observed=n,effect=self.effect,logs=logs)))
  public=visible(o,opponent)
  for c in public:self.note(c['id'],c['serial'],c['zone'])
  deck_known=(o.get('select') or {}).get('deck') or []
  deck_known=[c for c in deck_known if c.get('playerIndex')==opponent]
  for c in deck_known:self.note(c['id'],c['serial'],'D')
  snap=dict(op='snapshot',sizes=sizes,public=public,prizes=p.get('prize') or [],deck_known=deck_known,looking=looking if 'L' in sizes else [])
  for c in snap['prizes']:
   if c:self.note(c['id'],c['serial'],'P')
  self.emit(snap,out);self.events.extend(out)
  return dict(new_game=new,duplicate=False,operations=out,legal=o,game=g,signature=sig)
 def known_hand(self):return collections.Counter(c for c,z in self.known.values() if z=='H')
 def compatible(self,composition):return not (collections.Counter(self.lower)-collections.Counter(composition))

