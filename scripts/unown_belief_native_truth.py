"""Diagnostic-only truth joins. This module is never imported by the observer."""
from unown_belief_native_common import *
from unown_belief_native_adapter import legal

def meaningful(o):
 s=o.get('select')
 if not s or s['maxCount']==0:return False
 n=len(s['option']);return not(s['minCount']==s['maxCount']==n or n==1 and s['minCount']>=1)

def select_points(match_id,entries):
 groups=collections.defaultdict(list)
 for i,e in enumerate(entries):
  o=e['observation']
  if meaningful(o):
   g=o['current']['round'];j=len(groups[g]);key=hashlib.sha256(f'{match_id}:{g}:{j}'.encode()).hexdigest();groups[g].append((key,i,j))
 return {i:dict(game=g,decision_index=j,priority=key) for g,values in groups.items() for key,i,j in sorted(values)[:24]}

def key(cur):return (cur.get('round',1),cur['turn'],cur['turnActionCount'],cur['yourIndex'])
def counts(seq):return collections.Counter(c['id'] for c in seq or [] if c)
def copies(seq):return sorted((c['id'],c['serial']) for c in seq or [] if c)

def align(entries,frames,seat):
 """Join by the native state BEFORE the action, not frame.obs.

 The pinned wrapper attaches env.steps[i].observation to native frame i,
 although the latter describes the following selection state. We validate
 the actual current-state key, both public counts, and own physical hand.
 """
 index=collections.defaultdict(list)
 game=1
 for i,f in enumerate(frames):
  c=f.get('current')
  if c:
   if c.get('result',-1)>=0:game+=1;continue
   c=dict(c,round=game);index[key(c)].append((i,c))
 out=[]
 for e in entries:
  o=e['observation'];cur=o.get('current')
  if not cur or not o.get('select'):continue
  candidates=[]
  for i,t in index.get(key(cur),[]):
   if all((p['handCount'],p['deckCount'],len(p.get('prize') or []))==(q['handCount'],q['deckCount'],len(q.get('prize') or [])) for p,q in zip(cur['players'],t['players'])) and copies(cur['players'][seat]['hand'])==copies(t['players'][seat]['hand']):candidates.append((i,t))
  if len(candidates)!=1:raise InvariantFailure('Diagnostic alignment is missing/ambiguous '+str(dict(key=key(cur),candidates=len(candidates))))
  i,t=candidates[0];opp=t['players'][1-seat]
  for zone,number in (('hand',opp['handCount']),('deck',opp['deckCount']),('prize',len(opp.get('prize') or []))):
   if len(opp.get(zone) or [])!=number or any(c is None for c in opp.get(zone) or []):raise InvariantFailure('Native diagnostic hidden zone not fully exposed: '+zone)
  out.append(dict(key=list(key(cur)),frame_index=i,hand=dict(counts(opp['hand'])),deck=dict(counts(opp['deck'])),prize=dict(counts(opp['prize'])),hand_serials=[c['serial'] for c in opp['hand']]))
 return out


