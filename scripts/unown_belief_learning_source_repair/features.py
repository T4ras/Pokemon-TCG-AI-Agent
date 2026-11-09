"""Legal-only features; targets are joined by a separate caller afterwards."""
from .common import *
from .distributions import allocation,moments
from unown_belief_bayes_repair.adapter import legal,signature
from .baseline import Facts,hard_layout
from .allocation import allocation_one
from unown_belief_bayes_repair.deck import catalogue,deck_weights,mix_allocations
from unown_belief_bayes_repair.cache import bounded
import numpy as np

PUBLIC_DIM=20
HASH_DIM=16
GLOBAL_DIM=16*6+20+37
EVENT_DIM=16*6+20+24

def sketch(counts):
 out=np.zeros(16)
 for c,n in counts.items():
  h=stable('card-sketch-v1',int(c));out[int(h[:4],16)%16]+=(1 if int(h[4],16)%2 else -1)*n/60.
 return out

def public(o):
 c=o['current'];me=c['yourIndex'];op=1-me;p=c['players'];v=[]
 for who in (me,op):
  a=p[who];v.extend([a['handCount']/60,a['deckCount']/60,len(a.get('prize') or [])/6,sum(x is not None for x in a.get('active') or []),sum(x is not None for x in a.get('bench') or [])/8])
 first=c['firstPlayer'];v.extend([min(c['turn'],200)/200,min(c['turnActionCount'],2000)/2000,float(first==me),float(first==op)])
 v.extend(float(c.get(k,False)) for k in ('supporterPlayed','stadiumPlayed','energyAttached','retreated'))
 sel=o.get('select') or {};v.extend([float(sel.get('maxCount',0)>sel.get('minCount',0)),min(len(sel.get('option') or []),512)/512])
 return np.asarray(v,dtype=np.float32)

def tag(card_index):return np.array([(card_index>>b)&1 for b in range(8)],dtype=np.float32)

def support(facts,counts,indices,queries,known):
 upper=np.zeros((len(queries),3),int);lower=np.zeros_like(upper);fixed=hard_layout(facts)[0]
 for i,c in enumerate(queries):
  for z,zone in enumerate(('H','D','P')):
   outside=sum(cid==c and zone not in domain for cid,domain in facts.copies.values());hi=int(counts[indices,i].max())-outside
   lo=0 if zone=='H' else fixed[zone][c]
   if zone=='H':hi-=known[c]
   cap=facts.sizes[zone]-(known[c] if zone=='H' else 0);upper[i,z]=max(lo,min(cap,hi));lower[i,z]=lo
 return lower,upper

def encode(hist,selection,keep_conditionals=False):
 """No scheduled deck, hidden target, outcomes or match-score arguments."""
 started=time.process_time();decks,queries,counts,pop=catalogue();S=int(counts.max())+1;f=Facts(metadata()['cards']);q=queries;events=[];reset=[];selected=[];points=[];globals_=[];locals_=[];A=[];lower=[];upper=[];deckp=[];ledgers=[];knowns=[];conditionals=[];condindices=[];condoffsets=[0];games=[];zone_sizes=[];reference_A=[];uncertain=[];card_low=[];card_high=[];within={};game=None;latency=[]
 for index,entry in enumerate(hist):
  check_time();t=time.process_time();o=legal(entry['observation']);g=o['current']['round'];new=g!=game
  if new:within=collections.Counter();game=g
  changed=bounded(lambda:f.observe(o),30)
  if changed:
   for e in o.get('logs') or []:
    if e.get('cardId') is not None:within[e['cardId']]+=1
  zone=[collections.Counter(c for c,d in f.copies.values() if zone in d and len(d)==1) for zone in ('H','D','P','B')]
  hashes=np.concatenate([sketch(f.lower)]+[sketch(z) for z in zone]+[sketch(within)])
  types=np.zeros(24)
  if changed:
   for e in o.get('logs') or []:
    n=e.get('type',-1)
    if 0<=n<24:types[n]+=1/60.
  events.append(np.concatenate((hashes,public(o),types)).astype(np.float32));reset.append(new)
  if index not in selection and str(index) not in selection:
   latency.append(time.process_time()-t);continue
  known=f.known_hand();supported=[];pmfs=[];references=[];cache={}
  def readout():
   for d in range(len(decks)):
    if not f.compatible(prior()[decks[d]]['cards']):continue
    try:
     value=allocation(f,prior()[decks[d]]['cards'],q,S,cache)
     reference=allocation_one(f,prior()[decks[d]]['cards'],q)
    except IncompatibleAllocation:continue
    actual=moments(value);expected=np.array([reference[c] for c in q]);np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    supported.append(d);pmfs.append(value);references.append(expected)
  bounded(readout,10)
  if not supported:raise OutOfLibrary('No compatible per-deck allocation')
  w=deck_weights([f.lower.get(c,0) for c in q],np.isin(np.arange(37),supported),pop,beta=0)
  a=np.einsum('d,dqzs->qzs',w[supported],np.asarray(pmfs));am=mix_allocations(w[supported],np.array(references));lo,hi=support(f,counts,np.asarray(supported),q,known)
  k=np.arange(S);outside=(k[None,None,:]<lo[:,:,None])|(k[None,None,:]>hi[:,:,None])
  if np.any(a[outside]>1e-6):raise InvariantFailure('Baseline mass outside legal count bounds')
  a[outside]=0.;a/=a.sum(-1,keepdims=True)
  local=np.column_stack((np.array([tag(i) for i in range(len(q))]),am/np.array([1,60,60,60,3600,3600,3600]),np.array([known[c] for c in q])/60,lo/60,hi/60,counts.max(0)/60,a.reshape((len(q),-1))))
  fixed=hard_layout(f)[0];card_low.append([known[c]+fixed['D'][c]+fixed['P'][c] for c in q]);card_high.append([int(counts[supported,i].max())-sum(cid==c and not domain.intersection({'H','D','P'}) for cid,domain in f.copies.values()) for i,c in enumerate(q)]);reference_A.append(am);uncertain.append((am[:,0]>=.1)&(am[:,0]<=.9))
  globals_.append(np.concatenate((hashes,public(o),w)));locals_.append(local);A.append(a);lower.append(lo);upper.append(hi);deckp.append(w);ledgers.append([f.lower.get(c,0) for c in q]);knowns.append([known[c] for c in q]);selected.append(index);games.append(g);zone_sizes.append([f.sizes[z] for z in ('H','D','P')]);points.append(dict(index=index,key=[g,o['current']['turn'],o['current']['turnActionCount'],o['current']['yourIndex']],game=g,stage='setup' if o['current']['turn']==0 else 'early' if o['current']['turn']<=4 else 'middle' if o['current']['turn']<=12 else 'late',known_serials=sorted(s for s,(c,d) in f.copies.items() if d=={'H'})))
  if keep_conditionals:conditionals.extend(pmfs);condindices.extend(supported)
  condoffsets.append(len(conditionals));latency.append(time.process_time()-t)
 return dict(sequence=np.array(events,dtype=np.float32),reset=np.array(reset,bool),selected=np.array(selected,int),global_X=np.array(globals_,np.float32),local_X=np.array(locals_,np.float32),A=np.array(A,np.float32),lower=np.array(lower,np.uint8),upper=np.array(upper,np.uint8),deck_A=np.array(deckp,np.float32),ledger=np.array(ledgers,np.uint8),known=np.array(knowns,np.uint8),games=np.array(games,np.uint8),zone_sizes=np.array(zone_sizes,np.uint8),A_reference=np.array(reference_A,np.float64),uncertain=np.array(uncertain,bool),card_total_lower=np.array(card_low,np.uint8),card_total_upper=np.array(card_high,np.uint8),conditionals=np.array(conditionals,np.float32),conditional_indices=np.array(condindices,np.uint8),conditional_offsets=np.array(condoffsets,int),final_ledger=np.array([f.lower.get(c,0) for c in q],np.uint8)),dict(points=points,CPU_seconds=time.process_time()-started,latency_CPU_seconds=latency,all_callbacks=len(hist),full_prefix=True)

def target_arrays(targets,features,points,opponent_list):
 """Truth enters only here, after encode has produced every legal feature."""
 decks,q,counts,pop=catalogue();bykey={tuple(t['key']):t for t in targets};out=[];y=decks.index(opponent_list)
 for j,p in enumerate(points):
  t=bykey[tuple(p['key'])]
  if not set(p['known_serials'])<=set(t['hand_serials']):raise InvariantFailure('Known serial truth mismatch')
  row=np.array([[t[z].get(str(c),t[z].get(c,0)) for z in ('hand','deck','prize')] for c in q]);row[:,0]-=features['known'][j]
  if np.any(row<features['lower'][j]) or np.any(row>features['upper'][j]):raise InvariantFailure('Diagnostic target outside legal-only bounds')
  if features['deck_A'][j,y]<=0:raise InvariantFailure('True list excluded by allocation')
  out.append(row)
 return dict(counts=np.array(out,np.uint8),deck=np.full(len(out),y,np.uint8))
