"""Both players use the expanded current-pool adapter; blind inference only."""
import copy,importlib.util,sys,time,resource
from collections import Counter
import numpy as np
import torch
from unown_pilot_common import *
import unown_v2_runtime as rt
from unown_v2_training import state_hash,completed_game_returns
rt.OUT=OUT;rt.PRIVATE=PRIVATE


def load_model(path=None,device='cpu'):
 net=rt.make_model(text_path=PRIVATE/'experimental-text-tables.npz',technical_probe=False)
 if path:net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
 if hasattr(net,'_frozen_tables'):del net._frozen_tables
 return net.to(device).eval()


def freeze_inference(model):
 model.eval()
 if hasattr(model,'_frozen_tables'):del model._frozen_tables
 model.freeze_tables()


def physical_exposure(obs):
 cur=obs['current'];seat=cur['yourIndex'];out={'own':set(),'opponent':set(),'action_cards':set(),'action_attacks':set()}
 for who,p in enumerate(cur['players']):
  dest=out['own' if who==seat else 'opponent'];cards=[]
  for pk in p['active']+p['bench']:
   if pk:cards += [pk]+pk.get('energyCards',[])+pk.get('tools',[])+pk.get('preEvolution',[])
  cards += [c for c in p['discard']+p['prize']+(p.get('hand') or []) if c]
  dest.update(c['id'] for c in cards if c.get('id'))
 for c in (cur.get('stadium') or [])+(cur.get('looking') or [])+((obs.get('select') or {}).get('deck') or []):
  if c and c.get('id'):out['own' if c.get('playerIndex')==seat else 'opponent'].add(c['id'])
 for o in obs['select']['option']:
  if o.get('cardId'):out['action_cards'].add(o['cardId'])
  if o.get('attackId'):out['action_attacks'].add(o['attackId'])
 return out


class PilotActor(rt.Actor):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.exposure={k:Counter() for k in ['own','opponent','action_cards','action_attacks']}
  self.maximum_options=0;self.maximum_tokens=0;self.reset_count=0
 def act(self,raw):
  if raw.get('select') is None:self.reset_match();return list(self.deck)
  obs=rt.actor_view(raw);game=obs['current'].get('round',1)
  if game!=self.last_game:self.reset_count+=1
  self.memory.update(obs)
  if self.last_game!=game:self.forced_run=0;self.last_game=game
  for k,ids in physical_exposure(obs).items():self.exposure[k].update(ids)
  self.calls+=1;sel=obs['select'];n=len(sel['option']);self.maximum_options=max(self.maximum_options,n)
  if sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1:
   self.forced_run+=1;self.forced+=1;return list(range(n))
  picked=[]
  while True:
   stop=len(picked)>=sel['minCount'] and (sel['maxCount']>1 or sel['minCount']==0)
   packet=rt.encode(obs,picked,stop,self.forced_run,self.memory.encoder_tracker())
   ti=packet.row[slice(*packet.layout.offsets['tok_int'])].reshape(packet.layout.tokens,3+packet.layout.tools)
   self.maximum_tokens=max(self.maximum_tokens,int((ti[:,1]>0).sum()))
   with torch.no_grad():
    c,_=self.model.core(packet);logits=c['logits'][0];v=self.model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze()
    if not torch.isfinite(logits).all() or not torch.isfinite(v):raise FloatingPointError('Nonfinite inference logits/value')
    prob=torch.softmax(logits,-1).cpu();lp=torch.log_softmax(logits,-1).cpu()
    action=int(torch.multinomial(prob,1,generator=self.generator)) if self.sample else int(logits.argmax())
   if self.collect:self.records.append(dict(packet=packet,action=action,log_probability=float(lp[action]),value=float(v),
    old_probs=prob.numpy().astype(np.float32),game=game,seat=obs['current']['yourIndex']))
   if action==packet.layout.stop:
    if not stop:raise AssertionError('Illegal STOP')
    break
   if not 0<=action<n or action in picked:raise AssertionError('Invalid/repeated action')
   picked.append(action)
   if len(picked)>=sel['maxCount']:break
  self.forced_run=0
  if not sel['minCount']<=len(picked)<=sel['maxCount']:raise AssertionError('Selection count mismatch')
  return picked


def environment():
 from kaggle_environments import make,register
 name='unown_pilot_pinned_playground'
 if name not in sys.modules:
  folder=PARENT/'current-environment';spec=importlib.util.spec_from_file_location(name,folder/'cabt.py',submodule_search_locations=[str(folder)])
  module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  register(name,dict(interpreter=module.interpreter,renderer=module.renderer,html_renderer=module.html_renderer,specification=module.specification,agents=module.agents))
 return make(name,configuration=dict(bo=3,version=2,runTimeout=300),debug=True),sys.modules[name]


def play(job,learner,opponent,collect=False,limit=None):
 """One exclusive native attempt; no retries and no unsupported original adapter."""
 def validate_action(select,answer):
  if not isinstance(answer,list) or len(answer)!=len(set(answer)) or not select['minCount']<=len(answer)<=select['maxCount'] or any(not isinstance(i,int) or not 0<=i<len(select['option']) for i in answer):raise ValueError('Illegal conditional action')
 local_deadline() if limit is None else None
 stage=job['stage'];folder=OUT/stage;exclusive(folder/'intents'/(job['id']+'.json'),dict(job,at_utc=iso()))
 manifest=read(OUT/'manifest.json');decks=manifest['decks'];seat=job['learner_seat']
 a=PilotActor(learner,decks[job['learner_deck']]['cards'],sample=collect,seed=job['seed'],collect=collect)
 b=PilotActor(opponent,decks[job['opponent_deck']]['cards'],sample=False,seed=job['seed']+100000,collect=False)
 before=state_hash(learner);opp_before=state_hash(opponent);start=time.perf_counter();metadata={};errors=[];last=None
 times=[0.,0.];env,module=environment()
 def measured(actor,who):
  def call(obs,cfg):
   nonlocal last
   if limit and time.monotonic()>=limit:raise TimeoutError('Pilot active execution budget')
   if time.perf_counter()-start>300:raise TimeoutError('Per-BO3 compatibility/runtime cap')
   t=time.perf_counter();obs=copy.deepcopy(dict(obs));last=obs
   try:
    answer=actor.act(obs)
    if obs.get('select'):
     validate_action(obs['select'],answer);cur=obs['current'];rid=cur.get('round',1)
     g=metadata.setdefault(rid,dict(round=rid,chooser=None,first_player=-1))
     if cur['firstPlayer']>=0:g['first_player']=cur['firstPlayer']
     if obs['select']['context']==41:g.update(chooser=who,chooser_chose_first=obs['select']['option'][answer[0]]['type']==1)
    return answer
   except Exception as e:errors.append(dict(actor=who,type=type(e).__name__,error=str(e)));raise
   finally:times[who]+=time.perf_counter()-t
  return call
 agents=[None,None];agents[seat]=measured(a,seat);agents[1-seat]=measured(b,1-seat)
 status='FAILED';reason=None;games=[];winner=None
 try:
  env.run(agents)
  for frame in module.Battle.vis:
   cur=frame.get('current') or {}
   if cur.get('result',-1)>=0:
    rid=len(games)+1;games.append(dict(metadata.get(rid,dict(round=rid)),result=cur['result'],finish_reason=cur.get('finishReason','native_terminal_result'),completion_status='DONE'))
  assert [x.status for x in env.state]==['DONE','DONE'] and not errors
  assert len(games) in [2,3] and len(games)==sum(env.result)
  assert [sum(g['result']==i for g in games) for i in [0,1,2]]==list(env.result)
  assert games[0]['chooser']==0 and all(g['first_player'] in [0,1] for g in games)
  assert a.reset_count==len(games) and b.reset_count==len(games)
  assert before==state_hash(learner) and opp_before==state_hash(opponent)
  winner=0 if env.result[0]>env.result[1] else 1 if env.result[1]>env.result[0] else 2
  status='DONE'
 except Exception as e:reason=f'{type(e).__name__}: {e}'
 row=dict(job,status=status,error=reason,result=winner,games=games,actor_errors=errors,
  started_at_utc=(now()-dt.timedelta(seconds=time.perf_counter()-start)).isoformat(),ended_at_utc=iso(),
  wall_seconds=time.perf_counter()-start,learner_inference_seconds=times[seat],opponent_inference_seconds=times[1-seat],
  meaningful_decisions=len(a.records),forced_observations=a.forced,learner_hash=before,opponent_hash=opp_before,
  exposure={who:{k:dict(c) for k,c in actor.exposure.items()} for who,actor in [('learner',a),('opponent',b)]},
  maxima=dict(learner_options=a.maximum_options,learner_tokens=a.maximum_tokens,opponent_options=b.maximum_options,opponent_tokens=b.maximum_tokens),
  peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 append(folder/'matches.jsonl',row)
 if status!='DONE':
  import gzip
  with gzip.open(PRIVATE/(job['id']+'-failure.json.gz'),'wt') as f:json.dump(dict(record=row,last_actor_observation=last),f)
 del env;module.Battle.vis=[]
 return row,completed_game_returns(a.records,games,seat) if status=='DONE' and collect else []
