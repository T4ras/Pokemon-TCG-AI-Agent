"""Exact frozen inference in a pinned official BO3, compact per-game evidence."""
from unown_belief_native_common import *
from unown_belief_native_corpus import finish
import sys,copy,gzip,resource,importlib.util,math,traceback
class ProtocolViolation(RuntimeError):pass
MODEL=ACTOR=MODULE=ENVIRONMENT=CLOCK=TORCH=None

def state_hash():
 h=hashlib.sha256()
 for k,v in sorted(MODEL.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def initialize():
 global MODEL,ACTOR,MODULE,ENVIRONMENT,CLOCK,TORCH
 os.environ['CUDA_VISIBLE_DEVICES']='';import torch;TORCH=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
 policy=POLICY;sys.path.insert(0,str(policy));from frozen_unown.runtime import load
 template,MODEL,_=load(policy);ACTOR=type(template);assert state_hash()==TENSOR
 # Reuse the previously verified awake-clock helper; observing power never
 # changes the power plan or inhibits laptop sleep.
 import types
 clock=types.ModuleType('deck_screen_awake_clock');clock.__dict__.update(dict(OUT=OUT,read=read,write=write,append=append,iso=iso,now=now,parse=lambda s:dt.datetime.fromisoformat(s),time=time,os=os));clock.__file__=str(R/'scripts/unown_runner/units/clock.py')
 exec(compile(Path(clock.__file__).read_text(),clock.__file__,'exec'),clock.__dict__);CLOCK=clock.AwakeClock()
 import kaggle_environments.core as kgcore,kaggle_environments.agent as kgagent
 kgcore.perf_counter=CLOCK;kgagent.perf_counter=CLOCK
 from kaggle_environments import make,register
 folder=R/'data/private/astra-unown-new-card-embedding-continuation-001/assets/environment'
 spec=importlib.util.spec_from_file_location('deck_screen_official',folder/'cabt.py',submodule_search_locations=[str(folder)]);MODULE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MODULE;spec.loader.exec_module(MODULE)
 register('deck_screen_official',dict(interpreter=MODULE.interpreter,renderer=MODULE.renderer,html_renderer=MODULE.html_renderer,specification=MODULE.specification,agents=MODULE.agents))
 ENVIRONMENT=lambda:make('deck_screen_official',configuration=dict(bo=3,version=2,runTimeout=6000),debug=True)
 original=MODEL.core
 def checked(packet):
  import numpy as np
  if not np.isfinite(packet.row).all():raise ProtocolViolation('Nonfinite encoded observation')
  a,b=packet.layout.offsets['oracle']
  if packet.row[a:b].any():raise ProtocolViolation('Privileged oracle features entered actor inference')
  out=original(packet);logits=out[0]['logits'][0];a,b=packet.layout.offsets['opt_mask'];mask=packet.row[a:b]
  if not TORCH.isfinite(logits).all() or not mask.any() or mask[int(logits.argmax())]<=0:raise ProtocolViolation('Illegal/nonfinite conditional argmax or empty mask')
  return out
 MODEL.core=checked
 write(OUT/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),at_utc=iso(),tensor=TENSOR,startup='Exact copied submission inference, CPU float32, argmax',peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def one(job):
 start=time.monotonic();begun=now();awake=CLOCK();cpu=time.process_time();decks=read(OUT/'decks.json');seat=job['seat'];errors=[];games=[];metadata={};scanned=0;last_observations=collections.deque(maxlen=8);hist=[[],[]];belief_hist=[]
 actors=[None,None];actors[seat]=ACTOR(MODEL,decks[job['a']]['cards'],sample=False,collect=False,seed=job['seed']);actors[1-seat]=ACTOR(MODEL,decks[job['b']]['cards'],sample=False,collect=False,seed=job['seed']+1)
 assert actors[0].memory is not actors[1].memory and actors[0].order is not actors[1].order
 for a in actors:
  if a.sample or a.collect:raise ProtocolViolation('Decoder/collection mode mismatch')
 exclusive(OUT/'intents'/(job['id']+'.json'),dict(at_utc=iso(),pid=os.getpid(),job=job))
 write(OUT/'inflight'/f'{os.getpid()}.json',dict(at_utc=iso(),job=job))
 env=ENVIRONMENT();before=state_hash();assert before==TENSOR
 last_progress=[-1.,-1]
 def harvest():
  nonlocal scanned
  for frame in MODULE.Battle.vis[scanned:]:
   cur=frame.get('current') or {}
   if cur.get('result',-1)>=0:
    rid=len(games)+1;games.append(dict(metadata.get(rid,dict(game=rid)),result=cur['result'],finish_reason=cur.get('finishReason','native_terminal_result'),completion_status='DONE'))
  scanned=len(MODULE.Battle.vis)
  if len(games)!=last_progress[1] or time.monotonic()-last_progress[0]>2:
   write(OUT/'inflight'/f'{os.getpid()}.json',dict(at_utc=iso(),job=job,games=games));last_progress[:]=[time.monotonic(),len(games)]
 inference=[0.,0.];actual_resets=[[],[]];previous_game=[None,None];old_lower=[{},{}]
 def agent(who):
  a=actors[who]
  def call(raw,cfg):
   harvest();t=time.process_time()
   try:
    if now()>=deadline():raise TimeoutError('UTC experiment deadline')
    obs=copy.deepcopy(dict(raw));last_observations.append(dict(seat=who,observation={k:v for k,v in obs.items() if k in ('current','select','logs')}))
    if obs.get('select'):
     rid=obs['current'].get('round',1);tracker=a.memory.tracker;new=rid!=previous_game[who]
    answer=a.act(obs)
    if who==seat and obs.get('select'):belief_hist.append(dict(observation=copy.deepcopy(obs),action=answer))
    if obs.get('select'):
     sel=obs['select'];n=len(sel['option'])
     if not isinstance(answer,list) or len(answer)!=len(set(answer)) or not sel['minCount']<=len(answer)<=sel['maxCount'] or any(type(i) is not int or not 0<=i<n for i in answer):raise ProtocolViolation('Illegal selected action')
     if new:
      if tracker is not None and tracker is a.memory.tracker:raise ProtocolViolation('Per-game tracker failed to reset')
      actual_resets[who].append(rid)
     if any(a.memory.lower_bounds[k]<v for k,v in old_lower[who].items()):raise ProtocolViolation('Legal deck lower bounds lost within BO3')
     old_lower[who]=dict(a.memory.lower_bounds);previous_game[who]=rid
     cur=obs['current'];g=metadata.setdefault(rid,dict(game=rid,chooser=None,first_player=-1))
     if cur['firstPlayer']>=0:g['first_player']=cur['firstPlayer']
     if sel['context']==41:
      g.update(chooser=who,chooser_chose_first=sel['option'][answer[0]]['type']==1)
      if rid==1 and who!=0:raise ProtocolViolation('Initial chooser is not pinned physical seat0')
     if job.get('startup_check'):hist[who].append(dict(observation=obs,action=answer))
    harvest();return answer
   except BaseException as e:errors.append(dict(seat=who,error_type=type(e).__name__,message=str(e)[:300]));raise
   finally:inference[who]+=time.process_time()-t
  return call
 status='FAILED';winner=None;failure_kind=None;reason=None
 try:
  env.run([agent(0),agent(1)]);harvest()
  if errors:
   if all(x['error_type'] in ('TimeoutError',) for x in errors):raise TimeoutError('Timing limit during native BO3')
   raise ProtocolViolation('Actor error: '+str(errors[0]))
  statuses=[s.status for s in env.state]
  if 'TIMEOUT' in statuses:raise TimeoutError('Official agent/runtime timing limit')
  if statuses!=['DONE','DONE']:raise ProtocolViolation('Official execution status '+str(statuses))
  if len(games) not in (2,3) or len(games)!=sum(env.result):raise ProtocolViolation('Terminal game/match count mismatch')
  if [sum(g['result']==i for g in games) for i in (0,1,2)]!=list(env.result):raise ProtocolViolation('Terminal games do not map to match result')
  if games[0].get('chooser')!=0 or any(g.get('first_player') not in (0,1) or g.get('chooser') not in (0,1) for g in games):raise ProtocolViolation('Missing/mismatched chooser/starter evidence')
  if any(len(x)!=len(games) for x in actual_resets):raise ProtocolViolation('Game tracker reset count mismatch')
  if state_hash()!=before:raise ProtocolViolation('Frozen policy weights changed')
  winner=0 if env.result[0]>env.result[1] else 1 if env.result[1]>env.result[0] else 2
  # Only allocated startup matches replay their legal histories, no extra native
  # diagnostics. Independent actor objects must reproduce exact argmax actions.
  if job.get('startup_check'):
   for who in (0,1):
    replay=ACTOR(MODEL,actors[who].deck,sample=False,collect=False,seed=job['seed']+who)
    for e in hist[who]:
     if replay.act(e['observation'])!=e['action']:raise ProtocolViolation('Startup faithful-prefix action parity mismatch')
    if dict(replay.memory.lower_bounds)!=dict(actors[who].memory.lower_bounds):raise ProtocolViolation('Startup match knowledge replay mismatch')
    path=PRIVATE/'startup-histories'/f'{job["id"]}-{who}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(hist[who],f)
  status='DONE'
 except BaseException as e:
  harvest();reason=type(e).__name__+': '+str(e);failure_kind='TIMING' if isinstance(e,TimeoutError) else 'CORRECTNESS' if isinstance(e,ProtocolViolation) else 'EXECUTION'
 priority=int(hashlib.sha256(f'{SEED}:reservoir:{job["id"]}'.encode()).hexdigest(),16)/2**256
 detail=None;detail_error=None
 try:
  retention=read(OUT/'retention.json') if (OUT/'retention.json').exists() else dict(priority_cutoff=1.,failure_counts={})
  signature=(errors[0]['error_type']+':'+errors[0]['message'][:100]) if errors else reason or 'success'
  wanted=(status=='DONE' and priority<retention.get('priority_cutoff',1.)) or (status!='DONE' and retention.get('failure_counts',{}).get(signature,0)<10)
  if wanted:
   target=PRIVATE/'detail-temp'/(job['id']+'.json.gz');target.parent.mkdir(exist_ok=True)
   data=dict(job=job,games=games,environment=env.toJSON()) if status=='DONE' else dict(job=job,games=games,errors=errors,last_actor_observations=list(last_observations))
   with gzip.open(target,'wt',compresslevel=3) as f:
    for chunk in json.JSONEncoder(default=str).iterencode(data):
     f.write(chunk)
     if target.stat().st_size>32*1024**2:raise ValueError('Per-detail32MiB compressed cap')
   detail=dict(path=str(target.relative_to(R)),bytes=target.stat().st_size,priority=priority,success=status=='DONE',signature=signature)
 except Exception as e:detail_error=type(e).__name__+': '+str(e)
 for a in actors:
  a.act({'select':None})
  if a.memory.lower_bounds or a.memory.game is not None or a.order.first is not None:failure_kind='CORRECTNESS';status='FAILED';reason='Match reset violation'
 result=dict(job,status=status,result=winner,games=games,error=reason,failure_kind=failure_kind,actor_errors=errors,started_at_utc=begun.isoformat(),ended_at_utc=iso(),UTC_seconds=(now()-begun).total_seconds(),awake_seconds=CLOCK()-awake,CPU_core_seconds=time.process_time()-cpu,inference_CPU_seconds=inference,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,policy_tensor=before,policy_unchanged=state_hash()==TENSOR,independent_native_deals=True,detail=detail,detail_error=detail_error,startup_parity=bool(job.get('startup_check') and status=='DONE'))
 # One compact record per attempt is appended by the parent; this worker stores
 # only its current progress slot, not one unbounded directory per match.
 write(OUT/'inflight'/f'{os.getpid()}.json',dict(at_utc=iso(),job=job,completed=result['id'],games=games))
 result=finish(job,belief_hist,MODULE.Battle.vis,result)
 MODULE.Battle.vis=[];del env
 return result
