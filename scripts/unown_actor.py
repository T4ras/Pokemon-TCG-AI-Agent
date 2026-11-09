"""Isolated adapter around the attributed, unmodified submitted Python runtime."""
import sys,os,json,time,ctypes,inspect,copy,resource,hashlib,importlib.util,random
from unown_common import *

class CompatibilityError(RuntimeError):pass

def tensor_hash(network):
 h=hashlib.sha256()
 for k,v in sorted(network.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def check_supported(obs,supported,attacks):
 def visit(v):
  if isinstance(v,dict):
   for key,x in v.items():
    if key in ('id','cardId') and isinstance(x,int) and x>0 and x not in supported:raise CompatibilityError(f'Unsupported card identity {x}')
    if key=='attackId' and isinstance(x,int) and x>0 and x not in attacks:raise CompatibilityError(f'Unsupported attack identity {x}')
    if isinstance(x,(dict,list)):visit(x)
  elif isinstance(v,list):
   for x in v:visit(x)
 visit({k:obs[k] for k in ('current','select','logs') if k in obs})
 sel=obs.get('select')
 if sel:
  n=len(sel['option']);forced=sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1
  if n>63 and not forced:raise CompatibilityError(f'Non-forced option overflow: {n}>63')

def load_module(name):
 path=UP/'submission/main.py';spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def initialize(guarded=True,name='unown_released_main'):
 os.chdir(REPO);started=time.perf_counter();sys.path[:0]=[str(PRIVATE/'original-sdk'),str(UP/'submission')]
 from cg.sim import lib
 lib.AllCard.restype=ctypes.c_char_p;lib.AllAttack.restype=ctypes.c_char_p
 import torch
 torch.manual_seed(SEED);random.seed(SEED)
 module=load_module(name)
 assert module.HEADS==8 and module.SA.cfg is None and module.SA.temp==0 and module.SA.device.type=='cpu'
 baseline=tensor_hash(module.SA.policy)
 metrics=dict(startup_seconds=time.perf_counter()-started,model_load_seconds=time.time()-module._LOAD_T0,decisions=0,forced_decisions=0,model_encodes=0,max_tokens=0,max_options=0,game_resets=0,match_resets=0,CPU_seconds=0.,inference_seconds=0.,max_decision_seconds=0.,fallback_calls=0,oracle_nonzero=0)
 old=read(PRIVATE/'original-tables.json');current=read(PRIVATE/'current-tables.json')
 cards={c['cardId']:c for c in current['cards']};att={a['attackId']:a for a in current['attacks']}
 supported={c['cardId'] for c in old['cards'] if normalize(c)==normalize(cards.get(c['cardId']))}
 attack_ids={a['attackId'] for a in old['attacks'] if normalize(a)==normalize(att.get(a['attackId']))}
 supported={c['cardId'] for c in old['cards'] if c['cardId'] in supported and all(a in attack_ids for a in c['attacks'])}
 from ptcg.rl import search_agent,encoder
 from ptcg.rl.buffers import OFFSETS,MAX_TOKENS,TOK_INT
 original_encode=encoder.encode
 if guarded:
  source=inspect.getsource(original_encode);needle='        if nt >= MAX_TOKENS:\n            return None';assert source.count(needle)==1
  ns=dict(encoder.__dict__);ns['CompatibilityError']=CompatibilityError
  exec(source.replace(needle,'        if nt >= MAX_TOKENS:\n            raise CompatibilityError("State token overflow >160")'),ns)
  exact_guarded=ns['encode']
  def checked_encode(row,*args,**kw):
   exact_guarded(row,*args,**kw);metrics['model_encodes']+=1
   a,b=OFFSETS['tok_int'];tokens=int((row[a:b].reshape(MAX_TOKENS,TOK_INT)[:,1]>0).sum());metrics['max_tokens']=max(metrics['max_tokens'],tokens)
   a,b=OFFSETS['oracle'];assert not row[a:b].any(),'Privileged critic inputs populated during inference'
  search_agent.encode=checked_encode
  legal_random=module._legal_random
  def observed_random(sel):metrics['fallback_calls']+=1;return legal_random(sel)
  module._legal_random=observed_random
 return module,metrics,baseline,supported,attack_ids

class Adapter:
 def __init__(self):
  self.module,self.metrics,self.identity,self.supported,self.attacks=initialize();self.round=None;self.faults=[]
  original=self.module.SA.agent
  def watched(obs,*a,**k):
   try:return original(obs,*a,**k)
   except Exception as e:self.faults.append((type(e).__name__,str(e)));raise
  self.module.SA.agent=watched
 def reset(self,seed=SEED):
  import torch
  random.seed(seed);torch.manual_seed(seed);self.round=None;self.faults=[]
  self.metrics['max_tokens']=0;self.metrics['max_options']=0;self.metrics['max_decision_seconds']=0.
  self.metrics['match_resets']+=1
  answer=self.module.agent({'select':None})
  assert not self.module.SA.tracker.opp_known_hand and not self.module.SA.tracker.opp_known_deck and self.module.SA.tracker.my_prize_known is None
  return answer
 def act(self,obs):
  if obs.get('select') is None:return self.reset()
  check_supported(obs,self.supported,self.attacks)
  rid=obs['current'].get('round',1)
  if self.round!=rid:
   self.module.agent({'select':None});self.metrics['game_resets']+=1;self.round=rid
   assert not self.module.SA.tracker.opp_known_hand and not self.module.SA.tracker.opp_known_deck
  sel=obs['select'];n=len(sel['option']);self.metrics['max_options']=max(n,self.metrics['max_options'])
  self.metrics['forced_decisions']+=int(sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1)
  t=time.perf_counter();cpu=time.process_time();ans=self.module.agent(obs);elapsed=time.perf_counter()-t
  self.metrics['decisions']+=1;self.metrics['inference_seconds']+=elapsed;self.metrics['CPU_seconds']+=time.process_time()-cpu;self.metrics['max_decision_seconds']=max(self.metrics['max_decision_seconds'],elapsed)
  if self.faults:raise CompatibilityError('Original inference error: '+str(self.faults[-1]))
  if not isinstance(ans,list) or len(ans)!=len(set(ans)) or not sel['minCount']<=len(ans)<=sel['maxCount'] or any(not isinstance(j,int) or not 0<=j<n for j in ans):raise CompatibilityError('Invalid released action')
  return ans
 def stats(self):
  return dict(self.metrics,guard=dict(self.module.GUARD),tensor_sha256=tensor_hash(self.module.SA.policy),weights_unchanged=tensor_hash(self.module.SA.policy)==self.identity,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_CPU_seconds=time.process_time(),tracker_known_hand=len(self.module.SA.tracker.opp_known_hand) if self.module.SA.tracker else 0)

def serve():
 actor=Adapter();print(json.dumps(dict(ready=True,stats=actor.stats())),flush=True)
 for line in sys.stdin:
  try:
   job=json.loads(line);cmd=job['command']
   if cmd=='reset':out=dict(action=actor.reset(job['seed']))
   elif cmd=='act':out=dict(action=actor.act(job['observation']))
   elif cmd=='stats':out=dict(stats=actor.stats())
   elif cmd=='stop':print(json.dumps(dict(stats=actor.stats())),flush=True);break
   else:raise ValueError('Unknown actor command')
   print(json.dumps(out),flush=True)
  except Exception as e:
   print(json.dumps(dict(error_type=type(e).__name__,error=str(e))),flush=True)

if __name__=='__main__':serve()
