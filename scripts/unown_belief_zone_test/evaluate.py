"""Explicitly authorized test-only bridge; historical sealed loaders stay unchanged."""
from .common import *
import numpy as np,functools,pickle,traceback,resource,signal
from unown_belief_full_training.models import make_neural,point_weights
from unown_belief_full_training.fit import predict
from unown_belief_full_training.scoring import aggregate

@functools.lru_cache(maxsize=4)
def shard(path,expected):
 if sha(R/path)!=expected:raise InvariantFailure('Test shard changed')
 import gzip
 with gzip.open(R/path,'rt') as f:return [json.loads(line) for line in f]

def test_component(item,component):
 allowed={x['id']:x for x in read(OUT/'test-manifest.json')}
 if item['id'] not in allowed or item!=allowed[item['id']]:raise PermissionError('Outside frozen test allocation')
 if component not in ('legal','targets','provenance'):raise ValueError(component)
 row=item['storage'];p=row['paths'][component];v=shard(p['path'],p['sha256'])[row['index']]
 if v['id']!=item['id']:raise InvariantFailure('Test shard ID mismatch')
 return v

def load_models():
 import torch
 bank={'A':(None,'A')}
 for item in read(OUT/'models.json'):
  path=R/item['path'];family=item['job']['family']
  if sha(path)!=item['sha256']:raise InvariantFailure('Candidate changed')
  if family=='Bayes':
   with np.load(path) as f:model={k:f[k] for k in f.files}
  elif family=='Trees':
   with path.open('rb') as f:model=pickle.load(f)
  else:
   model=make_neural(family,item['job']['seed']);state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);model.eval()
  bank[item['key']]=(model,family)
 return bank

METRICS=tuple(f'{metric}_{zone}' for metric in ('presence','uncertain_presence','count_multiclass') for zone in ('hand','draw_pile','prizes'))+('total_hand_presence',)
def zone_scores(f,t,p):
 if not np.isfinite(p).all() or np.any(p< -1e-8) or not np.allclose(p.sum(-1),1,atol=1e-5):raise InvariantFailure('Invalid count probability')
 if p.shape!=f['A'].shape or p.shape[:3]!=t['counts'].shape:raise InvariantFailure('Zone/count shape mismatch')
 prob=1-p[...,0];truth=t['counts']>0;bs=(prob-truth)**2;A=1-f['A'][...,0];uncertain=(A>=.1)&(A<=.9);den=uncertain.sum(1);ubs=np.divide((bs*uncertain).sum(1),den,out=np.full(den.shape,np.nan),where=den>0)
 correct=np.take_along_axis(p,t['counts'][...,None],axis=-1)[...,0];multi=(p*p).sum(-1)-2*correct+1
 known=f['known']>0;total=np.where(known,1.,prob[:,:,0]);ytotal=known|truth[:,:,0]
 values=np.column_stack((bs.mean(1),ubs,multi.mean(1),((total-ytotal)**2).mean(1)))
 return dict(zip(METRICS,aggregate(values,f['games']))),dict(uncertain_queries=den.sum(0).tolist(),all_queries=int(np.prod(p.shape[:2])),points=len(p))

BANK=None
def initialize():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 # Replace only the expired historical orchestration deadline in this process.
 # Feature math, legal observer, reset logic and truth join are unchanged.
 import unown_belief_full_training.features as features
 features.check_time=guard
 global BANK
 BANK=load_models()
 write(OUT/'workers'/f'{os.getpid()}.json',dict(at_utc=iso(),models=len(BANK),threads=1))

def one(item):
 started=time.process_time();wall=time.monotonic();ident=item['id'];result=dict(id=ident,opponent_list=item['opponent_list'],chooser=item['chooser'],state='FAILED')
 try:
  guard();append(OUT/'test-access.jsonl',dict(at_utc=iso(),id=ident,event='authorized fixed test history opened',pid=os.getpid()))
  legal=test_component(item,'legal')
  from unown_belief_full_training.features import encode,target_arrays
  f,info=encode(legal['entries'],legal['selection'],True)
  # Only after legal features are fixed may diagnostic truth be joined.
  targets=test_component(item,'targets');assert targets['opponent_list']==item['opponent_list'];t=target_arrays(targets['points'],f,info['points'],targets['opponent_list'])
  fp=PRIVATE/'features'/(ident+'.npz');tp=PRIVATE/'targets'/(ident+'.npz');npwrite(fp,**f);npwrite(tp,**t)
  scores={};coverage=None
  for key,(model,family) in BANK.items():
   guard();d,p=predict(model,f,family);scores[key],coverage=zone_scores(f,t,p)
  result.update(state='COMPLETE',metrics=scores,coverage=coverage,feature_path=str(fp.relative_to(R)),feature_sha256=sha(fp),target_path=str(tp.relative_to(R)),target_sha256=sha(tp),callbacks=len(f['sequence']),feature_CPU_seconds=info['CPU_seconds'])
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 result.update(CPU_seconds=time.process_time()-started,elapsed_seconds=time.monotonic()-wall);write(OUT/'results'/(ident+'.json'),result);return result
