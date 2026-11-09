"""Portable CPU belief check inside the already bounded GPU session."""
from .common import *
from .math_cache import ExactBeliefCache
import torch,numpy as np

def run():
 from unown_belief_correction.belief import current_features,frozen_belief_model,vector_from_predictions
 from unown_belief_full_training.fit import predict
 from .deadline_hooks import install
 install()
 states=torch.load(PRIVATE/'belief-workload.pt',map_location='cpu',weights_only=False)['states'];frozen_belief_model()
 def measure():
  start=time.process_time();out=[]
  for x in states:
   f=current_features(x['facts'],x['observation'],x['within']);d,p=predict(frozen_belief_model(),f,'MLP');out.append((f,vector_from_predictions(f,d,p)))
  return time.process_time()-start,out
 before,a=measure()
 with ExactBeliefCache():after,b=measure()
 for (f,x),(g,y) in zip(a,b):
  assert x.dtype==y.dtype and x.tobytes()==y.tobytes()
  for k in f:assert f[k].dtype==g[k].dtype and f[k].tobytes()==g[k].tobytes(),k
 result=dict(at_utc=iso(),state='PASS',snapshots=len(states),uncached_CPU_seconds=before,cached_CPU_seconds=after,speedup=before/after,all_feature_bytes_exact=True,all_vector_bytes_exact=True,native_attempts=0)
 write(OUT/'cloud-cache-profile.json',result);print('BELIEF_RECOVERY_CACHE_PASS '+json.dumps(result),flush=True)
