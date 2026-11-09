"""Full fixed-history equivalence and unprofiled recovery-path timing."""
from . import audit as a
from .math_cache import ExactBeliefCache
from unown_belief_correction.common import *
from unown_belief_full_training.storage import load as load_record
import torch,numpy as np,copy,resource,signal,traceback
O=a.O;P=a.P/'exact';P.mkdir(parents=True,exist_ok=True)

def main():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(900,901))
 def cpu_limit(*args):raise TimeoutError('Registered900CPU-second exact audit cap')
 signal.signal(signal.SIGXCPU,cpu_limit);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 try:
  reg=read(O/'repair-registration.json');index={x['id']:x for x in read(R/'experiments/astra-unown-belief-classifier-full-training-001/dataset-index.json')};histories=[(ident,load_record(index[ident],'legal')) for ident in reg['retained_history_ids']];a.frozen_belief_model();base,bv,bf,_=a.replay(histories,True)
  with ExactBeliefCache() as cache:fast,fv,ff,_=a.replay(histories,True);cache_info={k:v.cache_info()._asdict() for k,v in cache.math.items()}
  assert all(x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(bv,fv)) and len(bv)==len(fv)
  for x,y in zip(bf,ff):
   for key in x:assert x[key].dtype==y[key].dtype and x[key].tobytes()==y[key].tobytes(),key
  states=torch.load(a.P/'belief-workload.pt',map_location='cpu',weights_only=False)['states']
  def timing():
   from unown_belief_full_training.fit import predict
   from unown_belief_correction.belief import current_features,vector_from_predictions
   t=time.process_time();out=[]
   for x in states:
    f=current_features(x['facts'],x['observation'],x['within']);d,p=predict(a.frozen_belief_model(),f,'MLP');out.append(vector_from_predictions(f,d,p))
   return time.process_time()-t,out
  slow,sv=timing()
  with ExactBeliefCache():quick,qv=timing()
  assert all(x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(sv,qv))
  # Exercise a nonzero correction so probability parity cannot hide behind zero heads.
  from unown_belief_correction_retry.runtime import TrainingRuntime
  from unown_belief_correction_retry.common import context,PRIVATE as RP
  rt=TrainingRuntime(context('belief'),'belief');model=rt.load_corrected();rows=torch.load(RP/'encoded-probes.pt',map_location='cpu',weights_only=False)['rows'];error=0
  with torch.no_grad():model.correction.actor[-1].weight.copy_(torch.linspace(-.01,.01,256)[None,:]);model.correction.critic[-1].weight.fill_(.01)
  try:
   with torch.no_grad():
    for i,row in enumerate(rows):
     p=copy.copy(row['packet']);p.belief=sv[i%len(sv)];x,_=model.core(p);p.belief=qv[i%len(qv)];y,_=model.core(p);px=x['logits'].softmax(-1);py=y['logits'].softmax(-1);assert torch.equal(px,py) and torch.equal(x['pilot_value'],y['pilot_value']);assert torch.equal(px.argmax(-1),py.argmax(-1));error=max(error,float((px-py).abs().max()))
  finally:rt.close()
  result=dict(at_utc=iso(),passed=True,callbacks=base['callbacks'],forced_callbacks=base['forced_callbacks'],genuine_callbacks=len(bv),baseline=base,cached=fast,snapshot_uncached_CPU_seconds=slow,snapshot_cached_CPU_seconds=quick,snapshot_count=len(states),snapshot_speedup=slow/quick,exact_features=True,exact_vectors=True,nonzero_correction_probability_max_error=error,nonzero_correction_argmax_exact=True,math_cache_info=cache_info,CPU_seconds=time.process_time()-start,native_attempts=0,GPU_seconds=0);write(O/'exact-cache-parity-timing.json',result);print(json.dumps(result),flush=True)
 except BaseException as e:write(O/'exact-audit-failure.json',dict(at_utc=iso(),reason=str(e),traceback=traceback.format_exc(),CPU_seconds=time.process_time()-start));raise
if __name__=='__main__':main()
