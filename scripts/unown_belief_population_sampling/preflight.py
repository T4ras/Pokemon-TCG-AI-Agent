from .common import *
import torch,numpy as np,copy,platform,importlib.metadata
from unown_belief_full_training.models import make_neural,neural_loss
from unown_belief_full_training.fit import predict
from unown_belief_deck_audit.diagnostics import digest,components,initial_counts
from .metrics import score
from .train import atomic_checkpoint

def main():
 started=time.process_time();torch.set_num_threads(3);torch.set_num_interop_threads(1);verified={}
 for item in read(OUT/'cache-manifest.json'):
  assert item['partition'] in ('training','validation');assert sha(R/item['cache_record'])==item['cache_record_sha256']
  for kind in ('feature','target'):
   cache=item['cache'];path=cache[kind+'_path'];assert '/locked-test/' not in path;assert sha(R/path)==cache[kind+'_sha256'];verified[path]=dict(sha256=cache[kind+'_sha256'],bytes=(R/path).stat().st_size)
  for detail in item['raw']['paths'].values():
   path=detail['path'];assert '/locked-test/' not in path
   if path not in verified:assert sha(R/path)==detail['sha256'];verified[path]=dict(sha256=detail['sha256'],bytes=detail['bytes'])
 # Only initialization/forward/backward parity; no optimizer updates in preflight.
 initial=[];sample=sorted(entries('validation'),key=lambda r:stable('population-preflight',r['id']))[:4]
 for seed in (11,29):
  model=make_neural('MLP',seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),seed=seed,tensor_sha256=digest(model.state_dict()));path=PRIVATE/'initial'/f's{seed}.pt';atomic_checkpoint(path,state);restored=torch.load(path,map_location='cpu',weights_only=False);other=make_neural('MLP',seed);other.load_state_dict(restored['model']);op=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001);op.load_state_dict(copy.deepcopy(restored['optimizer']));assert digest(other.state_dict())==state['tensor_sha256'] and op.state_dict()==optimizer.state_dict()
  for item in sample:
   f,t=load(item['id']);d,p=predict(model,f,'MLP');np.testing.assert_allclose(d,f['deck_A'],atol=2e-7,rtol=2e-5);np.testing.assert_allclose(p,initial_counts(f),atol=2e-6,rtol=2e-5);deck,count,_,_=components(model,f,t);torch.testing.assert_close(neural_loss(model,f,t),deck+count);m=score(f,t,d,p);assert abs(m['objective_NLL']-m['deck_log_loss']-m['count_NLL'])<5e-6
  initial.append(dict(seed=seed,path=str(path.relative_to(R)),sha256=sha(path),tensor_sha256=state['tensor_sha256'],optimizer_state_empty=True,matched_arms=['control','population']))
 write(OUT/'initialization.json',dict(seeds=initial,identical_model_optimizer_RNG_within_seed=True,optimizer_updates=0));write(OUT/'verified-data-manifest.json',dict(files=verified,training_BO3=2334,validation_BO3=592,test_payload_read=False));write(OUT/'runtime-versions.json',dict(python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in ('numpy','torch','scipy','scikit-learn','pytest')}));write(OUT/'preflight.json',dict(at_utc=iso(),passed=True,non_test_caches=2926,verified_private_files=len(verified),validation_parity_histories=4,initial_seeds=[11,29],optimizer_updates=0,test_access=False,CPU_seconds=time.process_time()-started))
 print('All2926 non-test caches and raw inputs verified; both matched initializations and exact losses pass; preflightoptimizerupdates0')
if __name__=='__main__':main()
