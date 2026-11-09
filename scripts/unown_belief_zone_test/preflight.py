from .common import *
import numpy as np,torch,time
from .evaluate import load_models,zone_scores
from unown_belief_full_training.fit import predict
from unown_belief_full_training.preprocess import load_features

def main():
 started=time.process_time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 full=R/'experiments/astra-unown-belief-classifier-full-training-001';entries=[x for x in read(full/'split-manifest.json')['entries'] if x['partition']=='validation'];item=min(entries,key=lambda x:stable(SEED,'parity',x['id']));ident=item['id'];f,t=load_features(ident);bank=load_models();errors={}
 for key,(model,family) in bank.items():
  d,p=predict(model,f,family);saved=read(full/'validation-results'/(key+'.json'));r=next(x for x in saved['records'] if x['id']==ident);path=R/r['prediction_path'];assert sha(path)==r['prediction_sha256']
  with np.load(path) as a:
   np.testing.assert_allclose(d,a['deck'],atol=2e-6,rtol=2e-5);np.testing.assert_allclose(p,a['counts'],atol=2e-6,rtol=2e-5);errors[key]=dict(deck=float(np.max(np.abs(d-a['deck']))),counts=float(np.max(np.abs(p-a['counts']))))
  m,c=zone_scores(f,t,p);assert abs(m['presence_hand']-r['metrics']['all_brier'])<2e-6;assert abs(m['uncertain_presence_hand']-r['metrics']['uncertain_brier'])<2e-6
 # Re-encode one validation legal history to verify the exact reused feature path.
 from unown_belief_full_training.storage import load
 index=next(x for x in read(full/'dataset-index.json') if x['id']==ident);legal=load(index,'legal')
 import unown_belief_full_training.features as features
 features.check_time=guard
 replay,info=features.encode(legal['entries'],legal['selection'],True)
 for k in ('global_X','local_X','sequence','reset','selected','A','lower','upper','deck_A','ledger','known','games','zone_sizes','conditional_indices','conditional_offsets','conditionals'):
  np.testing.assert_allclose(replay[k],f[k],atol=2e-6,rtol=2e-5)
 write(OUT/'preflight.json',dict(at_utc=iso(),passed=True,models=len(bank),fixed_validation_ID=ident,model_max_errors=errors,feature_replay_parity=True,old_hand_Brier_parity=True,test_feature_or_target_access=False,CPU_seconds=time.process_time()-started))
 print('All22 frozen models reproduce saved validation predictions; feature replay and Brier parity pass; no test model pass yet')
if __name__=='__main__':main()
