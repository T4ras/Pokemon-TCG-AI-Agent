"""Before-update gates on the exact pending rollout; no threshold exceptions."""
from .common import *
from .runtime import TrainingRuntime
import torch,numpy as np

def run(device='cpu'):
 rt=TrainingRuntime(context('belief'),'belief');engines=[];start=time.monotonic()
 try:
  model,opt,saved=rt.restore(PRIVATE/'initial-correction.pt',device);rt.configure_precision();rt.runtime.freeze_inference(model)
  source=PRIVATE/'pending-cohort.pt';cohort=torch.load(source,map_location='cpu',weights_only=False);rt.epochs.verify_cohort(cohort);assert len(cohort['rows'])==2876 and rt.state_hash(model)==cohort['collection_policy_hash']
  assert sha(source)==read(OUT/'registration.json')['pending_cohort_sha256'] and saved['step']==1052 and saved['extra']['execution']['cohort_exposures']==0
  results={}
  for kind in ('bucket64','baseline'):
   engine=rt.engine.Engine(model,cohort,kind,phase=9);engines.append(engine);results[kind]=engine.reference_check(cohort['rows']);engine.close();engines.remove(engine)
  record=dict(at_utc=iso(),backend=device,rows=2876,results=results,thresholds_unchanged=True,old_probabilities_rebased=False,source_step=1052,source_collected=35852,source_updated=32976,elapsed_seconds=time.monotonic()-start,native_matches=0,state='PASS' if results['baseline']['status']=='PASS' else 'FAILED')
  write(OUT/('pending-reference-'+device+'.json'),record)
  if record['state']!='PASS':raise FloatingPointError('Exact-length baseline failed original reference tolerances; pending rollout retained')
  assert sha(source)==read(OUT/'registration.json')['pending_cohort_sha256'];print('BELIEF_PENDING_REFERENCE_PASS '+device+' '+json.dumps(results),flush=True)
  return record
 finally:
  for engine in engines:engine.close()
  rt.close()
