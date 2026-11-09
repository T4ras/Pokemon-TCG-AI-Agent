"""Arithmetic-only GPU admission checks on immutable retained encoded rows."""
from .common import *
from .runtime import TrainingRuntime
import torch,numpy as np

def run(arm,device="cuda"):
 backend="GPU" if device=="cuda" else "CPU"
 ctx=context(arm);ctx.out.mkdir(parents=True,exist_ok=True);ctx.private.mkdir(parents=True,exist_ok=True);rt=TrainingRuntime(ctx,arm);engine=None
 try:
  model,opt,s=rt.restore(PRIVATE/'initial-correction.pt',device);source=torch.load(PRIVATE/'encoded-probes.pt',map_location='cpu',weights_only=False);assert source['software_verification_only'];rows=prepare_rows(rt,model,source['rows']);cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));engine=rt.make_engine(model,opt,cohort,9);ref=engine.reference_check(rows);write(OUT/(arm+'-'+backend+'-probe-reference.json'),ref);assert ref['status']=='PASS'
  e=dict(step=s['step']);before=rt.batch.snapshot(model,opt,e);fp=rt.scalar.state_fingerprints(before);a=engine.transaction(model,opt,rows,cohort['advantages'],e,backend+'-disposable-probe');assert a['accepted']
  saved=PRIVATE/(arm+'-'+backend+'-probe.pt');rt.legacy.save(saved,model,opt,e['step'],extra=dict(software_verification_only=True,candidate_eligible=False));resumed,ropt,state=rt.restore(saved,device,allow_disposable=True);other=rt.make_engine(resumed,ropt,cohort,9)
  try:
   ea=dict(step=e['step']);eb=dict(step=e['step']);rt.restore_rng(state,device);x=engine.transaction(model,opt,rows,cohort['advantages'],ea,backend+'-uninterrupted');fa=rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,ea));rt.restore_rng(state,device);y=other.transaction(resumed,ropt,rows,cohort['advantages'],eb,backend+'-resumed');fb=rt.scalar.state_fingerprints(rt.batch.snapshot(resumed,ropt,eb));assert x['accepted'] and y['accepted'] and fa==fb
  finally:other.close()
  rt.batch.restore(model,opt,before,e);diag=engine.diagnose;engine.diagnose=lambda *a,**kw:dict(diag(*a,**kw),phase_KL=.04);rejected=engine.transaction(model,opt,rows,cohort['advantages'],e,backend+'-forced-rollback');assert not rejected['accepted'] and rejected['rollback_exact'];assert rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,e))==fp
  write(OUT/(arm+'-'+backend+'-probe.json'),dict(at_utc=iso(),state='PASS',native_attempts=0,fresh_training_decisions=0,reference=ref,uninterrupted_resumed_exact=True,complete_rollback_exact=True,probe_weights_discarded=True));print('BELIEF_CORRECTION_'+backend+'_PROBE_PASS '+arm,flush=True)
 finally:
  if engine:engine.close()
  rt.close()
  if device=='cuda':torch.cuda.empty_cache()


def prepare_rows(rt,model,rows):
 """Software probes only: rebase retained encoded inputs to the saved candidate."""
 rows=copy.deepcopy(rows);cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));engine=rt.make_engine(model,None,cohort,0)
 try:outputs=engine.outputs(rows)
 finally:engine.close()
 for i,row in enumerate(rows):
  row.update(collection_batch=i,collection_index=0,collection_batch_size=1,collection_length=rt.prefix.host_length([row['packet']]),collection_variant='revised')
  logits,value=outputs[i];lp=logits.log_softmax(-1);row['old_probs']=lp.exp().numpy().astype(np.float32);row['log_probability']=float(lp[row['action']]);row['value']=float(value)
 return rows
