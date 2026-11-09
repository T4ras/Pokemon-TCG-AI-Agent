"""Readiness on retained legal histories/packets; never fresh learning evidence."""
from .common import *
from .belief import OnlineBelief,current_features,frozen_belief_model,vector_from_predictions
from .integration import PilotRuntime,context
from .correction import configure,frozen_digest,learning_rate
from unown_belief_full_training.fit import predict
from unown_belief_full_training.preprocess import load_features
from unown_belief_full_training.storage import load as load_record
from unown_runner.packets import Packet,Layout
import torch,numpy as np,copy,resource,random,traceback

def main():
 start=time.process_time();torch.set_num_threads(1);torch.set_num_interop_threads(1);errors={};rt=None
 try:
  full=R/'experiments/astra-unown-belief-classifier-full-training-001';index={r['id']:r for r in read(full/'dataset-index.json')};valid=[r for r in read(full/'split-manifest.json')['entries'] if r['partition']=='validation'];prior_info=prior();ids=[]
  for family in ('Alakazam/Dudunsparce draw','Dragapult spread','Festival Lead swarm','Ogerpon/Kangaskhan energy-switch'):
   candidates=[r['id'] for r in valid if prior_info[r['opponent_list']]['archetype']==family];ids.append(min(candidates,key=lambda i:stable('belief-pilot-readiness',i)))
  last_vector=None;callbacks=forced=0;points=0
  for ident in ids:
   guard();row=index[ident];assert row['partition']=='validation';legal=load_record(row,'legal');f,t=load_features(ident);del t;selected={int(v):i for i,v in enumerate(f['selected'])};observer=OnlineBelief();before=None
   for i,e in enumerate(legal['entries']):
    observer.observe(e['observation']);callbacks+=1;sel=e['observation']['select'];forced+=int(sel['minCount']==sel['maxCount']==len(sel['option']) or len(sel['option'])==1 and sel['minCount']>=1)
    if i not in selected:continue
    at=selected[i];actual=current_features(observer.facts,observer.current,observer.within)
    for key in ('global_X','local_X','A','lower','upper','deck_A','ledger','known','games','zone_sizes'):
     err=float(np.max(np.abs(actual[key].astype(float)-f[key][at:at+1].astype(float))));errors[key]=max(errors.get(key,0),err);np.testing.assert_allclose(actual[key],f[key][at:at+1],atol=2e-6,rtol=2e-5)
    direct={k:v[at:at+1] for k,v in f.items() if k in actual};dd,pp=predict(frozen_belief_model(),direct,'MLP');expected=vector_from_predictions(direct,dd,pp);np.testing.assert_allclose(observer.readout(),expected,atol=2e-6,rtol=2e-5);last_vector=expected;points+=1
   known=dict(observer.facts.lower);observer.reset_match();assert not observer.facts.lower and observer.game is None
  before_classifier=tensor_digest(frozen_belief_model().state_dict())
  rt=PilotRuntime(context('belief','preflight'),'belief');rt.context.out.mkdir(parents=True,exist_ok=True);rt.context.private.mkdir(parents=True,exist_ok=True);base=rt.load_base();model=rt.load_corrected();opt=configure(model);initial_frozen=frozen_digest(model);assert initial_frozen==tensor_digest({k:v for k,v in base.state_dict().items() if not k.startswith('value_mlp.')})
  inventory={name:p.numel() for name,p in model.named_parameters() if p.requires_grad};assert sum(inventory.values())==806403;write(OUT/'trainable-inventory.json',dict(parameters=inventory,total=sum(inventory.values()),actor=sum(v for k,v in inventory.items() if k.startswith(('correction.actor.','correction.actor_belief.'))),new_critic=sum(v for k,v in inventory.items() if k.startswith(('correction.critic.','correction.critic_belief.'))),existing_value=sum(v for k,v in inventory.items() if k.startswith('value_mlp.'))))
  saved=R/'data/private/astra-unown-gpu-batching-compilation-retry-001/development.pt';original=torch.load(saved,map_location='cpu',weights_only=False);rows=[];maxprob=0.;base_values=[]
  for source in original[:17]:
   p=source['packet'];packet=Packet(p.row.copy(),Layout(p.layout.tokens,p.layout.options,p.layout.tools,tuple(p.layout.oracle)));packet.belief=np.array(last_vector,copy=True);packet.belief.flags.writeable=False
   with torch.no_grad():
    b,_=base.core(packet);s,_=model.core(packet);old=b['logits'][0].softmax(-1);new=s['logits'][0].softmax(-1);maxprob=max(maxprob,float((old-new).abs().max()));assert int(b['logits'][0].argmax())==int(s['logits'][0].argmax());np.testing.assert_allclose(new.numpy(),old.numpy(),atol=1e-7,rtol=1e-6);bv=base.value_mlp(torch.cat([b['r'],b['g']],-1)).squeeze();torch.testing.assert_close(s['pilot_value'].squeeze(),bv,atol=1e-6,rtol=1e-6)
   mask=packet.row[slice(*packet.layout.offsets['opt_mask'])]>0;action=int(new.argmax());value=float(s['pilot_value'][0]);rows.append(dict(packet=packet,belief=packet.belief,action=action,old_probs=new.numpy().copy(),log_probability=float(s['logits'][0].log_softmax(-1)[action]),value=value,return_target=value+.1,advantage=.1 if len(rows)%2 else -.1,game=1,seat=0,software_verification_only=True))
  # Gradient isolation with nonzero disposable final layers to exercise both projections.
  scratch=rt.clone_model(model,'cpu')
  with torch.no_grad():scratch.correction.actor[-1].weight.fill_(.001);scratch.correction.critic[-1].weight.fill_(.001)
  s,_=scratch.core(rows[0]['packet']);scratch.zero_grad(set_to_none=True);s['pilot_value'].sum().backward();assert all(p.grad is None for n,p in scratch.named_parameters() if n.startswith(('correction.actor.','correction.actor_belief.')));assert all(p.grad is None for n,p in scratch.named_parameters() if not n.startswith(('correction.critic.','correction.critic_belief.','value_mlp.')))
  scratch.zero_grad(set_to_none=True);s,_=scratch.core(rows[0]['packet']);s['logits'][s['pilot_mask']].sum().backward();assert all(p.grad is None for n,p in scratch.named_parameters() if not n.startswith(('correction.actor.','correction.actor_belief.')));del scratch
  # Eager bucket64, actual remainder, reference/transaction/export and rollback.
  cohort=rt.epochs.new_cohort(rows,rt.state_hash(model));execution=dict(step=0);engine=rt.make_engine(model,opt,cohort,0)
  try:
   ref=engine.reference_check(rows);assert ref['status']=='PASS',ref;before=rt.batch.snapshot(model,opt,execution);finger=rt.scalar.state_fingerprints(before);adv=cohort['advantages'];learning_rate(opt,0);accepted=engine.transaction(model,opt,rows,adv,execution,'software-readiness-accepted');assert accepted['accepted'];assert frozen_digest(model)==initial_frozen
   checkpoint=PRIVATE/'preflight-export.pt';rt.legacy.save(checkpoint,model,opt,execution['step'],extra=dict(software_verification_only=True));restored,ropt,rstate=rt.restore(checkpoint,allow_disposable=True);assert rt.state_hash(restored)==rt.state_hash(model);assert rt.scalar.digest(ropt.state_dict())==rt.scalar.digest(opt.state_dict())
   after=engine.outputs(rows);second=rt.make_engine(restored,ropt,cohort,0)
   try:
    again=second.outputs(rows)
    for i in after:torch.testing.assert_close(after[i][0],again[i][0],atol=0,rtol=0);torch.testing.assert_close(after[i][1],again[i][1],atol=0,rtol=0)
   finally:second.close()
   # Same-backend uninterrupted versus checkpoint-resumed next optimizer update.
   rng_saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
   next_a=rt.make_engine(model,opt,cohort,0);next_b=rt.make_engine(restored,ropt,cohort,0)
   try:
    ea=dict(step=execution['step']);eb=dict(step=execution['step']);rt.restore_rng(rng_saved,'cpu')
    ra=next_a.transaction(model,opt,rows,adv,ea,'uninterrupted-next-update');fa=rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,ea))
    rt.restore_rng(rng_saved,'cpu');rb=next_b.transaction(restored,ropt,rows,adv,eb,'resumed-next-update');fb=rt.scalar.state_fingerprints(rt.batch.snapshot(restored,ropt,eb))
    assert ra['accepted'] and rb['accepted'] and fa==fb,(ra,rb,fa,fb)
   finally:next_a.close();next_b.close()
   # Recompute references only on disposable software copies, never a native cohort.
   rt.batch.restore(model,opt,before,execution);assert rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,execution))==finger
   diagnose=engine.diagnose
   def reject(*args,**kw):return dict(diagnose(*args,**kw),phase_KL=.04)
   engine.diagnose=reject;rejected=engine.transaction(model,opt,rows,adv,execution,'software-readiness-rejected');assert not rejected['accepted'] and rejected['rollback_exact'];assert rt.scalar.state_fingerprints(rt.batch.snapshot(model,opt,execution))==finger
  finally:engine.close()
  assert frozen_digest(model)==initial_frozen and tensor_digest(frozen_belief_model().state_dict())==before_classifier
  write(OUT/'preflight.json',dict(at_utc=iso(),passed=True,retained_validation_histories=ids,callback_count=callbacks,forced_callbacks=forced,feature_points=points,feature_max_errors=errors,belief_dim=1479,trainable_parameters=806403,zero_correction_probability_max_error=maxprob,zero_correction_greedy_action_parity=True,value_gradient_isolation=True,actor_gradient_isolation=True,production_bucket64_reference=ref,disposable_accepted_update=accepted['accepted'],disposable_forced_rollback_exact=rejected['rollback_exact'],export_reload_parity=True,uninterrupted_resumed_next_update_exact=True,frozen_policy_tensor_identity=True,frozen_classifier_identity=True,training_experience_created=0,native_attempts=0,GPU_seconds=0,software_probe_weights_candidate_eligible=False,CPU_seconds=time.process_time()-start));print('PRELIGHT PASSED',flush=True)
 except BaseException as e:
  write(OUT/'preflight-failure.json',dict(at_utc=iso(),reason=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),CPU_seconds=time.process_time()-start,native_attempts=0,GPU_seconds=0));raise
 finally:
  if rt is not None:rt.close()
if __name__=='__main__':main()
