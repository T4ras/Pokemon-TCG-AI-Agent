"""One bounded raw-batched PPO continuation with cohort-wise experience cosine."""
import resource,shutil,traceback
from unown_cosine_core import *
from unown_pilot_execution import cuda_startup
def run():
 begin=float(os.environ.get('COSINE_JOB_START',time.monotonic()));absolute=parse(os.environ['COSINE_ABSOLUTE_DEADLINE']);limit=min(begin+TRAIN_CAP,time.monotonic()+max(0,(absolute-now()).total_seconds()-900))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 cuda_startup(torch,lambda x:write(OUT/'gpu-runtime.json',dict(x,at_utc=iso())))
 for name,digest in read(OUT/'source-freeze.json')['files'].items():assert sha(ROOT/name)==digest,name
 exclusive(OUT/'training-execution-intent.json',dict(at_utc=iso(),pid=os.getpid(),one_training_process=True,no_restart=True))
 assert sha(PRIVATE/'start.pt')==START_SHA and sha(PRIVATE/'initialization.pt')==ANCHOR_SHA
 model,opt,start=restore(PRIVATE/'start.pt','cuda');assert state_hash(model)==START_TENSOR and batch.digest(opt.state_dict())==read(OUT/'provenance.json')['optimizer_digest']
 anchor=load_model(PRIVATE/'initialization.pt','cuda')
 for p in anchor.parameters():p.requires_grad_(False)
 fixed=torch.load(PRIVATE/'fixed-diagnostic.pt',weights_only=False,map_location='cpu');core.reference(fixed,anchor);diag=core.fixed_drift(model,fixed);append(OUT/'initialization-diagnostics.jsonl',dict(stage='startup',at_utc=iso(),**diag));restore_rng(start,'cuda')
 original_step=start['step'];execution=dict(step=original_step,additional_collected=0,completed_cohort_decisions=0,accepted_used_decisions=0,phase=0,offset=0,zero_streak=0,cohort_lr=rate(0),last_accepted_lr=None);phase=0;completed=0;meaningful=0;actual_collected=0;actual_complete=0;partial=None;fatal=None;reason='registered_time_endpoint';timing=collections.Counter();stops=collections.Counter();rejected=0;lastphase=0.;lastdiag=time.monotonic();max_journal_step=original_step
 buffers=batch.digest(dict(model.named_buffers()));disabled=batch.digest({n:p for n,p in model.named_parameters() if not p.requires_grad});anchor_hash=state_hash(anchor)
 def save(name='latest.pt'):
  t=time.monotonic();digest=legacy.save(PRIVATE/name,model,opt,execution['step'],extra=dict(execution=copy.deepcopy(execution),scheduler=dict(target=TARGET,additional_collected=execution['additional_collected'],completed_cohort_decisions=execution['completed_cohort_decisions'],applied_lr=execution['cohort_lr']),continuation_start_SHA256=START_SHA));timing['checkpoint_seconds']+=time.monotonic()-t;return digest
 save();write(OUT/'training-start.json',dict(at_utc=iso(),start_step=original_step,optimizer_digest=batch.digest(opt.state_dict()),initial_tensor_hash=state_hash(model),optimizer_resumed=True))
 try:
  while execution['additional_collected']<TARGET and time.monotonic()+max(360,lastphase*1.25)<limit:
   phase_start=time.monotonic();prior=execution['completed_cohort_decisions'];applied=rate(prior);execution.update(phase=phase,offset=0,cohort_lr=applied,cohort_start_progress=prior)
   for g in opt.param_groups:g['lr']=applied
   runtime.freeze_inference(model);cohort_hash=state_hash(model);rows=[];records=[];t=time.monotonic()
   for job in jobs(phase):
    if time.monotonic()+120>=limit:partial=dict(phase=phase,completed=len(records),planned=16,reason='timebeforematch');break
    row,experience=core.play(job,model,anchor,True,limit);records.append(row)
    if row['status']!='DONE':
     partial=dict(phase=phase,attempted=len(records),completed=sum(r['status']=='DONE' for r in records),planned=16,error=row['error'])
     if 'Timeout' in str(row['error']):reason='time_limit_in_final_match';break
     raise RuntimeError('Training correctness failure: '+str(row['error']))
    rows+=experience;completed+=1;actual_collected+=len(experience);meaningful+=sum(int((r['packet'].row[slice(*r['packet'].layout.offsets['opt_mask'])]>0).sum()>1) for r in experience)
   timing['collection_seconds']+=time.monotonic()-t;execution['additional_collected']=actual_collected
   if len(records)!=16 or partial:
    torch.save(rows,PRIVATE/'partial-final-rollout.pt');partial.update(decisions=len(rows),chooser_counts=dict(collections.Counter(r['learner_seat'] for r in records)));append(OUT/'partial-cohorts.jsonl',dict(at_utc=iso(),**partial));break
   assert collections.Counter(r['learner_seat'] for r in records)=={0:8,1:8} and all(r['learner_hash']==cohort_hash and r['opponent_hash']==anchor_hash for r in records)
   actual_complete+=len(rows);execution['completed_cohort_decisions']=actual_complete;t=time.monotonic();torch.save(rows,PRIVATE/'last-rollout.pt');timing['rollout_serialization_seconds']+=time.monotonic()-t;save('phase-input.pt')
   if phase==0:
    shutil.copy2(PRIVATE/'last-rollout.pt',PRIVATE/'first-rollout.pt');t=time.monotonic();check=live_check(model,opt,rows,execution);write(OUT/'live-interface.json',check);timing['live_interface_seconds']+=time.monotonic()-t;print('COSINE_LIVE_INTERFACE_PASS '+json.dumps(dict(decisions=len(rows),start_step=original_step,disposable_weights_not_promoted=True)),flush=True)
   # Old probabilities remain immutable; the registered raw convention is used.
   t=time.monotonic();likelihood=check['likelihood'] if phase==0 else lean.diagnose(model,rows)
   if phase>0:assert likelihood['phase_KL']<=1e-6 and likelihood['max_old_policy_KL']<=1e-4,likelihood
   timing['cohort_likelihood_seconds']+=time.monotonic()-t;write(OUT/f'phase-{phase:04d}-likelihood.json',likelihood)
   adv=core.advantages(rows);order=np.random.default_rng(SEED+phase).permutation(len(rows));before_step=execution['step'];rollout_stop=None;time_stop=False
   append(OUT/'cohort-inputs.jsonl',dict(at_utc=iso(),phase=phase,policy_hash=cohort_hash,decisions=len(rows),progress_before=prior,progress_after=execution['completed_cohort_decisions'],applied_lr=applied,rollout_SHA256=sha(PRIVATE/'last-rollout.pt')))
   for offset in range(0,len(rows),128):
    if time.monotonic()+90>=limit:reason='time_limit_between_updates';time_stop=True;break
    ii=order[offset:offset+128];execution.update(offset=offset,order=order)
    exclusive(OUT/'update-intents'/f'{phase:04d}-{offset:06d}.json',dict(at_utc=iso(),phase=phase,offset=offset,step_before=execution['step'],cohort_progress_before=prior,applied_lr=applied))
    old=copy.deepcopy(lean.METERS);t=time.monotonic();result=lean.transaction(model,opt,[rows[i] for i in ii],adv[ii],execution,'rolling-rejection');timing['transaction_seconds']+=time.monotonic()-t
    for k in lean.METERS:timing[k]+=lean.METERS[k]-old[k]
    if result['accepted']:execution['accepted_used_decisions']+=len(ii);execution['last_accepted_lr']=applied;max_journal_step=execution['step']
    append(OUT/'updates.jsonl',dict(at_utc=iso(),phase=phase,offset=offset,rows=len(ii),global_step=execution['step'],additional_accepted=execution['step']-original_step,progress_before=prior,additional_collected=execution['additional_collected'],applied_lr=applied,**result))
    if not result['accepted']:rollout_stop=result['reason'];stops[rollout_stop]+=1;rejected+=int(rollout_stop.startswith('phase_post'));break
    if (execution['step']-original_step)%8==0:
     checksum=save();write(OUT/'durable-progress.json',dict(at_utc=iso(),global_step=execution['step'],checkpoint_SHA256=checksum,scheduler_progress=execution['additional_collected'],applied_lr=applied))
   accepted=execution['step']-before_step;execution['zero_streak']=execution['zero_streak']+1 if accepted==0 else 0;save()
   assert batch.digest(dict(model.named_buffers()))==buffers and batch.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})==disabled
   if time.monotonic()-lastdiag>=3600:
    t=time.monotonic();diag=core.fixed_drift(model,fixed);timing['initialization_diagnostic_seconds']+=time.monotonic()-t;lastdiag=time.monotonic();append(OUT/'initialization-diagnostics.jsonl',dict(at_utc=iso(),stage='hourly',phase=phase,**diag));save(f'hourly-{int((time.monotonic()-begin)//3600):02d}.pt')
   lastphase=time.monotonic()-phase_start;append(OUT/'cohorts.jsonl',dict(at_utc=iso(),phase=phase,BO3=16,decisions=len(rows),accepted_updates=accepted,accepted_total=execution['step']-original_step,progress_before=prior,progress_after=execution['completed_cohort_decisions'],applied_lr=applied,rollout_stop=rollout_stop,zero_streak=execution['zero_streak'],elapsed_seconds=lastphase,training_WDL=dict(collections.Counter('W' if r['result']==r['learner_seat'] else 'D' if r['result']==2 else 'L' for r in records))))
   phase+=1;progress('TRAINING',additional_collected=execution['additional_collected'],target=TARGET,completed_BO3=completed,additional_accepted=execution['step']-original_step,applied_lr=applied,remaining_seconds=max(0,limit-time.monotonic()))
   if execution['additional_collected']>=TARGET:reason='additional_experience_target';break
   if execution['zero_streak']>=3:reason='three_consecutive_fresh_zero_update_cohorts';break
   if time_stop:break
   if sum(p.stat().st_size for p in PRIVATE.glob('*.pt'))>2*1024**3:reason='registered_mutable_recovery_storage_cap';break
 except BaseException as e:
  fatal=type(e).__name__+': '+str(e);reason='correctness_failure';write(OUT/'training-failure.json',dict(at_utc=iso(),error=fatal,traceback=traceback.format_exc(),no_replacement=True,observed_collected=actual_collected));model,opt,recovery=restore(PRIVATE/'latest.pt','cuda');execution=copy.deepcopy(recovery['extra']['execution']);execution.update(additional_collected=actual_collected,completed_cohort_decisions=actual_complete)
 finally:
  t=time.monotonic();diag=core.fixed_drift(model,fixed);timing['initialization_diagnostic_seconds']+=time.monotonic()-t;append(OUT/'initialization-diagnostics.jsonl',dict(at_utc=iso(),stage='endpoint',**diag));final_hash=state_hash(model);accepted=execution['step']-original_step;evaluate=fatal is None and accepted>0 and final_hash!=START_TENSOR;digest=save('final.pt')
  write(OUT/'training-completion.json',dict(at_utc=iso(),status='FAILED' if fatal else 'COMPLETE',error=fatal,stop_reason=reason,completed_BO3=completed,complete_cohorts=phase,additional_collected_decisions=execution['additional_collected'],completed_cohort_decisions=execution['completed_cohort_decisions'],accepted_used_decisions=execution['accepted_used_decisions'],meaningful_collected_decisions=meaningful,overshoot=max(0,execution['additional_collected']-TARGET),incomplete_final_cohort=partial,additional_accepted_updates=accepted,global_step=execution['step'],accepted_but_not_durable_updates_lost=max(0,max_journal_step-execution['step']),rejected_updates=rejected,rollout_stops=dict(stops),final_progress=min(1,execution['additional_collected']/TARGET),last_applied_lr=execution['last_accepted_lr'],last_configured_cohort_lr=execution['cohort_lr'],rate_at_final_collected_progress=rate(execution['additional_collected']),start_tensor_hash=START_TENSOR,final_tensor_hash=final_hash,checkpoint_SHA256=digest,evaluate=evaluate,skip_evaluation='correctnessfailedtraining' if fatal else 'noadditionalretainedmodelchange' if not evaluate else None,elapsed_seconds=time.monotonic()-begin,timing=dict(timing),CPU_core_hours=(resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)/3600,max_GPU_memory_bytes=torch.cuda.max_memory_allocated(),fixed_diagnostic=diag,initialization_KL_diagnostic_only=True))
 if fatal:raise RuntimeError(fatal)
if __name__=='__main__':
 try:run()
 except BaseException as e:write(OUT/'trainer-terminal-failure.json',dict(at_utc=iso(),error=str(e),traceback=traceback.format_exc(),automatic_replacement=False));raise
