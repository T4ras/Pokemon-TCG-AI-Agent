"""One time-bounded fresh PPO arm; phase stops discard only that rollout."""
import resource,traceback
from unown_corrected_ppo_core import *
from unown_pilot_execution import cuda_startup
def run():
 begin=float(os.environ.get('CORRECTED_JOB_MONOTONIC_START',time.monotonic()));absolute=parse(os.environ['CORRECTED_PPO_ABSOLUTE_DEADLINE_UTC']);limit=min(begin+TRAIN_CAP,time.monotonic()+max(0,(absolute-now()).total_seconds()-900))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 cuda_startup(torch,lambda v:write(OUT/'gpu-runtime.json',dict(v,at_utc=iso())))
 for name,digest in read(OUT/'source-freeze.json')['files'].items():assert sha(ROOT/name)==digest,'Frozen source changed: '+name
 assert sha(PRIVATE/'initialization.pt')==INITIAL_SHA
 model,opt=fresh('cuda');anchor=load_model(PRIVATE/'initialization.pt','cuda')
 for p in anchor.parameters():p.requires_grad_(False)
 probe=torch.load(PRIVATE/'verification.pt',map_location='cpu',weights_only=False)
 import unown_batched_verify as verifier
 verifier.OUT=OUT;verifier.PRIVATE=PRIVATE
 parity=verifier.compare(model,probe[:128]);assert not opt.state
 fixed=torch.load(PRIVATE/'fixed-diagnostic.pt',map_location='cpu',weights_only=False);reference(fixed,anchor);first=fixed_drift(model,fixed);assert abs(first['initialization_KL'])<1e-6
 write(OUT/'gpu-preflight.json',dict(status='PASS',elapsed_seconds=time.monotonic()-begin,fresh_optimizer_empty=True,initial_hash=state_hash(model),corrected_packet_scalar_optimized=parity,reused_numerics=read(OUT/'reused-numerics.json'),no_probe_model_promoted=True))
 append(OUT/'initialization-diagnostics.jsonl',dict(at_utc=iso(),stage='startup',**first));lastdiag=time.monotonic();last_diagnostic=first
 buffers=batch.digest(dict(model.named_buffers()));disabled=batch.digest({k:p for k,p in model.named_parameters() if not p.requires_grad});anchor_hash=state_hash(anchor)
 execution=dict(step=0,phase=0,offset=0);phase=0;completed=0;trained_BO3=0;decisions=0;rejected=0;stops=collections.Counter();zero=ZeroCohorts();timing=collections.Counter();lastphase=0.;fatal=None;partial=None;reason='registered_time_endpoint'
 legacy.save(PRIVATE/'latest.pt',model,opt,0,extra=dict(execution=execution,fresh_optimizer=True))
 try:
  while time.monotonic()+max(360,lastphase*1.25)<limit:
   phase_start=time.monotonic();runtime.freeze_inference(model);rows=[];cohort_records=[];cohort_hash=state_hash(model);t=time.monotonic()
   for job in cohort_jobs(phase):
    if time.monotonic()+120>=limit:partial=dict(phase=phase,completed=len(cohort_records),planned=16,chooser_counts=dict(collections.Counter(r['learner_seat'] for r in cohort_records)));break
    record,experience=play(job,model,anchor,True,limit);cohort_records.append(record)
    if record['status']!='DONE':
     partial=dict(phase=phase,completed=sum(r['status']=='DONE' for r in cohort_records),attempted=len(cohort_records),planned=16,chooser_counts=dict(collections.Counter(r['learner_seat'] for r in cohort_records)))
     if 'Timeout' in str(record['error']):reason='time_limit_in_final_match';break
     raise RuntimeError('Native training correctness failure: '+str(record['error']))
    rows+=experience;completed+=1
   timing['collection_seconds']+=time.monotonic()-t;decisions+=len(rows)
   if len(cohort_records)!=16 or partial:
    torch.save(rows,PRIVATE/'partial-final-rollout.pt');append(OUT/'partial-cohorts.jsonl',dict(at_utc=iso(),**partial));break
   assert collections.Counter(r['learner_seat'] for r in cohort_records)=={0:8,1:8}
   assert all(r['learner_hash']==cohort_hash and r['opponent_hash']==anchor_hash for r in cohort_records)
   assert all(r['feature_semantics']=='starting-order-v1' and r['seat']==r['physical_seat'] for r in rows)
   t=time.monotonic();torch.save(rows,PRIVATE/'last-rollout.pt');legacy.save(PRIVATE/'phase-input.pt',model,opt,execution['step'],extra=dict(phase=phase,execution=execution,cohort_sha256=sha(PRIVATE/'last-rollout.pt')))
   if phase==0:
    import shutil;shutil.copy2(PRIVATE/'last-rollout.pt',PRIVATE/'first-rollout.pt')
   timing['checkpoint_seconds']+=time.monotonic()-t;t=time.monotonic();likelihood=phase_metrics(model,rows)
   write(OUT/f'phase-{phase:04d}-likelihood.json',likelihood)
   if likelihood['max_chosen_logprob_difference']>1e-5 or likelihood['max_old_policy_KL']>1e-6:raise FloatingPointError('Actual CUDA rollout/PPO likelihood mismatch; saved probabilities unchanged')
   timing['diagnostic_seconds']+=time.monotonic()-t;adv=advantages(rows);order=np.random.default_rng(SEED+phase).permutation(len(rows));accepted_start=execution['step'];rollout_stop=None;stop_time=False
   for offset in range(0,len(rows),128):
    if time.monotonic()+90>=limit:stop_time=True;reason='time_limit_between_updates';break
    ii=order[offset:offset+128];execution.update(phase=phase,offset=offset,order=order);attempt=f'{phase:04d}-{offset:06d}'
    exclusive(OUT/'update-intents'/(attempt+'.json'),dict(at_utc=iso(),phase=phase,offset=offset,accepted_step_before=execution['step'],policy_hash=state_hash(model)))
    t=time.monotonic();old_diag=METERS['phase_diagnostic_seconds'];old_step=METERS['optimizer_seconds'];result=transaction(model,opt,[rows[i] for i in ii],adv[ii],execution,'rolling-rejection');total=time.monotonic()-t;diag=METERS['phase_diagnostic_seconds']-old_diag;step_seconds=METERS['optimizer_seconds']-old_step
    timing['update_seconds']+=step_seconds;timing['diagnostic_seconds']+=diag;timing['transaction_snapshot_rollback_overhead_seconds']+=max(0,total-diag-step_seconds)
    append(OUT/'updates.jsonl',dict(at_utc=iso(),phase=phase,offset=offset,rows=len(ii),accepted_step=execution['step'],**result))
    if not result['accepted']:
     rollout_stop=result['reason'];stops[rollout_stop]+=1;rejected+=int(rollout_stop.startswith('phase_post'));break
    t=time.monotonic();checksum=legacy.save(PRIVATE/'latest.pt',model,opt,execution['step'],extra=dict(execution=execution,cohort_sha256=sha(PRIVATE/'last-rollout.pt')));timing['checkpoint_seconds']+=time.monotonic()-t
    write(OUT/'update-commits'/f"{execution['step']:06d}.json",dict(at_utc=iso(),attempt=attempt,step=execution['step'],checkpoint_sha256=checksum))
   accepted=execution['step']-accepted_start;trained_BO3+=16;stop_zero=zero.record(accepted)
   assert batch.digest(dict(model.named_buffers()))==buffers and batch.digest({k:p for k,p in model.named_parameters() if not p.requires_grad})==disabled
   if time.monotonic()-lastdiag>=3600:
    t=time.monotonic();last_diagnostic=fixed_drift(model,fixed);timing['diagnostic_seconds']+=time.monotonic()-t;lastdiag=time.monotonic()
    append(OUT/'initialization-diagnostics.jsonl',dict(at_utc=iso(),stage='hourly',phase=phase,**last_diagnostic));t=time.monotonic();legacy.save(PRIVATE/f'hourly-{int((time.monotonic()-begin)//3600):02d}.pt',model,opt,execution['step'],extra=dict(execution=execution));timing['checkpoint_seconds']+=time.monotonic()-t
   lastphase=time.monotonic()-phase_start
   append(OUT/'cohorts.jsonl',dict(at_utc=iso(),phase=phase,BO3=16,decisions=len(rows),accepted_updates=accepted,accepted_total=execution['step'],rollout_early_stop=rollout_stop,zero_update_streak=zero.consecutive,unused_optimization_discarded=bool(rollout_stop),elapsed_seconds=lastphase,phase_policy_hash=cohort_hash,training_WDL=dict(collections.Counter('W' if r['result']==r['learner_seat'] else 'D' if r['result']==2 else 'L' for r in cohort_records))))
   progress('TRAINING',cohort=phase,completed_BO3=completed,accepted_updates=execution['step'],zero_cohorts=zero.consecutive,remaining_training_seconds=max(0,limit-time.monotonic()))
   phase+=1
   if stop_zero:reason='three_consecutive_fresh_cohorts_zero_accepted_updates';break
   if stop_time:break
   if sum(p.stat().st_size for p in PRIVATE.glob('*.pt'))>2*1024**3:reason='registered_recovery_storage_cap';break
 except BaseException as e:
  fatal=f'{type(e).__name__}: {e}';reason='correctness_failure';write(OUT/'training-failure.json',dict(at_utc=iso(),error=fatal,traceback=traceback.format_exc(),no_restart=True))
  # Recovery/export only, never resume learning after a failure.
  model,opt,s=batch.restore_checkpoint(PRIVATE/'latest.pt','cuda',rng=True);execution.clear();execution.update(s.get('extra',{}).get('execution',dict(step=s['step'])));execution['step']=s['step']
 finally:
  t=time.monotonic();last_diagnostic=fixed_drift(model,fixed);timing['diagnostic_seconds']+=time.monotonic()-t;append(OUT/'initialization-diagnostics.jsonl',dict(at_utc=iso(),stage='endpoint',**last_diagnostic))
  final_hash=state_hash(model);changed=final_hash!=INITIAL_TENSOR;evaluate=fatal is None and execution['step']>0 and changed
  t=time.monotonic();checksum=legacy.save(PRIVATE/'final.pt',model,opt,execution['step'],extra=dict(execution=execution,stop_reason=reason));timing['checkpoint_seconds']+=time.monotonic()-t
  write(OUT/'training-completion.json',dict(status='FAILED' if fatal else 'COMPLETE',stop_reason=reason,error=fatal,completed_BO3=completed,optimized_cohort_BO3=trained_BO3,completed_cohorts=phase,incomplete_final_cohort=partial,learner_decisions=decisions,accepted_updates=execution['step'],rejected_updates=rejected,rollout_early_stops=dict(stops),zero_cohort_streak=zero.consecutive,initial_tensor_hash=INITIAL_TENSOR,final_tensor_hash=final_hash,model_tensors_changed=changed,evaluate=evaluate,skip_evaluation='correctness failure' if fatal else 'no retained model change' if not evaluate else None,checkpoint_sha256=checksum,elapsed_seconds=time.monotonic()-begin,preflight_seconds=read(OUT/'gpu-preflight.json')['elapsed_seconds'],timing=dict(timing),remote_CPU_core_hours=(resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime)/3600,max_GPU_memory_bytes=torch.cuda.max_memory_allocated(),devices_used=[0],allocated_devices=2,learning_rate=1e-5,scheduler=None,initialization_KL_diagnostic_only=True,final_fixed_diagnostic=last_diagnostic,static_text_tables_unchanged=True))
 if fatal:raise RuntimeError(fatal)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  write(OUT/'training-failure.json',dict(at_utc=iso(),error=str(e),traceback=traceback.format_exc(),automatic_restart=False));raise
