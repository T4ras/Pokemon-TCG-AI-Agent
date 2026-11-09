"""Exactly six fixed passes worth of updates; no early stopping or test access."""
from .common import *
import torch,numpy as np,collections,signal,traceback,copy
from unown_belief_full_training.models import make_neural,neural_loss
from unown_belief_deck_audit.diagnostics import digest
from .metrics import evaluate

def atomic_checkpoint(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()));torch.save(value,tmp);tmp.replace(path)
def main(job):
 torch.set_num_threads(3);torch.set_num_interop_threads(1);key=job['key'];root=PRIVATE/'models'/key;root.mkdir(parents=True,exist_ok=True);exclusive(OUT/'fit-intents'/(key+'.json'),dict(at_utc=iso(),pid=os.getpid(),job=job));updates=0;started=time.process_time();wall=time.monotonic();state='INCOMPLETE';reason='';exposure=collections.Counter();strata=collections.Counter();curve=[];block_loss=0.;model=None;optimizer=None
 def check():guard()
 def abort(*_):raise RuntimeError('Controller cancelled at resource cap or interruption')
 signal.signal(signal.SIGTERM,abort)
 def save(path):
  atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),job=job,updates=updates,next_schedule_index=updates,schedule_sha256=sha(OUT/'schedules.json'),initial_checkpoint_sha256=sha(PRIVATE/'initial'/f's{job["seed"]}.pt'),exposure_counts=dict(exposure),stratum_exposure=dict(strata)))
 try:
  check();initial=torch.load(PRIVATE/'initial'/f's{job["seed"]}.pt',map_location='cpu',weights_only=False);model=make_neural('MLP',job['seed']);model.load_state_dict(initial['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);optimizer.load_state_dict(copy.deepcopy(initial['optimizer']));torch.set_rng_state(initial['torch_rng']);assert digest(model.state_dict())==initial['tensor_sha256'];assert optimizer.state_dict()['state']=={}
  sequence=read(OUT/'schedules.json')[key];assert len(sequence)==UPDATES;train={x['id']:x for x in entries('training')};assert set(sequence)<=set(train);assert len(train)==2334
  write(OUT/'fit-startup'/(key+'.json'),dict(at_utc=iso(),passed=True,initial_tensor_sha256=digest(model.state_dict()),initial_checkpoint_sha256=sha(PRIVATE/'initial'/f's{job["seed"]}.pt'),empty_optimizer_state=True,updates=0,eligible_training_BO3=2334,sampling_seed=job['sampling_seed'],sampling_seed_separate_from_initialization=job['sampling_seed']!=job['seed']))
  for checkpoint in CHECKPOINTS:
   while updates<checkpoint:
    check();ident=sequence[updates];item=train[ident];f,t=load(ident);model.train();optimizer.zero_grad(set_to_none=True);scalar=item['neural_loss_weight'] if job['arm']=='control' else 1.;loss=neural_loss(model,f,t)*scalar
    if not torch.isfinite(loss):raise InvariantFailure('Nonfinite supervised loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise InvariantFailure('Nonfinite gradient')
    optimizer.step();updates+=1;block_loss+=float(loss.detach());exposure[ident]+=1;strata[item['opponent_list']+':'+str(item['chooser'])]+=1
    if updates%100==0:
     save(root/'recovery.pt');write(OUT/'active-fit.json',dict(at_utc=iso(),job=job,updates=updates,target=UPDATES,phase='training',unique_training_BO3=len(exposure),CPU_seconds=time.process_time()-started))
   save(root/f'checkpoint-{updates:05d}.pt');write(OUT/'active-fit.json',dict(at_utc=iso(),job=job,updates=updates,target=UPDATES,phase='validation',CPU_seconds=time.process_time()-started))
   val=evaluate(model,'validation',check,initial=updates==0,save_baseline=updates==0 and key=='control-s11');row=dict(updates=updates,online_training_combined_mean=block_loss/2334 if updates else None,**val);block_loss=0.;curve.append(row);write(OUT/'validation'/f'{key}-u{updates:05d}.json',row)
   if 'A_records' in row:write(OUT/'baseline-validation.json',dict(summary=row['A_summary'],records=row['A_records']))
   if updates in (0,UPDATES):
    write(OUT/'active-fit.json',dict(at_utc=iso(),job=job,updates=updates,target=UPDATES,phase='stationary_training',CPU_seconds=time.process_time()-started));station=evaluate(model,'training',check,initial=updates==0);write(OUT/'stationary-training'/f'{key}-u{updates:05d}.json',station)
   write(OUT/'curves'/(key+'.json'),[{k:v for k,v in x.items() if k not in ('records','A_records')} for x in curve]);save(root/'recovery.pt')
  assert updates==UPDATES and sum(exposure.values())==UPDATES
  if job['arm']=='control':assert len(exposure)==2334 and set(exposure.values())=={6}
  state='COMPLETE';reason='Exactly14004updates and allseven fixed validation checkpoints complete.'
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);write(OUT/'failures'/(key+'.json'),dict(at_utc=iso(),updates=updates,reason=reason,traceback=traceback.format_exc()))
  if model is not None and optimizer is not None:save(root/'failure-recovery.pt')
 finally:
  counts={x['id']:exposure[x['id']] for x in entries('training')};write(OUT/'exposures'/(key+'.json'),dict(job=job,total_updates=updates,eligible_training_BO3=2334,unique_histories=len(exposure),exposure_counts=counts,deck_chooser_counts=dict(strata),sampling_schedule_sha256=sha(OUT/'schedules.json')));write(OUT/'fit-results'/(key+'.json'),dict(at_utc=iso(),state=state,reason=reason,job=job,updates=updates,CPU_seconds=time.process_time()-started,elapsed_seconds=time.monotonic()-wall,unique_histories=len(exposure),model_directory=str(root.relative_to(R)),checkpoints=[dict(updates=int(p.stem.split('-')[1]),path=str(p.relative_to(R)),sha256=sha(p)) for p in sorted(root.glob('checkpoint-*.pt'))],test_access=False))
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':
 import sys
 main(read(OUT/'jobs.json')[int(sys.argv[1])])
