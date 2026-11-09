"""Persistent synchronous collector/update lifecycle, with durable run ownership."""
import copy,fcntl,hashlib,json,os,resource,time,traceback
from pathlib import Path
import numpy as np
import torch
from .context import *
from .runtime import Runtime,NativeCollector

class RecoveryConflict(RuntimeError):pass

class Runner:
 def __init__(self,context,registration,authorization,*,device,collector=None,runtime_factory=Runtime):
  self.context=context;self.reg,self.auth=context.authorize(registration,authorization);self.device=device
  if self.reg.get('device')!=device:raise PermissionError('Device is not the registered execution device')
  if context.mode=='software_test' and device!='cpu':raise PermissionError('Software verification is CPU-only')
  if context.mode=='software_test' and (collector is None or not getattr(collector,'software_only',False)):raise PermissionError('Software tests cannot instantiate native collection')
  if context.mode=='learning' and collector is not None:raise PermissionError('Learning mode cannot inject a test/replay collector')
  for key in ('start_checkpoint','opponent_checkpoint','text_tables','identity_directory','current_tables','original_tables','released_submission','original_sdk','environment'):
   if key not in context.assets:raise ValueError('Missing current-run input: '+key)
  # The future registration must pin every private runtime input, not merely code.
  required=set()
  for value in context.assets.values():
   path=Path(value).resolve()
   if path.is_dir():required.update(str(p.resolve()) for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
   else:required.add(str(path))
  pinned=self.reg.get('inputs',{})
  if not required.issubset(pinned):raise ValueError('Registration does not pin all runtime assets')
  for name,digest in pinned.items():
   if sha(name)!=digest:raise ValueError('Input changed: '+name)
  schedule=self.reg.get('cohorts',[])
  if len(schedule)!=context.limits['max_cohorts']:raise ValueError('Missing exact registered cohort schedule')
  ids=[]
  for jobs in schedule:
   if len(jobs)!=16 or sorted(j['learner_seat'] for j in jobs)!=[0]*8+[1]*8:raise ValueError('Require16 jobs with8/8 chooser assignments')
   ids.extend(j['id'] for j in jobs)
  if len(ids)!=len(set(ids)):raise ValueError('Duplicate registered match IDs')
  self.source_hashes=implementation_hashes()
  if self.reg.get('implementation')!=self.source_hashes:raise PermissionError('Registered implementation hashes differ')
  self.rt=runtime_factory(context);self.collector=collector or NativeCollector(self.rt,authorized=True)
  self.model=self.opt=self.anchor=None;self.execution=None;self.active=None;self.reference=None;self.engine=None
  self.journal=context.out/'journal.jsonl';self.seq=0;self.intents=set();self.lock=None;self.state='NEW';self.last_disk=0.
  self.started=float(os.environ.get('UNOWN_JOB_START',os.environ.get('REUSE_JOB_START',self.rt.clock())));self.deadline=self.started+context.limits['session_seconds']-context.limits['export_reserve_seconds']
  external=os.environ.get('UNOWN_ABSOLUTE_DEADLINE') or os.environ.get('REUSE_ABSOLUTE_DEADLINE')
  if context.mode=='learning' and not external:raise PermissionError('Owner-enforced absolute deadline required')
  if context.mode=='learning' and not (os.environ.get('UNOWN_JOB_START') or os.environ.get('REUSE_JOB_START')):raise PermissionError('Owner session-start clock required')
  self.absolute=min(parse(external),parse(self.auth['expires_utc'])) if external else parse(self.auth['expires_utc'])
  self.latest=context.private/'checkpoints/latest.pt';self.phase=0
 def event(self,kind,**fields):
  self.seq+=1;append(self.journal,dict(fields,seq=self.seq,kind=kind,at_utc=iso()))
 def safety(self):
  if self.rt.clock()>=self.deadline or now()>=self.absolute:raise TimeoutError('Registered run deadline; no automatic extension')
  if torch.device(self.device).type=='cuda':
   if torch.cuda.max_memory_allocated()>10*1024**3 or torch.cuda.max_memory_reserved()>12*1024**3:raise MemoryError('Validated GPU memory cap')
  # Linux RSS includes parent and all collector workers, with shared pages counted per process.
  pids=[os.getpid()]+([p.pid for p in self.collector.pool.workers] if getattr(self.collector,'pool',None) else [])
  total=0
  for pid in pids:
   try:total+=int(Path(f'/proc/{pid}/stat').read_text().split()[23])*os.sysconf('SC_PAGE_SIZE')
   except FileNotFoundError:pass
  if total>24*1024**3:raise MemoryError('Validated24GiB summed RSS cap')
  if self.rt.clock()-self.last_disk>5:
   self.last_disk=self.rt.clock()
   if sum(p.stat().st_size for p in self.context.private.rglob('*') if p.is_file())>self.context.limits['storage_bytes']:raise MemoryError('Registered private storage cap')
   if sum(p.stat().st_size for p in self.context.out.rglob('*') if p.is_file())>self.context.limits['log_bytes']:raise MemoryError('Registered log cap; no silent evidence truncation')
 def acquire(self):
  self.context.out.mkdir(parents=True,exist_ok=True);self.context.private.mkdir(parents=True,exist_ok=True)
  for name in ('cohorts','checkpoints'):
   if (self.context.private/name).resolve()!=self.context.private.resolve()/name:raise RecoveryConflict('Mutable recovery directory is a symlink')
  self.lock=(self.context.out/'runner.lock').open('a+b')
  try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:self.lock.close();self.lock=None;raise RecoveryConflict('Another persistent owner holds this run')
 def manifest(self):
  value=dict(context=self.context.spec(),implementation=self.source_hashes,registration=self.reg,
   runtime=dict(torch=str(torch.__version__),numpy=np.__version__,device=self.device,TF32=False,flash_SDP=False,memory_efficient_SDP=False,math_SDP=True),
   controller_pid=int(os.environ.get('UNOWN_OWNER_PID',os.getpid())),runner_pid=os.getpid(),software_verification_only=self.context.mode=='software_test',compilation=self.context.execution.compiled,
   no_fallback=True,collector='validated revised concurrent16',decks=self.reg['decks'])
  path=self.context.out/'manifest.json'
  if path.exists():
   old=read(path)
   for key in ('context','implementation','registration','runtime'):
    if old[key]!=json.loads(json.dumps(value[key])):raise RecoveryConflict('Manifest changed: '+key)
  else:write(path,value)
  self.manifest_sha=sha(path)
 def save_cohort(self,cohort):
  self.rt.epochs.verify_cohort(cohort);rows=cohort['rows']
  if not rows or len(rows)>20000 or sum(r['packet'].row.nbytes for r in rows)>512*1024**2:raise MemoryError('Cohort bound; no truncation')
  path=self.context.private/'cohorts'/f'{self.phase:06d}.pt';path.parent.mkdir(exist_ok=True)
  with path.open('xb') as stream:torch.save(cohort,stream);stream.flush();os.fsync(stream.fileno())
  return dict(path=str(path.relative_to(self.context.private)),sha256=sha(path),bytes=path.stat().st_size,immutable_digest=cohort['immutable_digest'])
 def read_cohort(self,ref):
  path=(self.context.private/ref['path']).resolve()
  if not path.is_relative_to((self.context.private/'cohorts').resolve()) or not path.is_file():raise RecoveryConflict('Missing/invalid cohort reference')
  if path.stat().st_size!=ref['bytes'] or sha(path)!=ref['sha256']:raise RecoveryConflict('Corrupt immutable cohort')
  cohort=torch.load(path,map_location='cpu',weights_only=False);self.rt.epochs.verify_cohort(cohort)
  if cohort['immutable_digest']!=ref['immutable_digest']:raise RecoveryConflict('Cohort digest mismatch')
  return cohort
 def commit(self):
  if self.active is not None:self.rt.epochs.verify_cohort(self.active)
  raw=self.journal.read_bytes() if self.journal.exists() else b''
  self.rt.legacy.save(self.latest,self.model,self.opt,self.execution['step'],extra=dict(
   execution=copy.deepcopy(self.execution),cohort_reference=self.reference,runner_phase=self.phase,runner_state=self.state,
   run_manifest_sha256=self.manifest_sha,journal_bytes=len(raw),journal_sha256=hashlib.sha256(raw).hexdigest(),journal_seq=self.seq,
   software_verification_only=self.context.mode=='software_test',candidate_eligible=False,
   **(self.rt.save_extra(self) if hasattr(self.rt,'save_extra') else {})))
  if getattr(self,'supervisor_contract',None):
   from experiment_supervisor.runner_hook import publish
   publish(self,self.supervisor_contract)
 def restore(self):
  self.model,self.opt,saved=self.rt.restore(self.latest,self.device);extra=saved['extra']
  if extra['run_manifest_sha256']!=self.manifest_sha:raise RecoveryConflict('Recovery manifest mismatch')
  raw=self.journal.read_bytes() if self.journal.exists() else b'';count=extra['journal_bytes']
  if len(raw)<count or hashlib.sha256(raw[:count]).hexdigest()!=extra['journal_sha256']:raise RecoveryConflict('Missing/corrupt durable journal')
  if len(raw)>count:raise RecoveryConflict('Work beyond durable cursor: refusing ambiguous update/match replay')
  self.seq=extra['journal_seq'];self.intents={tuple(x['token']) for x in records(self.journal) if x['kind']=='update-intent'}
  self.execution=extra['execution'];self.phase=extra['runner_phase'];self.state=extra['runner_state'];self.reference=extra['cohort_reference']
  if self.state in ('FAILED','COLLECTING'):raise RecoveryConflict('Interrupted/failed collection cannot be silently replayed')
  self.active=self.read_cohort(self.reference) if self.reference else None
 def validate_cohort(self):
  label=f'phase-{self.phase:06d}'
  if self.active['collection_policy_hash']!=self.rt.state_hash(self.model):raise RecoveryConflict('Collection policy identity changed before validation')
  # Save/recovery reference exists before either correctness checker runs.
  ref=dict(self.reference,path=str((self.context.private/self.reference['path']).relative_to(self.context.root)))
  result=self.rt.checks.collected(self.model,self.active['rows'],label,ref,'revised')
  engine=self.rt.make_engine(self.model,self.opt,self.active,self.phase)
  try:actual=engine.reference_check(self.active['rows'])
  finally:engine.close()
  write(self.context.out/(label+'-learner-reference.json'),actual)
  if actual['status']!='PASS':raise FloatingPointError('Selected learner path reference mismatch')
  self.rt.epochs.verify_cohort(self.active);self.event('validated',collector=result['status'],learner=actual['status'])
 def train_active(self):
  self.engine=self.rt.make_engine(self.model,self.opt,self.active,self.phase)
  def emit(kind,event):
   if kind=='intent':
    token=(self.phase,event['epoch'],event['offset'])
    if token in self.intents:raise RecoveryConflict('Duplicate logical update')
    self.intents.add(token);self.event('update-intent',token=token,**event)
   else:self.event(kind,**event)
   self.safety()
  try:
   return self.rt.epochs.optimize(self.model,self.opt,self.active,self.execution,transact=self.engine.transaction,
    measured=self.engine.measure,emit=emit,commit=self.commit,should_stop=lambda:self.rt.clock()+60>=self.deadline or now()>=self.absolute)
  finally:self.engine.close();self.engine=None;self.rt.batch.clear(self.model)
 def run(self,resume=False):
  try:
   self.acquire()
   self.rt.configure_precision()
   self.manifest()
   if resume:self.restore()
   else:
    if self.latest.exists() or self.journal.exists():raise RecoveryConflict('Existing run: explicit resume required')
    self.model,self.opt,saved=self.rt.restore(self.context.assets['start_checkpoint'],self.device)
    self.execution=dict(step=saved['step'],original_step=saved['step'],phase=0,additional_collected=0,accepted_used_decisions=0,unique_used_decisions=0,meaningful_unique_used=0,meaningful_exposures=0)
    self.state='BOUNDARY';self.commit()
   self.anchor=self.rt.load_model(self.context.assets['opponent_checkpoint'],self.device)
   for p in self.anchor.parameters():p.requires_grad_(False)
   anchor_hash=self.rt.state_hash(self.anchor)
   # Model construction may consume global RNG; restore the committed learner RNG.
   committed=torch.load(self.latest,map_location='cpu',weights_only=False);self.rt.restore_rng(committed,self.device)
   while self.phase<self.context.limits['max_cohorts']:
    self.safety()
    if self.active is None and hasattr(self.rt,'may_start_cohort') and not self.rt.may_start_cohort(self):break
    if self.rt.state_hash(self.anchor)!=anchor_hash:raise RuntimeError('Frozen opponent changed')
    if self.active is None:
     self.state='COLLECTING';self.event('collect-intent',phase=self.phase,learner_hash=self.rt.state_hash(self.model),opponent_hash=anchor_hash)
     self.rt.batch.clear(self.model);self.rt.batch.clear(self.anchor)
     version=self.rt.state_hash(self.model)
     rows,matches,metrics=self.collector.collect(self.reg['cohorts'][self.phase],dict(learner=self.model,opponent=self.anchor),self.deadline,self.safety)
     if self.rt.state_hash(self.model)!=version or self.rt.state_hash(self.anchor)!=anchor_hash:raise RuntimeError('Policy changed within cohort')
     self.active=self.rt.epochs.new_cohort(rows,version);self.reference=self.save_cohort(self.active)
     if hasattr(self.rt,'begin_cohort_update'):self.execution['last_collection_seconds']=metrics['elapsed_seconds']
     expected={j['id']:j['learner_seat'] for j in self.reg['cohorts'][self.phase]}
     if len(matches)!=16 or {m['id'] for m in matches}!=set(expected) or any(m['status']!='DONE' or m['learner_hash']!=version or m['opponent_hash']!=anchor_hash or m['learner_seat']!=expected[m['id']] for m in matches):raise RuntimeError('Incomplete/misrouted cohort')
     self.execution['phase']=self.phase;self.execution['additional_collected']+=len(rows);self.rt.epochs.initialize_cursor(self.execution,len(rows))
     self.state='VALIDATING';self.event('collected',phase=self.phase,version=version,cohort=self.reference,metrics=metrics);self.commit()
    if hasattr(self.rt,'begin_cohort_update'):self.rt.begin_cohort_update(self)
    if self.state=='VALIDATING':self.validate_cohort();self.state='UPDATING';self.commit()
    terminal=self.train_active();self.commit()
    if terminal=='registered_time_endpoint':self.state='TIME_ENDPOINT';self.commit();break
    registered_stop=self.rt.cohort_finished(self,terminal) if hasattr(self.rt,'cohort_finished') else None
    expired=self.reference;self.active=None;self.reference=None;self.phase+=1;self.execution['phase']=self.phase;self.state='BOUNDARY'
    self.event('cohort-boundary',phase=self.phase,stop_reason=terminal,learner_hash=self.rt.state_hash(self.model));self.commit()
    # Only the current run's now-unreferenced bounded rollout is released.
    path=(self.context.private/expired['path']).resolve()
    if not path.is_relative_to((self.context.private/'cohorts').resolve()):raise RecoveryConflict('Invalid retention path')
    path.unlink()
    if registered_stop:self.execution['registered_stop']=registered_stop;break
   self.state='COMPLETE' if self.phase==self.context.limits['max_cohorts'] or self.execution.get('registered_stop') in ('fresh_decision_target','three_zero_actor_cohorts') else 'INCOMPLETE';self.commit()
   write(self.context.out/'runner-completion.json',dict(status=self.state,phase=self.phase,step=self.execution['step'],execution=self.context.execution.learner,software_test=self.context.mode=='software_test',no_automatic_followup=True))
   return self.execution
  except BaseException as error:
   write(self.context.out/'runner-failure.json',dict(status='FAILED',type=type(error).__name__,error=str(error),phase=self.phase,traceback=traceback.format_exc(),no_automatic_replay=True))
   raise
  finally:
   if self.engine:self.engine.close()
   self.collector.close()
   if self.lock:fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None

def main():
 import argparse
 parser=argparse.ArgumentParser(description='Requires a separately registered and explicitly authorized run; never launches GPU jobs itself.')
 parser.add_argument('--context',required=True);parser.add_argument('--registration',required=True);parser.add_argument('--authorization',required=True);parser.add_argument('--device',required=True,choices=['cpu','cuda']);parser.add_argument('--resume',action='store_true')
 args=parser.parse_args();context=RunContext.from_spec(read(args.context))
 runner=Runner(context,args.registration,args.authorization,device=args.device)
 try:runner.run(args.resume)
 finally:runner.rt.close()

if __name__=='__main__':main()
