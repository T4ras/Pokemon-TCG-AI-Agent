"""Opt-in future-run hooks. Historical experiment sources are untouched."""
import copy, pickle, sys, traceback
from pathlib import Path
from .io import *
class ObserverCapture:
 """Wrap observe(obs), keeping the actual legal state BEFORE each transition.

The supplied getter must return only the observer's legal internal state (e.g.
 facts/current/within). Never pass simulator state. Serialization is bounded and
 performed before the observer can mutate its input. Unsupported state fails
 capture explicitly; synthetic data is never labelled as actual evidence.
 """
 def __init__(self,observer,folder,get_state,*,max_bytes=8*1024**2,max_failures=8):
  self.observer=observer;self.folder=Path(folder);self.get_state=get_state;self.max_bytes=max_bytes;self.max_failures=max_failures;self.sequence=0
 def __getattr__(self,name):return getattr(self.observer,name)
 def reset_match(self):
  self.last_raw=None;self.last_capture_error=None
  return self.observer.reset_match()
 def observe(self,legal_observation):
  self.sequence+=1;raw=None;capture_error=None
  try:
   raw=pickle.dumps({'pre_event_state':self.get_state(self.observer),'ordered_input':[legal_observation],'sequence':self.sequence,'actual_capture':True,'legal_only':True},protocol=5)
   if len(raw)>self.max_bytes:raw=None;capture_error='Pre-event capture exceeds bound; actual reconstruction unavailable'
  except Exception as e:capture_error=type(e).__name__
  self.last_raw=raw;self.last_capture_error=capture_error
  try:return self.observer.observe(legal_observation)
  except BaseException as error:self.persist(error,'observe');raise
 def readout(self):
  try:return self.observer.readout()
  except BaseException as error:self.persist(error,'readout');raise
 def persist(self,error,stage):
  raw=getattr(self,'last_raw',None);self.folder.mkdir(parents=True,exist_ok=True)
  with Lock(self.folder/'capture.lock'):
   counts=read(self.folder/'counts.json') if (self.folder/'counts.json').exists() else {'failures':0,'retained':0}
   counts['failures']+=1
   if counts['retained']<self.max_failures:
    name=f'failure-{counts["failures"]:06d}';counts['retained']+=1
    if raw is not None:
     p=self.folder/(name+'.pickle')
     with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    atomic(self.folder/(name+'.json'),{'sequence':self.sequence,'stage':stage,'actual_pre_event_capture':raw is not None,'capture_error':getattr(self,'last_capture_error',None),'bytes':len(raw or b''),'sha256':hashlib.sha256(raw).hexdigest() if raw else None,'error':scrub(error),'traceback':scrub(traceback.format_exc()),'diagnostic_only':True})
   atomic(self.folder/'counts.json',counts)

def replay_observer(capture,*,expected_sha256,factory,restore_state,stage):
 # Trusted private pickles only; verify against independently retained receipt.
 if sha(capture)!=expected_sha256:raise ValueError('Untrusted observer capture')
 record=pickle.loads(Path(capture).read_bytes())
 if not record['actual_capture'] or not record['legal_only']:raise ValueError('Not an actual legal pre-event capture')
 observer=factory();restore_state(observer,record['pre_event_state'])
 for obs in record['ordered_input']:observer.observe(obs)
 if stage=='readout':return observer.readout()
 return observer

def fatal_export(stop_accepting,close_workers,verified_reference,export,folder,error):
 """Use from the remote top-level except/finally. Export before process exit.

`export` is the existing pinned owner export callback, not an agent command.
Returns a nonzero exit code; caller must raise SystemExit(result). No GPU wait
for an agent and no provider terminate call. Abrupt provider kill cannot run it.
 """
 from .boundary import verify
 failure={'at_utc':utc(),'error':scrub(error),'traceback':scrub(traceback.format_exc()),'new_work_stopped':True}
 try:stop_accepting()
 except Exception as e:failure['stop_error']=scrub(e);failure['new_work_stopped']=False
 try:
  close_workers();failure['boundary']=verify(verified_reference);failure['boundary_reference']=verified_reference
 except Exception as e:failure['preservation_error']=scrub(e)
 atomic(Path(folder)/'fatal.json',failure)
 try:export();atomic(Path(folder)/'export-complete.json',{'at_utc':utc(),'exit_required':True})
 except Exception as e:atomic(Path(folder)/'export-failed.json',{'at_utc':utc(),'error':scrub(e),'exit_required':True})
 return 1
