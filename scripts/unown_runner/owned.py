"""Inner-process adapter for an ordinary persistent owner, never a GPU launcher."""
import fcntl,os,signal,subprocess,sys,threading,time
from pathlib import Path
from .context import *

def run_owned(context_path,registration,authorization,*,device,session_started,absolute_deadline,resume=False,process_factory=subprocess.Popen):
 context=RunContext.from_spec(read(context_path));reg,auth=context.authorize(registration,authorization)
 if reg.get('device')!=device or reg.get('implementation')!=implementation_hashes():raise PermissionError('Owner configuration/source is not registered')
 if context.mode!='learning' and process_factory is subprocess.Popen:raise PermissionError('Software verification cannot launch a training subprocess')
 absolute=min(parse(absolute_deadline),parse(auth['expires_utc']))
 stop_at=min(float(session_started)+context.limits['session_seconds']-context.limits['export_reserve_seconds'],time.monotonic()+max(0,(absolute-now()).total_seconds()-context.limits['export_reserve_seconds']))
 if time.monotonic()>=stop_at:raise TimeoutError('No registered execution time remains')
 context.out.mkdir(parents=True,exist_ok=True);lock=(context.out/'owner.lock').open('a+b')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 intent=context.out/'owned-process-intent.json'
 if intent.exists() and not resume:lock.close();raise RuntimeError('Existing owner intent; explicit recovery required, no duplicate launch')
 if not intent.exists():exclusive(intent,dict(at_utc=iso(),context_sha256=sha(context_path),registration_sha256=sha(registration),owner_pid=os.getpid(),automatic_followup=False))
 command=[sys.executable,'-m','unown_runner.runner','--context',str(Path(context_path).resolve()),'--registration',str(Path(registration).resolve()),'--authorization',str(Path(authorization).resolve()),'--device',device]+(['--resume'] if resume else [])
 env=os.environ.copy();env.update(UNOWN_JOB_START=str(session_started),UNOWN_ABSOLUTE_DEADLINE=absolute.isoformat(),UNOWN_OWNER_PID=str(os.getpid()))
 env['PYTHONPATH']=str(Path(__file__).parent.parent)+os.pathsep+env.get('PYTHONPATH','')
 child=None;thread=None;overflow=threading.Event();written=0;omitted=0;status='FAILED';failure=None
 try:
  child=process_factory(command,cwd=context.root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
  def pump():
   nonlocal written,omitted
   with (context.out/'owned-runner.log').open('ab') as stream:
    written=stream.tell()
    while True:
     data=(child.stdout.read1(65536) if hasattr(child.stdout,'read1') else child.stdout.read(65536))
     if not data:break
     available=max(0,context.limits['log_bytes']-written);keep=data[:available];stream.write(keep);stream.flush();written+=len(keep)
     if len(data)>available:omitted+=len(data)-available;overflow.set()
  thread=threading.Thread(target=pump,daemon=True);thread.start()
  while child.poll() is None:
   if overflow.is_set():raise RuntimeError('Registered owner log cap exceeded; stopping child')
   if time.monotonic()>=stop_at or now()>=absolute:raise TimeoutError('Independent owner deadline')
   time.sleep(.1)
  thread.join(timeout=5)
  if overflow.is_set():raise RuntimeError('Registered owner log cap exceeded')
  if child.returncode:raise RuntimeError('Runner exited '+str(child.returncode))
  completion=context.out/'runner-completion.json'
  if completion.exists():
   status=read(completion)['status']
   if status not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('Runner reported '+status)
  elif context.mode=='learning':raise RuntimeError('Missing runner completion record')
  else:status='COMPLETE' # Explicit software process mock only.
 except BaseException as error:
  failure=type(error).__name__+': '+str(error)
  if child and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   except ProcessLookupError:pass
  raise
 finally:
  if thread:thread.join(timeout=5)
  write(context.out/'owned-process-receipt.json',dict(at_utc=iso(),status=status,error=failure,command=command,bytes_written=written,omitted_log_bytes=omitted,owner_pid=os.getpid(),resume=resume,no_GPU_job_created=True,no_automatic_followup=True,software_mock=context.mode=='software_test'))
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 return read(context.out/'owned-process-receipt.json')
