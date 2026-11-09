"""Single Windows owner: persistent WSL execution, no automatic restart."""
from .common import *
import subprocess,traceback,shutil,threading,types

def clock_module():
 m=types.ModuleType('belief_learning_awake_clock');m.__dict__.update(dict(OUT=OUT,read=read,write=write,append=append,iso=iso,now=now,parse=lambda x:dt.datetime.fromisoformat(x),time=time,os=os));source=R/'scripts/unown_runner/units/clock.py';exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),m.__dict__);return m

def start_clock():
 m=clock_module()
 # A synchronous valid sample exists before the WSL process can dispatch work.
 import ctypes
 ctypes.windll.kernel32.GetTickCount64.restype=ctypes.c_ulonglong
 write(OUT/'power-clock.json',dict(at_utc=iso(),awake_seconds=m.awake_windows(),uptime_seconds=ctypes.windll.kernel32.GetTickCount64()/1000,power=m.power_windows(),controller_pid=os.getpid()))
 stop=threading.Event();thread=threading.Thread(target=m.telemetry,args=(stop,),daemon=True);thread.start()
 return stop,thread

def main():
 for name in ('STOP.json','completion.json','runner-intent.json','owner-intent.json'):
  if (OUT/name).exists():raise RuntimeError('Refuse stale operational state: '+name)
 proc=None
 start=now();cpu=time.process_time();clock_stop=clock_thread=None;exclusive(OUT/'owner-intent.json',dict(at_utc=iso(),pid=os.getpid(),automatic_restart=False));write(OUT/'lifecycle.json',dict(started_at_utc=read(OUT/'continuation-window.json')['started_at_utc'],resumed_at_utc=start.isoformat(),deadline_utc=read(OUT/'continuation-window.json')['deadline_utc'],CPU_cap_seconds=CPU_CAP,fit_CPU_cap_seconds=FIT_CAP,workers=3,worker_threads=1,total_compute_threads_max=4));write(OUT/'launch.json',dict(at_utc=iso(),owner_pid=os.getpid(),source_manifest_sha256=sha(OUT/'source-manifest.json'),registration_sha256=sha(OUT/'registration.json'),split_manifest_sha256=sha(OUT/'split-manifest.json')))
 try:
  clock_stop,clock_thread=start_clock()
  sample=read(OUT/'power-clock.json')
  if (now()-dt.datetime.fromisoformat(sample['at_utc'])).total_seconds()>5:raise RuntimeError('Clock telemetry not ready; no child launch')
  command=['wsl','-d','Ubuntu','--cd','/path/to/ptcg-agent','-e','env','PYTHONPATH=scripts:src:.','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','.venv/bin/python','-m','unown_belief_full_training.run']
  with (OUT/'runner.log').open('w',encoding='utf-8') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);write(OUT/'active-local.json',dict(pid=proc.pid,at_utc=iso()))
   while proc.poll() is None:
    write(OUT/'owner-meter.json',dict(at_utc=iso(),CPU_seconds=time.process_time()-cpu,pid=os.getpid()))
    if (deadline()-now()).total_seconds()<60:
     if (OUT/'pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-TERM','--','-'+str(read(OUT/'pid.json')['group'])],capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill()
     raise TimeoutError('Registered24-hour UTC deadline')
    time.sleep(1)
   if proc.returncode:raise RuntimeError('Controller ended incomplete; inspect retained completion/failure, no restart')
 except BaseException as exc:
  if proc is not None and proc.poll() is None:
   if (OUT/'pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-TERM','--','-'+str(read(OUT/'pid.json')['group'])],capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:
    if (OUT/'pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-KILL','--','-'+str(read(OUT/'pid.json')['group'])],capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    proc.kill();proc.wait()
  write(OUT/'owner-failure.json',dict(at_utc=iso(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
  if not (OUT/'completion.json').exists():
   write(OUT/'owner-emergency-completion.json',dict(at_utc=iso(),state='FAILED_OWNER',reason=str(exc),unknown_native_attempts_require_reconciliation=True,automatic_restart=False))
   try:
    from .report_full import render
    render('FAILED_OWNER','Persistent owner detected an interrupted controller: '+str(exc)+'. Native intents must be reconciled; no automatic restart.')
   except BaseException as report_error:write(OUT/'emergency-report-failure.json',dict(error=str(report_error)))
 finally:
  if clock_stop is not None:clock_stop.set();clock_thread.join(timeout=3)
  write(OUT/'owner-completion.json',dict(at_utc=iso(),CPU_seconds=time.process_time()-cpu,elapsed_seconds=(now()-start).total_seconds(),automatic_retry=False,no_permanent_power_change=True));mirror=Path('C:/path/to/ptcg-agent')/'experiments'/RUN;mirror.mkdir(parents=True,exist_ok=True)
  for name in ('completion.json','owner-completion.json','launch.json','failure.json','owner-failure.json','owner-emergency-completion.json'):
   if (OUT/name).exists():shutil.copy2(OUT/name,mirror/name)
if __name__=='__main__':main()
