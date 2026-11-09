"""Windows owner for a single capped, private local offline benchmark."""
from .common import *
import subprocess,traceback,shutil

def main():
 started=now();cpu=time.process_time();exclusive(OUT/'owner-intent.json',dict(pid=os.getpid(),at_utc=iso(),automatic_restart=False));write(OUT/'lifecycle.json',dict(started_at_utc=started.isoformat(),deadline_utc=(started+dt.timedelta(hours=8)).isoformat(),CPU_limit_seconds=28800,workers=4,thread_limit=1));write(OUT/'launch.json',dict(at_utc=iso(),pid=os.getpid(),source_sha256=sha(OUT/'source-manifest.json'),registration_sha256=sha(OUT/'registration.json'),schedule_sha256=sha(OUT/'schedule.json')))
 try:
  command=['wsl','-d','Ubuntu','--cd','/path/to/ptcg-agent','-e','env','PYTHONPATH=scripts:src:.','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','.venv/bin/python','-m','unown_belief_rb.run']
  with (OUT/'runner.log').open('w',encoding='utf-8') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);write(OUT/'active-local.json',dict(pid=proc.pid,at_utc=iso()))
   while proc.poll() is None:
    write(OUT/'owner-meter.json',dict(at_utc=iso(),CPU_seconds=time.process_time()-cpu,pid=os.getpid()))
    if (deadline()-now()).total_seconds()<=60:
     if (OUT/'pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-TERM','--','-'+str(read(OUT/'pid.json')['group'])],capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill()
     raise TimeoutError('Eight UTC elapsed-hour cap')
    time.sleep(.5)
   if proc.returncode:raise RuntimeError('Offline controller exited '+str(proc.returncode)+'; see retained failure and runner.log')
 except BaseException as exc:
  write(OUT/'owner-failure.json',dict(at_utc=iso(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
  if not (OUT/'completion.json').exists():
   write(OUT/'completion.json',dict(at_utc=iso(),state='BUDGET_EXHAUSTED' if isinstance(exc,TimeoutError) else 'FAILED',reason=str(exc),elapsed_seconds=(now()-started).total_seconds(),aggregate_CPU_seconds=time.process_time()-cpu,completed=0,expected_total_replays=208,unresolved_intents=[],private_bytes=0))
   from .report import render
   render(read(OUT/'completion.json')['state'],str(exc))
 finally:
  write(OUT/'owner-completion.json',dict(at_utc=iso(),CPU_seconds=time.process_time()-cpu,elapsed_seconds=(now()-started).total_seconds(),no_keep_awake_or_power_change=True));mirror=Path('C:/path/to/ptcg-agent')/'experiments'/RUN;mirror.mkdir(parents=True,exist_ok=True)
  for name in ('completion.json','owner-completion.json','launch.json','failure.json','owner-failure.json'):
   if (OUT/name).exists():shutil.copy2(OUT/name,mirror/name)
if __name__=='__main__':main()
