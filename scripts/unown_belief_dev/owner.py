"""Windows owner with read-only awake telemetry and fresh24-hour deadline."""
from .common import *
import subprocess,threading,types,traceback
WSL='/path/to/ptcg-agent';ENV=WSL+'/artifacts/astra-stage-d-repair-001/torch211:'+WSL+'/scripts:'+WSL+'/src:'+WSL

def telemetry(stop):
 m=types.ModuleType('belief_dev_clock');m.__dict__.update(dict(OUT=OUT,read=read,write=write,append=append,iso=iso,now=now,parse=lambda x:dt.datetime.fromisoformat(x),time=time,os=os));source=R/'scripts/unown_runner/units/clock.py';exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),m.__dict__);m.telemetry(stop)

def main():
 exclusive(OUT/'owner-intent.json',dict(at_utc=iso(),pid=os.getpid(),automatic_restart=False));write(OUT/'launch.json',dict(at_utc=iso(),pid=os.getpid(),new_native_limit=290,inherited_native=6,held_out=0,deadline_utc=deadline().isoformat(),workers=4,GPU=0))
 from .report import render
 stop=threading.Event();thread=threading.Thread(target=telemetry,args=(stop,),daemon=True);thread.start();start=now();cpu=time.process_time()
 try:
  assert read(OUT/'native-gate.json')['passed'];render('RUNNING','Initial repairs validated. Collecting only290 unattempted development IDs; offline analysis follows.')
  with (OUT/'runner.log').open('w',encoding='utf-8') as log:
   command=['wsl','-d','Ubuntu','--cd',WSL,'-e','env','PYTHONPATH='+ENV,'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','.venv/bin/python','-m','unown_belief_dev.run'];process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);write(OUT/'active-local.json',dict(at_utc=iso(),pid=process.pid))
   while process.poll() is None:
    if (deadline()-now()).total_seconds()<90:
     if (OUT/'local-pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-TERM','--','-'+str(read(OUT/'local-pid.json')['process_group'])],capture_output=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
     try:process.wait(timeout=10)
     except subprocess.TimeoutExpired:process.kill()
     raise TimeoutError('Fresh24-hour execution deadline; no extension')
    stop.wait(.5)
   if process.returncode:raise RuntimeError('Development controller exited '+str(process.returncode)+'; see runner.log')
 except BaseException as exc:
  write(OUT/'owner-failure.json',dict(at_utc=iso(),error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()))
  if not (OUT/'completion.json').exists():write(OUT/'completion.json',dict(state='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',reason=str(exc),at_utc=iso(),held_out_accessed=False))
 finally:
  stop.set();thread.join(timeout=3);append(OUT/'costs.jsonl',dict(stage='Windows_owner',at_utc=iso(),wall_seconds=(now()-start).total_seconds(),CPU_seconds=time.process_time()-cpu,GPU=0,power_changes=False,sleep_inhibition=False));write(OUT/'owner-completion.json',dict(at_utc=iso(),owner_stopped=True))
  if (OUT/'completion.json').exists():c=read(OUT/'completion.json');render(c['state'],c['reason'],True)
