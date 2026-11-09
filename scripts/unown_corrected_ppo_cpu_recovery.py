"""Windows persistent CPU-only recovery; never uploads or launches GPU jobs."""
from pathlib import Path
import ctypes,datetime as dt,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(r'\\wsl.localhost\Ubuntu\home\taras\projects\ptcg-agent')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'.venv/lib/python3.12/site-packages')]
from unown_corrected_ppo_common import *
WIN=Path(r'C:\path\to\ptcg-agent')
REC=OUT/'cpu-evaluation-recovery';WSL='/path/to/ptcg-agent'
EXPECTED='96a0a8d44e5439f437c5ff41fcc988e418ebecde276a7d7baa13055581581fd2'

def remaining():return max(0,(parse(read(REC/'registration.json')['deadline_utc'])-now()).total_seconds())
def mirror():
 shutil.copytree(OUT,WIN/'experiments'/RUN,dirs_exist_ok=True)
 shutil.copy2(REPORT,WIN/'reports'/REPORT.name);shutil.copy2(ROOT/'docs/STATUS.md',WIN/'docs/STATUS.md')
def status(state,detail):
 write(REC/'progress.json',dict(at_utc=iso(),state=state,detail=detail,pid=os.getpid()))
 marker='## Corrected Hydrapple CPU evaluation recovery'
 path=ROOT/'docs/STATUS.md';text=path.read_text();start=text.find(marker)
 if start>=0:
  end=text.find('\n## ',start+len(marker));text=text[:start]+(text[end+1:] if end>=0 else '')
 path.write_text(marker+'\n\n'+iso()+': **'+state+'**. '+detail+' No new GPU job/training/upload. [Report](../reports/astra-unown-corrected-hydrapple-ppo-6h.md).\n\n'+text)
 REPORT.write_text('# Corrected Hydrapple PPO six-hour experiment\n\nState: **'+state+'**. '+detail+'\n\nOriginal frozen512-BO3 evaluation:256 per checkpoint,128/128 chooser rights. User authorized CPU recovery on [removed]; new four-hour CPU-only ceiling includes retrieval and analysis. Historical reports and failed download preserved in cpu-evaluation-recovery/preserved-local.\n')
 mirror()

def download():
 from hydrapple_submission_network import api
 import requests
 from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
 send=requests.Session.send
 def bounded(self,request,**kw):kw['timeout']=(8,25);return send(self,request,**kw)
 requests.Session.send=bounded
 a=api();identity=read(OUT/'gpu-launch.json');assert identity['version']==1
 provider=a.kernels_status(identity['kernel']).to_dict();assert provider.get('status','').upper()=='COMPLETE',provider
 files={}
 with a.build_kaggle_client() as k:
  q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=identity['kernel'].split('/');q.version_label='v1';q.page_size=100
  result=k.kernels.kernels_api_client.list_kernel_session_output(q)
  if result.log:(PRIVATE/'recovery-provider.log').write_text(result.log)
  for f in result.files or []:
   name=Path(f.file_name).name
   if name not in ['pilot-receipt.json','pilot-results.tar.gz']:continue
   target=PRIVATE/'remote-download'/name;target.parent.mkdir(exist_ok=True)
   if name=='pilot-results.tar.gz' and target.exists() and sha(target)==EXPECTED:files[name]=dict(path=str(target),sha256=EXPECTED,bytes=target.stat().st_size);continue
   tmp=target.with_suffix(target.suffix+'.recovery-partial')
   with requests.get(f.url,stream=True) as response:
    response.raise_for_status()
    with tmp.open('wb') as out:
     for chunk in response.iter_content(1024*1024):
      if remaining()<120:raise TimeoutError('CPU recovery deadline during retrieval')
      out.write(chunk)
   if name=='pilot-results.tar.gz':assert sha(tmp)==EXPECTED,'Export archive checksum mismatch'
   tmp.replace(target);files[name]=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size)
 receipt=read(PRIVATE/'remote-download/pilot-receipt.json');assert receipt['state']=='COMPLETE' and receipt['archive_sha256']==EXPECTED
 assert set(files)=={'pilot-results.tar.gz','pilot-receipt.json'}
 write(REC/'download.json',dict(at_utc=iso(),verified_archive_sha256=EXPECTED,files=files,receipt=receipt,provider=provider,GPU_launches=0))

def extract():
 import tarfile
 receipt=read(PRIVATE/'remote-download/pilot-receipt.json');archive=PRIVATE/'remote-download/pilot-results.tar.gz';assert sha(archive)==EXPECTED==receipt['archive_sha256']
 allowed={'progress.json','effect-coverage.json','corrected-match-checks.jsonl'}
 with tarfile.open(archive) as t:
  members=t.getmembers();names=set()
  for m in members:
   target=(ROOT/m.name).resolve();assert m.isfile() and not m.issym() and not m.islnk()
   assert target.is_relative_to(OUT.resolve()) or target.is_relative_to(PRIVATE.resolve())
   assert m.name in receipt['files'] and m.name not in names;names.add(m.name)
   expected=receipt['files'][m.name]['sha256']
   if target.exists() and sha(target)!=expected:assert target.name in allowed,'Immutable local conflict: '+m.name
  assert names==set(receipt['files'])
  for m in members:
   target=ROOT/m.name;expected=receipt['files'][m.name]['sha256']
   if target.exists() and sha(target)==expected:continue
   raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==expected
   if target.exists():
    saved=REC/'preserved-local'/m.name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,saved)
   target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.recovery-tmp');tmp.write_bytes(raw);tmp.replace(target)
 for name,info in receipt['files'].items():assert sha(ROOT/name)==info['sha256'],name
 training=read(OUT/'training-completion.json');assert training['status']=='COMPLETE' and sha(PRIVATE/'final.pt')==training['checkpoint_sha256']
 for name,digest in read(OUT/'source-freeze.json')['files'].items():assert sha(ROOT/name)==digest,'Frozen source changed: '+name
 write(OUT/'retrieval.json',dict(at_utc=iso(),identity=read(OUT/'gpu-launch.json'),verified_all_remote_hashes=True,receipt=receipt,recovery=True))
 write(REC/'extraction.json',dict(at_utc=iso(),state='VERIFIED',files=len(receipt['files']),final_checkpoint_SHA256=training['checkpoint_sha256'],accepted_updates=training['accepted_updates'],completed_training_BO3=training['completed_BO3'],evaluate=training['evaluate']))

def worker():
 env=WSL+'/artifacts/astra-stage-d-repair-001/torch211:'+WSL+'/scripts:'+WSL+'/src:'+WSL
 cmd=['wsl','-d','Ubuntu','--cd',WSL,'-e','env','PYTHONPATH='+env,'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','.venv/bin/python','scripts/unown_corrected_ppo_cpu_worker.py']
 with (REC/'worker.log').open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);write(REC/'worker-process.json',dict(pid=p.pid,at_utc=iso()))
  try:rc=p.wait(timeout=max(1,remaining()-90))
  except BaseException:
   if (REC/'worker-pid.json').exists():subprocess.run(['wsl','-d','Ubuntu','-e','kill','-TERM',str(read(REC/'worker-pid.json')['pid'])],timeout=15,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
   p.kill();raise
 if rc:raise RuntimeError('CPU worker failed; see retained worker.log')

def main():
 REC.mkdir(exist_ok=True);start=now();cpu=time.process_time()
 exclusive(REC/'registration.json',dict(at_utc=iso(),deadline_utc=(start+dt.timedelta(hours=4)).isoformat(),authority='Human instruction [removed]: launch cpu evaluation; explicit new CPU-only recovery authorization after historical network failure',CPU_local_elapsed_cap_seconds=14400,GPU_usage=0,training=False,uploads=False,evaluation_matches=512,checkpoints=['corrected initialization','final committed trained checkpoint'],matches_per_checkpoint=256,chooser_per_checkpoint=[128,128],schedule_SHA256={n:sha(OUT/n) for n in ['evaluation-initial-schedule.json','evaluation-final-schedule.json']},analysis_registration_sha256=sha(OUT/'registration.json'),archive_sha256=EXPECTED,no_replacement_matches=True,no_intermediate_selection=True,execution='Same frozen sequential CPU evaluation and policies; only authorized deadline binding and read-only retrieval recovery added',worker_source_sha256=sha(ROOT/'scripts/unown_corrected_ppo_cpu_worker.py'),runner_source_sha256=sha(Path(__file__))))
 for label in ['initial','final']:assert not list((OUT/f'evaluation-{label}/intents').glob('*.json')),'Prior evaluation attempt; no duplicate launch'
 for name in ['completion.json','costs.json','gpu-costs.json']:
  if (OUT/name).exists():target=REC/'preserved-local'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/name,target)
 shutil.copy2(REPORT,REC/'preserved-local/report.md');ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 state='FAILED';reason='Recovery incomplete'
 try:
  status('RETRIEVING','Downloading and verifying the existing frozen export; evaluation not started.')
  p=subprocess.Popen([sys.executable,__file__,'download'],stdout=(REC/'download.log').open('w'),stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  try:rc=p.wait(timeout=min(900,max(1,remaining()-120)))
  except BaseException:p.kill();raise
  if rc:raise RuntimeError('Bounded artifact download failed; no GPU replacement')
  extract();training=read(OUT/'training-completion.json')
  status('CPU_EVALUATION_RUNNING',f"Export verified. Training completed{training['completed_BO3']}BO3 with{training['accepted_updates']}accepted updates. Launching the original fixed evaluation, gated by retained tensor change.")
  worker();result=read(REC/'worker-completion.json');state=result['state'];reason=result['reason']
 except BaseException as e:
  reason=type(e).__name__+': '+str(e);write(REC/'failure.json',dict(at_utc=iso(),error=reason,traceback=traceback.format_exc(),automatic_retry=False));status('CPU_RECOVERY_FAILED',reason+' Historical training export remains preserved; no replacement matches.')
 finally:
  write(REC/'controller-costs.json',dict(at_utc=iso(),elapsed_seconds=(now()-start).total_seconds(),CPU_core_hours=(time.process_time()-cpu)/3600,GPU_seconds=0,paid_model_API_requests=0))
  write(REC/'completion.json',dict(at_utc=iso(),state=state,reason=reason,automatic_followup=False,GPU_jobs=0,paid_model_API_requests=0));mirror();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
 if sys.argv[1:] == ['download']:
  try:download()
  except BaseException as e:
   write(REC/'download-failure.json',dict(at_utc=iso(),error_type=type(e).__name__,detail='Read-only artifact retrieval failed; signed URLs intentionally omitted',automatic_retry=False));print('Download failed:',type(e).__name__);raise SystemExit(1)
 else:main()
