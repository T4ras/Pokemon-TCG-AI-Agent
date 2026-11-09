"""One bounded private GPU process; accept Kaggle's archive auto-extraction."""
from pathlib import Path
import datetime as dt,hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback
RUN='astra-unown-local-init-learning-pilot-execution-repair-001'
def iso():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,default=str))

def child_environment(repo):
 env=os.environ.copy();env.update(PYTHONPATH=str(repo/'scripts')+':'+str(repo),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0')
 return env

def stage_inputs(input_root,repo,expected_archive,expected_manifest):
 repo=Path(repo);repo.mkdir(parents=True,exist_ok=True);input_root=Path(input_root)
 archives=list(input_root.rglob('pilot-inputs.tar.gz'));expanded=not bool(archives)
 if archives:
  assert len(archives)==1 and sha(archives[0])==expected_archive,'Input archive hash mismatch'
  with tarfile.open(archives[0]) as t:
   for m in t.getmembers():
    assert not m.issym() and not m.islnk() and (repo/m.name).resolve().is_relative_to(repo.resolve()),'Unsafe input archive path'
   t.extractall(repo,filter='data')
 else:
  # Kaggle automatically exposes .tar.gz contents under pilot-inputs/.
  found=[p for p in input_root.rglob('upload-payload-manifest.json') if sha(p)==expected_manifest]
  assert len(found)==1,'Missing or changed expanded payload manifest'
  manifest_path=found[0];root=manifest_path.parents[2];manifest=json.loads(manifest_path.read_text())
  for name in list(manifest['files'])+[f'experiments/{RUN}/upload-payload-manifest.json']:
   source=(root/name).resolve();dest=(repo/name).resolve()
   assert source.is_relative_to(root.resolve()) and dest.is_relative_to(repo.resolve()),'Unsafe expanded input path'
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 manifest_path=repo/'experiments'/RUN/'upload-payload-manifest.json'
 assert sha(manifest_path)==expected_manifest,'Payload manifest hash mismatch'
 manifest=json.loads(manifest_path.read_text())
 for name,info in manifest['files'].items():
  path=(repo/name).resolve();assert path.is_relative_to(repo.resolve()) and sha(path)==info['sha256'],name
 return dict(archive_sha256=expected_archive,payload_manifest_sha256=expected_manifest,autoexpanded=expanded,verified_files=len(manifest['files']))

def main():
 start=time.monotonic();started=iso();repo=Path('/kaggle/temp/unown-pilot');repo.mkdir(parents=True,exist_ok=True)
 out=repo/'experiments'/RUN;private=repo/'data/private'/RUN;child=None;state='FAILED';error=None
 try:
  verified=stage_inputs('/kaggle/input',repo,'EXPECTED_ARCHIVE_SHA256','EXPECTED_PAYLOAD_MANIFEST_SHA256')
  write(out/'remote-start.json',dict(at_utc=started,pid=os.getpid(),inputs=verified,one_job=True,internal_seconds=2700,process_timeout_seconds=3000,provisioned_pool='T4x2',learner_visible_device='0'))
  env=child_environment(repo)
  with (out/'remote-runner.log').open('w') as log:
   child=subprocess.Popen([sys.executable,str(repo/'scripts/unown_pilot_training.py')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
   rc=child.wait(timeout=max(1,3000-(time.monotonic()-start)))
  if rc:raise RuntimeError(f'GPU trainer exited {rc}; see retained log/failure')
  assert (private/'final.pt').exists() and (out/'training-completion.json').exists()
  state='COMPLETE'
 except BaseException as e:
  error=str(e)
  if child and child.poll() is None:child.kill();child.wait(timeout=15)
  write(out/'remote-failure.json',dict(at_utc=iso(),error=error,traceback=traceback.format_exc(),no_restart=True))
 finally:
  result=Path('/kaggle/working');result.mkdir(exist_ok=True)
  write(out/'remote-completion.json',dict(state=state,started_at_utc=started,finished_at_utc=iso(),elapsed_seconds=time.monotonic()-start,error=error))
  files=sorted(p for folder in [out,private] if folder.exists() for p in folder.rglob('*') if p.is_file())
  inventory={str(p.relative_to(repo)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
  archive=result/'pilot-results.tar.gz'
  with tarfile.open(archive,'w:gz',compresslevel=1) as t:
   for p in files:t.add(p,arcname=str(p.relative_to(repo)),recursive=False)
  write(result/'pilot-receipt.json',dict(state=state,started_at_utc=started,export_finished_at_utc=iso(),elapsed_with_export_seconds=time.monotonic()-start,
   archive_sha256=sha(archive),bytes=archive.stat().st_size,files=inventory,error=error))
  print(json.dumps(dict(state=state,error=error,elapsed_seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
