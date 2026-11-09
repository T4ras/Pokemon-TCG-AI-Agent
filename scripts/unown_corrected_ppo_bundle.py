"""Explicit necessary-only private bundle and extracted-root cold start."""
import os,shutil,subprocess,sys,tarfile,time
from unown_corrected_ppo_common import *
def bundle():
 assert read(OUT/'CPU-compatibility.json')['status']=='PASS' and read(OUT/'pytest.json')['exit_code']==0
 old=read(ROOT/'experiments/astra-unown-batched-continuation-execution-repair-001/upload-payload-manifest.json')['files'];paths=[]
 for name in old:
  if name.startswith('data/private/astra-unown-batched-continuation-'):continue
  if name.startswith('experiments/') and not name.startswith('experiments/astra-unown-current-pool-adaptation-001/'):continue
  if name.startswith('scripts/') and name.endswith(('unown_batched_train.py','unown_batched_verify.py')):continue
  paths.append(ROOT/name)
 paths += [ROOT/'scripts/unown_batched_verify.py',ROOT/'scripts/unown_start_order_actor.py',ROOT/'scripts/unown_stability_stats.py']+list(ROOT.glob('scripts/unown_corrected_ppo_*.py'))
 paths += [ROOT/'tests/test_unown_corrected_ppo.py']
 write(OUT/'source-freeze.json',dict(at_utc=iso(),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.suffix=='.py'}))
 # Only training/local/data stages needed remotely; transport/report modules are
 # small necessary reproducibility sources, never credentials or unrelated assets.
 paths += [PRIVATE/name for name in ['initialization.pt','experimental-text-tables.npz','fixed-diagnostic.pt','verification.pt']]
 paths += [OUT/name for name in ['registration.json','manifest.json','diagnostic-freeze.json','CPU-compatibility.json','pytest.json','reused-numerics.json','source-freeze.json','evaluation-initial-schedule.json','evaluation-final-schedule.json']]
 paths=sorted(set(paths));assert all(p.is_file() for p in paths)
 inventory={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
 write(OUT/'upload-payload-manifest.json',dict(private=True,destination=DATASET,files=inventory,excludes=['credentials','raw observations','old PPO checkpoints/optimizers/rollouts','G/R1/R2 weights','unrelated assets'],authority='Current human request explicitly permits necessary private uploads and one GPU job <=6 session-hours'))
 folder=PRIVATE/'upload';folder.mkdir(exist_ok=False);archive=folder/'pilot-inputs.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
  t.add(OUT/'upload-payload-manifest.json',arcname=str((OUT/'upload-payload-manifest.json').relative_to(ROOT)))
 write(OUT/'bundle.json',dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(paths),dataset=DATASET))
 write(folder/'bundle-manifest.json',dict(archive_sha256=sha(archive),payload_manifest_sha256=sha(OUT/'upload-payload-manifest.json')))
 write(folder/'dataset-metadata.json',dict(id=DATASET,title='PTCG Unown Corrected Hydrapple PPO 6h Inputs',licenses=[dict(name='other')],description='Private necessary source, official SDK/card metadata, unchanged initialization, corrected encoded diagnostic states and disposable compatibility packets. No credentials, raw observations or prior PPO optimizer/checkpoint.'))
 source=(ROOT/'scripts/unown_pilot_remote.py').read_text();source=source.replace("RUN='astra-unown-local-init-learning-pilot-execution-repair-001'",f"RUN='{RUN}'")
 source=source.replace('unown_pilot_training.py','unown_corrected_ppo_train.py').replace('internal_seconds=2700',f'internal_seconds={TRAIN_CAP}').replace('process_timeout_seconds=3000',f'process_timeout_seconds={PROCESS_CAP}')
 source=source.replace('3000-(time.monotonic()-start)',f'min({PROCESS_CAP}-(time.monotonic()-start),(dt.datetime.fromisoformat(ABSOLUTE_DEADLINE_UTC)-dt.datetime.now(dt.timezone.utc)).total_seconds()-600)')
 source=source.replace("OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0'","OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'")
 source=source.replace('env=child_environment(repo)',"env=child_environment(repo);env['CORRECTED_JOB_MONOTONIC_START']=str(start);env['CORRECTED_PPO_ABSOLUTE_DEADLINE_UTC']=ABSOLUTE_DEADLINE_UTC")
 source=source.replace("RUN='"+RUN+"'","RUN='"+RUN+"'\nABSOLUTE_DEADLINE_UTC='TO_BE_FROZEN_AT_PUSH'")
 source=source.replace('EXPECTED_ARCHIVE_SHA256',sha(archive)).replace('EXPECTED_PAYLOAD_MANIFEST_SHA256',sha(OUT/'upload-payload-manifest.json'))
 kernel=PRIVATE/'kernel';kernel.mkdir(exist_ok=False);(kernel/'main-template.py').write_text(source)
 write(kernel/'kernel-metadata.json',dict(id=KERNEL,title=TITLE,code_file='main.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',docker_image=IMAGE,docker_image_pinning_type='original',dataset_sources=[DATASET+'/1'],competition_sources=[],kernel_sources=[]))
def smoke():
 import unown_pilot_remote as transport
 transport.RUN=RUN;root=PRIVATE/'cold-extracted';folder=PRIVATE/'upload';m=read(folder/'bundle-manifest.json');transport.stage_inputs(folder,root,m['archive_sha256'],m['payload_manifest_sha256'])
 code="""
from pathlib import Path
import sys,json,torch
from unown_corrected_ppo_core import *
assert ROOT.resolve()==Path.cwd().resolve()
m,o=fresh();assert not o.state and state_hash(m)==INITIAL_TENSOR
rows=torch.load(PRIVATE/'verification.pt',weights_only=False)
p=phase_metrics(m,rows);assert p['max_chosen_logprob_difference']<=1e-5 and p['max_old_policy_KL']<=1e-6
tracker=runtime.rt.MatchTracker(read(OUT/'manifest.json')['decks']['Hydro']['cards']);tracker.reset_game(1);assert tracker.tracker is not None
paths={n:str(Path(v.__file__).resolve()) for n,v in sys.modules.items() if getattr(v,'__file__',None) and (n.startswith('unown_') or n.startswith('ptcg.'))}
assert all(Path(p).is_relative_to(ROOT.resolve()) for p in paths.values()),paths
print(json.dumps(dict(status='PASS',fresh_optimizer=True,tensor_hash=state_hash(m),likelihood=p,agent_modules=len(paths),checkout_fallback=False,native_attempts=0)))
"""
 env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'artifacts/astra-stage-d-repair-001/torch211')+':'+str(root/'scripts')+':'+str(root),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 t=time.monotonic();result=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=180)
 write(OUT/'isolated-archive-log.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 assert result.returncode==0,'Extracted bundle cold start failed'
 write(OUT/'isolated-bundle-smoke.json',dict(status='PASS',bundle_sha256=m['archive_sha256'],elapsed_seconds=time.monotonic()-t,result=json.loads(result.stdout.splitlines()[-1]),no_checkout_source_fallback=True))
