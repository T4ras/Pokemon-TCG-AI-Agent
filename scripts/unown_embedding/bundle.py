"""Necessary-only private training payload and isolated cold initialization."""
import resource,shutil,subprocess,sys,tarfile
from .common import *

def bundle():
 assert read(OUT/'readiness.json')['status']=='PASS' and read(OUT/'runner-regression.json')['returncode']==0
 assert read(OUT/'action-schema-check.json')['status']=='PASS'
 assert not (OUT/'gpu-intent.json').exists() and not (OUT/'upload-intent-valid-metadata.json').exists()
 if not (OUT/'registration-initial.json').exists():shutil.copy2(OUT/'registration.json',OUT/'registration-initial.json')
 registration=read(OUT/'registration.json')
 registration['execution_freeze']=dict(at_utc=iso(),before_retry_native_and_GPU=True,CPU_workers=4,evaluation='First2 scheduled initial attempts verify startup, then complete the same512; no extra integration matches',
  required_before_GPU='All512 initial CPU attempts done without correctness/runtime failure; no strength-outcome gate',
  collection_reserve='Do not start a cohort with less than max(600s,1.5*(last measured collection+update seconds)+60s) before the trainer deadline; export reserve already removed',
  diagnostics='All compact failures retained; at most10 before/rejected tensor pairs per signature and2GiB total detail, within3GiB scientific outputs',
  precision='Pinned Torch2.11.0+cu128/Python3.13/T4, float32, mathSDPA, noTF32/flash/memory-efficientSDPA',
  checkpoint='Durable every8 accepted updates, epoch/cohort/stop boundaries, exact sparse/optimizer/RNG/cursor recovery; canonical effective final export',
  initial_fresh_optimizer='Zero state, actor sparse parameter group1e-4 and detached blind-value group1e-5; seed2026103001',
  current_configuration_throughput='Not extrapolated from historical Hydrapple or compilation runs; collect and report actual cohort times',
  post_GPU='One fixed final candidate; no retained actor change skips redundant final512; no source/weights changes, replacement job or automatic follow-up')
 write(OUT/'registration.json',registration)
 paths=[p for p in (PRIVATE/'assets').rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
 paths += [PRIVATE/name for name in ('initialization.pt','text-tables.npz','old-probes.pt','new-probes.pt','asset-paths.json')]
 paths += list((ROOT/'scripts/unown_runner').rglob('*.py'))+list((ROOT/'scripts/unown_embedding').glob('*.py'))
 paths += [ROOT/name for name in ('scripts/unown_train.py','scripts/unown_embedding_owner.py','scripts/unown_embedding_remote.py','scripts/unown_pilot_remote.py','scripts/unown_pilot_execution.py','tests/test_unown_embedding.py','tests/test_unown_embedding_collection.py')]
 paths += [OUT/name for name in ('registration.json','registration-initial.json','decks.json','trainable-rows.json','training-schedule.json','evaluation-initial-schedule.json','evaluation-final-schedule.json','initialization-provenance.json','mechanic-readiness.json','action-schema-check.json','probe-provenance.json','readiness.json','runner-regression.json','parameter-verification.json','recovery-verification.json','retry-provenance.json','collector-repair-verification.json')]
 inventory={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))}
 write(OUT/'source-freeze.json',dict(at_utc=iso(),files={n:v['sha256'] for n,v in inventory.items() if n.endswith('.py')}))
 inventory[str((OUT/'source-freeze.json').relative_to(ROOT))]=dict(sha256=sha(OUT/'source-freeze.json'),bytes=(OUT/'source-freeze.json').stat().st_size)
 write(OUT/'input-manifest.json',dict(at_utc=iso(),files=inventory,scope='Exact one-arm configuration, initialization and frozen base text/SDK assets. Encoded probes are software diagnostics, not training experience. Initial512 CPU receipt is injected by owner only after completion; no outcome-based gate.'))
 inventory[str((OUT/'input-manifest.json').relative_to(ROOT))]=dict(sha256=sha(OUT/'input-manifest.json'),bytes=(OUT/'input-manifest.json').stat().st_size)
 write(OUT/'upload-payload-manifest.json',dict(private=True,destination=DATASET,files=inventory,contents='Source, pinned SDK/runtime/card metadata/released runtime assets, exact original experimental initialization and fixed text tables, selected semantic-row/deck/schedule manifests, bounded retained encoded old/new software probes; no credentials, raw observations, G/R1/R2 or later PPO weights.',authority='Current user requested one six-hour GPU experiment; exact-payload private upload approval must be recorded before launching if required by automatic approval review.'))
 folder=PRIVATE/'upload';folder.mkdir(exist_ok=True);archive=folder/'pilot-inputs.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as stream:
  for name in inventory:stream.add(ROOT/name,arcname=name,recursive=False)
  stream.add(OUT/'upload-payload-manifest.json',arcname=str((OUT/'upload-payload-manifest.json').relative_to(ROOT)))
 write(OUT/'bundle.json',dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(inventory),dataset=DATASET,one_GPU_job=True,session_cap_seconds=SESSION_CAP,allocated_device_hours_cap=SESSION_CAP*2/3600))
 write(folder/'dataset-metadata.json',dict(id=DATASET,title='PTCG Unown Embedding Retry Inputs',licenses=[dict(name='other')],description='Private one-arm sparse new-card embedding experiment; no raw replay observations or credentials.'))
 import unown_pilot_remote as transport
 transport.RUN=RUN;cold=PRIVATE/('cold-'+sha(archive)[:12]);transport.stage_inputs(folder,cold,sha(archive),sha(OUT/'upload-payload-manifest.json'))
 code='''
from pathlib import Path
import json,sys
from unown_embedding.common import *
from unown_embedding.runtime import AdaptationRuntime
assert ROOT.resolve()==Path.cwd().resolve()
rt=AdaptationRuntime(context())
try:
 m,o,s=rt.restore(PRIVATE/'initialization.pt','cpu')
 assert rt.state_hash_from_canonical(m)==INITIAL_TENSOR and not o.state
 assert len(o.param_groups)==2 and [g['lr'] for g in o.param_groups]==[1e-4,1e-5]
 assert len(read(OUT/'training-schedule.json'))==MAX_COHORTS
 assert len(torch.load(PRIVATE/'new-probes.pt',map_location='cpu',weights_only=False))>=32
 for name,module in list(sys.modules.items()):
  if getattr(module,'__file__',None) and (name.startswith('unown_runner') or name.startswith('unown_embedding') or name.startswith('ptcg.') or name.startswith('cg.')):assert Path(module.__file__).resolve().is_relative_to(ROOT.resolve()),name
 print(json.dumps(dict(status='PASS',initial_tensor=INITIAL_TENSOR,fresh_optimizer=True,frozen_hash_exact=True,no_checkout_fallback=True,native_attempts=0,GPU_seconds=0)))
finally:rt.close()
'''
 env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'artifacts/astra-stage-d-repair-001/torch211')+':'+str(cold/'scripts')+':'+str(cold),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 result=subprocess.run([sys.executable,'-c',code],cwd=cold,env=env,capture_output=True,text=True,timeout=120)
 write(OUT/'cold-start.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,bundle_SHA256=sha(archive)));assert result.returncode==0,'Cold payload verification failed'
 source=(ROOT/'scripts/unown_embedding_remote.py').read_text().replace('EXPECTED_ARCHIVE_SHA256',sha(archive)).replace('EXPECTED_PAYLOAD_MANIFEST_SHA256',sha(OUT/'upload-payload-manifest.json')).replace('EXPECTED_TRANSPORT_SHA256',sha(ROOT/'scripts/unown_pilot_remote.py'))
 compile(source,'<embedding-GPU>','exec');kernel=PRIVATE/'kernel';kernel.mkdir(exist_ok=True);(kernel/'main-template.py').write_text(source)
 metadata=dict(id=KERNEL,title='PTCG Unown New Card Embedding Retry',code_file='main.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',docker_image=IMAGE,docker_image_pinning_type='original',dataset_sources=[DATASET+'/1'],competition_sources=[],kernel_sources=[])
 from unown_pilot_execution import validate_kernel_metadata
 validate_kernel_metadata(metadata);write(kernel/'kernel-metadata.json',metadata)
 print(json.dumps(dict(status='COLD_VERIFIED',bundle=read(OUT/'bundle.json'))))

if __name__=='__main__':
 start=time.monotonic();cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 try:bundle()
 finally:
  after=resource.getrusage(resource.RUSAGE_CHILDREN);append(OUT/'local-costs.jsonl',dict(stage='bundle-cold-restore',at_utc=iso(),elapsed_seconds=time.monotonic()-start,CPU_core_hours=(time.process_time()-cpu+after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime)/3600,GPU_seconds=0,native_attempts=0))
