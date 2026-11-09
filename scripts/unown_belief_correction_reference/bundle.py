"""Necessary private payload, cold restore and source isolation checks."""
from .common import *
import tarfile,subprocess,resource
import torch

def run():
 assert not (OUT/'gpu-intent.json').exists() and not (OUT/'upload-intent.json').exists()
 assert (PRIVATE/'encoded-probes.pt').exists()
 files={ROOT/name for name in read(ROOT/'experiments'/OLD/'source-manifest.json')['files']};files.update(ROOT/name for name in read(ROOT/'experiments'/OLD/'input-manifest.json')['files'])
 files.update((ROOT/'scripts/unown_belief_correction_reference').glob('*.py'));files.update((ROOT/'src').rglob('*.py'))
 for name in ('scripts/unown_belief_reference_remote.py','scripts/unown_belief_reference_owner.py','scripts/unown_pilot_remote.py','scripts/unown_pilot_execution.py','scripts/unown_train.py','tests/test_belief_correction_reference.py','configs/compute-usage-policy.json'):
  files.add(ROOT/name)
 for name in ('assets.json','frozen-components.json','registration.json','lifecycle.json','decks.json'):
  files.add(ROOT/'experiments'/OLD/name)
 files.add(ROOT/'experiments/astra-unown-belief-tracker-validation-001/prior.json');files.add(ROOT/'experiments/astra-unown-belief-classifier-full-training-001/schema.json')
 for name in ('registration.json','training-schedule.json','evaluation-schedule.json','decks.json','initialization.json','frozen-components.json','preflight.json','pytest-latest.json','exact-cache-parity-timing.json','recovery-forecast.json','repair-registration.json','repair-amendment-002.json','diagnostic-registration.json','budget-amendment-003.json','numerical-recovery-amendment.json','CPU-reference-diagnostics.json'):
  files.add(OUT/name)
 for name in ('initial-correction.pt','encoded-probes.pt','asset-paths.json','belief-workload.pt','pending-cohort.pt'):files.add(PRIVATE/name)
 files={p for p in files if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 inventory={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
 write(OUT/'upload-payload-manifest.json',dict(at_utc=iso(),private=True,destination=DATASET,contents='Required source, pinned SDK/released runtime/card metadata, exact current policy and saved step1052 correction model/optimizer/RNG boundary, frozen seed11/update14004 MLP and text tables,32 retained encoded arithmetic probes,40 encoded legal-fact/public-scalar profiling states, immutable2,876-row pending cohort and fixed deck/schedules/registration; completed state weights remain local. No raw histories, credentials, G/R1/R2, paid API payload or submission.',files=inventory))
 upload=PRIVATE/'upload';upload.mkdir(exist_ok=True);archive=upload/'pilot-inputs.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for name in inventory:t.add(ROOT/name,arcname=name,recursive=False)
  t.add(OUT/'upload-payload-manifest.json',arcname=str((OUT/'upload-payload-manifest.json').relative_to(ROOT)),recursive=False)
 write(upload/'dataset-metadata.json',dict(id=DATASET,title='PTCG Unown Belief Reference Inputs',licenses=[dict(name='other')],description='Private registered belief-conditioned PPO correction pilot, frozen current policy/classifier and encoded arithmetic probes. No raw observations or credentials.'))
 write(OUT/'bundle.json',dict(at_utc=iso(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(inventory),dataset=DATASET,kernel=KERNEL,session_cap_seconds=SESSION,allocated_device_hours_cap=2*SESSION/3600,one_job=True))
 import unown_pilot_remote as transport
 transport.RUN=RUN;cold=PRIVATE/('cold-'+sha(archive)[:12]);transport.stage_inputs(upload,cold,sha(archive),sha(OUT/'upload-payload-manifest.json'))
 code="""
from pathlib import Path
import sys,torch
from unown_belief_correction_reference.common import *
from unown_belief_correction_reference.runtime import TrainingRuntime
from unown_belief_correction.belief import frozen_belief_model,OnlineBelief
assert ROOT.resolve()==Path.cwd().resolve();verify_inputs()
assert not (ROOT/'experiments/astra-unown-belief-classifier-full-training-001/lifecycle.json').exists()
from unown_belief_correction_reference.cloud_profile import run as check_cache
check_cache()
from unown_belief_correction_reference.train import register_runtime
from unown_runner.runner import Runner
class NoNative:
 software_only=True
 def close(self):pass
 def collect(self,*args,**kwargs):raise AssertionError('No native matches in cold verification')
for arm in ('state','belief'):
 ctx=register_runtime(arm,'cpu',software_only=True)
 runner=Runner(ctx,ctx.out/'runtime-registration.json',ctx.out/'runtime-authorization.json',device='cpu',collector=NoNative(),runtime_factory=TrainingRuntime)
 assert len(runner.reg['cohorts'])==(115 if arm=='belief' else 128) and sum(map(len,runner.reg['cohorts']))==(1840 if arm=='belief' else 2048)
 from unown_belief_correction_reference.runtime import PilotRunner
 PilotRunner.seed_boundary(runner);runner.restore()
 assert runner.execution['step']==1052 and runner.execution['additional_collected']==35852 and runner.execution['unique_used_decisions']==32976
 assert runner.state=='VALIDATING' and len(runner.active['rows'])==2876 and runner.execution['phase']==9
 assert runner.rt.state_hash(runner.model)==read(OUT/'initialization.json')['tensor_sha256']
 assert len(runner.opt.state)==16
 runner.rt.close()

for arm in ('state','belief'):
 rt=TrainingRuntime(context(arm),arm)
 try:
  m,o,s=rt.restore(PRIVATE/'initial-correction.pt','cpu');assert len(o.state)==16 and len(o.param_groups)==3
  assert rt.state_hash(m)==read(OUT/'initialization.json')['tensor_sha256']
  assert sum(p.numel() for p in m.parameters() if p.requires_grad)==806403
  from unown_belief_correction_reference.probe import prepare_rows
  rows=prepare_rows(rt,m,torch.load(PRIVATE/'encoded-probes.pt',map_location='cpu',weights_only=False)['rows']);c=rt.epochs.new_cohort(rows,rt.state_hash(m));e=rt.make_engine(m,o,c,0)
  try:assert e.reference_check(rows)['status']=='PASS'
  finally:e.close()
 finally:rt.close()
assert sum(p.numel() for p in frozen_belief_model().parameters())==57691
OnlineBelief()
from unown_belief_correction_reference.recovery_gate import run as pending_gate
pending_gate('cpu')
from unown_belief_correction_reference.validation import rare_path
rare_path()
from unown_belief_correction_reference.probe import run as arithmetic_probe
arithmetic_probe('state','cpu')
for name,module in list(sys.modules.items()):
 if name.startswith(('unown_','ptcg.','cg.')) and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(ROOT.resolve()),name
print('COLD_RESTORE_AND_PROBABILITY_PASS no_native no_GPU')
"""
 env=os.environ.copy();env.update(PYTHONPATH=str(cold/'scripts')+':'+str(cold/'src')+':'+str(cold),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='');result=subprocess.run([sys.executable,'-c',code],cwd=cold,env=env,capture_output=True,text=True,timeout=180);write(OUT/'cold-start.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,bundle_sha256=sha(archive)));assert result.returncode==0,'Cold-start verification failed; see retained result'
 source=(ROOT/'scripts/unown_belief_reference_remote.py').read_text().replace('EXPECTED_ARCHIVE_SHA256',sha(archive)).replace('EXPECTED_MANIFEST_SHA256',sha(OUT/'upload-payload-manifest.json')).replace('EXPECTED_TRANSPORT_SHA256',sha(ROOT/'scripts/unown_pilot_remote.py'));compile(source,'<private-remote>','exec');kernel=PRIVATE/'kernel';kernel.mkdir(exist_ok=True);(kernel/'main-template.py').write_text(source)
 metadata=dict(id=KERNEL,title='PTCG Unown Belief Reference',code_file='main.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,machine_shape='NvidiaTeslaT4',docker_image=IMAGE,docker_image_pinning_type='original',dataset_sources=[DATASET+'/1'],competition_sources=[],kernel_sources=[])
 from unown_pilot_execution import validate_kernel_metadata
 validate_kernel_metadata(metadata);write(kernel/'kernel-metadata.json',metadata);progress('COLD_VERIFIED',bundle=read(OUT/'bundle.json'))
if __name__=='__main__':run()
