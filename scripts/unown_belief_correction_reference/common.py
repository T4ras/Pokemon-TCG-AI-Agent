"""Explicit budget amendment; scientific protocol remains frozen."""
from unown_runner.context import *
import collections,copy,math,random,sys,traceback,shutil
ROOT=Path(chr(92)*2+'wsl.localhost/Ubuntu/path/to/ptcg-agent') if os.name=='nt' else Path(__file__).resolve().parents[2]
RUN='astra-unown-belief-correction-ppo-execution-007';OLD='astra-unown-belief-correction-ppo-pilot-001'
OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
REPORT=ROOT/'reports/astra-unown-belief-correction-ppo-pilot.md'
SESSION=43200;ARM_CAPS={'state':12563,'belief':43200};LOCAL_CPU_CAP=float('inf');TARGET=250000;SEED=904096
DATASET='portfolio-owner/ptcg-unown-belief-reference-inputs';KERNEL='portfolio-owner/ptcg-unown-belief-reference'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
POLICY_TENSOR='5b43fff134b12aef91361b18d504a04e9baa408c36d5fea89caf493566905cb8'
def context(arm,mode='learning'):
 a={k:str(ROOT/v) for k,v in read(PRIVATE/'asset-paths.json').items()};a['start_checkpoint']=str(PRIVATE/'initial-correction.pt')
 return RunContext(ROOT,RUN+'-'+arm,a,Execution('baseline'),SEED,dict(FIXED_RECIPE,lr=1e-4),dict(max_cohorts=115 if arm=='belief' else 128,native_attempts=1840 if arm=='belief' else 2048,session_seconds=ARM_CAPS[arm],export_reserve_seconds=300,storage_bytes=4*1024**3,log_bytes=256*1024**2),mode)
def guard():
 if (OUT/'STOP.json').exists():raise RuntimeError(read(OUT/'STOP.json')['reason'])
 # Study elapsed budget withdrawn; active provider limits remain in the runner.
def verify_inputs():
 for name,info in read(OUT/'upload-payload-manifest.json')['files'].items():
  if sha(ROOT/name)!=info['sha256']:raise ValueError('Frozen payload changed: '+name)
def progress(stage,**kw):
 x=dict(kw,at_utc=iso(),stage=stage);write(OUT/'progress.json',x);print('BELIEF_CORRECTION_PROGRESS '+json.dumps(x,default=str),flush=True)
