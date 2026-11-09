"""Frozen single-pilot state. No paid model APIs or credential lookup."""
from pathlib import Path
import datetime as dt,hashlib,json,os
REPO=Path(__file__).resolve().parent.parent
RUN='astra-unown-local-init-learning-pilot-execution-repair-001'
OUT=REPO/'experiments'/RUN
PRIVATE=REPO/'data/private'/RUN
PARENT=REPO/'data/private/astra-unown-gradiant-benchmark-001'
UP=PARENT/'upstream/a176778d99da74a44b495bf19299b4795faf12f2'
OLD=REPO/'experiments/astra-unown-current-pool-adaptation-001'
LOCAL=REPO/'experiments/astra-unown-local-text-001'
OLDPRIVATE=REPO/'data/private/astra-unown-current-pool-adaptation-001'
SEED=904105
START='[removed]'
DEADLINE='[removed]'
DATASET='portfolio-owner/ptcg-unown-pilot-execution-repair-inputs'
KERNEL='portfolio-owner/ptcg-unown-pilot-execution-repair'
KERNEL_TITLE='PTCG Unown Pilot Execution Repair'
TRAINING_CAP_SECONDS=2700
PROCESS_CAP_SECONDS=3000
PROVIDER_CAP_SECONDS=3270
SESSION_CAP_SECONDS=3570
ALLOCATION_GPU_COUNT=2
PINNED_IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
REPORT=REPO/'reports/astra-unown-local-init-learning-pilot.md'
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(s):return dt.datetime.fromisoformat(s.replace('Z','+[removed]'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('w') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
 q.replace(p)
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,default=str)+'\n');f.flush();os.fsync(f.fileno())
def progress(stage,**kw):
 write(OUT/'progress.json',dict(stage=stage,at_utc=iso(),pid=os.getpid(),**kw));print(iso(),stage,flush=True)
def local_deadline():
 if now()>=effective_deadline():raise TimeoutError('Four-hour local deadline excluding bounded remote GPU session')

def effective_deadline():
 approval=OUT/'launch-authorization.json'
 registration=OUT/'registration.json'
 base=parse(read(approval)['local_deadline_utc']) if approval.exists() else parse(read(registration)['preparation_deadline_utc']) if registration.exists() else parse(DEADLINE)
 p=OUT/'remote-clock-exclusion.json'
 seconds=read(p)['excluded_remote_seconds'] if p.exists() else 0.
 if not 0<=seconds<=SESSION_CAP_SECONDS:raise ValueError('Invalid remote clock exclusion')
 return base+dt.timedelta(seconds=seconds)

def require_launch_authorization():
 p=OUT/'launch-authorization.json'
 if not p.exists():raise RuntimeError('Replacement job and local execution window need explicit approval; prepared artifacts only')
 value=read(p)
 if value.get('approved') is not True or value.get('registration_sha256')!=sha(OUT/'registration.json') or value.get('bundle_sha256')!=read(OUT/'bundle.json')['sha256']:raise RuntimeError('Launch approval does not match frozen proposal')
 if parse(value['local_deadline_utc'])-parse(value['at_utc'])>dt.timedelta(hours=1):raise RuntimeError('Approval exceeds proposed one-hour local window')
 local_deadline()
