"""One preregistered, bounded continuation. No model-service calls."""
from pathlib import Path
import datetime as dt, hashlib, json, os
REPO=Path(__file__).resolve().parent.parent
RUN='astra-unown-batched-continuation-execution-repair-001'
OUT=REPO/'experiments'/RUN; PRIVATE=REPO/'data/private'/RUN
PREV=REPO/'experiments/astra-unown-update-stability-repair-001'
PREV_PRIVATE=REPO/'data/private/astra-unown-update-stability-repair-001'
REPORT=REPO/'reports/astra-unown-batched-continuation.md'
START='[removed]'; DEADLINE='[removed]'
SEED=904095
PRIOR_SESSION_SECONDS=64.792662
PRIOR_QUOTA_SECONDS=44.263
REMAINING_SESSION_SECONDS=14400-PRIOR_SESSION_SECONDS
PROVIDER_TIMEOUT_SECONDS=13680
PRETRAINING_SCRIPT_CAP_SECONDS=1100
DATASET='portfolio-owner/ptcg-unown-batched-continuation-inputs'
KERNEL='portfolio-owner/ptcg-unown-batched-continuation'
TITLE='PTCG Unown Batched Continuation'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
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
 with p.open('a') as f:f.write(json.dumps(v,default=str)+'\n');f.flush()
def deadline():
 p=OUT/'remote-clock.json';extra=read(p)['excluded_seconds'] if p.exists() else 0
 assert 0<=extra<=14400
 return parse(DEADLINE)+dt.timedelta(seconds=extra)
def check_deadline():
 if now()>=deadline():raise TimeoutError('Four-hour local deadline excluding bounded remote session')
def progress(stage,**v):
 write(OUT/'progress.json',dict(at_utc=iso(),stage=stage,pid=os.getpid(),**v));print(iso(),stage,flush=True)

