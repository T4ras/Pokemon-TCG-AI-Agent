"""One explicitly authorized six-session-hour corrected Hydrapple PPO arm."""
from pathlib import Path
import datetime as dt,hashlib,json,os
ROOT=Path(__file__).resolve().parent.parent;REPO=ROOT
RUN='astra-unown-corrected-hydrapple-ppo-6h-001';OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
PARENT=ROOT/'experiments/astra-unown-starting-order-broad-validation-001'
INIT_PRIVATE=ROOT/'data/private/astra-unown-batched-continuation-execution-repair-001'
REPORT=ROOT/'reports/astra-unown-corrected-hydrapple-ppo-6h.md'
DATASET='portfolio-owner/ptcg-unown-corrected-hydrapple-ppo-6h-inputs'
KERNEL='portfolio-owner/ptcg-unown-corrected-hydrapple-ppo-6h';TITLE='PTCG Unown Corrected Hydrapple PPO 6h'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
SEED=900005;BOOT=904101
SESSION_CAP=21600;PROVIDER_CAP=20700;PROCESS_CAP=20100;TRAIN_CAP=19500;ABSOLUTE_CAP=21000
INITIAL_SHA='718d9c2346f17e4e324ede41b2be360c99f9b58ccbfc7e6b3b5833c166037193'
INITIAL_TENSOR='72b4ad7e65e154aa5736ec9facec5d43ffb8f39dc8c7041dd6cebe23fcc9ef78'
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(s):return dt.datetime.fromisoformat(s.replace('Z','+[removed]'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(v,indent=2,default=str),encoding='utf-8');q.replace(p)
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(v,default=str)+'\n');f.flush();os.fsync(f.fileno())
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def deadline():
 r=read(OUT/'registration.json');extra=read(OUT/'remote-clock.json')['excluded_seconds'] if (OUT/'remote-clock.json').exists() else 0.
 assert 0<=extra<=SESSION_CAP
 return parse(r['local_execution_start_utc'])+dt.timedelta(seconds=14400+extra)
def check_time(reserve=120):
 if (deadline()-now()).total_seconds()<reserve:raise TimeoutError('Four-hour local cap excluding only bounded remote GPU window')
def progress(stage,**v):write(OUT/'progress.json',dict(at_utc=iso(),stage=stage,pid=os.getpid(),**v));print(iso(),stage,flush=True)
