"""One local inference-only experiment, no network or training entry point."""
from pathlib import Path
import datetime as dt,hashlib,json,os
ROOT=Path(__file__).resolve().parent.parent
RUN='astra-unown-starting-order-feature-test-001'
OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
PARENT=ROOT/'experiments/astra-unown-batched-continuation-execution-repair-001'
PARENT_PRIVATE=ROOT/'data/private/astra-unown-batched-continuation-execution-repair-001'
SDK=ROOT/'data/private/astra-unown-gradiant-benchmark-001/current-environment'
HIST=PARENT_PRIVATE/'actor-histories'
REPORT=ROOT/'reports/astra-unown-starting-order-feature-test.md'
SEED=904109;BOOT=904110
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
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
def deadline():return dt.datetime.fromisoformat(read(OUT/'launch.json')['deadline_utc'])
def check_time(reserve=120):
 if (deadline()-now()).total_seconds()<=reserve:raise TimeoutError('Four-hour execution cap; analysis/export reserve')
def progress(stage,**v):write(OUT/'progress.json',dict(at_utc=iso(),stage=stage,pid=os.getpid(),**v))
