"""One approved experience-driven continuation; local evaluation has no wall cap."""
from pathlib import Path
import datetime as dt,hashlib,json,math,os,time
ROOT=Path(__file__).resolve().parent.parent
RUN='astra-unown-corrected-hydrapple-cosine-continuation-001';OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
SOURCE=ROOT/'experiments/astra-unown-corrected-hydrapple-ppo-6h-001';SAVED=ROOT/'data/private/astra-unown-corrected-hydrapple-ppo-6h-001'
AUDIT=ROOT/'experiments/astra-unown-round-effect-throughput-001'
REPORT=ROOT/'reports/astra-unown-corrected-hydrapple-cosine-continuation.md'
KERNEL='portfolio-owner/ptcg-unown-hydrapple-cosine-continuation';DATASET=KERNEL+'-inputs'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
START_SHA='597fdcf0e596336939297050293768cecddd9a17701037e559f696e222bf327f';START_TENSOR='124ed3be69b954a06c0567830b5bd92be171feca759ab7a8a833655c0ac62d42'
ANCHOR_SHA='718d9c2346f17e4e324ede41b2be360c99f9b58ccbfc7e6b3b5833c166037193'
SEED=904102;TARGET=300000;SESSION_CAP=28800;PROVIDER_CAP=28200;PROCESS_CAP=27900;TRAIN_CAP=27300;ABSOLUTE_CAP=28500
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(x):return dt.datetime.fromisoformat(x.replace('Z','+[removed]'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(v,indent=2,default=str));q.replace(p)
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,default=str)+'\n');f.flush();os.fsync(f.fileno())
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def rate(decisions):
 p=min(1.,max(0.,decisions/TARGET));return 3e-6+.5*(1e-5-3e-6)*(1+math.cos(math.pi*p))
def progress(stage,**kw):write(OUT/'progress.json',dict(at_utc=iso(),stage=stage,pid=os.getpid(),**kw));print(iso(),stage,flush=True)
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
