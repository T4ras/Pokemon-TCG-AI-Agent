"""Fresh CPU implementation/readiness and validation scope; historical run untouched."""
from pathlib import Path
import os,json,hashlib,datetime as dt,time,collections,math
R=Path(__file__).resolve().parents[2]
OLD=R/'experiments/astra-unown-belief-tracker-validation-001'
RUN='astra-unown-belief-development-repair-003';OUT=R/'experiments'/RUN;PRIVATE=R/'data/private'/RUN
READY=R/'experiments/astra-unown-belief-native-observer-readiness-001'
SCREEN=R/'experiments/astra-unown-top100-deck-playability-screen-001';SP=R/'data/private/astra-unown-top100-deck-playability-screen-001'
POLICY=SP/'policy';TENSOR='5b43fff134b12aef91361b18d504a04e9baa408c36d5fea89caf493566905cb8';SEED=900001
REPORT=R/'reports/astra-unown-belief-development-repair.md';READY_REPORT=R/'reports/astra-unown-belief-native-observer-readiness.md'
WORKERS=4;DETAIL_CAP=2*1024**3;CORPUS_CAP=8*1024**3

def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8');q.replace(p)
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,default=str);f.flush();os.fsync(f.fileno())
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(v,default=str)+'\n');f.flush();os.fsync(f.fileno())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def prior():return read(OLD/'prior.json')['lists']
def metadata():return read(POLICY/'current/current-tables.json')
def logsumexp(values):
 if not values:return -math.inf
 m=max(values)
 return m+math.log(sum(math.exp(x-m) for x in values)) if math.isfinite(m) else m
class UnsupportedEvent(RuntimeError):pass
class IncompatibleAllocation(ValueError):pass
class InvariantFailure(AssertionError):pass
class Depletion(RuntimeError):pass
class OutOfLibrary(RuntimeError):pass

def deadline():return dt.datetime.fromisoformat(read(OUT/'lifecycle.json')['deadline_utc'])
def check_time():
 if now()>=deadline():raise TimeoutError('Fresh24-hour UTC deadline reached')
def progress(stage,**kwargs):write(OUT/'progress.json',dict(stage=stage,at_utc=iso(),**kwargs))

PARENT=R/'experiments/astra-unown-belief-native-validation-002'
PARENT_PRIVATE=R/'data/private/astra-unown-belief-native-validation-002'
