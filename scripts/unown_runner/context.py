"""Run-local paths, immutable execution selection, and fail-closed authorization."""
from dataclasses import dataclass,asdict
from pathlib import Path
import datetime as dt,hashlib,json,os,time
import math,re

def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(value):
 result=dt.datetime.fromisoformat(value.replace('Z','+[removed]'))
 if result.tzinfo is None:raise ValueError('Timezone-aware deadline required')
 return result
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with temp.open('w') as f:json.dump(value,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
 temp.replace(path)
def append(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(value,default=str)+'\n');f.flush();os.fsync(f.fileno())
def exclusive(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,default=str);f.flush();os.fsync(f.fileno())
def records(path):return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []
def implementation_hashes():
 package=Path(__file__).parent
 result={str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}
 result['../unown_train.py']=sha(package.parent/'unown_train.py');return result

FIXED_RECIPE=dict(logical_minibatch=128,epochs=4,precision='float32',gamma=1.,gae_lambda=.95,value_coefficient=2.,entropy_coefficient=.01,clip=.2,grad_clip=1.5,pre_KL=.02,post_KL=.03,optimizer='AdamW(.95,.999),eps1e-8,decay0,foreachFalse',features='starting-order-v1',targets='completed-game',learner_decoder='conditional-sample-T1',opponent_decoder='conditional-argmax')

@dataclass(frozen=True)
class Execution:
 learner:str
 collector:str='revised-concurrent16'
 compiled:bool=False
 boundaries:tuple=(64,72,96,160)
 def validate(self):
  if self.learner=='bucket64-compiled':
   if self.collector!='revised-concurrent16' or self.compiled is not True or tuple(self.boundaries)!=(64,72,96,160):raise ValueError('Invalid compiled execution choice')
   from .compiled_option import certificate
   certificate();return
  if self.learner not in ('baseline','bucket64') or self.collector!='revised-concurrent16' or self.compiled or tuple(self.boundaries)!=(64,72,96,160):raise ValueError('Unsupported/unvalidated execution choice; no fallback')

@dataclass(frozen=True)
class RunContext:
 root:Path
 run_id:str
 assets:dict
 execution:Execution
 seed:int
 recipe:dict
 limits:dict
 mode:str
 @property
 def out(self):return self.root/'experiments'/self.run_id
 @property
 def private(self):return self.root/'data/private'/self.run_id
 def validate(self):
  if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',self.run_id):raise ValueError('Invalid run ID')
  for path in (self.out,self.private):
   if path.resolve()!=self.root.resolve()/path.relative_to(self.root):raise ValueError('Mutable run paths must not be symlinks')
  if self.mode not in ('software_test','learning'):raise ValueError('Explicit run mode required')
  self.execution.validate()
  if set(self.recipe)!=set(FIXED_RECIPE)|{'lr'} or any(self.recipe.get(k)!=v for k,v in FIXED_RECIPE.items()):raise ValueError('Unsupported recipe; execution never selects a different learning recipe')
  if not isinstance(self.recipe.get('lr'),(float,int)) or not math.isfinite(self.recipe['lr']) or self.recipe['lr']<=0:raise ValueError('Learning rate must be separately registered')
  for key in ('max_cohorts','native_attempts','session_seconds','export_reserve_seconds','storage_bytes','log_bytes'):
   if not isinstance(self.limits.get(key),int) or self.limits[key]<0:raise ValueError('Missing/invalid budget: '+key)
  if self.limits['session_seconds']<=self.limits['export_reserve_seconds'] or self.limits['max_cohorts']<=0:raise ValueError('Invalid deadline/cohort budget')
  if self.mode=='learning' and self.limits['native_attempts']!=16*self.limits['max_cohorts']:raise ValueError('Native allocation must cover complete16-match cohorts')
 def spec(self):
  return dict(root=str(self.root),run_id=self.run_id,assets=self.assets,execution=asdict(self.execution),seed=self.seed,recipe=self.recipe,limits=self.limits,mode=self.mode)
 @classmethod
 def from_spec(cls,v):
  v=dict(v);v['root']=Path(v['root']).resolve();v['execution']=Execution(**v['execution']);return cls(**v)
 def authorize(self,registration,authorization):
  self.validate()
  if not registration or not authorization:raise PermissionError('Separate run registration and explicit authorization required')
  reg=read(registration);auth=read(authorization)
  if reg.get('context')!=json.loads(json.dumps(self.spec())):raise PermissionError('Registration/context mismatch')
  if auth.get('registration_sha256')!=sha(registration) or auth.get('run_id')!=self.run_id or auth.get('mode')!=self.mode or auth.get('approved') is not True:raise PermissionError('Authorization does not bind this registered run')
  if self.mode=='learning' and (auth.get('scope')!='explicitly_authorized_training_run' or not auth.get('human_authority')):raise PermissionError('No authorized learning experiment')
  if self.mode=='software_test' and auth.get('scope')!='local_software_verification_no_native':raise PermissionError('Software-only authorization required')
  if parse(auth['expires_utc'])<=now():raise PermissionError('Run authorization expired')
  return reg,auth
