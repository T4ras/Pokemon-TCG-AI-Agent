"""Run-local instances of the validated algorithms; no benchmark imports or resets."""
import collections,contextlib,copy,ctypes,functools,gzip,hashlib,importlib.util,inspect,math
import multiprocessing as mp,os,random,resource,signal,sys,textwrap,threading,time,traceback,types,uuid
from multiprocessing.connection import wait
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .context import *
from .packets import Layout,Packet

class Runtime:
 def __init__(self,context):
  context.validate();self.context=context;self.modules={};self.key='unown_runner._instance_'+uuid.uuid4().hex
  self._reference_value=None;self.current_model=None
  self.base=dict(globals(),ROOT=context.root,REPO=context.root,OUT=context.out,PRIVATE=context.private,SEED=context.seed,LR=context.recipe['lr'],MAX_EPOCHS=4,
   MAX_PACKET_BYTES=16*1024**2,MAX_ROLLOUT_BYTES=512*1024**2,MATCH_TIMEOUT=300.,REQUEST_TIMEOUT=45.,BATCH_WAIT=.002,MAX_RAM=24*1024**3,
   active_time=time.monotonic,METERS=collections.Counter(),MIX_LENGTHS=False,nullcontext=contextlib.nullcontext,Counter=collections.Counter,defaultdict=collections.defaultdict,
   NATIVE_AUTHORIZED=context.mode=='learning',WORKER_SCOPE='software' if context.mode=='software_test' else 'local' if os.environ.get('COLLECT_LOCAL_CLOCK')=='1' else 'gpu')
  a={k:Path(v).resolve() for k,v in context.assets.items()}
  self.assets=a
  clock=self.unit('clock')
  if os.environ.get('COLLECT_LOCAL_CLOCK')=='1':self.base['active_time']=clock.AwakeClock()
  self.clock=self.base['active_time'];self.scalar=self.unit('scalar');legacy=self.unit('legacy')
  targets=self.unit('targets');migration=self.unit('migration')
  self.rt=self.unit('rt',reference=self.reference,extend_rows=migration.extend_rows,PREVIOUS=a['identity_directory'],PREVIOUS_PRIVATE=a['current_tables'].parent,PARENT=a['original_tables'].parent,UP=a['released_submission'].parent)
  self.runtime=self.unit('runtime',rt=self.rt,state_hash=targets.state_hash,completed_game_returns=targets.completed_game_returns,ENVIRONMENT=a['environment'],TEXT_TABLES=a['text_tables'])
  order=self.unit('order');self.corrected=self.unit('corrected',runtime=self.runtime,actor_class=order.actor_class,replace_feature=order.replace_feature)
  self.batch=self.unit('batch',runtime=self.runtime,legacy=legacy)
  self.batch.scalar=self.scalar;self.batch.digest=self.scalar.digest;self.batch.snapshot=self.scalar.snapshot;self.batch.restore=self.scalar.restore
  self.batch.advantages=self.scalar.advantages;self.batch.state_hash=targets.state_hash
  def optimizer(model):
   opt=legacy.configure(model)
   for group in opt.param_groups:group['lr']=context.recipe['lr']
   return opt
  self.batch.optimizer=optimizer
  self.lean=self.unit('lean',batch=self.batch,runtime=self.runtime,METERS=collections.Counter())
  self.core=types.SimpleNamespace(advantages=self.scalar.advantages)
  self.epochs=self.unit('epochs',core=self.core,batch=self.batch,runtime=self.runtime,lean=self.lean)
  self.prefix=self.unit('prefix',runtime=self.runtime)
  self.baseline=self.unit('baseline',runtime=self.runtime,batch=self.batch,lean=self.lean,host_length=self.prefix.host_length)
  def boundaries(layout):return sorted({min(v,layout.tokens) for v in context.execution.boundaries}|{layout.tokens})
  self.inputs=self.unit('inputs',runtime=self.runtime,batch=self.batch,epochs=self.epochs,host_length=self.prefix.host_length,boundaries=boundaries)
  self.tensor=self.unit('tensor',runtime=self.runtime)
  self.engine=self.unit('engine',runtime=self.runtime,batch=self.batch,epochs=self.epochs,lean=self.lean,baseline=self.baseline,
   Resident=self.inputs.Resident,groups=self.inputs.groups,host_length=self.prefix.host_length,boundaries=boundaries,
   region=self.tensor.region,arguments=self.tensor.arguments,summarize=self.tensor.summarize)
  self.diagnose=self.unit('diagnose',runtime=self.runtime,batch=self.batch,DevicePacket=self.prefix.DevicePacket)
  self.checks=self.unit('checks',runtime=self.runtime,batch=self.batch,lean=self.lean,revised=self.baseline,
   DevicePacket=self.prefix.DevicePacket,outputs=self.diagnose.outputs,clone_model=self.clone_model,PRECISION_ENVELOPE=True)
  self.service=self.unit('service',runtime=self.runtime,state_hash=targets.state_hash)
  self.revised=self.unit('revised',original=self.service,host_length=self.prefix.host_length)
  profile=self.unit('profile',runtime=self.runtime)
  self.worker=self.unit('worker',runtime=self.runtime,batch=self.batch,core=self.corrected,bind=self.bind,
   CollectorProfile=profile.CollectorProfile,AwakeClock=clock.AwakeClock,RuntimeClock=clock.RuntimeClock,
   RemotePolicy=None,**{k:getattr(self.service,k) for k in ('ProtocolError','CollectionCancelled','array_sha')})
  self.pool=self.unit('pool',runtime=self.runtime,state_hash=targets.state_hash,run_worker=self.worker.run_worker,
   InferenceService=self.revised.Revised,Ledger=self.service.Ledger,ProtocolError=self.service.ProtocolError)
  self.legacy=legacy;self.state_hash=targets.state_hash
 def unit(self,name,**injected):
  path=Path(__file__).parent/'units'/(name+'.py');module=types.ModuleType(self.key+'_'+name)
  module.__dict__.update(self.base);module.__dict__.update(injected);module.__name__=self.key+'_'+name;module.__file__=str(path)
  sys.modules[module.__name__]=module
  exec(compile(path.read_text(),str(path),'exec'),module.__dict__);self.modules[name]=module;return module
 def bind(self,out,private):
  for module in self.modules.values():module.OUT=out;module.PRIVATE=private
 def reference(self):
  if self._reference_value is not None:return self._reference_value
  a=self.assets;submission=a['released_submission'];sdk=a['original_sdk']
  # Released imports use absolute ptcg/cg names. Refuse a different private SDK
  # already loaded into this process instead of silently crossing asset versions.
  for name in ('ptcg','cg'):
   existing=sys.modules.get(name)
   if existing is not None and getattr(existing,'__file__',None):
    path=Path(existing.__file__).resolve()
    if not path.is_relative_to(submission) and not path.is_relative_to(sdk):raise RuntimeError('Different pinned SDK already imported: '+name)
  sys.path[:0]=[str(sdk),str(submission)]
  from cg.sim import lib
  lib.AllCard.restype=ctypes.c_char_p;lib.AllAttack.restype=ctypes.c_char_p
  module_name=self.key+'_released';spec=importlib.util.spec_from_file_location(module_name,submission/'main.py');module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
  cwd=Path.cwd()
  try:os.chdir(self.context.root);spec.loader.exec_module(module)
  finally:os.chdir(cwd)
  if module.HEADS!=8 or module.SA.cfg is not None or module.SA.temp!=0 or module.SA.device.type!='cpu':raise ValueError('Pinned released runtime mismatch')
  from ptcg.rl import model,encoder,buffers
  from ptcg.tracker import InfoTracker
  self._reference_value=(module,model,encoder,buffers,InfoTracker);return self._reference_value
 def load_model(self,path,device):
  model=self.runtime.load_model(path,device);self.configure_precision();self.prefix.install(model);return model
 def configure_precision(self):
  self.batch.deterministic_backend();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 def clone_model(self,source,device):
  model=self.load_model(None,device);model.load_state_dict(source.state_dict());return model
 def restore_rng(self,saved,device):
  torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
  if torch.device(device).type=='cuda' and saved.get('CUDA_rng'):torch.cuda.set_rng_state_all([v.cpu() for v in saved['CUDA_rng']])
 def restore(self,path,device='cpu'):
  saved=torch.load(path,map_location=device,weights_only=False)
  if self.context.mode=='learning' and saved.get('extra',{}).get('software_verification_only'):raise PermissionError('Software probe weights cannot initialize learning/candidates')
  if any(v.is_floating_point() and v.dtype!=torch.float32 for v in saved['state_dict'].values()):raise ValueError('Checkpoint precision differs from registered float32 recipe')
  model=self.load_model(None,device);opt=self.batch.optimizer(model)
  model.load_state_dict(saved['state_dict']);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
  for group in opt.param_groups:
   if (group['lr'],group['betas'],group['eps'],group['weight_decay'])!=(self.context.recipe['lr'],(.95,.999),1e-8,0) or group.get('foreach') is not False:raise ValueError('Registered optimizer differs from checkpoint')
  self.restore_rng(saved,device);return model,opt,saved
 def make_engine(self,model,opt,cohort,phase):
  if self.context.execution.learner=='bucket64-compiled':
   from .compiled_option import make_engine
   return make_engine(self,model,opt,cohort,phase)
  return self.engine.Engine(model,cohort,self.context.execution.learner,phase=phase)
 def close(self):
  compiler=getattr(self,'_bucket64_compiler',None)
  if compiler:compiler.close();self._bucket64_compiler=None
  for module in self.modules.values():sys.modules.pop(module.__name__,None)
  self.modules.clear()

def environment_worker(context_spec,connection,index,manifest):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 runtime=Runtime(RunContext.from_spec(context_spec))
 try:runtime.worker.run_worker(connection,index,manifest)
 finally:runtime.close()

class NativeCollector:
 software_only=False
 def __init__(self,runtime,authorized=False):
  if not authorized or runtime.context.mode!='learning':raise PermissionError('Native collection needs a separately authorized learning run')
  self.runtime=runtime;self.pool=None
 def collect(self,schedule,models,deadline,safety_check=None):
  if self.pool is None:self.pool=self.runtime.pool.ConcurrentCollector(16,worker_target=functools.partial(environment_worker,self.runtime.context.spec()),manifest=read(self.runtime.context.out/'manifest.json'))
  return self.pool.collect(schedule,models,concurrency=16,deadline=deadline,safety_check=safety_check)
 def close(self):
  if self.pool:self.pool.close()
