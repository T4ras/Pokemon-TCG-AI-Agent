"""Run-local additions to the validated collector/eager bucket64; no original edits."""
from .common import *
from .correction import attach,corrected_outputs,configure,frozen_digest
from .belief import OnlineBelief
from unown_runner.context import RunContext,Execution,FIXED_RECIPE
from unown_runner.runtime import Runtime
import torch,numpy as np,inspect,textwrap,types,copy,functools

ZERO=np.zeros(BELIEF_DIM,np.float32);ZERO.flags.writeable=False

def check_privileged_checkpoint(saved,allow=False):
 if saved.get('extra',{}).get('privileged_oracle') and not allow:raise PermissionError('Private oracle diagnostic checkpoint cannot enter an ordinary belief/submission path')


def context(arm='state',suffix='readiness'):
 assets=read(OUT/'assets.json');return RunContext(R,RUN+'-'+suffix+'-'+arm,assets,Execution('bucket64'),SEED,dict(FIXED_RECIPE,lr=1e-4),dict(max_cohorts=128,native_attempts=2048,session_seconds=10800,export_reserve_seconds=300,storage_bytes=4*1024**3,log_bytes=256*1024**2),'learning')

def packet_belief(packet,device,dtype):
 if hasattr(packet,'belief_tensor'):value=packet.belief_tensor
 elif hasattr(packet,'belief'):value=packet.belief
 elif hasattr(packet,'rows'):value=np.stack([r['belief'] for r in packet.rows])
 else:raise InvariantFailure('Correctedmodelpacketmissingcollectionbelief')
 out=torch.as_tensor(np.array(value,copy=True) if isinstance(value,np.ndarray) and not value.flags.writeable else value,device=device,dtype=dtype)
 if out.ndim==1:out=out[None,:]
 return out

def with_belief(packet,actor):
 if actor.pilot_arm is None:return packet
 vector=actor.belief_observer.readout() if actor.belief_observer is not None else ZERO
 if vector.shape!=(BELIEF_DIM,) or not np.isfinite(vector).all():raise InvariantFailure('Badactorbeliefinput')
 out=copy.copy(packet);out.belief=vector;return out

def actor_type(rt,remote=False):
 base=rt.runtime.PilotActor;source=rt._pilot_actor_source
 old="""with torch.no_grad():
   c,_=self.model.core(packet);logits=c['logits'][0];v=self.model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze()
   if not torch.isfinite(logits).all() or not torch.isfinite(v):raise FloatingPointError('Nonfinite inference logits/value')
   prob=torch.softmax(logits,-1).cpu();lp=torch.log_softmax(logits,-1).cpu()
   action=int(torch.multinomial(prob,1,generator=self.generator)) if self.sample else int(logits.argmax())"""
 assert source.count(old)==1
 if remote:
  new="""with torch.no_grad():
   c,_=self.model.core(packet);logits=c['logits'][0];v=c['value'].squeeze()
   if not torch.isfinite(logits).all() or not torch.isfinite(v):raise FloatingPointError('Nonfinite inference logits/value')
   prob=c['probs'][0];lp=c['log_probs'][0]
   action=int(torch.multinomial(prob,1,generator=self.generator)) if self.sample else int(logits.argmax())"""
  source=source.replace(old,new).replace("obs=rt.actor_view(raw);game=obs['current'].get('round',1)","obs=self.current;assert obs is not None;game=obs['current'].get('round',1)")
  source=source.replace("old_probs=prob.numpy().astype(np.float32),game=game", "old_probs=prob.numpy().astype(np.float32),request_id=c['request_id'],belief=packet.belief,game=game")
 else:
  source=source.replace("v=self.model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze()", "v=c['pilot_value'].squeeze() if 'pilot_value' in c else self.model.value_mlp(torch.cat([c['r'],c['g']],-1)).squeeze()")
  source=source.replace("old_probs=prob.numpy().astype(np.float32),game=game", "old_probs=prob.numpy().astype(np.float32),belief=packet.belief,game=game")
 source=source.replace('self.memory.update(obs)',"self.memory.update(obs)\n if self.belief_observer is not None:self.belief_observer.observe(obs)")
 needle='packet=rt.encode(obs,picked,stop,self.forced_run,self.memory.encoder_tracker())';assert source.count(needle)==1;source=source.replace(needle,needle+'\n  packet=with_belief(packet,self)')
 namespace=dict(rt.runtime.__dict__,with_belief=with_belief);exec(compile(source,'<belief-pilot-conditional-act>','exec'),namespace)
 class Pilot(base):
  def __init__(self,model,*a,**kw):
   super().__init__(model,*a,**kw);self.pilot_arm=getattr(model,'pilot_arm',None);self.belief_observer=OnlineBelief() if self.pilot_arm=='belief' else None;rt._pilot_actors.append(self)
  def reset_match(self):
   super().reset_match()
   if getattr(self,'belief_observer',None) is not None:self.belief_observer.reset_match()
 Pilot.act=namespace['act'];return Pilot


def value_functions(rt):
 # Same replay/grouping/check functions; only use the permitted critic residual.
 for name,functions in [('batch',['scalar_outputs','forward']),('baseline',['forward']),('diagnose',['outputs']),('checks',['collected'])]:
  module=getattr(rt,name)
  for function in functions:
   source=textwrap.dedent(inspect.getsource(getattr(module,function)))
   for var in ('c','s','state'):
    expression="model.value_mlp(torch.cat(["+var+"['r'],"+var+"['g']],-1)).squeeze(-1)";source=source.replace(expression,"("+var+"['pilot_value'] if 'pilot_value' in "+var+" else "+expression+")")
   exec(compile(source,'<belief-pilot-permitted-value-readout>','exec'),module.__dict__)
 # Collection checker uses outputs function captured at assembly.
 rt.checks.outputs=rt.diagnose.outputs

class PilotRuntime(Runtime):
 def __init__(self,ctx,arm='state'):
  super().__init__(ctx);self.arm=arm;self._pilot_actor_source=textwrap.dedent(inspect.getsource(self.runtime.PilotActor.act));self._pilot_actors=[]
  value_functions(self)
  original_repository=self.baseline.Repository
  class BeliefRepository(original_repository):
   def packet(self,rows):
    p=super().packet(rows);p.belief=np.stack([r['belief'] for r in rows]);return p
  self.baseline.Repository=BeliefRepository
  original_resident=self.inputs.Resident
  class BeliefResident(original_resident):
   def __init__(repo,cohort,device,config,phase=100):
    super().__init__(cohort,device,config,phase)
    for layout,table in repo.tables.items():
     rows=[r for r in repo.rows if r['packet'].layout==layout]
     for row in rows:
      if not np.array_equal(row['belief'],row['packet'].belief):raise InvariantFailure('Rolloutbeliefchanged')
     arr=np.stack([r['belief'] for r in rows]);table['belief']=torch.as_tensor(arr,device=device);repo.counts['cohort_transfer_bytes']+=arr.nbytes
  self.inputs.Resident=BeliefResident;self.engine.Resident=BeliefResident
  old_arguments=self.tensor.arguments
  self.engine.arguments=lambda view,tables:(*old_arguments(view,tables),view['belief'])
  def region(model,layout,length):
   base=self.tensor.core_with_tables(layout,length)
   def run(observation,card_table,attack_table,old,old_lp,actions,mask,returns,advantages,real_mask,denominator,belief):
    with torch.no_grad():state=base(model,observation,(card_table,attack_table))
    state['pilot_mask']=observation[:,slice(*layout.offsets['opt_mask'])]>0
    logits,value,_=corrected_outputs(model,state,belief);loss,metrics=self.tensor.objective(logits,value,old,old_lp,actions,mask,returns,advantages,real_mask,denominator);return logits,value,loss,metrics
   return run
  self.engine.region=region
  # Preserve existing service batching/routing; add the immutable belief tensor.
  original_prepared=self.service.PreparedPacket;runtime=self
  class BeliefPrepared(original_prepared):
   def __init__(prepared,packets,device,metrics):
    super().__init__(packets,device,metrics);prepared.valid_length=runtime.prefix.host_length(packets)
    if any(hasattr(p,'belief') for p in packets):
     if not all(hasattr(p,'belief') for p in packets):raise InvariantFailure('Mixedmissingbeliefpayloads')
     for p in packets:p.belief.flags.writeable=False
     prepared.belief_tensor=torch.as_tensor(np.stack([p.belief for p in packets]),device=device);metrics['belief_transfer_bytes']+=prepared.belief_tensor.numel()*4
  self.service.PreparedPacket=BeliefPrepared;self.revised.Prepared=BeliefPrepared
  source=textwrap.dedent(inspect.getsource(self.service.InferenceService.predict));needle="values=model.value_mlp(torch.cat([state['r'],state['g']],-1)).reshape(-1)";assert source.count(needle)==1;source=source.replace(needle,"values=state['pilot_value'].reshape(-1) if 'pilot_value' in state else model.value_mlp(torch.cat([state['r'],state['g']],-1)).reshape(-1)");namespace=dict(self.service.__dict__);exec(compile(source,'<belief-pilot-service-value>','exec'),namespace);self.service.InferenceService.predict=namespace['predict']
  original_reconstruct=self.service.InferenceService.reconstruct
  def reconstruct(service,reduced):
   rows=original_reconstruct(service,reduced)
   for r in rows:
    if not np.array_equal(r['belief'],r['packet'].belief):raise InvariantFailure('IPCbeliefcontentmismatch')
    r['belief'].flags.writeable=False;r['packet'].belief.flags.writeable=False
   return rows
  self.service.InferenceService.reconstruct=reconstruct
  remote=self.worker.RemotePolicy
  class BeliefRemote(remote):
   def __init__(proxy,connection,job,role,version):
    super().__init__(connection,job,role,version)
    if role=='learner':proxy.pilot_arm=job['arm']
  self.worker.RemotePolicy=BeliefRemote;self.worker.remote_actor_class=lambda:actor_type(self,True)
  old_play=self.corrected.play
  def play(*args,**kw):
   self._pilot_actors=[];start=time.process_time();result,rows=old_play(*args,**kw);actors=self._pilot_actors;costs=collections.Counter()
   for actor in actors:
    if actor.belief_observer is not None:costs.update(actor.belief_observer.costs)
   result['belief_costs']=dict(costs);result['complete_worker_CPU_seconds']=time.process_time()-start
   return result,rows
  self.corrected.play=play
  self.batch.optimizer=configure
 def load_base(self,device='cpu'):
  model=super().load_model(self.context.assets['start_checkpoint'],device);assert self.state_hash(model)==POLICY_TENSOR
  for p in model.parameters():p.requires_grad_(False)
  return model
 def load_corrected(self,path=None,device='cpu'):
  model=self.load_base(device);attach(model);model.pilot_arm=self.arm
  if path:
   saved=torch.load(path,map_location=device,weights_only=False);check_privileged_checkpoint(saved,getattr(self,'allow_privileged_oracle',False));model.load_state_dict(saved['state_dict'],strict=True)
  original=model.core
  def core(packet):
   with torch.no_grad():state,value=original(packet)
   state['pilot_mask']=value[:,slice(*packet.layout.offsets['opt_mask'])]>0
   belief=packet_belief(packet,next(model.parameters()).device,state['r'].dtype);logits,v,delta=corrected_outputs(model,state,belief);state=dict(state,base_logits=state['logits'],logits=logits,pilot_value=v,pilot_delta=delta);return state,value
  model.core=core;return model
 def restore(self,path,device='cpu',allow_disposable=False):
  saved=torch.load(path,map_location=device,weights_only=False)
  check_privileged_checkpoint(saved,getattr(self,'allow_privileged_oracle',False))
  if saved.get('extra',{}).get('software_verification_only') and not allow_disposable:raise PermissionError('Disposable software weights cannot initialize pilot training')
  model=self.load_corrected(device=device);model.load_state_dict(saved['state_dict'],strict=True);opt=configure(model);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
  for group in opt.param_groups:
   if group['betas']!=(.95,.999) or group['eps']!=1e-8 or group['weight_decay']!=0 or group.get('foreach') is not False:raise ValueError('Pilotoptimizerrecipe mismatch')
  self.restore_rng(saved,device);return model,opt,saved
 def clone_model(self,source,device):
  model=self.load_corrected(device=device);model.load_state_dict(source.state_dict(),strict=True);return model
 def local_actor_type(self):return actor_type(self,False)


def environment_worker(context_spec,arm,connection,index,manifest):
 os.environ['CUDA_VISIBLE_DEVICES']='';rt=PilotRuntime(RunContext.from_spec(context_spec),arm)
 try:rt.worker.run_worker(connection,index,manifest)
 finally:rt.close()
