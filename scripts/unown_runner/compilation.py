"""Experimental bucket64 compilation; production selection stays disabled pending evidence."""
import copy,hashlib,importlib.util,inspect,linecache,sys,textwrap,time,types,uuid
from pathlib import Path
import torch
from .context import write,sha

def counters():
 from torch._dynamo.utils import counters as values
 return {k:dict(v) for k,v in values.items()}

def graphs():return counters().get('stats',{}).get('unique_graphs',0)

def sync(model):
 if next(model.parameters()).is_cuda:torch.cuda.synchronize()

class Bucket64Compiler:
 """Real per-shape modules, literal lengths, explicit differentiable table inputs."""
 def __init__(self,runtime,model,cohort,keys,*,backend='inductor',compile_limit=600):
  self.runtime=runtime;self.model=model;self.keys=[tuple(k) for k in keys];self.backend=backend
  self.device=next(model.parameters()).device;self.modules=[];self.functions={};self.table_calls=0;self.calls=0
  self.before=graphs();self.start_counters=counters();self.initialized=False;self.seconds=0.;self.timings=[];self.compile_limit=compile_limit
  self.max_graphs=2*(len(self.keys)+1)+2;self.warm_graphs=None
  if backend!='inductor' and (backend!='eager' or runtime.context.mode!='software_test'):raise ValueError('Only an explicitly labeled local tracing backend is allowed')
  if backend=='inductor' and self.device.type!='cuda':raise ValueError('Pinned GPU verification required for Inductor benchmark')
  if len(self.keys)>4 or not self.keys:raise ValueError('At most four fixed shape keys required')
  if not hasattr(torch._dynamo.config,'error_on_recompile'):raise RuntimeError('Pinned compiler cannot enforce post-warm guard stability')
  torch._dynamo.config.suppress_errors=False
  for name in ('recompile_limit','cache_size_limit'):
   if hasattr(torch._dynamo.config,name):setattr(torch._dynamo.config,name,16)
  if backend=='inductor':
   import torch._inductor.config as config
   if not hasattr(config,'pattern_matcher'):raise RuntimeError('Cannot preserve declared attention backend')
   config.pattern_matcher=False
  self.folder=runtime.context.private/'execution-cache/generated';self.folder.mkdir(parents=True,exist_ok=True)
  self.example_engine=runtime.engine.Engine(model,cohort,'bucket64',phase=100)
  self.examples={}
  for _,group in runtime.inputs.groups(cohort['rows'],'bucket64'):
   layout=group[0]['packet'].layout;length=runtime.prefix.host_length([r['packet'] for r in group]);length=next(n for n in runtime.inputs.boundaries(layout) if n>=length)
   key=(repr(layout),length)
   if key in self.keys:self.examples.setdefault(key,group[:64])
  if set(self.examples)!=set(self.keys):raise ValueError('Missing valid warm examples for a declared compiled shape')
  original=type(model)._card_table;source=textwrap.dedent(inspect.getsource(original));lines=source.splitlines()
  matches=[i for i,line in enumerate(lines) if line.strip()=='ft = getattr(self, "_frozen_tables", None)']
  if len(matches)!=1:raise RuntimeError('Changed pinned inference-table semantics')
  i=matches[0]
  if lines[i+1].strip()!='if ft is not None:' or lines[i+2].strip()!='return ft':raise RuntimeError('Changed table-cache guard')
  source='\n'.join(lines[:i]+lines[i+3:])+'\n'
  source+='\ndef table_call():\n return _card_table(model)\n'
  module=self.register('tables',dict(original.__globals__,model=model),source)
  self.table_function=self.capture(module.table_call)
  for key in self.keys:
   layout=self.examples[key][0]['packet'].layout;length=key[1]
   namespace=dict(runtime.rt.functions(layout)[0].__globals__)
   source=textwrap.dedent(inspect.getsource(runtime.rt.reference()[1].PTCGTransformer._core))
   replacements={
    'def _core(self, obs):':'def _core(self, obs, tables):',
    'card_vec, attack_vec = self._card_table()':'card_vec, attack_vec = tables',
    'L = int(valid.sum(1).max().item())':f'L = {int(length)}',
    '    x = self.token_proj(torch.cat([':'    for extra_tool in range(5, TOK_INT):\n        et = tok_i[:, :, extra_tool]\n        tool_vec = tool_vec + card_vec[et] * (et > 0).unsqueeze(-1)\n    x = self.token_proj(torch.cat(['}
   for old,new in replacements.items():
    if source.count(old)!=1:raise RuntimeError('Changed pinned tensor semantics: '+old)
    source=source.replace(old,new)
   # The wrapper belongs to this actual module, not ptcg.rl.model's copied name.
   source+='''
def region(observation,card_table,attack_table,old,old_lp,actions,mask,returns,advantages,real_mask,denominator):
 state=_core(model,observation,(card_table,attack_table))
 values=model.value_mlp(torch.cat([state['r'],state['g']],-1)).squeeze(-1)
 loss,metrics=objective(state['logits'],values,old,old_lp,actions,mask,returns,advantages,real_mask,denominator)
 return state['logits'],values,loss,metrics
'''
   namespace.update(model=model,objective=runtime.tensor.objective)
   module=self.register('L'+str(length),namespace,source,rebind_take=True)
   self.functions[key]=self.capture(module.region)
 def register(self,label,namespace,source,rebind_take=False):
  name='unown_compiled_'+uuid.uuid4().hex+'_'+label;path=self.folder/(name+'.py')
  path.write_text(source);module=types.ModuleType(name);module.__dict__.update(namespace)
  module.__name__=name;module.__file__=str(path);module.__spec__=importlib.util.spec_from_file_location(name,path)
  module.__loader__=module.__spec__.loader;module.__package__='';module.__cached__=None
  module.__dict__.pop('_host_length',None)
  sys.modules[name]=module
  if rebind_take:
   original=namespace['_take'];module._take=types.FunctionType(original.__code__,module.__dict__,original.__name__,original.__defaults__,original.__closure__)
  linecache.cache[str(path)]=(len(source),None,source.splitlines(keepends=True),str(path))
  exec(compile(source,str(path),'exec'),module.__dict__)
  self.modules.append(module);return module
 def capture(self,fn):
  kwargs=dict(backend=self.backend,fullgraph=True,dynamic=False)
  if self.backend=='inductor':kwargs['mode']='reduce-overhead'
  return torch.compile(fn,**kwargs)
 def eligible(self,key):return tuple(key) in self.functions
 def function(self,layout,length):self.calls+=1;return self.functions[repr(layout),length]
 def iteration_begin(self):
  if self.device.type=='cuda' and hasattr(torch.compiler,'cudagraph_mark_step_begin'):torch.compiler.cudagraph_mark_step_begin()
 def tables(self):
  self.table_calls+=1
  # Clones are outside graph capture; CloneBackward retains table gradients.
  return tuple(t.clone() for t in self.table_function())
 def warm(self,opt,deadline):
  snapshot=self.runtime.scalar.snapshot(self.model,opt,{})
  before=self.runtime.scalar.state_fingerprints(snapshot);started=time.monotonic()
  torch._dynamo.config.error_on_recompile=False # Only this declared cold-build interval.
  try:
   for key in self.keys:
    group=self.examples[key];view=self.example_engine.repo.view(group,128,pad_to=64);fn=self.functions[key]
    for training in (True,False):
     for repeat in range(2):
      if time.monotonic()>=deadline:raise TimeoutError('Registered compilation allowance exhausted')
      self.iteration_begin();self.model.train(training);self.runtime.batch.clear(self.model);self.model.zero_grad(set_to_none=True);t=time.monotonic()
      if training:
       tables=self.tables();outputs=tuple(x.clone() for x in fn(*self.runtime.tensor.arguments(view,tables)));outputs[2].backward()
      else:
       with torch.no_grad():tables=self.tables();outputs=tuple(x.clone() for x in fn(*self.runtime.tensor.arguments(view,tables)))
      sync(self.model);self.timings.append(dict(shape=key,training=training,repeat=repeat,seconds=time.monotonic()-t));del tables,outputs
      self.check()
   if graphs()<=self.before:raise RuntimeError('No compiled graph generated')
   self.warm_graphs=graphs();self.initialized=True
  finally:
   self.seconds=time.monotonic()-started
   self.example_engine.close();self.example_engine=None
   self.runtime.scalar.restore(self.model,opt,snapshot,{})
   after=self.runtime.scalar.state_fingerprints(self.runtime.scalar.snapshot(self.model,opt,{}))
   if before!=after:raise RuntimeError('Compilation warmup changed model/optimizer/gradients/RNG')
   torch._dynamo.config.error_on_recompile=True
  self.check()
 def check(self):
  current=counters()
  if graphs()-self.before>self.max_graphs:raise RuntimeError('Registered graph bound exceeded')
  if self.initialized and graphs()!=self.warm_graphs:raise RuntimeError('Unexpected post-warm recompilation')
  if sum(current.get('graph_break',{}).values())>sum(self.start_counters.get('graph_break',{}).values()):raise RuntimeError('Fullgraph region broke')
  if self.seconds>self.compile_limit:raise TimeoutError('Compilation cap exceeded')
 def note_transaction(self,before_graphs,seconds):
  if graphs()!=before_graphs:raise RuntimeError('Post-warm transaction recompiled')
  self.check()
 def summary(self):
  return dict(status='PASS' if self.initialized else 'FAILED',backend=self.backend,mode='reduce-overhead' if self.backend=='inductor' else 'LOCAL_TRACE_ONLY',
   keys=self.keys,physical_batch=64,other_shapes='explicit eager; compiled/eager calls counted',cold_seconds=self.seconds,graphs=graphs()-self.before,max_graphs=self.max_graphs,
   post_warm_recompilations=graphs()-self.warm_graphs if self.warm_graphs is not None else None,graph_counters=counters(),warmups=self.timings,
   compiled_function_calls=self.calls,compiled_table_calls=self.table_calls,modules=[dict(name=m.__name__,path=str(Path(m.__file__).relative_to(self.runtime.context.root)),sha256=sha(m.__file__)) for m in self.modules],
   shared_mutable_length=False,table_cache_across_updates=False,fullgraph=True,suppress_errors=False)
 def close(self):
  if self.example_engine:self.example_engine.close();self.example_engine=None
  self.functions.clear()
  for module in self.modules:sys.modules.pop(module.__name__,None)
  self.modules.clear()

def experimental_engine(runtime,model,cohort,compiler=None,phase=100):
 engine=runtime.engine.Engine(model,cohort,'bucket64',phase=phase)
 if compiler is not None:
  if not compiler.initialized:engine.close();raise RuntimeError('Unverified cold compiler')
  engine.compiler=compiler
 return engine
