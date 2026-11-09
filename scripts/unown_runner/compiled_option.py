"""Dormant portable option. Only a successful frozen benchmark activates it.

No benchmark paths or checkpoint defaults are used by the execution provider.
The activation certificate ships alongside the package, not its compiler caches.
"""
import sys,time
from pathlib import Path
from .context import read,sha,append

PINNED_FILES=('compilation.py','compiled_option.py','units/engine.py','units/tensor.py','units/inputs.py','units/prefix.py')

def certificate():
 package=Path(__file__).parent;path=package/'compiled-validation.json'
 if not path.exists():raise ValueError('Optional compilation is unvalidated and disabled')
 value=read(path)
 if value.get('adopted') is not True or value.get('each_speedup') is None or len(value['each_speedup'])!=3 or min(value['each_speedup'])<1.10 or not 0<=value.get('break_even_stages',99)<=3:raise ValueError('Invalid compiler adoption certificate')
 if value.get('sources')!={name:sha(package/name) for name in PINNED_FILES}:raise ValueError('Compiler implementation differs from validated sources')
 return value

def make_engine(runtime,model,opt,cohort,phase):
 import torch
 from .compilation import Bucket64Compiler,experimental_engine,graphs
 proof=certificate();context=runtime.context
 if next(model.parameters()).device.type!='cuda' or str(torch.__version__)!=proof['torch'] or torch.version.cuda!=proof['CUDA'] or list(sys.version_info[:2])!=proof['python']:raise ValueError('Compiled option requires validated CUDA backend versions')
 if torch.cuda.get_device_capability()!=tuple(proof['capability']):raise ValueError('Compiled GPU capability differs')
 cap=context.limits.get('compilation_seconds')
 if not isinstance(cap,int) or cap<=0:raise ValueError('Future run must explicitly allocate compilation time')
 compiler=getattr(runtime,'_bucket64_compiler',None)
 if compiler is None:
  started=time.monotonic();compiler=Bucket64Compiler(runtime,model,cohort,proof['keys'],compile_limit=cap)
  runtime._bucket64_compiler=compiler
  try:compiler.warm(opt,started+cap)
  finally:append(context.out/'compilation.jsonl',dict(seconds=time.monotonic()-started,compiler=compiler.summary(),recovery_cache_rebuilt=True))
 if compiler.model is not model:raise ValueError('Compiler/model identity changed; explicit cache rebuild required')
 compiler.check();engine=experimental_engine(runtime,model,cohort,compiler,phase)
 original=engine.transaction
 def transaction(*args,**kwargs):
  before=graphs();started=time.monotonic();result=original(*args,**kwargs);compiler.note_transaction(before,time.monotonic()-started);return result
 engine.transaction=transaction
 return engine
