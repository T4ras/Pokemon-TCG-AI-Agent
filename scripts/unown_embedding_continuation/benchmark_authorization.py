"""Explicit permission to evaluate a frozen checkpoint after a training stop."""
from .common import *

def final_evaluation_allowed():
 training=read(OUT/'training-completion.json')
 if training['evaluate']:return True
 path=OUT/'checkpoint-benchmark-registration.json'
 if not path.exists():return False
 registration=read(path)
 if registration.get('scope')!='explicit_frozen_checkpoint_CPU_evaluation' or registration.get('attempts')!=512:
  raise PermissionError('Invalid CPU benchmark authorization')
 if registration.get('training_allowed') is not False or registration.get('GPU_allowed') is not False:
  raise PermissionError('Benchmark must authorize CPU evaluation only')
 for name,expected in registration['frozen_files'].items():
  if sha(ROOT/name)!=expected:raise RuntimeError('Frozen benchmark input changed: '+name)
 if registration['final_tensor_hash']!=training['final_tensor_hash']:
  raise RuntimeError('Benchmark checkpoint identity mismatch')
 return True
