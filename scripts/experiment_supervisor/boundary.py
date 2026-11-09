"""Immutable recovery commits layered over existing runner checkpoints.

No checkpoint is unpickled until its trusted expected SHA has passed. Call the
runner inspector only on locally trusted artifacts. Never accept an agent's SHA.
"""
import os, shutil, uuid
from pathlib import Path
from .io import *
REQUIRED={'model','optimizer','rng','schedule','counters','pending'}

def _json_sidecar(value):
 import numpy as np
 if isinstance(value,np.ndarray):
  return {'__numpy_array__':True,'dtype':value.dtype.str,'shape':list(value.shape),'values':value.tolist()}
 if isinstance(value,np.generic):return _json_sidecar(value.item())
 if isinstance(value,dict):return {k:_json_sidecar(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [_json_sidecar(v) for v in value]
 return value
def commit(root,components,metadata):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if set(components)!=REQUIRED:raise ValueError('Incomplete recovery components')
 for k in ('updates','trained_decisions','schedule_cursor','remaining_target','matches','resources'):
  if k not in metadata:raise ValueError('Missing recovery metadata: '+k)
 token=uuid.uuid4().hex;stage=root/('.staging-'+token);stage.mkdir()
 inventory={};copied={}
 for role,source in components.items():
  source=Path(source).resolve()
  if not source.is_file():raise ValueError('Missing recovery component: '+role)
  # A single existing .pt contains multiple roles; copy it only once.
  if str(source) not in copied:
   dest=stage/(str(len(copied))+'-'+source.name);shutil.copyfile(source,dest)
   with dest.open('r+b') as f:os.fsync(f.fileno())
   copied[str(source)]=dest.name
  name=copied[str(source)];inventory[role]={'path':name,'sha256':sha(stage/name)}
 record={'format':1,'components':inventory,'metadata':metadata,'created_utc':utc()}
 atomic(stage/'commit.json',record);sync_dir(stage)
 target=root/token;os.replace(stage,target);sync_dir(root)
 # Pointer moves only after an immutable committed boundary is complete.
 atomic(root/'latest.json',{'directory':token,'commit_sha256':sha(target/'commit.json')})
 return {'root':str(target.resolve()),'commit_sha256':sha(target/'commit.json')}
def verify(reference):
 root=Path(reference['root']);p=root/'commit.json'
 if sha(p)!=reference['commit_sha256']:raise ValueError('Commit marker checksum mismatch')
 record=read(p)
 if record['format']!=1 or set(record['components'])!=REQUIRED:raise ValueError('Incomplete recovery marker')
 for item in record['components'].values():
  if sha(relative(root,item['path']))!=item['sha256']:raise ValueError('Corrupt recovery component')
 for k in ('updates','trained_decisions','schedule_cursor','remaining_target','matches','resources'):
  if k not in record['metadata']:raise ValueError('Missing counters')
 return record

def from_unown(checkpoint,journal,root,*,expected_sha256,schedule_path,run_manifest,matches,resources,target):
 """Opt-in adapter; does not edit historical .pt or journals. CPU only."""
 if sha(checkpoint)!=expected_sha256:raise ValueError('Untrusted checkpoint')
 import torch
 data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 for key in ('state_dict','optimizer','step','torch_rng','numpy_rng','python_rng','CUDA_rng','extra'):
  if key not in data:raise ValueError('Missing checkpoint field: '+key)
 e=data['extra'];required=('execution','cohort_reference','runner_phase','runner_state','journal_bytes','journal_sha256','journal_seq','run_manifest_sha256')
 if any(k not in e for k in required):raise ValueError('Incomplete runner recovery fields')
 if e['runner_state'] in ('COLLECTING','FAILED'):raise ValueError('Interrupted work needs explicit scientific disposition')
 if sha(run_manifest)!=e['run_manifest_sha256']:raise ValueError('Runner manifest does not match checkpoint')
 raw=Path(journal).read_bytes();n=e['journal_bytes']
 if len(raw)!=n or hashlib.sha256(raw).hexdigest()!=e['journal_sha256']:raise ValueError('Journal work beyond or before durable cursor; do not replay')
 x=e['execution'];trained=x.get('unique_used_decisions');collected=x.get('additional_collected')
 if not isinstance(trained,int) or not isinstance(collected,int) or trained<0 or collected<trained:raise ValueError('Missing decision counter')
 # Pending rollouts are included and hashed, not silently discarded.
 ref=e['cohort_reference'];pending=Path(journal).parent/'supervisor-pending.json'
 # Use a new sidecar directory; no mutations to original experiment evidence.
 side=Path(root)/('runner-sidecars-'+uuid.uuid4().hex);side.mkdir(parents=True)
 atomic(side/'counters.json',_json_sidecar(dict(step=data['step'],execution=x,journal_seq=e['journal_seq'])))
 atomic(side/'pending.json',ref)
 if ref:
  cohort=relative(Path(checkpoint).parents[1],ref['path'])
  if sha(cohort)!=ref['sha256']:raise ValueError('Corrupt pending cohort')
  shutil.copyfile(cohort,side/'pending-cohort.pt')
  atomic(side/'pending.json',{'reference':ref,'cohort_sha256':sha(side/'pending-cohort.pt')})
 # Journal and schedule bytes are copied together as one schedule envelope.
 shutil.copyfile(journal,side/'journal.jsonl');shutil.copyfile(schedule_path,side/'schedule.json')
 atomic(side/'schedule-envelope.json',{'schedule':read(schedule_path),'journal_sha256':sha(journal),'journal':raw.decode('utf-8')})
 # Pending data must be embedded in a single role file for generic verification.
 if ref:
  pending_file=side/'pending-cohort.pt'
 else:pending_file=side/'pending.json'
 result=commit(root,{'model':checkpoint,'optimizer':checkpoint,'rng':checkpoint,'schedule':side/'schedule-envelope.json','counters':side/'counters.json','pending':pending_file},dict(updates=data['step'],trained_decisions=trained,schedule_cursor=e['runner_phase'],remaining_target=max(0,target-collected),collected_decisions=collected,stopping_counter=collected,target_counter='additional_collected',matches=matches,resources=resources,runner_state=e['runner_state'],pending_reference=ref,source_checkpoint_sha256=expected_sha256))
 # Scratch bytes are already verified inside the committed boundary.
 assert side.resolve().parent==Path(root).resolve() and side.name.startswith('runner-sidecars-')
 shutil.rmtree(side)
 return result


def prune_boundaries(root,keep):
 """Optional, explicit new-execution retention. Never prune pinned incident data."""
 if not 2<=keep<=4:raise ValueError('Retain at least two verified boundaries')
 root=Path(root).resolve();pointer=read(root/'latest.json')['directory']
 commits=sorted((p for p in root.iterdir() if p.is_dir() and (p/'commit.json').exists()),key=lambda p:p.stat().st_mtime_ns,reverse=True)
 ledger=read(root/'compact-commits.json') if (root/'compact-commits.json').exists() else {}
 for p in commits:
  record=read(p/'commit.json');ledger[p.name]={'commit_sha256':sha(p/'commit.json'),'metadata':record['metadata']}
 atomic(root/'compact-commits.json',ledger)
 for p in commits[keep:]:
  if p.name!=pointer and not (p/'PINNED').exists():
   assert p.parent==root and len(p.name)==32
   shutil.rmtree(p)
