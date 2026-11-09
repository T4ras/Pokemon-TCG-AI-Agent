"""Explicit future-run integration, never enabled for historical registrations."""
from pathlib import Path
from .io import *
from .boundary import from_unown,verify,prune_boundaries
from .integration import fatal_export,ObserverCapture

def publish(runner,contract):
 """Called after the existing runner has atomically saved its completed boundary.

Contract is installed by the future registered runner factory, never an env flag.
No collection, optimizer step, model rewrite, upload or provider call occurs here.
 """
 if not contract.get('enabled'):return
 if contract['target']<=0:raise ValueError('Explicit stopping target required')
 ref=from_unown(runner.latest,runner.journal,contract['boundary_root'],expected_sha256=sha(runner.latest),schedule_path=contract['schedule_path'],run_manifest=runner.context.out/'manifest.json',matches=contract['match_counters'](runner),resources=contract['resource_counters'](runner),target=contract['target'])
 atomic(Path(contract['boundary_root'])/'verified-reference.json',ref)
 runner.supervisor_boundary=ref
 if 'keep_boundaries' in contract:prune_boundaries(contract['boundary_root'],contract['keep_boundaries'])

def capture_observer(observer,private_folder):
 return ObserverCapture(observer,private_folder,lambda o:{'facts':o.facts,'current':o.current,'within':o.within,'game':o.game,'cached':o.cached})

def run_and_export(runner,contract,export,stop_accepting,*,resume=False):
 """Top-level remote composition: return nonzero then SystemExit, no repair wait.

The existing runner closes its workers in finally. The remote owner keeps its
supported export implementation. Register its source and callbacks in advance.
 """
 runner.supervisor_contract=contract
 try:runner.run(resume);return 0
 except BaseException as error:
  ref=getattr(runner,'supervisor_boundary',None)
  if ref is None:
   pointer=Path(contract['boundary_root'])/'verified-reference.json'
   if pointer.exists():ref=read(pointer)
  return fatal_export(stop_accepting,lambda:None,ref,export,contract['evidence_root'],error)
