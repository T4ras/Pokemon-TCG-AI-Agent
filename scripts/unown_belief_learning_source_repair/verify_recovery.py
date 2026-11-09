"""Fixed pretraining parity checks on retained non-test inputs."""
from .common import *
from .storage import load
from .features import encode
import numpy as np

def main():
 start=time.process_time();parent=R/'experiments/astra-unown-belief-learning-dataset-v1-execution-003';files=read(parent/'source-manifest.json')['files']
 for p,v in files.items():
  if sha(R/p)!=v['sha256']:raise InvariantFailure('Historical source changed: '+p)
 entries={x['id']:x for x in read(OUT/'split-manifest.json')['entries']};library=prior();records=[read(p) for p in (OUT/'feature-results').glob('*.json')];index={r['id']:r for r in read(OUT/'inherited-dataset-index.json')};selected=[]
 for family in sorted({v['archetype'] for v in prior().values()}):
  for partition in ('training','validation'):
   candidates=[r for r in records if r['partition']==partition and library[entries[r['id']]['opponent_list']]['archetype']==family]
   selected.append(min(candidates,key=lambda r:stable(SEED,'recovery-parity',r['id'])))
 write(OUT/'allocator-parity-sample.json',dict(at_utc=iso(),ids=[r['id'] for r in selected],rule='one hashed existing cache per archetype/partition; legal inputs only; no model scores/test access'))
 checks=[]
 for r in selected:
  legal=load(index[r['id']],'legal');actual,info=encode(legal['entries'],legal['selection'],r['partition']=='validation')
  if sha(R/r['feature_path'])!=r['feature_sha256']:raise InvariantFailure('Historical feature cache changed')
  with np.load(R/r['feature_path']) as old:
   assert set(actual)==set(old.files)
   for key,value in actual.items():np.testing.assert_array_equal(value,old[key],err_msg=r['id']+':'+key)
  checks.append(dict(id=r['id'],callbacks=len(legal['entries']),selected_points=len(info['points']),CPU_seconds=info['CPU_seconds'],all_arrays_exact=True))
  print(r['id'],'exact',flush=True)
 write(OUT/'allocator-parity-gate.json',dict(at_utc=iso(),passed=True,CPU_seconds=time.process_time()-start,historical_source_files_verified=len(files),BO3=len(checks),callbacks=sum(x['callbacks'] for x in checks),selected_points=sum(x['selected_points'] for x in checks),all_feature_arrays_bitwise_equal=True,checks=checks,locked_test_access=False,new_native_matches=0))
if __name__=='__main__':main()
