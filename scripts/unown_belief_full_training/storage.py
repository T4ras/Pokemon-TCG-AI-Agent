"""Atomic compressed components, a commit ledger, and locked-test access control."""
from .common import *
from .truth import select_points,align
from .audit import integrity
from unown_belief_bayes_repair.adapter import legal
import gzip,functools

def finish(job,hist,frames,result):
 started=time.process_time();hist=[dict(observation=legal(e['observation']),action=e['action']) for e in hist];selection=select_points(job['id'],hist)
 if result['status']!='DONE':return dict(result,payload=None,partition=job['partition'])
 targets=align([hist[i] for i in sorted(selection)],frames,job['seat']);checks=integrity(hist,targets,job['b'])
 # All locked-test target accesses stop at this predetermined integrity check.
 payload=dict(legal=dict(id=job['id'],entries=hist,selection={str(i):v for i,v in selection.items()}),targets=dict(id=job['id'],opponent_list=job['b'],points=targets),provenance=dict(id=job['id'],partition=job['partition'],observer_deck=job['a'],opponent_list=job['b'],observer_seat=job['seat'],chooser=job['seat'],policy_tensor=TENSOR,source_manifest_sha256=sha(OUT/'source-manifest.json'),games=result['games'],match_result=result['result'],native_receipt={k:v for k,v in result.items() if k not in ('detail','actor_errors')},status='COMPLETE',checks=checks,CPU_core_seconds=result['CPU_core_seconds'],UTC_seconds=result['UTC_seconds'],integrity_CPU_seconds=time.process_time()-started,policy_unchanged=result['policy_unchanged'],origin='new'))
 return dict(result,payload=payload,partition=job['partition'],integrity_CPU_seconds=time.process_time()-started)

class Shards:
 def __init__(self):
  self.groups={};self.committed=read(OUT/'inherited-dataset-index.json') if (OUT/'inherited-dataset-index.json').exists() else [];self.manifest=read(OUT/'inherited-dataset-shards.json')['shards'] if (OUT/'inherited-dataset-shards.json').exists() else []
 def append(self,payload):
  ident=payload['legal']['id'];part=payload['provenance']['partition'];group=self.groups.setdefault(part,[])
  if any(x['id']==ident for x in self.committed):raise InvariantFailure('Duplicate dataset match')
  group.append(payload)
  if len(group)>=16:self.flush(part)
 def flush(self,part):
  group=self.groups.get(part,[])
  if not group:return
  shard=sum(x['partition']==part for x in self.manifest);paths={};summaries=[]
  for component in ('legal','targets','provenance'):
   path=PRIVATE/'dataset'/part/(component+f'-{shard:04d}.jsonl.gz');path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
   with gzip.open(tmp,'wt',compresslevel=3) as f:
    for payload in group:f.write(json.dumps(payload[component],separators=(',',':'))+'\n')
   tmp.replace(path)
   # Roundtrip verifies byte encoding and record ordering for every component.
   with gzip.open(path,'rt') as f:roundtrip=[json.loads(line) for line in f]
   if [x['id'] for x in roundtrip]!=[x['legal']['id'] for x in group]:raise InvariantFailure('Shard roundtrip/order mismatch')
   paths[component]=dict(path=str(path.relative_to(R)),sha256=sha(path),bytes=path.stat().st_size)
  for index,x in enumerate(group):
   row=dict(id=x['legal']['id'],partition=part,shard=shard,index=index,paths=paths,selected_points=len(x['targets']['points']),callbacks=len(x['legal']['entries']))
   self.committed.append(row);append(OUT/'dataset-commit-ledger.jsonl',row);summaries.append(row)
  self.manifest.append(dict(partition=part,shard=shard,records=len(group),paths=paths));self.groups[part]=[]
  write(OUT/'dataset-shards.json',dict(schema_version='belief-dataset-v1',shards=self.manifest,committed_BO3=len(self.committed),partition_counts=dict(collections.Counter(x['partition'] for x in self.committed))))
  write(OUT/'dataset-index.json',self.committed)
  for x in group:
   p=PRIVATE/'pending-results'/(x['legal']['id']+'.json.gz')
   if p.exists():p.unlink()
 def flush_all(self):
  for part in tuple(self.groups):self.flush(part)

def import_payload(item):
 e=item['corpus'];hist=gzip_read(R/e['legal_path']);targets=gzip_read(R/e['truth_path']);assert sha(R/e['legal_path'])==e['legal_sha256'] and sha(R/e['truth_path'])==e['truth_sha256']
 return dict(legal=dict(id=item['id'],entries=hist,selection=e['selection']),targets=dict(id=item['id'],opponent_list=item['opponent_list'],points=targets),provenance=dict(item['provenance'],id=item['id'],partition='training',observer_deck=anchor(),opponent_list=item['opponent_list'],observer_seat=item['chooser'],chooser=item['chooser'],origin=item['origin'],source_legal_sha256=e['legal_sha256'],source_truth_sha256=e['truth_sha256'],checks=item['checks']))

@functools.lru_cache(maxsize=4)
def read_shard(path,expected):
 if sha(R/path)!=expected:raise InvariantFailure('Shard checksum changed')
 with gzip.open(R/path,'rt') as f:return [json.loads(line) for line in f]

def load(row,component='legal',integrity_only=False):
 if (row['partition']=='locked-test' or any('/locked-test/' in v['path'] for v in row['paths'].values())) and not integrity_only:raise PermissionError('Locked-test data are sealed; model loaders cannot open them')
 if component not in ('legal','targets','provenance'):raise ValueError(component)
 p=row['paths'][component];out=read_shard(p['path'],p['sha256'])[row['index']]
 if out['id']!=row['id']:raise InvariantFailure('Dataset index/record mismatch')
 return out
