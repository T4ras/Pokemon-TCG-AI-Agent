"""Common legal-observer/allocation pass, independent of any true deck label."""
from .common import *
from .baseline import Facts
from .allocation import allocation_one
from .deck import catalogue
import gzip,signal,resource,traceback,numpy as np

class ResourceStop(RuntimeError):pass
class OperationLimit(RuntimeError):pass

def guard():
 if (OUT/'STOP.json').exists():raise ResourceStop('Registered aggregate resource stop')
 check_time()
def alarm(*_):raise OperationLimit('Registered per-operation CPU guard')
def bounded(fn,seconds):
 signal.signal(signal.SIGPROF,alarm);signal.setitimer(signal.ITIMER_PROF,seconds)
 try:return fn()
 finally:signal.setitimer(signal.ITIMER_PROF,0)
def gzwrite(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(value,f,separators=(',',':'))
 tmp.replace(path)
def initialize():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));write(OUT/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),at_utc=iso(),max_memory_bytes=1024**3,compute_threads=1))
 def terminate(*_):raise ResourceStop('Aggregate budget termination')
 signal.signal(signal.SIGTERM,terminate)

def cache_one(job):
 cpu=time.process_time();start=now();updates=[];readouts=[];rows=[];indices=[];offsets=[0];points=[];error=None;status='FAILED';f=None;decks,cards,deck_counts,prevalence=catalogue();p=prior();q=read(OUT/'query-set.json')['card_ids'];where={c:i for i,c in enumerate(cards)};alloc_path=PRIVATE/'allocations'/(job['id']+'.npz');feature_path=PRIVATE/'features'/(job['id']+'.json.gz');evidence=job['corpus'];selected={int(i) for i in evidence['selection']}
 try:
  guard();assert sha(R/evidence['legal_path'])==evidence['legal_sha256'];hist=json.load(gzip.open(R/evidence['legal_path'],'rt'));f=Facts(metadata()['cards'])
  for i,entry in enumerate(hist):
   guard();tick=time.process_time();bounded(lambda:f.observe(entry['observation']),30);updates.append(time.process_time()-tick)
   if i not in selected:continue
   tick=time.process_time();known=bounded(f.known_hand,10);compatible=[];start_index=len(rows)
   def allocations():
    for d,deck in enumerate(decks):
     if not f.compatible(p[deck]['cards']):continue
     try:out=allocation_one(f,p[deck]['cards'],q)
     except IncompatibleAllocation:continue
     matrix=np.array([out[c] for c in q],dtype=float)
     if not np.isfinite(matrix).all() or np.any(matrix[:,0]<-1e-8) or np.any(matrix[:,0]>1+1e-8) or np.any(matrix[:,1:]<-1e-8):raise InvariantFailure('Invalid per-deck allocation')
     compatible.append(d);indices.append(d);rows.append(matrix)
   bounded(allocations,10)
   if not compatible:raise OutOfLibrary('No per-deck A allocation is valid')
   readouts.append(time.process_time()-tick);cur=entry['observation']['current'];game=cur['round'];stage='setup' if cur['turn']==0 else 'early' if cur['turn']<=4 else 'middle' if cur['turn']<=12 else 'late';ledger=[f.lower.get(c,0) for c in cards];points.append(dict(index=i,key=[game,cur['turn'],cur['turnActionCount'],cur['yourIndex']],game=game,stage=stage,lower=ledger,known=[known[c] for c in q],known_serials=sorted(s for s,(cid,domain) in f.copies.items() if domain=={'H'}),compatible=compatible));offsets.append(len(rows))
  if len(points)!=evidence['selected_decisions']:raise InvariantFailure('Registered selected points missing')
  status='COMPLETE'
 except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());status='BUDGET_EXHAUSTED' if isinstance(exc,(ResourceStop,TimeoutError)) else 'FAILED'
 finally:
  alloc_path.parent.mkdir(parents=True,exist_ok=True);tmp=alloc_path.with_name(alloc_path.name+'.tmp-'+str(os.getpid()))
  with tmp.open('wb') as file:np.savez_compressed(file,allocations=np.array(rows,dtype=float).reshape((-1,len(q),7)),decks=np.array(indices,dtype=np.int16),offsets=np.array(offsets,dtype=np.int32))
  tmp.replace(alloc_path);gzwrite(feature_path,dict(id=job['id'],points=points,final_ledger=[f.lower.get(c,0) for c in cards] if f else None,query_ids=q,vocabulary=cards))
  record=dict(id=job['id'],status=status,error=error,expected_points=evidence['selected_decisions'],completed_points=len(points),feature_path=str(feature_path.relative_to(R)),feature_sha256=sha(feature_path),allocation_path=str(alloc_path.relative_to(R)),allocation_sha256=sha(alloc_path),CPU_seconds=time.process_time()-cpu,elapsed_seconds=(now()-start).total_seconds(),update_CPU_samples=updates,readout_CPU_samples=readouts,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_legal_sha256=evidence['legal_sha256']);write(OUT/'common-results'/(job['id']+'.json'),record)
 return record
