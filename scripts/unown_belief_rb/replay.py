"""CPU-only replay. Scoring labels are loaded after inference has finished."""
from .common import *
from .baseline import Facts,baseline
from .filter import Observer as PF
from .candidate import Observer as RB
from .collapsed import CollapsedState
from .scoring import cell_scores,hierarchical
import gzip,signal,resource,traceback,copy

class BudgetExhausted(RuntimeError):pass
class UpdateLimit(RuntimeError):pass

def alarm(*_):raise UpdateLimit('Registered per-operation CPU limit')
def check():
 if (OUT/'STOP.json').exists():raise BudgetExhausted('Aggregate resource stop')
 check_time()
def bounded(fn,seconds):
 signal.signal(signal.SIGPROF,alarm);signal.setitimer(signal.ITIMER_PROF,seconds)
 try:return fn()
 finally:signal.setitimer(signal.ITIMER_PROF,0)
def gzwrite(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(value,f,separators=(',',':'))
 tmp.replace(p)
def label_key(cur):return (cur['round'],cur['turn'],cur['turnActionCount'],cur['yourIndex'])
def query_matrix(pred,q):return [[pred[c][k] for k in ('p_unknown_hand','hand_unknown_mean','deck_mean','prize_mean')] for c in q]
def matrix_prediction(matrix,q):return {c:dict(zip(('p_unknown_hand','hand_unknown_mean','deck_mean','prize_mean'),v)) for c,v in zip(q,matrix)}

def initialize():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
 # No Torch, policy, native engine, GPU or network is loaded in this process.
 write(OUT/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),at_utc=iso(),memory_limit_bytes=1024**3,threads=1))

def run_one(task):
 started=now();cpu=time.process_time();oldintegration=CollapsedState.integration_CPU;oldsteps=CollapsedState.branch_steps;item=next(x for x in read(OUT/'development-inputs.json') if x['id']==task['id']);evidence=item['corpus'];path=PRIVATE/'predictions'/(task['task_id']+'.json.gz');expected=evidence['selected_decisions'];result=dict(task,at_utc=iso(),expected=expected,attempted=True,completed=False,scored=0,supported=0,status='FAILED',method_metrics={},analysis_error=None);predictions=[];updates=[];readouts=[];fallback_causes=collections.Counter();ob=None;facts=None
 try:
  check();assert sha(R/evidence['legal_path'])==evidence['legal_sha256'];hist=json.load(gzip.open(R/evidence['legal_path'],'rt'));selection={int(k):v for k,v in evidence['selection'].items()};p=prior();lib={k:v['cards'] for k,v in p.items()};mass={k:v['prior'] for k,v in p.items()};q=read(OUT/'query-set.json')['card_ids'];facts=Facts(metadata()['cards']) if task['method']=='A' else None;ob=None if facts else (PF if task['method']=='PF' else RB)(lib,mass,metadata()['cards'],task['seed'],task['particles']);observations_supported=0
  for i,entry in enumerate(hist):
   check();tick=time.process_time();limit=None
   try:
    if ob:bounded(lambda:ob.observe(entry['observation']),30)
    else:bounded(lambda:facts.observe(entry['observation']),30)
   except UpdateLimit as exc:
    if not ob or 'baseline.py' in traceback.format_exc():raise
    ob.semantic_failure=dict(kind='INTEGRATION_RESOURCE_LIMIT',message=str(exc),callback=i);ob.status();limit=str(exc)
   updates.append(time.process_time()-tick);f=ob.facts if ob else facts;support=ob is None or not ob.degraded;observations_supported+=support
   if i not in selection:continue
   known=bounded(f.known_hand,10);known_sids=sorted(s for s,(c,d) in f.copies.items() if d=={'H'});tick=time.process_time()
   if support:pr=bounded(lambda:ob.predict(q) if ob else baseline(f,lib,mass,q),10)
   else:pr=bounded(lambda:baseline(f,lib,mass,q),10);fallback_causes[ob.degraded['kind']]+=1
   readouts.append(time.process_time()-tick);cur=entry['observation']['current'];predictions.append(dict(index=i,key=list(label_key(cur)),game=cur['round'],supported=support,prediction=query_matrix(pr,q),known={str(k):v for k,v in known.items()},known_serials=known_sids,variance=[dict(total=pr[c].get('hand_total_variance'),within=pr[c].get('within_component_variance'),between=pr[c].get('between_component_variance')) for c in q] if task['method']=='RB' and support else None));result['supported']+=support
  result.update(completed=True,callbacks=len(hist),supported_callbacks=observations_supported)
  # Labels and A probabilities are deliberately below the completed inference
  # loop; neither can reach a tracker transition or query.
  assert sha(R/evidence['truth_path'])==evidence['truth_sha256'];labels=json.load(gzip.open(R/evidence['truth_path'],'rt'));labelmap={tuple(t['key']):t for t in labels};spec=read(OUT/'query-set.json');Apoints={x['index']:x for x in predictions} if task['method']=='A' else {x['index']:x for x in json.load(gzip.open(PRIVATE/'predictions'/('A-'+task['id']+'.json.gz'),'rt'))['predictions']};decisions=[]
  for point in predictions:
   truth=labelmap[tuple(point['key'])];assert set(point['known_serials'])<=set(truth['hand_serials']);known={int(k):v for k,v in point['known'].items()};A=matrix_prediction(Apoints[point['index']]['prediction'],q);pred=matrix_prediction(point['prediction'],q);pr=dict(A=A,Full_plus_fallback=pred);metrics=cell_scores(pr,truth,known,spec,point['supported']);point['metrics']={k:v for k,v in metrics.items() if '/Full_plus_fallback/' in k};decisions.append(dict(game=point['game'],metrics=point['metrics']));result['scored']+=1
  if result['scored']!=expected:raise InvariantFailure('Expected selected points missing')
  result.update(status='COMPLETE',method_metrics=hierarchical(decisions))
 except BaseException as exc:
  result['status']='BUDGET_EXHAUSTED' if isinstance(exc,(BudgetExhausted,TimeoutError)) else 'FAILED';result['analysis_error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());result['shared_correctness_failure']='baseline.py' in result['analysis_error']['traceback'] and not isinstance(exc,UpdateLimit)
 finally:
  result.update(CPU_seconds=time.process_time()-cpu,UTC_seconds=(now()-started).total_seconds(),update_CPU_samples=updates,readout_CPU_samples=readouts,nonfallback_fraction=result['supported']/expected if expected else None,fallback_causes=dict(fallback_causes),depletion_count=sum(x.get('event')=='STRATUM_DEPLETED' for x in ob.transitions) if ob else 0,recovery_attempts=len(ob.recoveries) if ob else 0,recovery_successes=sum(x['success'] for x in ob.recoveries) if ob else 0,unsupported_count=int(bool(ob and ob.semantic_failure)),peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exact_integration_CPU_seconds=CollapsedState.integration_CPU-oldintegration,exact_branch_steps=CollapsedState.branch_steps-oldsteps,prediction_path=str(path.relative_to(R)))
  gzwrite(path,dict(task=task,predictions=predictions,analysis_error=result['analysis_error'],recoveries=ob.recoveries if ob else [],transitions=ob.transitions if ob else []));result['prediction_sha256']=sha(path);result.update(CPU_seconds=time.process_time()-cpu,UTC_seconds=(now()-started).total_seconds());write(OUT/'results'/(task['task_id']+'.json'),result)
 return result
