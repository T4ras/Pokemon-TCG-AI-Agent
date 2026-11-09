"""Development only: collect missing IDs once, then independent offline analysis."""
from .common import *
import concurrent.futures as cf,multiprocessing as mp,resource,traceback,gzip,copy
import numpy as np

def isolate_process_group():
 try:os.setsid()
 except PermissionError:
  if os.getpgrp()!=os.getpid():raise
 assert os.getpgrp()==os.getpid()

def verify_sources():
 m=read(OUT/'source-manifest.json');bad=[p for p,v in m['files'].items() if sha(R/p)!=v['sha256']]
 if bad:raise InvariantFailure('Frozen repair source/policy changed: '+str(bad))

def worker_init():
 verify_sources()
 from .worker import initialize
 initialize()

def collect_one(job):
 from .worker import one
 return one(job)

def remaining_jobs():
 anchor=next(k for k,v in prior().items() if v['group']=='anchor');native={r['id'] for r in read(OUT/'inherited-native-records.json')};schedule=read(OUT/'development-schedule.json');remaining=[r for r in schedule if r['id'] not in native];assert len(native)==6 and len(remaining)==290
 return [dict(id=r['id'],partition='development',a=anchor,b=r['opponent_list'],seat=r['anchor_seat'],seed=r['policy_seed'],tracker_seed=r['tracker_seed'],startup_check=i<2) for i,r in enumerate(remaining)]

def storage_check():
 corpus=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
 details=sum(p.stat().st_size for p in (PRIVATE/'detail-temp').rglob('*') if p.is_file())
 if corpus>CORPUS_CAP:raise RuntimeError('Registered8GiB compressed corpus cap reached')
 if details>DETAIL_CAP:raise RuntimeError('Registered2GiB detail cap reached')
 return corpus

def collection_record(result):
 append(OUT/'native-matches.jsonl',result);append(OUT/'attempt-ledger.jsonl',dict(event='NATIVE_RESULT',at_utc=iso(),id=result['id'],native=result.get('native_collection_status'),truth=result.get('truth_alignment_status'),status=result['status']))
 storage_check()
 if result['status']!='DONE' or result.get('native_collection_status')!='COMPLETE' or result.get('truth_alignment_status')!='COMPLETE':raise RuntimeError('Collection/truth/actor protocol failure: '+result['id'])

def collect():
 jobs=remaining_jobs();issued={r['id'] for r in rows(OUT/'attempt-ledger.jsonl') if r['event']=='ISSUED'}
 if issued:raise RuntimeError('Existing issued intents require reconciliation; no ambiguous native retry')
 new=[];pending={};index=0
 with cf.ProcessPoolExecutor(WORKERS,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
  while index<len(jobs) or pending:
   check_time();cap=2 if index<2 or len(new)<2 else WORKERS
   while len(pending)<cap and index<len(jobs) and (index<2 or len(new)>=2):
    job=jobs[index];index+=1;append(OUT/'attempt-ledger.jsonl',dict(event='ISSUED',at_utc=iso(),id=job['id'],job=job));pending[pool.submit(collect_one,job)]=job
   finished,_=cf.wait(pending,timeout=10,return_when=cf.FIRST_COMPLETED);failure=None
   for future in finished:
    job=pending.pop(future)
    try:
     result=future.result();collection_record(result);new.append(result)
    except BaseException as exc:
     failure=exc;append(OUT/'collection-failures.jsonl',dict(at_utc=iso(),id=job['id'],error=type(exc).__name__+': '+str(exc)))
   if len(new)==2 and not (OUT/'startup.json').exists():write(OUT/'startup.json',dict(at_utc=iso(),passed=all(r.get('startup_parity') for r in new),new_native_attempts=2,ids=[r['id'] for r in new],inherited_native=6,observer_not_on_collection_path=True,source_manifest_sha256=sha(OUT/'source-manifest.json')))
   progress('collection',issued=index,new_complete=len(new),new_limit=290,total_development_complete=6+len(new),held_out_accessed=False)
   if failure:
    for future,job in pending.items():
     try:collection_record(future.result())
     except BaseException as exc:append(OUT/'collection-failures.jsonl',dict(at_utc=iso(),id=job['id'],error=type(exc).__name__+': '+str(exc)))
    raise failure
 return new

def inherited():
 result=[]
 for row in read(OUT/'inherited-native-records.json'):
  r=copy.deepcopy(row);r.update(source_run='astra-unown-belief-native-validation-002',source_record_status=r['status'],status='DONE',native_collection_status='COMPLETE',truth_alignment_status='COMPLETE',observer_status='PENDING',scoring_status='PENDING');result.append(r)
 return result

def analyze_one(record,N=None,seed=None,stability=False):
 from .corpus import replay,gzwrite
 N=N or 512;seed=record['tracker_seed'] if seed is None else seed;start=time.process_time()
 try:r=replay(record['corpus'],seed,N)
 except BaseException as exc:r=dict(observer_status='FAILED',scoring_status='FAILED',belief_metrics={},analysis_error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),selected_expected=record['corpus']['selected_decisions'],selected_scored=0,supported_selected=0,selected_support_fraction=0.,observation_expected=record['corpus']['legal_observations'],observation_scored=0,observation_support_fraction=0.,predictions=[],tracker_CPU_seconds=time.process_time()-start)
 suffix=f'-{N}-{seed}' if stability else '';path=PRIVATE/('stability' if stability else 'analysis')/(record['id']+suffix+'.json.gz');gzwrite(path,r);out=copy.deepcopy(record)
 # Historical scores are kept in run002. Only the new analysis populates
 # this row; a failed reanalysis never retains an old successful score.
 for key in ('observer_error','failure_kind','belief_metrics','partial_metrics','recoveries','transitions','update_timing_samples','meaningful_update_with_readout_samples','readout_samples','belief_counts'):out.pop(key,None)
 out.update({k:v for k,v in r.items() if k not in ('predictions','analysis_error')});out.update(analysis_path=str(path.relative_to(R)),analysis_sha256=sha(path),analysis_CPU_seconds=time.process_time()-start,particles=N,inference_seed=seed)
 if r.get('analysis_error'):out['analysis_failure_signature']=r['analysis_error']['type']+': '+r['analysis_error']['message'].split('{')[0].split('[')[0][:120]
 out.setdefault('belief_metrics',{})
 for kind,winner in (('BO3',record['result']),('first_game',record['games'][0]['result'])):
  for label,value in (('W',record['seat']),('D',2),('L',1-record['seat'])):out['belief_metrics'][f'native/{kind}_{label}']=float(winner==value)
 out['belief_metrics']['coverage/full_supported']=r.get('selected_support_fraction') or 0.
 out['belief_metrics']['coverage/observation_supported']=r.get('observation_support_fraction') or 0.
 return out

def stability_task(task):return analyze_one(*task,stability=True)

def summarize_stability(rows,records):
 ids=read(OUT/'stability-schedule.json')['development_matches'];results=[]
 for ident in ids:
  rs=[r for r in rows if r['id']==ident];pairs=[]
  for a in range(len(rs)):
   pa=json.load(gzip.open(R/rs[a]['analysis_path'],'rt'))['predictions'];am={p['index']:p for p in pa}
   for b in range(a+1,len(rs)):
    pb=json.load(gzip.open(R/rs[b]['analysis_path'],'rt'))['predictions'];bm={p['index']:p for p in pb};common=sorted(am.keys()&bm.keys());expected=records[ident]['corpus']['selected_decisions'];shared=[i for i in common if am[i]['supported'] and bm[i]['supported']]
    pairs.append(dict(a=[rs[a]['particles'],rs[a]['inference_seed']],b=[rs[b]['particles'],rs[b]['inference_seed']],paired_selected_fraction=len(common)/expected,shared_full_points=len(shared),mean_abs_probability_difference=float(np.mean([np.abs(np.array(am[i]['p_full'])-bm[i]['p_full']).mean() for i in common])) if common else None,shared_full_difference=float(np.mean([np.abs(np.array(am[i]['p_full'])-bm[i]['p_full']).mean() for i in shared])) if shared else None,support_disagreement=sum(am[i]['supported']!=bm[i]['supported'] for i in common)/len(common) if common else None))
  results.append(dict(id=ident,pairs=pairs,run_coverage=[dict(particles=r['particles'],seed=r['inference_seed'],supported=r.get('selected_support_fraction'),observer_status=r['observer_status']) for r in rs]))
 ds=[p['mean_abs_probability_difference'] for r in results for p in r['pairs'] if p['mean_abs_probability_difference'] is not None];coverage=[p['paired_selected_fraction'] for r in results for p in r['pairs']]
 return dict(at_utc=iso(),histories=16,replays=len(rows),budgets=[512,2048],seeds=[11,29,47],new_native_matches=0,mean_abs_probability_discrepancy=float(np.mean(ds)) if ds else None,mean_paired_point_coverage=float(np.mean(coverage)),per_history=results,larger_budget_is_not_ground_truth=True,fallback_included_in_all_in_differences=True)

def main():
 isolate_process_group();write(OUT/'local-pid.json',dict(pid=os.getpid(),process_group=os.getpgrp(),at_utc=iso()));exclusive(OUT/'runner-intent.json',dict(at_utc=iso(),pid=os.getpid(),restart_automatic=False));start=now();cpu=time.process_time();state='INCOMPLETE';reason='Not finished';analyses=[]
 try:
  verify_sources();assert read(OUT/'native-gate.json')['passed'];new=collect();native=inherited()+new;assert len({r['id'] for r in native})==296;write(OUT/'development-corpus.json',native);verify_sources()
  with cf.ProcessPoolExecutor(WORKERS,mp_context=mp.get_context('spawn')) as pool:
   for result in pool.map(analyze_one,native,chunksize=1):
    check_time();storage_check();analyses.append(result);append(OUT/'analysis-records.jsonl',result);progress('offline_analysis',completed=len(analyses),planned=296,held_out_accessed=False)
   from .scoring import summarize
   summary=summarize(analyses,'development');summary.update(analysis_failures=sum(r['scoring_status']!='COMPLETE' for r in analyses),complete_field_inference_valid=all(r['scoring_status']=='COMPLETE' for r in analyses),exploratory_development=True,held_out_accessed=False);write(OUT/'development-analysis.json',summary)
   lookup={r['id']:r for r in native};ids=read(OUT/'stability-schedule.json')['development_matches'];tasks=[(lookup[i],N,s) for i in ids for N in (512,2048) for s in (11,29,47)];stable=[]
   for result in pool.map(stability_task,tasks,chunksize=1):check_time();storage_check();stable.append(result);append(OUT/'stability-records.jsonl',result);progress('offline_stability',completed=len(stable),planned=96)
   write(OUT/'stability-summary.json',summarize_stability(stable,lookup))
  state='COMPLETE';reason='All296 development native completions retained; repaired analyses and96 stability replays attempted. Analysis failures, if any, remain explicit. Held-out untouched; no automatic follow-up.'
 except BaseException as exc:
  state='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED';reason=type(exc).__name__+': '+str(exc);write(OUT/'runner-failure.json',dict(at_utc=iso(),reason=reason,traceback=traceback.format_exc()))
 finally:
  native=rows(OUT/'native-matches.jsonl');seen={r['id'] for r in native}
  for path in (OUT/'native-results').glob('*.json'):
   row=read(path)
   if row['id'] not in seen:
    append(OUT/'native-matches.jsonl',row);append(OUT/'attempt-ledger.jsonl',dict(event='RECONCILED_RESULT',at_utc=iso(),id=row['id'],native=row.get('native_collection_status'),truth=row.get('truth_alignment_status')));native.append(row);seen.add(row['id'])
  ledger=rows(OUT/'attempt-ledger.jsonl');cost=dict(at_utc=iso(),state=state,reason=reason,inherited_native=6,new_issued=sum(r['event']=='ISSUED' for r in ledger),new_native_attempts=len(list((OUT/'intents').glob('*.json'))),new_native_complete=sum(r.get('native_collection_status')=='COMPLETE' for r in native),new_truth_complete=sum(r.get('truth_alignment_status')=='COMPLETE' for r in native),development_total_complete=6+sum(r.get('native_collection_status')=='COMPLETE' for r in native),analysis_attempted=len(rows(OUT/'analysis-records.jsonl')),held_out_attempts=0,held_out_accessed=False,wall_seconds=(now()-start).total_seconds(),controller_CPU_seconds=time.process_time()-cpu,worker_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,storage_bytes=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()),GPU_seconds=0,training_updates=0,paid_model_API=0,power_changes=False,sleep_inhibition=False,automatic_followup=False);write(OUT/'completion.json',cost)
  from .report import render
  render(state,reason,True)
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
