"""Native corpus persistence and offline observer replay, after actors finish."""
from unown_belief_native_common import *
from unown_belief_native_adapter import legal
from unown_belief_native_truth import select_points,align,key
from unown_belief_native_filter import Observer
from unown_belief_native_baseline import baseline
from unown_belief_native_scoring import cell_scores,hierarchical
import gzip,resource

def gzwrite(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
 with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(value,f,separators=(',',':'))
 temp.replace(path)

def capture(job,hist,frames,status):
 hist=[dict(observation=legal(e['observation']),action=e['action']) for e in hist];selected=select_points(job['id'],hist);path=PRIVATE/'legal'/f'{job["id"]}.json.gz';gzwrite(path,hist)
 evidence=dict(legal_path=str(path.relative_to(R)),legal_sha256=sha(path),legal_bytes=path.stat().st_size,legal_observations=len(hist),selected_decisions=len(selected),selection={str(k):v for k,v in selected.items()})
 if status!='DONE':return dict(evidence,truth_status='not_aligned_after_native_failure')
 chosen=[hist[i] for i in sorted(selected)];labels=align(chosen,frames,job['seat']);target=PRIVATE/'labels'/f'{job["id"]}.json.gz';gzwrite(target,labels)
 return dict(evidence,truth_path=str(target.relative_to(R)),truth_sha256=sha(target),truth_bytes=target.stat().st_size,truth_status='aligned',aligned_decisions=len(labels))

def replay(evidence,seed,particles=512,persist=False):
 start=time.process_time();rss_before=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')));rss_max=rss_before;hist=json.load(gzip.open(R/evidence['legal_path'],'rt'));p=prior();lib={k:v['cards'] for k,v in p.items()};m={k:v['prior'] for k,v in p.items()};ob=Observer(lib,m,metadata()['cards'],seed,particles);spec=read(OUT/'query-set.json');q=spec['card_ids'];selected={int(k):v for k,v in evidence['selection'].items()};labels=json.load(gzip.open(R/evidence['truth_path'],'rt'));labelmap={tuple(t['key']):t for t in labels};predictions=[];decisions=[];counts=collections.Counter();fallbacks=collections.Counter();meaningful_times=[];readout_times=[]
 for i,e in enumerate(hist):
  check_time();result=ob.observe(e['observation']);rss_max=max(rss_max,int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))));counts[result['status']]+=1
  if i not in selected:continue
  true=labelmap.get(key(e['observation']['current']))
  if true is None:raise InvariantFailure('Selected legal decision has no separate truth label')
  known=ob.facts.known_hand();known_serials={s for s,(c,d) in ob.facts.copies.items() if d=={'H'}}
  if not known_serials<=set(true['hand_serials']):raise InvariantFailure('Certified hand serial not present in diagnostic hand')
  at=time.perf_counter();A=baseline(ob.facts,lib,m,q);A_seconds=time.perf_counter()-at;supported=result['status']=='SUPPORTED'
  if supported:
   ft=time.perf_counter();F=ob.predict(q);feature_seconds=time.perf_counter()-ft;B=baseline(ob.facts,lib,ob.masses(),q)
  else:
   F=B=A;feature_seconds=A_seconds;fallbacks[(ob.degraded or {}).get('kind','unknown')]+=1
  readout_times.append(feature_seconds);meaningful_times.append(ob.timings[-1]+feature_seconds)
  pr=dict(A=A,B_plus_fallback=B,Full_plus_fallback=F);metrics=cell_scores(pr,true,known,spec,supported);decisions.append(dict(game=selected[i]['game'],metrics=metrics));predictions.append(dict(index=i,game=selected[i]['game'],supported=supported,p_full=[F[c]['p_unknown_hand'] for c in q],known=dict(known),prediction=pr if persist else None))
 timings=sorted(ob.timings);recovery=sorted(r['elapsed_seconds'] for r in ob.recoveries)
 def pct(xs,p):return xs[min(len(xs)-1,int((len(xs)-1)*p))] if xs else None
 return dict(belief_metrics=hierarchical(decisions),belief_counts=dict(counts),selected_decisions=len(decisions),fallback_counts=dict(fallbacks),tracker_CPU_seconds=time.process_time()-start,incremental_RSS_MiB=(rss_max-rss_before)/1024,update_timing_samples=ob.timings,meaningful_update_with_readout_samples=meaningful_times,readout_samples=readout_times,update_latency=dict(p50=pct(timings,.5),p95=pct(timings,.95),p99=pct(timings,.99)),reconstruction_latency=dict(p50=pct(recovery,.5),p95=pct(recovery,.95),maximum=max(recovery) if recovery else None),recoveries=ob.recoveries,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,predictions=predictions)

def finish(job,hist,frames,result):
 cpu_start=time.process_time();result.update(opponent_list=job['b'],anchor_seat=job['seat'],partition=job['partition'],policy_seed=job['seed'],tracker_seed=job['tracker_seed'])
 try:
  result['corpus']=capture(job,hist,frames,result['status'])
  if result['status']=='DONE':
   diagnostics=replay(result['corpus'],job['tracker_seed']);private=PRIVATE/'predictions'/f'{job["id"]}.json.gz';gzwrite(private,diagnostics.pop('predictions'));result.update(diagnostics)
   for kind,winner in [('BO3',result['result']),('first_game',result['games'][0]['result'])]:
    for label,value in [('W',job['seat']),('D',2),('L',1-job['seat'])]:result['belief_metrics'][f'native/{kind}_{label}']=float(winner==value)
   result['prediction_path']=str(private.relative_to(R))
 except BaseException as e:
  import traceback
  private_error=PRIVATE/'failures'/f'{job["id"]}.json';write(private_error,dict(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()))
  result.update(status='FAILED',failure_kind='OBSERVER_IMPLEMENTATION',observer_error=type(e).__name__+': see private diagnostic',observer_diagnostic_path=str(private_error.relative_to(R)),observer_diagnostic_sha256=sha(private_error))
 result['post_collection_CPU_seconds']=time.process_time()-cpu_start
 return result
