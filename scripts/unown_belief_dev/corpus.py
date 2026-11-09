"""Native corpus persistence and offline observer replay, after actors finish."""
from unown_belief_dev.common import *
from unown_belief_dev.adapter import legal
from unown_belief_dev.truth import select_points,align,key
from unown_belief_dev.filter import Observer
from unown_belief_dev.baseline import baseline
from unown_belief_dev.scoring import cell_scores,hierarchical
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
 start=time.process_time();rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;hist=json.load(gzip.open(R/evidence['legal_path'],'rt'));p=prior();lib={k:v['cards'] for k,v in p.items()};m={k:v['prior'] for k,v in p.items()};ob=Observer(lib,m,metadata()['cards'],seed,particles);spec=read(OUT/'query-set.json');q=spec['card_ids'];selected={int(k):v for k,v in evidence['selection'].items()};labels=json.load(gzip.open(R/evidence['truth_path'],'rt'));labelmap={tuple(t['key']):t for t in labels};predictions=[];decisions=[];counts=collections.Counter();fallbacks=collections.Counter();timings=[];readouts=[];coverage=collections.defaultdict(collections.Counter);first=None;error=None
 for index,entry in enumerate(hist):
  cur=entry['observation']['current'];stage='setup' if cur['turn']==0 else 'early' if cur['turn']<=4 else 'middle' if cur['turn']<=12 else 'late';group=f'game{cur["round"]}/{stage}';coverage[group]['observations_expected']+=1;coverage[group]['selected_expected']+=index in selected
 for i,e in enumerate(hist):
  check_time();cur=e['observation']['current'];game=cur['round'];turn=cur['turn'];stage='setup' if turn==0 else 'early' if turn<=4 else 'middle' if turn<=12 else 'late'
  try:
   result=ob.observe(e['observation']);supported=not ob.degraded;counts['SUPPORTED' if supported else 'FALLBACK']+=1
   group=f'game{game}/{stage}';coverage[group]['observations']+=1;coverage[group]['supported_observations']+=supported
   if not supported and first is None:
    transition=next((x for x in reversed(ob.transitions) if x.get('event')=='STRATUM_DEPLETED'),None)
    first=dict(observation_index=i,game=game,turn=turn,stage=stage,reason=ob.degraded,transition=transition)
   if i not in selected:continue
   true=labelmap.get(key(cur))
   if true is None:raise InvariantFailure('Selected decision has no separate truth label')
   known=ob.facts.known_hand();known_serials={sid for sid,(c,d) in ob.facts.copies.items() if d=={'H'}}
   if not known_serials<=set(true['hand_serials']):raise InvariantFailure('Certified serial absent from truth hand')
   tick=time.perf_counter();A=baseline(ob.facts,lib,m,q);At=time.perf_counter()-tick
   if supported:
    tick=time.perf_counter();F=ob.predict(q);ft=time.perf_counter()-tick;B=baseline(ob.facts,lib,ob.masses(),q)
   else:F=B=A;ft=At;fallbacks[ob.degraded['kind']]+=1
   if any(v['hand_unknown_mean']<-1e-8 for v in F.values()):raise InvariantFailure('Full states contradict certified identity count')
   times=ob.timings[-1]+ft;timings.append(times);readouts.append(ft);pr=dict(A=A,B_plus_fallback=B,Full_plus_fallback=F);metrics=cell_scores(pr,true,known,spec,supported);decisions.append(dict(game=game,metrics=metrics));coverage[group]['selected']+=1;coverage[group]['supported_selected']+=supported
   predictions.append(dict(index=i,game=game,stage=stage,supported=supported,p_full=[F[c]['p_unknown_hand'] for c in q],known=dict(known)))
  except Exception as exc:
   import traceback
   error=dict(index=i,game=game,stage=stage,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());break
 expected=len(selected);scored=len(decisions);unsupported=ob.semantic_failure is not None
 status='FAILED' if error else 'UNSUPPORTED' if unsupported else 'COMPLETE'
 result=dict(observer_status=status,scoring_status='FAILED' if error else 'COMPLETE',belief_metrics=hierarchical(decisions) if not error else {},partial_metrics=hierarchical(decisions) if error else None,belief_counts=dict(counts),selected_expected=expected,selected_scored=scored,supported_selected=sum(v['supported'] for v in predictions),selected_support_fraction=sum(v['supported'] for v in predictions)/expected if expected else None,observation_expected=len(hist),observation_scored=sum(counts.values()),observation_support_fraction=counts['SUPPORTED']/len(hist) if hist else None,coverage_by_game_stage=dict(coverage),first_fallback=first,later_selected_affected=sum(not x['supported'] and x['index']>first['observation_index'] for x in predictions) if first else 0,fallback_counts=dict(fallbacks),analysis_error=error,tracker_CPU_seconds=time.process_time()-start,update_timing_samples=ob.timings,meaningful_update_with_readout_samples=timings,readout_samples=readouts,recoveries=ob.recoveries,transitions=ob.transitions,recovery_steps_used=2000000*(particles//512)-ob.budget['remaining'],incremental_RSS_MiB=max(0,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss_before)/1024,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,predictions=predictions)
 return result

def finish(job,hist,frames,result):
 # This function is on the native worker path. It never instantiates an
 # observer and never scores labels; valid collection proceeds independently.
 result.update(opponent_list=job['b'],anchor_seat=job['seat'],partition='development',policy_seed=job['seed'],tracker_seed=job['tracker_seed'],native_collection_status='COMPLETE' if result['status']=='DONE' else 'FAILED',observer_status='PENDING',scoring_status='PENDING')
 try:
  result['corpus']=capture(job,hist,frames,result['status']);result['truth_alignment_status']='COMPLETE' if result['corpus'].get('truth_status')=='aligned' else 'FAILED'
  if result['truth_alignment_status']!='COMPLETE':result['status']='FAILED';result['failure_kind']='COLLECTION_TRUTH'
 except Exception as exc:
  import traceback
  target=PRIVATE/'failures'/f'{job["id"]}-collection.json';write(target,dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()));result.update(status='FAILED',truth_alignment_status='FAILED',failure_kind='COLLECTION_TRUTH',diagnostic_path=str(target.relative_to(R)))
 write(OUT/'native-results'/(job['id']+'.json'),result)
 return result
