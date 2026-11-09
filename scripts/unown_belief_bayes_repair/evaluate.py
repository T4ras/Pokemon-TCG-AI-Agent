"""Legal A/B inference is separate from true-deck and hidden-label scoring."""
from .common import *
from .deck import catalogue,deck_weights,mix_allocations
from .cache import guard,bounded,gzwrite
from .scoring import cell_scores,hierarchical
import numpy as np,gzip,time,traceback,resource

FIELDS=('p_unknown_hand','hand_unknown_mean','deck_mean','prize_mean')
def as_dict(matrix,queries):return {c:dict(zip(FIELDS,v[:4])) for c,v in zip(queries,matrix)}

def legal_prediction(point,indices,allocations,prevalence,theta,beta):
 mask=np.zeros(len(prevalence),dtype=bool);mask[indices]=True;times={};tick=time.process_time();wa=deck_weights(point['lower'],mask,prevalence,beta=0);times['A_weight']=time.process_time()-tick;tick=time.process_time();a=mix_allocations(wa[indices],allocations);times['A_readout']=time.process_time()-tick;tick=time.process_time();wb=deck_weights(point['lower'],mask,prevalence,theta,beta);times['B_weight']=time.process_time()-tick;tick=time.process_time();b=mix_allocations(wb[indices],allocations);times['B_readout']=time.process_time()-tick
 return a,b,wa,wb,times

def evaluate_one(item,common_record,theta,beta,features):
 started=time.process_time();decks,cards,counts,priorw=catalogue();q=read(OUT/'query-set.json')['card_ids'];spec=read(OUT/'query-set.json');data=np.load(R/common_record['allocation_path']);aa=data['allocations'];ds=data['decks'];offsets=data['offsets'];raw=[];timings=collections.Counter();latencies=[]
 # This whole inference loop has no true-deck index or hidden labels.
 for i,point in enumerate(features['points']):
  guard();lo,hi=offsets[i:i+2];indices=ds[lo:hi];alloc=aa[lo:hi];a,b,wa,wb,t=bounded(lambda:legal_prediction(point,indices,alloc,priorw,theta,beta),10);timings.update(t);latencies.append(t);raw.append(dict(point=point,A=a,B=b,A_deck=wa,B_deck=wb,indices=indices,alloc=alloc))
 # Scorer-only information enters here, after legal predictions exist.
 y=decks.index(item['opponent_list']);labels={tuple(t['key']):t for t in json.load(gzip.open(R/item['corpus']['truth_path'],'rt'))};methods={m:[] for m in ('A','B','known_deck_diagnostic')};bygame={m:collections.defaultdict(list) for m in methods};bystage={m:collections.defaultdict(list) for m in methods};saved=[];scoring_start=time.process_time()
 for e in raw:
  p=e['point'];truth=labels[tuple(p['key'])];known={c:k for c,k in zip(q,p['known'])};assert set(p['known_serials'])<=set(truth['hand_serials']);matches=np.where(e['indices']==y)[0]
  if len(matches)!=1:raise InvariantFailure('True list missing from shared allocation support')
  oracle=e['alloc'][matches[0]];a_dict=as_dict(e['A'],q);out={}
  for name,pred in (('A',e['A']),('B',e['B']),('known_deck_diagnostic',oracle)):
   scores=cell_scores(dict(A=a_dict,Full_plus_fallback=as_dict(pred,q)),truth,known,spec,True);scores={k.replace('/Full_plus_fallback/','/prediction/'):v for k,v in scores.items() if '/Full_plus_fallback/' in k};scores['coverage/scored']=1.
   if name!='known_deck_diagnostic':
    w=e[name+'_deck'];scores.update({'deck/log_loss':float(-np.log(max(w[y],np.finfo(float).tiny))),'deck/Brier':float(np.sum(w*w)-2*w[y]+1),'deck/true_probability':float(w[y]),'deck/top1_accuracy':float(int(np.argmax(w))==y),'deck/remaining_composition_MAE':float(np.mean(np.abs((w@counts-np.array(p['lower']))-(counts[y]-np.array(p['lower'])))))})
   # Count variances are output features, not treated as independent samples.
   out[name]=pred.tolist();methods[name].append(dict(game=p['game'],metrics=scores));bygame[name][str(p['game'])].append(dict(game=p['game'],metrics=scores));bystage[name][p['stage']].append(dict(game=p['game'],metrics=scores))
  saved.append(dict(index=p['index'],game=p['game'],stage=p['stage'],known_hand_counts=p['known'],total_hand_means={m:[v[1]+k for v,k in zip(matrix,p['known'])] for m,matrix in out.items()},methods=out,deck_A=e['A_deck'].tolist(),deck_B=e['B_deck'].tolist()))
 path=PRIVATE/'predictions'/(item['id']+'.json.gz');gzwrite(path,dict(id=item['id'],methods=saved,known_deck_diagnostic_scorer_only=True));record=dict(id=item['id'],opponent_list=item['opponent_list'],chooser=item['chooser'],outer_fold=item['outer_fold'],status='COMPLETE',expected=item['corpus']['selected_decisions'],scored=len(saved),metrics={m:hierarchical(v) for m,v in methods.items()},by_game={m:{g:hierarchical(v) for g,v in groups.items()} for m,groups in bygame.items()},by_stage={m:{g:hierarchical(v) for g,v in groups.items()} for m,groups in bystage.items()},timing=dict(timings),mixture_latency_samples=latencies,common_CPU_seconds=common_record['CPU_seconds'],scoring_and_export_CPU_seconds=time.process_time()-scoring_start,CPU_seconds=time.process_time()-started,prediction_path=str(path.relative_to(R)),prediction_sha256=sha(path),peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write(OUT/'evaluation-results'/(item['id']+'.json'),record);return record
