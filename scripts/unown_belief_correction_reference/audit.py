"""Fixed retained-history timing/parity; zero new native matches or labels."""
from unown_belief_correction.common import *
from unown_belief_correction.belief import OnlineBelief,current_features,frozen_belief_model
from unown_belief_full_training.storage import load as load_record
from .cache import FixedDataCache
import cProfile,pstats,io,copy,resource,signal,numpy as np,torch,traceback
RUN4='astra-unown-belief-correction-ppo-execution-007';O=R/'experiments'/RUN4;P=R/'data/private'/RUN4

def replay(histories,save=False):
 cpu=time.process_time();wall=time.monotonic();vectors=[];features=[];snapshots=[];callbacks=forced=0;costs=collections.Counter()
 for ident,hist in histories:
  observer=OnlineBelief()
  for index,entry in enumerate(hist['entries']):
   observer.observe(entry['observation']);callbacks+=1;sel=entry['observation']['select'];n=len(sel['option']);is_forced=sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1;forced+=is_forced
   if is_forced:continue
   vector=observer.readout();vectors.append(vector.copy())
   if save:
    # Full features at every genuine callback are kept private solely for parity.
    feature=current_features(observer.facts,observer.current,observer.within);features.append(feature)
    key=hashlib.sha256(('belief-recovery-profile:'+ident+':'+str(index)).encode()).hexdigest();snapshots.append(dict(history=ident,index=index,game=observer.game,key=key,facts=copy.deepcopy(observer.facts),within=observer.within.copy(),observation=copy.deepcopy(observer.current)))
  costs.update(observer.costs);observer.reset_match();assert not observer.facts.lower
 return dict(CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,callbacks=callbacks,forced_callbacks=forced,readouts=len(vectors),costs=dict(costs)),vectors,features,snapshots

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(1200,1201))
 def cpu_limit(*args):raise TimeoutError('Registered1200CPU-second audit cap')
 signal.signal(signal.SIGXCPU,cpu_limit)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);P.mkdir(parents=True,exist_ok=True);start=time.process_time();reg=read(O/'repair-registration.json');index={x['id']:x for x in read(R/'experiments/astra-unown-belief-classifier-full-training-001/dataset-index.json')};histories=[(ident,load_record(index[ident],'legal')) for ident in reg['retained_history_ids']];frozen_belief_model();prof=cProfile.Profile()
 try:
  prof.enable();base,bv,bf,snapshots=replay(histories,True);prof.disable();prof.dump_stats(str(P/'uncached-profile.pstats'));stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumtime').print_stats(25);(O/'uncached-profile.txt').write_text(stream.getvalue())
  with FixedDataCache() as cache:fast,fv,ff,_=replay(histories,True);calls=dict(cache.calls)
  assert len(bv)==len(fv) and len(bf)==len(ff);error=0.
  for a,b in zip(bv,fv):assert np.array_equal(a,b);error=max(error,float(np.max(np.abs(a-b))))
  for a,b in zip(bf,ff):
   assert a.keys()==b.keys()
   for key in a:assert np.array_equal(a[key],b[key]),key
  # Timed comparison has the same work; profiling inflates baseline, so retime
  # immutable snapshots without profiling for an unbiased readout speed ratio.
  selected=[]
  for ident in reg['retained_history_ids']:
   games=sorted({x['game'] for x in snapshots if x['history']==ident})
   for game in games:selected.extend(sorted([x for x in snapshots if x['history']==ident and x['game']==game],key=lambda x:x['key'])[:4])
  selected=selected[:48]
  from unown_belief_full_training.features import public
  for x in selected:
   old=x['observation'];c=old['current'];cur={k:c[k] for k in ('round','yourIndex','firstPlayer','turn','turnActionCount')}
   cur.update({k:c[k] for k in ('supporterPlayed','stadiumPlayed','energyAttached','retreated') if k in c})
   cur['players']=[dict(handCount=a['handCount'],deckCount=a['deckCount'],prize=[None]*len(a.get('prize') or []),active=[{}]*sum(v is not None for v in a.get('active') or []),bench=[{}]*sum(v is not None for v in a.get('bench') or [])) for a in c['players']]
   sel=old['select'];mini=dict(current=cur,select=dict(minCount=sel['minCount'],maxCount=sel['maxCount'],option=[None]*len(sel['option'])))
   assert np.array_equal(public(old),public(mini));x['observation']=mini;x['public_scalar_observation_only']=True
  torch.save(dict(states=selected,legal_actor_visible_only=True,no_hidden_targets=True,not_training_experience=True),P/'belief-workload.pt')
  def timing():
   from unown_belief_full_training.fit import predict
   from unown_belief_correction.belief import vector_from_predictions
   t=time.process_time();outputs=[]
   for x in selected:
    f=current_features(x['facts'],x['observation'],x['within']);d,p=predict(frozen_belief_model(),f,'MLP');outputs.append(vector_from_predictions(f,d,p))
   return time.process_time()-t,outputs
  slow_cpu,so=timing()
  with FixedDataCache() as cache:fast_cpu,fo=timing()
  assert all(np.array_equal(a,b) for a,b in zip(so,fo));result=dict(at_utc=iso(),passed=True,baseline=base,cached=fast,unprofiled_snapshot_CPU_seconds=slow_cpu,cached_snapshot_CPU_seconds=fast_cpu,speedup=slow_cpu/fast_cpu,snapshots=len(selected),exact_features=True,exact_vectors=True,max_vector_error=error,cache_calls=calls,native_attempts=0,GPU_seconds=0,CPU_seconds=time.process_time()-start)
  write(O/'cache-parity-timing.json',result);print(json.dumps(result),flush=True)
 except BaseException as e:write(O/'audit-failure.json',dict(at_utc=iso(),reason=str(e),traceback=traceback.format_exc(),CPU_seconds=time.process_time()-start));raise
if __name__=='__main__':main()
