"""Bounded local preparation/checks/evaluation; routine code calls no model APIs."""
import collections,copy,gzip,json,resource,shutil,sys,time,traceback
from unown_corrected_ppo_core import *
def diagnostic():
 from unown_start_order_actor import StartingOrder,replace_feature
 model,opt=fresh();deck=read(OUT/'manifest.json')['decks']['Hydro']['cards'];source=ROOT/'data/private/astra-unown-starting-order-broad-validation-001/histories';schedule=read(PARENT/'schedule.json')['bo3'];strata={};sources=[]
 for p in sorted(source.glob('bo3-*.jsonl.gz')):
  mid=p.name.split('-A.')[0].split('-B.')[0];job=next(j for j in schedule if j['id']==mid);memory=runtime.rt.MatchTracker(deck);state=StartingOrder();forced=0;game=None;sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
  with gzip.open(p,'rt') as f:history=[json.loads(s) for s in f]
  for index,r in enumerate(history):
   obs=r['observation']
   if not obs.get('current'):memory.reset_match();state.reset();continue
   info=state.observe(obs);memory.update(obs);sel=obs['select'];n=len(sel['option'])
   if game!=info['game']:forced=0;game=info['game']
   if sel['minCount']==sel['maxCount']==n or n==1 and sel['minCount']>=1:forced+=1;continue
   picked=[]
   for action in list(r['action'])+[None]:
    if len(picked)>=sel['maxCount']:break
    stop=len(picked)>=sel['minCount'] and (sel['maxCount']>1 or sel['minCount']==0);packet=replace_feature(runtime.rt.encode(obs,picked,stop,forced,memory.encoder_tracker()),state)
    key=(job['candidate_seat'],info['seat'],game);strata.setdefault(key,[]).append(dict(packet=packet,source=p.name,line=index,prefix=list(picked),game=game,seat=info['seat'],public_first=info['public_first']))
    if action is not None:picked.append(action)
   forced=0
 rng=np.random.default_rng(SEED+99)
 for rows in strata.values():rng.shuffle(rows)
 fixed=[];k=0
 while len(fixed)<1024:
  added=False
  for key in sorted(strata):
   if k<len(strata[key]) and len(fixed)<1024:fixed.append(strata[key][k]);added=True
  if not added:break
  k+=1
 assert len(fixed)>=128;torch.save(fixed,PRIVATE/'fixed-diagnostic.pt');reference(fixed,model)
 result=fixed_drift(model,fixed);assert abs(result['initialization_KL'])<1e-6
 write(OUT/'diagnostic-freeze.json',dict(states=len(fixed),sha256=sha(PRIVATE/'fixed-diagnostic.pt'),sources=sources,strata={str(k):len(v) for k,v in strata.items()},selected_packet_hash=batch.digest([r['packet'].row for r in fixed]),CPU_start=result,training_experience=False))
def compatibility():
 model,opt=fresh();anchor,_=fresh();rows=[]
 for job in read(OUT/'compatibility-schedule.json'):
  check_time();existing=OUT/'compatibility/matches.jsonl'
  previous=[json.loads(s) for s in existing.read_text().splitlines()] if existing.exists() else []
  saved=next((r for r in previous if r['id']==job['id']),None)
  if saved:
   assert saved['status']=='DONE';context=dict(candidate_model=model,actors=[],retain_history=False);actor=corrected_actor_class(context)(model,read(OUT/'manifest.json')['decks']['Hydro']['cards'],sample=True,seed=job['seed'],collect=True)
   with gzip.open(PRIVATE/'histories'/(job['id']+'-learner.jsonl.gz'),'rt') as f:
    for line in f:
     h=json.loads(line);obs=h['observation']
     if not obs.get('select'):continue
     assert actor.act(obs)==h['action'],'Retained conditional sampler replay changed'
   experience=runtime.completed_game_returns(actor.records,saved['games'],job['learner_seat']);record=saved
  else:record,experience=play(job,model,anchor,True,time.monotonic()+max(0,(deadline()-now()).total_seconds()-120))
  assert record['status']=='DONE',record['error'];rows+=experience
 parity=phase_metrics(model,rows);assert parity['max_chosen_logprob_difference']<=1e-5 and parity['max_old_policy_KL']<=1e-6
 assert all(r['feature_semantics']=='starting-order-v1' for r in rows)
 torch.save(rows[:128],PRIVATE/'verification.pt')
 before_hash=state_hash(model);probe=load_model(PRIVATE/'initialization.pt','cpu');po=batch.optimizer(probe);execution=dict(step=0,phase=-1,offset=0)
 result=transaction(probe,po,rows[:128],advantages(rows)[:128],execution,'cpu-probe')
 assert state_hash(model)==before_hash==INITIAL_TENSOR and not opt.state
 # Optimized save/resume equivalence with the corrected rows, not old PPO data.
 batch.step(probe,po,rows[:16],advantages(rows[:16]));legacy.save(PRIVATE/'cpu-resume.pt',probe,po,execution['step']);before_resume=dict(model=batch.digest(probe.state_dict()),optimizer=batch.digest(po.state_dict()),grad_flags={k:p.requires_grad for k,p in probe.named_parameters()})
 batch.step(probe,po,rows[:16],advantages(rows[:16]));expected=copy.deepcopy(dict(model=probe.state_dict(),optimizer=po.state_dict(),torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))
 resumed,ro,_=batch.restore_checkpoint(PRIVATE/'cpu-resume.pt',rng=True);after_resume=dict(model=batch.digest(resumed.state_dict()),optimizer=batch.digest(ro.state_dict()),grad_flags={k:p.requires_grad for k,p in resumed.named_parameters()});assert before_resume==after_resume,(before_resume,after_resume)
 batch.step(resumed,ro,rows[:16],advantages(rows[:16]));actual=dict(model=resumed.state_dict(),optimizer=ro.state_dict(),torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
 checks={k:dict(expected=batch.digest(v),resumed=batch.digest(actual[k]),equal=batch.digest(v)==batch.digest(actual[k])) for k,v in expected.items()};write(OUT/'CPU-resume-details.json',checks)
 if not all(v['equal'] for v in checks.values()):
  differences={k:float((v-actual['model'][k]).abs().max()) for k,v in expected['model'].items() if not torch.equal(v,actual['model'][k])};write(OUT/'CPU-resume-parameter-differences.json',differences)
 assert all(v['equal'] for v in checks.values()),checks
 write(OUT/'CPU-compatibility.json',dict(status='PASS',BO3=2,chooser=[1,1],decisions=len(rows),likelihood=parity,disposable_update=result,corrected_resume_exact=True,unchanged_actual_initialization=state_hash(model),optimizer_fresh=True,rollouts_used_for_training=False,verification_sha256=sha(PRIVATE/'verification.pt')))
def evaluate():
 completion=read(OUT/'training-completion.json')
 if not completion.get('evaluate'):
  write(OUT/'evaluation-analysis.json',dict(status='SKIPPED',reason=completion.get('skip_evaluation','no retained model change'),new_native_attempts=0));return
 from unown_stability_stats import analyze_records
 import unown_stability_stats as stats
 stats.SEED=BOOT-700;records={};anchor,_=fresh()
 for label in ['initial','final']:
  model=load_model(PRIVATE/('initialization.pt' if label=='initial' else 'final.pt'),'cpu');records[label]=[]
  for job in read(OUT/f'evaluation-{label}-schedule.json'):
   check_time();record,_=play(job,model,anchor,False,time.monotonic()+max(0,(deadline()-now()).total_seconds()-120));records[label].append(record)
   if record['status']!='DONE':
    write(OUT/'evaluation-extra.json',evaluation_details(records));write(OUT/'evaluation-analysis.json',dict(analyze_records(records),status='INCOMPLETE',reason=record['error'],no_replacement=True));raise RuntimeError('Fixed evaluation failure; no replacement: '+str(record['error']))
   progress('EVALUATION',checkpoint=label,completed=len(records[label]),target=256)
 result=analyze_records(records);result.update(status='COMPLETE',new_native_attempts=512,primary='final_minus_initial equal chooser BO3',method='Primary20000 independent whole-match chooser-stratified bootstrap; additional simultaneous exact CP conservative95% bound. Draws count as nonwins. Native deals independent.',previous_evaluations_pooled=False);write(OUT/'evaluation-analysis.json',result);write(OUT/'evaluation-extra.json',evaluation_details(records))

def evaluation_details(records):
 from unown_stability_stats import cp
 details={}
 def outcomes(rows):
  w=sum(x=='W' for x in rows);return dict(n=len(rows),wins=w,draws=rows.count('D'),losses=rows.count('L'),win_rate=w/len(rows) if rows else None,CI95=cp(w,len(rows)))
 for label,matches in records.items():
  cells={};conversion=[];recovery=[]
  for r in matches:
   seat=r['learner_seat'];games=r['games']
   def result(x):return 'W' if x==seat else 'D' if x==2 else 'L'
   if r['status']=='DONE' and games:
    first=result(games[0]['result'])
    if first=='W':conversion.append(result(r['result']))
    if first=='L':recovery.append(result(r['result']))
   for i,g in enumerate(games):
    key=f"game{i+1}/chooser{seat}/starter{g.get('first_player','unknown')}";cell=cells.setdefault(key,dict(outcomes=[],finish_reasons=collections.Counter()))
    cell['outcomes'].append(result(g['result']));cell['finish_reasons'][str(g.get('finish_reason','not recorded'))]+=1
  details[label]=dict(attempted=len(matches),completed=sum(r['status']=='DONE' for r in matches),failed=sum(r['status']!='DONE' for r in matches),game_cells={k:dict(outcomes(v['outcomes']),finish_reasons=dict(v['finish_reasons'])) for k,v in cells.items()},match_conversion_after_first_win=outcomes(conversion),match_recovery_after_first_loss=outcomes(recovery))
 return dict(checkpoints=details,interpretation='Later-game participation/start order and first-game conversion/recovery are descriptive outcome-dependent subsets. Cell intervals use one game per whole match, no pooled independent-game claim.')
def main():
 stage=sys.argv[1];t=time.monotonic();c=time.process_time();write(OUT/'local-pid.json',dict(stage=stage,pid=os.getpid(),at_utc=iso()))
 try:
  check_time()
  if stage=='diagnostic':diagnostic()
  elif stage=='compatibility':compatibility()
  elif stage=='tests':
   import pytest
   code=int(pytest.main(['tests/test_unown_corrected_ppo.py','-q']));write(OUT/'pytest.json',dict(exit_code=code,native_attempts=0));assert code==0
  elif stage=='bundle':
   from unown_corrected_ppo_bundle import bundle;bundle()
  elif stage=='smoke':
   from unown_corrected_ppo_bundle import smoke;smoke()
  elif stage=='evaluate':evaluate()
  else:raise ValueError(stage)
  write(OUT/(stage+'-completion.json'),dict(state='COMPLETE',at_utc=iso()))
 except BaseException as e:
  detail=dict(state='FAILED',error=str(e),traceback=traceback.format_exc(),at_utc=iso(),no_automatic_retry=True);write(OUT/(stage+'-failure-'+str(os.getpid())+'.json'),detail);write(OUT/(stage+'-failure.json'),detail);raise
 finally:append(OUT/'local-costs.jsonl',dict(stage=stage,at_utc=iso(),elapsed_seconds=time.monotonic()-t,CPU_core_hours=(time.process_time()-c)/3600,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU_seconds=0,paid_model_API_requests=0))
if __name__=='__main__':main()
