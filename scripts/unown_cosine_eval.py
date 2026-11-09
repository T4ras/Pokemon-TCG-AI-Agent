"""Durable fixed allocation; interrupted processes never silently replace deals."""
import contextlib,fcntl,resource,sys,traceback
from unown_cosine_core import *
from unown_cosine_clock import AwakeClock,RuntimeClock

def records(path):
 if not Path(path).exists():return []
 result=[];raw=Path(path).read_text();rows=raw.splitlines()
 for index,line in enumerate(rows):
  try:result.append(json.loads(line))
  except json.JSONDecodeError:
   write(OUT/'record-fragment.json',dict(file=str(path),line=index+1,fragment=line))
   if index+1!=len(rows):raise RuntimeError('Nontrailingrecordcorruption; stop before additionalattempts')
   preserved=OUT/'record-fragments'/('original-'+sha(path)+'.txt');preserved.parent.mkdir(exist_ok=True);preserved.write_text(raw);fixed=Path(path).with_name(Path(path).name+'.valid-prefix');fixed.write_text('\n'.join(rows[:index])+('\n' if index else ''));fixed.replace(path)
 return result
def recover(jobs):
 path=OUT/'evaluation-records.jsonl';known={r['id']:r for r in records(path)};lookup={r['id']:r for r in jobs};assert len(known)==len(records(path)),'DuplicatecanonicalmatchIDs'
 for stage in ['evaluation-start','evaluation-final']:
  for row in records(OUT/stage/'matches.jsonl'):
   assert row['id'] in lookup
   if row['id'] not in known:row.update(recovered_same_completed_attempt=True,power_context='Originalstagecompletion recovered afterprocessloss; extraCPUmetering unavailable');append(path,row);known[row['id']]=row
  intents=list((OUT/stage/'intents').glob('*.json'))+[p for p in (OUT/'evaluation-attempt-intents').glob('*.json') if read(p)['stage']==stage]
  for intent in sorted(intents):
   job=read(intent);mid=job['id'];assert mid in lookup
   if mid in known:continue
   games=read(OUT/'game-progress'/(mid+'.json'))['games'] if (OUT/'game-progress'/(mid+'.json')).exists() else []
   row=dict(lookup[mid],status='FAILED',error='INTERRUPTED_PROCESS_LOST: original native attempt consumed; no replacementdeal',result=None,games=games,wall_seconds=0,started_at_utc=job.get('at_utc'),ended_at_utc=iso(),interrupted=True,wall_measurement='Unknown afterprocessloss,not assumed zero processing',learner_hash=START_TENSOR if stage=='evaluation-start' else read(OUT/'training-completion.json')['final_tensor_hash'])
   append(path,row);known[mid]=row
 return known

class ObservedEnvironment:
 def __init__(self,env,module,job):self.env=env;self.module=module;self.job=job;self.metadata={};self.games=[];self.scanned=0
 def __getattr__(self,k):return getattr(self.env,k)
 def harvest(self):
  vis=self.module.Battle.vis
  for i in range(self.scanned,len(vis)):
   current=vis[i].get('current') or {}
   if current.get('result',-1)>=0:
    rid=len(self.games)+1;self.games.append(dict(self.metadata.get(rid,dict(round=rid)),result=current['result'],finish_reason=current.get('finishReason','native_terminal_result'),completion_status='DONE'))
    path=OUT/'game-progress'/(self.job['id']+'.json');write(path,dict(at_utc=iso(),games=self.games,diagnostic_only=True,actor_input_unchanged=True))
    with path.open('rb') as file:os.fsync(file.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
  self.scanned=len(vis)
 def run(self,agents):
  def observed(agent,who):
   def call(obs,cfg):
    self.harvest();answer=agent(obs,cfg)
    if obs.get('select'):
     cur=obs['current'];g=self.metadata.setdefault(cur.get('round',1),dict(round=cur.get('round',1),chooser=None,first_player=-1))
     if cur['firstPlayer']>=0:g['first_player']=cur['firstPlayer']
     if obs['select']['context']==41:g.update(chooser=who,chooser_chose_first=obs['select']['option'][answer[0]]['type']==1)
    self.harvest();return answer
   return call
  try:return self.env.run([observed(a,i) for i,a in enumerate(agents)])
  finally:self.harvest()

def analyze(known):
 import unown_stability_stats as stats
 from unown_corrected_ppo_local import evaluation_details
 stats.SEED=SEED;grouped={k:[r for r in known.values() if r['stage']=='evaluation-'+k] for k in ['start','final']};result=stats.analyze_records(dict(initial=grouped['start'],final=grouped['final']));result['checkpoints']['start']=result['checkpoints'].pop('initial')
 for value in result['difference'].values():value['final_minus_start']=value.pop('final_minus_initial')
 result.update(status='COMPLETE',attempts=len(known),failures=sum(r['status']!='DONE' for r in known.values()),primary='Final-minus-continuation-start equalchooserBO3',method='Primary20000 independent whole-match checkpoint/chooser-stratified bootstrap; simultaneous exact CP conservative sensitivity. Draws nonwins. Failedattempts explicit; validfirstgames after laterfailure retained. No old-evaluation pooling.',whole_match_dependence=True)
 write(OUT/'evaluation-analysis.json',result);write(OUT/'evaluation-extra.json',evaluation_details(grouped))

def run():
 lock=(OUT/'evaluation.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:write(OUT/'evaluation-worker-already-active.json',dict(at_utc=iso(),no_duplicate=True));return False
 write(OUT/'evaluation-worker.json',dict(at_utc=iso(),pid=os.getpid(),proc_birth=Path('/proc/self/stat').read_text().split()[21]))
 if (OUT/'evaluation-fatal.json').exists():raise RuntimeError('Prior correctnessfailure requires explicitreview; no unattendedresume')
 start=time.monotonic();cpu=time.process_time();completion=read(OUT/'training-completion.json')
 if not completion['evaluate']:write(OUT/'evaluation-analysis.json',dict(status='SKIPPED',reason=completion['skip_evaluation'],attempts=0));return
 for name,digest in read(OUT/'evaluation-freeze.json')['files'].items():assert sha(ROOT/name)==digest,name
 jobs=read(OUT/'evaluation-start-schedule.json')+read(OUT/'evaluation-final-schedule.json');assert len(jobs)==1024;known=recover(jobs);before_count=len(known);clock=AwakeClock()
 import kaggle_environments.core as kgcore,kaggle_environments.agent as kgagent
 kgcore.perf_counter=clock;kgagent.perf_counter=clock;runtime.time=RuntimeClock(clock)
 models={k:load_model(PRIVATE/('start.pt' if k=='start' else 'final.pt'),'cpu') for k in ['start','final']};anchor=load_model(PRIVATE/'initialization.pt','cpu');assert state_hash(models['start'])==START_TENSOR and state_hash(models['final'])==completion['final_tensor_hash']
 original_environment=runtime.environment;write(OUT/'evaluation-worker.json',dict(at_utc=iso(),pid=os.getpid(),proc_birth=Path('/proc/self/stat').read_text().split()[21]))
 try:
  for job in jobs:
   if job['id'] in known:continue
   awake=clock();utc=now();cpu0=time.process_time();label='start' if job['stage']=='evaluation-start' else 'final';exclusive(OUT/'evaluation-attempt-intents'/(job['id']+'.json'),dict(job,at_utc=utc.isoformat(),pid=os.getpid(),allocation_slot_committed_before_native=True));write(OUT/'active-evaluation-attempt.json',dict(id=job['id'],pid=os.getpid(),started_at_utc=utc.isoformat(),awake_start=awake,status='ACTIVE'))
   def environment():
    env,module=original_environment();return ObservedEnvironment(env,module,job),module
   runtime.environment=environment
   from unown_cosine_failure_detail import opener
   original_gzip_open=gzip.open;gzip.open=opener(original_gzip_open,job)
   try:row,_=core.play(job,models[label],anchor,False,awake+300)
   finally:runtime.environment=original_environment;gzip.open=original_gzip_open
   if row['status']!='DONE' and (OUT/'game-progress'/(job['id']+'.json')).exists():row['games']=read(OUT/'game-progress'/(job['id']+'.json'))['games'];row['valid_games_recovered_from_progress']=True
   row.update(started_at_utc=utc.isoformat(),ended_at_utc=iso(),UTC_elapsed_seconds=(now()-utc).total_seconds(),awake_elapsed_seconds=clock()-awake,CPU_core_seconds=time.process_time()-cpu0,power_at_completion=read(OUT/'power-clock.json')['power'],normal_match_memory=True)
   append(OUT/'evaluation-records.jsonl',row);known[job['id']]=row;write(OUT/'active-evaluation-attempt.json',dict(id=job['id'],status='COMMITTED',at_utc=iso()));progress('EVALUATION',completed_allocation=len(known),target=1024,checkpoint=label,checkpoint_attempts=sum(r['stage']==job['stage'] for r in known.values()))
   assert state_hash(models[label])==row['learner_hash'] and state_hash(anchor)==row['opponent_hash'],'Evaluationmodelchanged'
   if row['status']!='DONE' and (any(t in str(row.get('error')) for t in ['Illegal','Nonfinite','masked','UnresolvedIdentity']) or any(e.get('type')=='AssertionError' for e in row.get('actor_errors',[]))):raise RuntimeError('Evaluationcorrectnessfailure; preserveattempts,stop')
  analyze(known);write(OUT/'evaluation-completion.json',dict(at_utc=iso(),state='COMPLETE',attempts=len(known),additional_native_attempts=len(known)-before_count))
 except BaseException as e:
  detail=dict(at_utc=iso(),error=str(e),traceback=traceback.format_exc(),remaining_allocation=1024-len(known))
  if 'Awakeclocktelemetryunavailable' in str(e):write(OUT/'evaluation-telemetry-interruption.json',detail);raise SystemExit(76)
  write(OUT/'evaluation-fatal.json',detail);raise
 finally:
  append(OUT/'local-costs.jsonl',dict(stage='CPU-evaluation',at_utc=iso(),process_elapsed_seconds=time.monotonic()-start,CPU_core_hours=(time.process_time()-cpu)/3600,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,completed_allocation=len(known),GPU_seconds=0,clock='WindowsOSawake QueryUnbiasedInterruptTime; UTC and perattempt CPU retained'))
  lock.close()
if __name__=='__main__':run()
