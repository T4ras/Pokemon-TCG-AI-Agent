"""Fixed512+512 CPU attempts; persistent parent, no replacement matches."""
import fcntl,multiprocessing as mp,resource,sys,traceback
from .common import *
from unown_runner.runtime import Runtime
from .benchmark_authorization import final_evaluation_allowed

RT=MODELS=LABEL=None

class ProgressEnvironment:
 def __init__(self,environment,module,job,root):self.environment=environment;self.module=module;self.job=job;self.root=root;self.games=[];self.metadata={};self.scanned=0
 def __getattr__(self,name):return getattr(self.environment,name)
 def harvest(self):
  for frame in self.module.Battle.vis[self.scanned:]:
   cur=frame.get('current') or {}
   if cur.get('result',-1)>=0:
    rid=len(self.games)+1;self.games.append(dict(self.metadata.get(rid,dict(round=rid)),result=cur['result'],finish_reason=cur.get('finishReason','native_terminal_result'),completion_status='DONE'))
    write(self.root/'game-progress'/(self.job['id']+'.json'),dict(at_utc=iso(),games=self.games))
  self.scanned=len(self.module.Battle.vis)
 def run(self,agents):
  def wrap(agent,who):
   def call(obs,cfg):
    self.harvest();answer=agent(obs,cfg)
    if obs.get('select'):
     cur=obs['current'];g=self.metadata.setdefault(cur.get('round',1),dict(round=cur.get('round',1),chooser=None,first_player=-1))
     if cur['firstPlayer']>=0:g['first_player']=cur['firstPlayer']
     if obs['select']['context']==41:g.update(chooser=who,chooser_chose_first=obs['select']['option'][answer[0]]['type']==1)
    self.harvest();return answer
   return call
  try:return self.environment.run([wrap(a,i) for i,a in enumerate(agents)])
  finally:self.harvest()

def initialize(label):
 global RT,MODELS,LABEL
 os.environ['CUDA_VISIBLE_DEVICES']='';LABEL=label;RT=Runtime(context());RT.configure_precision()
 root=OUT/'evaluation-workers'/str(os.getpid());private=PRIVATE/'evaluation-workers'/str(os.getpid());root.mkdir(parents=True);private.mkdir(parents=True)
 write(root/'manifest.json',dict(decks=read(OUT/'decks.json')));RT.bind(root,private)
 # A single root awake-clock file is maintained by the Windows owner.
 RT.modules['clock'].OUT=OUT
 if os.environ.get('COLLECT_LOCAL_CLOCK')=='1':
  RT.runtime.time=RT.modules['clock'].RuntimeClock(RT.clock)
  import kaggle_environments.core as kgcore,kaggle_environments.agent as kgagent
  kgcore.perf_counter=RT.clock;kgagent.perf_counter=RT.clock
 path=PRIVATE/('initialization.pt' if label=='initial' else 'final.pt')
 MODELS=dict(learner=RT.load_model(path,'cpu'),opponent=RT.load_model(PRIVATE/'initialization.pt','cpu'))
 expected=INITIAL_TENSOR if label=='initial' else read(OUT/'training-completion.json')['final_tensor_hash']
 assert RT.state_hash(MODELS['learner'])==expected and RT.state_hash(MODELS['opponent'])==INITIAL_TENSOR
 for model in MODELS.values():
  for p in model.parameters():p.requires_grad_(False)
  RT.runtime.freeze_inference(model)
 append(OUT/'evaluation-processes.jsonl',dict(at_utc=iso(),pid=os.getpid(),proc_birth=Path('/proc/self/stat').read_text().split()[21],stage=label))

def one(job):
 start=RT.clock();utc=now();cpu=time.process_time();previous=RT.runtime.environment
 exclusive(OUT/'evaluation-native-starts'/(job['id']+'.json'),dict(at_utc=iso(),pid=os.getpid(),job=job))
 def environment():
  env,module=previous();return ProgressEnvironment(env,module,job,OUT),module
 RT.runtime.environment=environment
 try:row,_=RT.corrected.play(job,MODELS['learner'],MODELS['opponent'],False,start+300)
 finally:RT.runtime.environment=previous
 if row['status']!='DONE' and (OUT/'game-progress'/(job['id']+'.json')).exists():row['games']=read(OUT/'game-progress'/(job['id']+'.json'))['games']
 assert row['learner_hash']==RT.state_hash(MODELS['learner']) and row['opponent_hash']==RT.state_hash(MODELS['opponent'])
 row.update(checkpoint=LABEL,UTC_elapsed_seconds=(now()-utc).total_seconds(),awake_elapsed_seconds=RT.clock()-start,CPU_core_seconds=time.process_time()-cpu,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,greedy_both=True,independent_native_deals=True)
 return row

def run(label):
 assert label in ('initial','final') and read(OUT/'readiness.json')['status']=='PASS' and read(OUT/'runner-regression.json')['returncode']==0
 if label=='final' and not final_evaluation_allowed():
  write(OUT/'evaluation-final-completion.json',dict(state='SKIPPED',attempts=0,reason=read(OUT/'training-completion.json')['skip_final_evaluation']));return
 lock=(OUT/'evaluation.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 expected={j['id']:j for j in read(OUT/f'evaluation-{label}-schedule.json')};path=OUT/f'evaluation-{label}-records.jsonl';known={r['id']:r for r in records(path)}
 if len(known)!=len(records(path)):raise RuntimeError('Duplicate match records')
 for intent in (OUT/'evaluation-attempt-intents').glob('*.json'):
  job=read(intent)['job'];mid=job['id']
  if mid in expected and mid not in known:
   progress_path=OUT/'game-progress'/(mid+'.json');row=dict(job,status='FAILED',error='Interrupted allocated attempt; no replacement',result=None,games=read(progress_path)['games'] if progress_path.exists() else [],checkpoint=label)
   append(path,row);known[mid]=row
 jobs=[j for j in expected.values() if j['id'] not in known];start=time.monotonic();cpu=time.process_time();pool=None;inflight={};next_job=0
 write(OUT/'evaluation-active.json',dict(at_utc=iso(),pid=os.getpid(),proc_birth=Path('/proc/self/stat').read_text().split()[21],checkpoint=label))
 try:
  pool=mp.get_context('spawn').Pool(4,initializer=initialize,initargs=(label,))
  while next_job<len(jobs) or inflight:
   # First two planned matches provide bounded live startup evidence. They are
   # allocation members, never extra diagnostic matches.
   capacity=1 if len(known)<2 else 4
   while len(inflight)<capacity and next_job<len(jobs):
    job=jobs[next_job];next_job+=1;exclusive(OUT/'evaluation-attempt-intents'/(job['id']+'.json'),dict(at_utc=iso(),job=job,owner_pid=os.getpid()));inflight[job['id']]=(job,pool.apply_async(one,(job,)))
   changed=False
   for mid,(job,future) in list(inflight.items()):
    if not future.ready():continue
    row=future.get();append(path,row);known[mid]=row;inflight.pop(mid);changed=True
    if row['status']!='DONE':raise RuntimeError('CPU evaluation correctness/runtime failure; preserve allocation and stop before any further launch')
   if changed:
    progress('CPU_EVALUATION',checkpoint=label,completed=len(known),planned=512)
    if len(known)==2:write(OUT/f'evaluation-{label}-startup.json',dict(status='PASS',planned_attempts=2,extra_native_attempts=0,both_corrected=True,greedy_both=True))
   if not changed:time.sleep(.2)
  pool.close();pool.join();pool=None
  write(OUT/f'evaluation-{label}-completion.json',dict(at_utc=iso(),state='COMPLETE',attempts=len(known),completed=sum(r['status']=='DONE' for r in known.values()),failures=sum(r['status']!='DONE' for r in known.values()),UTC_elapsed_seconds=time.monotonic()-start,no_replacements=True))
 except BaseException as error:
  if pool:pool.terminate();pool.join();pool=None
  for mid,(job,future) in list(inflight.items()):
   if mid in known:continue
   evidence=OUT/'game-progress'/(mid+'.json');row=dict(job,status='FAILED',result=None,error='Allocated concurrent attempt interrupted after execution failure; no replacement',games=read(evidence)['games'] if evidence.exists() else [],checkpoint=label,attempt_consumed=True)
   append(path,row);known[mid]=row
  write(OUT/f'evaluation-{label}-failure.json',dict(at_utc=iso(),error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),completed_records=len(known),active_allocated=list(inflight),no_replacements=True));raise
 finally:
  if pool:pool.terminate();pool.join()
  append(OUT/'local-costs.jsonl',dict(at_utc=iso(),stage='CPU-evaluation-'+label,elapsed_seconds=time.monotonic()-start,parent_CPU_core_hours=(time.process_time()-cpu)/3600,match_CPU_core_hours=sum(r.get('CPU_core_seconds',0) for r in known.values())/3600,native_attempts=len(known),GPU_seconds=0));lock.close()

if __name__=='__main__':run(sys.argv[1])
