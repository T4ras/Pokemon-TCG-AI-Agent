"""Registration, offline compatibility and autonomous frozen CPU BO3 evaluation."""
import os,sys,time,random,copy,subprocess,select,signal,gzip,importlib.util,resource,traceback,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from unown_common import *
os.chdir(REPO)

def register():
 import importlib.metadata as md
 from ptcg_agent.engine import load_sdk,validate_deck
 from ptcg_agent.learning.rl_stage1 import load_model,tensor_sha
 import torch
 torch.set_num_threads(1)
 h=read(REPO/'experiments/astra-stage-h-hydrapple-readiness-001/manifest.json')
 league=read(REPO/'experiments/astra-hydrapple-league-comparison-001/recovery-artifacts.json')['league']['final']
 models={'BC34-Hydrapple':h['models']['BC'],'League32-Hydrapple':dict(path=league['path'],sha256=league['sha256']),'G30-OceanMix':h['models']['G']}
 packages={}
 for label,p in models.items():
  assert sha(p['path'])==p['sha256'];net=load_model(p['path']);assert net.observation_schema=='visible-v1'
  deck=h['packages']['G' if label=='G30-OceanMix' else 'Hydrapple-Meganium']['deck']
  packages[label]=dict(p,deck=deck,deck_sha256=h['packages']['G' if label=='G30-OceanMix' else 'Hydrapple-Meganium']['deck_sha256'],ordered_deck_compact_json_sha256=hashlib.sha256(json.dumps(deck,separators=(',',':')).encode()).hexdigest(),model_tensor_sha256=tensor_sha(net),decoder='conditional argmax',schema='visible-v1')
 assert packages['League32-Hydrapple']['sha256']=='1830c1a854e3865c9af26243727c6182d96b4b60a23ca1e62cd87e65875bac37'
 deck=list(map(int,(UP/'submission/deck.csv').read_text().split()))
 packages['Unown-Gradiant']=dict(path=str((UP/'submission/model.pt').relative_to(REPO)),sha256=sha(UP/'submission/model.pt'),deck=deck,deck_sha256=sha(UP/'submission/deck.csv'),decoder='Exact submitted main.py: cfg=None, heads8, temp0, twoTorchthreads; rejection/time guards retained')
 api,_=load_sdk();cards={c.cardId:c for c in api.all_card_data()}
 for p in packages.values():validate_deck(p['deck'],cards)
 shutil.copy2(REPO/'data/sdk/extracted/sample_submission/sample_submission/sample_submission/cg/libcg.so',CURRENT/'cg/libcg.so')
 assert sha(CURRENT/'cg/libcg.so')=='c699020e086eede189c445c7abf9d891a60005ceec6eb91dccd5bbdcd3c0de22'
 source={str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'scripts').glob('unown_*.py'))}
 source.update({str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'src/ptcg_agent').rglob('*.py'))})
 assets={str(p.relative_to(REPO)):dict(bytes=p.stat().st_size,sha256=sha(p)) for folder in [CURRENT,PRIVATE/'original-sdk'] for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
 for f in ['original-tables.json','current-tables.json']:assets[str((PRIVATE/f).relative_to(REPO))]=dict(sha256=sha(PRIVATE/f),bytes=(PRIVATE/f).stat().st_size)
 m=dict(at_utc=iso(),upstream_commit=COMMIT,license='MIT; Copyright2026 Dipam Chakraborty; retained unmodified',packages=packages,assets=assets,environment_commit=read(CURRENT/'source.json')['commit'],environment_version=2,source_sha256=source,dependencies={x:dict(version=md.version(x),metadata_sha256=sha(Path(md.distribution(x)._path)/'METADATA')) for x in ['torch','numpy','kaggle-environments']},python=sys.version,prior_report_sha256=sha(REPO/'reports/astra-hydrapple-league-comparison.md'),published_solutions_sha256=sha(REPO/'docs/PTCG_PUBLISHED_SOLUTIONS_FOR_ASTRA.md'),original_upstream_manifest_sha256=sha(ROOT/'upstream-manifest.json'))
 write(ROOT/'manifest.json',m)
 rng=random.Random(SEED);official=[];strength=[]
 combos=[(label,seat) for label in LABELS for seat in [0,1]];rng.shuffle(combos)
 for i,(label,seat) in enumerate(combos):official.append(dict(match_id=f'U-check-{i:03d}',opponent=label,candidate_seat=seat,seed=SEED*100000+i,component='official'))
 for block in range(48):
  combos=[(label,seat) for label in LABELS for seat in [0,1]];rng.shuffle(combos)
  for label,seat in combos:
   i=len(strength);strength.append(dict(match_id=f'U-strength-{i:04d}',block=block,opponent=label,candidate_seat=seat,seed=SEED*100000+1000+i,component='strength'))
 workers=min(4,max(1,len(os.sched_getaffinity(0))//2));write(ROOT/'official-schedule.json',official);write(ROOT/'strength-schedule.json',strength)
 exclusive(ROOT/'registration.json',dict(registered_at_utc=iso(),accounting_start_utc=START,deadline_utc=DEADLINE,manifest_sha256=sha(ROOT/'manifest.json'),official_schedule_sha256=sha(ROOT/'official-schedule.json'),strength_schedule_sha256=sha(ROOT/'strength-schedule.json'),workers=workers,official_attempts=6,strength_attempts=288,total_attempt_cap=294,additional_native_diagnostics=0,primary='Unown Gradiant BO3 win rate vsLeague32Hydrapple',secondary=['BC34Hydrapple','G30OceanMix'],gate='Primary estimate>=.60 and whole-match chooser-stratified95% lower>.50 and no correctness/runtime failure',bootstrap=dict(seed=BOOT,replicates=20000,unit='whole match',strata='opponent/initial chooser',point='equal chooser weights',draws='nonwins',first_games='same match resampling, valid first game retained after later failure',pairing='independent native deals, never paired'),forecast='After six checks, (288/workers)*(max(check BO3 wall)+max(cold candidate startup))*3 plus180s analysis reserve must fit remaining absolute deadline. No outcome information enters admission.',retention='All compact outcomes; candidate legal histories from six checks only, fixed recorded prefix parity; first3 failure details/signature up to1MiB each and256MiB total; no all-success visualizations persisted',policy='Original submitted module/config/weights/deck; own three frozen visible-v1 argmax packages. No semantic/tactical changes. Unrepresentable IDs or option/token overflow fail closed. No privileged critic truth.',GPU_seconds=0,training_steps=0,uploads=0,replacements=0,seed=SEED,CPU_only=True,CPU_scope='Measured process CPU for program children/workers; early model/metadata probes and human/model development CPU are unavailable, not zero. Wall clock includes all preparation.'))

def verify():
 m=read(ROOT/'manifest.json');r=read(ROOT/'registration.json')
 assert sha(ROOT/'manifest.json')==r['manifest_sha256']
 for p,h in m['source_sha256'].items():assert sha(REPO/p)==h,p
 for p,v in m['assets'].items():assert sha(REPO/p)==v['sha256'],p
 for p in m['packages'].values():assert sha(REPO/p['path'])==p['sha256']
 for p,v in read(ROOT/'upstream-manifest.json')['files'].items():assert sha(UP/p)==v['sha256']
 for component in ['official','strength']:assert sha(ROOT/(component+'-schedule.json'))==r[component+'_schedule_sha256']
 return m,r

def compatibility():
 import torch,numpy as np
 from unown_actor import Adapter,load_module,check_supported,CompatibilityError,tensor_hash
 started=now();cpu=time.process_time();m,r=verify();adapter=Adapter();reference=load_module('unown_original_reference')
 # Restore the untouched reference encoder for it, and explicitly select guarded one for adapted calls.
 from ptcg.rl import search_agent,encoder
 guarded_encode=search_agent.encode;original_encode=encoder.encode
 from ptcg.rl.cards import build_tables,load_text_tables
 from ptcg.rl.buffers import OFFSETS,MAX_TOKENS,TOK_INT
 sd=torch.load(UP/'submission/model.pt',map_location='cpu',weights_only=True)
 static={k:torch.equal(torch.from_numpy(v),sd[k]) for k,v in zip(['card_static','card_attacks','card_skills','att_static'],build_tables())}
 static.update({k:torch.equal(torch.from_numpy(v),sd[k]) for k,v in zip(['card_text','att_text','skill_text'],load_text_tables())});assert all(static.values())
 old=read(PRIVATE/'original-tables.json');cur=read(PRIVATE/'current-tables.json');oc={c['cardId']:c for c in old['cards']};cc={c['cardId']:c for c in cur['cards']}
 panel_ids=set(cid for p in m['packages'].values() for cid in p['deck']);assert panel_ids<=adapter.supported
 assert all(a in adapter.attacks for cid in panel_ids for a in oc[cid]['attacks'])
 differences={name:[dict(id=x[key],old=x,current=next((y for y in cur[name] if y[key]==x[key]),None)) for x in old[name] if normalize(x)!=normalize(next((y for y in cur[name] if y[key]==x[key]),None))] for name,key in [('cards','cardId'),('attacks','attackId')]}
 write(PRIVATE/'mapping-details.json',dict(substantive_differences=differences,new_cards=[c for c in cur['cards'] if c['cardId'] not in oc]))
 write(ROOT/'mapping-compatibility.json',dict(old_counts={k:len(v) for k,v in old.items()},current_counts={k:len(v) for k,v in cur.items()},panel_verified_ids=sorted(panel_ids),supported_card_ids=sorted(adapter.supported),supported_attack_ids=sorted(adapter.attacks),substantive_differences={k:[dict(id=d['id'],name=d['old']['name'],changed_fields=[key for key in d['old'] if normalize(d['old'][key])!=normalize(d['current'].get(key))]) for d in v] for k,v in differences.items()},unsupported_new_card_ids=[c['cardId'] for c in cur['cards'] if c['cardId'] not in oc],raw_card_metadata_differences=sum(oc[cid]!=cc[cid] for cid in oc),private_details_sha256=sha(PRIVATE/'mapping-details.json')))
 hist=REPO/'data/private/astra-hydrapple-league-submission-001/L-package-4-actor-history.json.gz'
 with gzip.open(hist,'rt') as f:history=json.load(f)
 adapter.reset(SEED);reference.agent({'select':None});rid=None;matched=0;row_matches=0
 for e in history:
  obs=copy.deepcopy(e['observation']);round_id=obs['current'].get('round',1)
  if rid!=round_id:reference.agent({'select':None});rid=round_id
  search_agent.encode=guarded_encode;a=adapter.act(copy.deepcopy(obs));a_row=adapter.module.SA.row.copy()
  search_agent.encode=original_encode;b=reference.agent(copy.deepcopy(obs))
  assert a==b,'Adapter differs from submitted Python decoder';matched+=1
  # Row equality is meaningful only if that select called the policy (forced calls retain previous row).
  assert np.array_equal(a_row,reference.SA.row);row_matches+=1
 search_agent.encode=guarded_encode
 row=torch.from_numpy(adapter.module.SA.row.copy());a,b=OFFSETS['oracle'];other=row.clone();other[:,a:b]=torch.arange(b-a).float()+1
 with torch.no_grad():assert torch.equal(adapter.module.SA.policy.forward_policy(row),adapter.module.SA.policy.forward_policy(other))
 bad=copy.deepcopy(history[0]['observation']);bad['select']['option']=[dict(type=0,number=i) for i in range(64)];bad['select']['minCount']=bad['select']['maxCount']=1
 try:check_supported(bad,adapter.supported,adapter.attacks);raise AssertionError('Overflow guard failed')
 except CompatibilityError:pass
 bad=copy.deepcopy(history[0]['observation']);seat=bad['current']['yourIndex'];bad['current']['players'][seat]['hand']=[dict(id=96,serial=10000+i,playerIndex=seat) for i in range(161)]
 from ptcg.rl.buffers import OBS_SIZE
 try:guarded_encode(np.zeros(OBS_SIZE,np.float32),bad,[],False,0,tracker=None);raise AssertionError('Token guard failed')
 except CompatibilityError:pass
 actor_before=adapter.identity;adapter.reset(SEED);assert adapter.module.SA.tracker.my_prize_known is None and adapter.module.SA.forced_run==0
 assert tensor_hash(adapter.module.SA.policy)==actor_before
 output=dict(status='PASSED',at_utc=iso(),native_matches=0,static_and_text_buffers_bit_equal=static,checkpoint_state_entries=len(sd),model_parameters=sum(p.numel() for p in adapter.module.SA.policy.parameters()),arch=dict(width=256,layers=4,heads=8,ffn=512),candidate_tensor_sha256=actor_before,panel_card_identities_verified=len(panel_ids),unsupported_new_cards=164,unsupported_new_attacks=199,substantive_old_card_differences=len(differences['cards']),substantive_old_attack_differences=len(differences['attacks']),retained_history=str(hist.relative_to(REPO)),retained_history_sha256=sha(hist),faithful_prefix_action_matches=matched,encoded_row_matches=row_matches,oracle_actor_independence=True,nonforced_64_option_guard=True,token_161_overflow_guard=True,game_and_match_resets=True,metadata_only_original_library_no_BattleStart=True,original_competition_download='403 access; no rules accepted automatically. Already-installed official legacy library used only for public metadata; static checkpoint buffers match exactly.',original_CSV='Unavailable; only read by optional search/threat code, disabled in submitted cfg=None. No policy inference path reads it.',scope='These fixed old-pool packages only; no representation for current new cards or changed mechanics, no universal training-pool integration claimed.',startup=adapter.stats(),UTC_seconds=(now()-started).total_seconds(),process_CPU_seconds=time.process_time()-cpu,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write(ROOT/'compatibility.json',output)

def environment():
 from kaggle_environments import make,register as kgregister
 name='unown_playground_official'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,CURRENT/'cabt.py',submodule_search_locations=[str(CURRENT)]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
  kgregister(name,dict(interpreter=module.interpreter,renderer=module.renderer,html_renderer=module.html_renderer,specification=module.specification,agents=module.agents))
 return make(name,configuration=dict(bo=3,version=2,runTimeout=300),debug=True),sys.modules[name]

class RemoteActor:
 def __init__(self,log):
  t=time.perf_counter();self.log=Path(log).open('a');self.p=subprocess.Popen([sys.executable,str(REPO/'scripts/unown_actor.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
  initial=self.receive(45);assert initial.get('ready'),initial;self.startup=initial['stats'];self.startup['pipe_startup_seconds']=time.perf_counter()-t
 def receive(self,timeout=45):
  if not select.select([self.p.stdout],[],[],timeout)[0]:raise TimeoutError('Candidate inference subprocess timeout')
  line=self.p.stdout.readline()
  if not line:raise RuntimeError('Candidate inference subprocess exited')
  out=json.loads(line)
  if out.get('error_type'):raise RuntimeError(out['error_type']+': '+out['error'])
  return out
 def call(self,command,**kw):
  self.p.stdin.write(json.dumps(dict(command=command,**kw))+'\n');self.p.stdin.flush();return self.receive()
 def close(self):
  try:self.call('stop');self.p.wait(timeout=10)
  except Exception:self.p.kill();self.p.wait()
  self.log.close()

def worker(payload):
 number,jobs,component=payload;started=now();cpu=time.process_time();out=ROOT/component/f'worker-{number}';private=PRIVATE/component/f'worker-{number}';out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
 allowed=sorted(os.sched_getaffinity(0));pair=allowed[2*number:2*number+2] or allowed[-2:];os.sched_setaffinity(0,pair)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from ptcg_agent.learning.rl_stage1 import load_model,tensor_sha
 from ptcg_agent.learning.decoder_matrix import EvaluationAgent
 from ptcg_agent.policies import validate_action
 m,r=verify();models={label:load_model(m['packages'][label]['path']) for label in LABELS};assert all(tensor_sha(net)==m['packages'][label]['model_tensor_sha256'] for label,net in models.items())
 parent_load=(now()-started).total_seconds();remote=RemoteActor(private/'candidate-stderr.log');initial=remote.startup
 write(out/'startup.json',dict(at_utc=iso(),pid=os.getpid(),candidate_pid=remote.p.pid,affinity=pair,parent_startup_seconds=parent_load,candidate=initial))
 try:
  for job in jobs:
   if now()>=parse(DEADLINE):raise TimeoutError('Absolute registered deadline before match')
   began=now();seat=job['candidate_seat'];pack=m['packages'][job['opponent']];other=EvaluationAgent(models[job['opponent']],pack['deck'],'argmax');other.reset_match();remote.call('reset',seed=job['seed']);before=remote.call('stats')['stats']
   candidate_times=[];own_times=[];metadata={};hist=[];errors=[];last_obs=None
   env,module=environment()
   def actor(obs,configuration):
    nonlocal last_obs
    if obs.get('select') is None:return m['packages']['Unown-Gradiant']['deck']
    return remote.call('act',observation=obs)['action']
   def comparator(obs,configuration):
    if obs.get('select') is None:other.reset_match();return pack['deck']
    return other(obs)
   def measured(fn,who,times):
    def call(obs,configuration):
     nonlocal last_obs
     if now()>=parse(DEADLINE) or (now()-began).total_seconds()>300:raise TimeoutError('Registered match/deadline cap')
     obs=copy.deepcopy(dict(obs));last_obs={k:obs[k] for k in ['current','select','logs'] if k in obs}
     t=time.perf_counter()
     try:
      action=fn(obs,configuration)
      if obs.get('select'):
       validate_action(obs['select'],action);cur=obs['current'];rid=cur.get('round',1)
       g=metadata.setdefault(rid,dict(round=rid,chooser=None,chooser_chose_first=None,first_player=-1,decisions=0));g['decisions']+=1
       if cur.get('firstPlayer',-1)>=0:g['first_player']=cur['firstPlayer']
       if obs['select']['context']==41:
        g['chooser']=who;g['chooser_chose_first']=obs['select']['option'][action[0]]['type']==1
       if component=='official' and who==seat:hist.append(dict(observation=obs,action=action))
      return action
     except Exception as e:
      errors.append(dict(actor='candidate' if who==seat else 'comparator',error_type=type(e).__name__,error=str(e)));raise
     finally:times.append(time.perf_counter()-t)
    return call
   agents=[None,None];agents[seat]=measured(actor,seat,candidate_times);agents[1-seat]=measured(comparator,1-seat,own_times)
   exclusive(out/(job['match_id']+'-intent.json'),dict(job,at_utc=iso()))
   status='FAILED';reason=None;games=[];result=None
   try:
    env.run(agents)
    for frame in module.Battle.vis:
     current=frame.get('current') or {}
     if current.get('result',-1)>=0:
      idx=len(games)+1;meta=metadata.get(idx,dict(round=idx,chooser=None,first_player=-1))
      games.append(dict(meta,result=current['result'],finish_reason=current.get('finishReason','native_terminal_result'),completion_status='DONE'))
    statuses=[x.status for x in env.state];counts=getattr(env,'result',None)
    assert statuses==['DONE','DONE'] and not errors,(statuses,errors)
    assert counts and len(games)==sum(counts) and len(games) in [2,3],(len(games),counts)
    assert [sum(g['result']==i for g in games) for i in [0,1,2]]==list(counts),'Game/match aggregation mismatch'
    result=0 if counts[0]>counts[1] else 1 if counts[1]>counts[0] else 2
    assert env.state[seat].reward==(0 if result==2 else 1 if result==seat else -1)
    assert games[0]['chooser']==0,'Initial chooser mismatch'
    assert all(g['chooser']==(g['round']-1)%2 and g['first_player'] in [0,1] for g in games),'Later-game chooser/starter mismatch'
    status='DONE'
   except Exception as e:reason=type(e).__name__+': '+str(e)
   after=remote.call('stats')['stats'];wall=(now()-began).total_seconds();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   metrics=dict(candidate_inference_seconds=sum(candidate_times),candidate_model_seconds=after['inference_seconds']-before['inference_seconds'],candidate_max_decision_seconds=max(candidate_times,default=0),comparator_inference_seconds=sum(own_times),comparator_max_decision_seconds=max(own_times,default=0),candidate_decisions=after['decisions']-before['decisions'],forced_decisions=after['forced_decisions']-before['forced_decisions'],candidate_encodes=after['model_encodes']-before['model_encodes'],game_resets=after['game_resets']-before['game_resets'],max_options=after['max_options'],max_tokens=after['max_tokens'],fallback_calls=after['fallback_calls']-before['fallback_calls'],panic_calls=after['guard']['panic']-before['guard']['panic'],weights_unchanged=after['weights_unchanged'],parent_peak_RSS_kib=rss,candidate_peak_RSS_kib=after['peak_RSS_kib'],combined_peak_RSS_kib=rss+after['peak_RSS_kib'],cold_candidate_startup_seconds=initial['pipe_startup_seconds'])
   runtime=metrics['combined_peak_RSS_kib']<12.2*1024**2 and sum(candidate_times)+initial['pipe_startup_seconds']<570 and metrics['fallback_calls']==0 and metrics['panic_calls']==0 and after['weights_unchanged']
   failure_class=None if status=='DONE' else 'adapter' if any(word in (reason or '')+str(errors) for word in ['Unsupported','overflow','reset count','chooser mismatch','aggregation mismatch']) else 'agent' if errors else 'infrastructure'
   row=dict(job,status=status,result=result,games=games,wall_seconds=wall,started_at_utc=began.isoformat(),ended_at_utc=iso(),error=reason,actor_errors=errors,failure_class=failure_class,metrics=metrics,runtime_pass=runtime,official_statuses=[x.status for x in env.state],CPU_affinity=pair)
   if status=='DONE' and metrics['game_resets']!=len(games):row.update(status='FAILED',error='Candidate game reset count mismatch')
   append(out/'outcomes.jsonl',row);write(out/'progress.json',dict(at_utc=iso(),last=job['match_id'],completed=len([x for x in (out/'outcomes.jsonl').read_text().splitlines()])))
   if component=='official':
    with gzip.open(private/(job['match_id']+'-actor-history.json.gz'),'wt') as f:json.dump(hist,f)
   if row['status']!='DONE':
    failure_files=list(private.glob('failure-*.json.gz'))
    if len(failure_files)<3:
     raw=json.dumps(dict(row=row,last_actor_observation=last_obs))[:1048576]
     with gzip.open(private/('failure-'+job['match_id']+'.json.gz'),'wt') as f:f.write(raw)
   del env;module.Battle.vis=[]
  assert all(tensor_sha(net)==m['packages'][label]['model_tensor_sha256'] for label,net in models.items())
 finally:remote.close()
 child=resource.getrusage(resource.RUSAGE_CHILDREN)
 summary=dict(worker=number,component=component,completed=len(jobs),UTC_seconds=(now()-started).total_seconds(),process_CPU_seconds=time.process_time()-cpu+child.ru_utime+child.ru_stime,peak_parent_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,candidate_startup=initial,model_tensors_unchanged=True)
 write(out/'completion.json',summary);return summary

def pool(component,jobs,workers):
 import multiprocessing as mp
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
  futures=[ex.submit(worker,(i,[j for k,j in enumerate(jobs) if k%workers==i],component)) for i in range(workers)]
  summaries=[]
  for f in as_completed(futures):summaries.append(f.result())
 write(ROOT/(component+'-completion.json'),dict(at_utc=iso(),workers=summaries))

def analysis():
 import numpy as np
 data=rows('strength');planned=read(ROOT/'strength-schedule.json');assert len(data)==288 and {x['match_id'] for x in data}=={x['match_id'] for x in planned}
 rng=np.random.default_rng(BOOT);cells={};all_good=all(x['status']=='DONE' and x['runtime_pass'] for x in data)
 for label in LABELS:
  group=[x for x in data if x['opponent']==label];assert len(group)==96
  counts=[dict(win=0,loss=0,draw=0,failure=0) for _ in range(2)];points=np.zeros(2);dist=np.zeros((20000,2));supported=[True,True];strata={};starters={}
  for seat in [0,1]:
   rr=[x for x in group if x['candidate_seat']==seat];assert len(rr)==48
   strata[str(seat)]=dict(attempts=48,completed=sum(x['status']=='DONE' for x in rr),valid_first_games=sum(bool(x['games']) for x in rr))
   # One whole-match draw per bootstrap index; masks retain valid game1 after a later failure.
   arr=np.array([[float(x['result']==seat) if x['status']=='DONE' else np.nan,float(x['games'][0]['result']==seat) if x['games'] else np.nan] for x in rr])
   idx=rng.integers(0,48,(20000,48));sample=arr[idx]
   for k in range(2):
    usable=arr[:,k][~np.isnan(arr[:,k])]
    if not len(usable):supported[k]=False;continue
    points[k]+=float(usable.mean())/2
    denominator=np.sum(~np.isnan(sample[:,:,k]),axis=1);supported[k]&=bool(np.all(denominator>0))
    dist[:,k]+=np.nansum(sample[:,:,k],axis=1)/np.maximum(denominator,1)/2
   for x in rr:
    for k,res in enumerate([x['result'] if x['status']=='DONE' else None,x['games'][0]['result'] if x['games'] else None]):
     key='failure' if res is None else 'draw' if res==2 else 'win' if res==seat else 'loss';counts[k][key]+=1
    if x['games']:
     key='first' if x['games'][0]['first_player']==seat else 'second';d=starters.setdefault(key,dict(first_game_W=0,L=0,D=0,n=0));d['n']+=1;res=x['games'][0]['result'];d['D' if res==2 else 'first_game_W' if res==seat else 'L']+=1
  rates={name:dict(estimate=float(points[k]) if supported[k] else None,interval_95=np.quantile(dist[:,k],[.025,.975]).tolist() if supported[k] else None) for k,name in enumerate(['bo3','first_game'])}
  cells[label]=dict(bo3_counts=counts[0],first_counts=counts[1],**rates,chooser_strata=strata,actual_first_game_starters=starters,failures=[x['match_id'] for x in group if x['status']!='DONE'],runtime=dict(match_seconds=sum(x['wall_seconds'] for x in group),max_match_seconds=max(x['wall_seconds'] for x in group),max_candidate_decision_seconds=max(x['metrics']['candidate_max_decision_seconds'] for x in group),max_combined_RSS_MiB=max(x['metrics']['combined_peak_RSS_kib'] for x in group)/1024,all_checks_pass=all(x['runtime_pass'] for x in group)),all_games=dict(W=sum(g['result']==x['candidate_seat'] for x in group for g in x['games']),L=sum(g['result']==1-x['candidate_seat'] for x in group for g in x['games']),D=sum(g['result']==2 for x in group for g in x['games'])))
 primary=cells['League32-Hydrapple']['bo3'];gate=all_good and primary['estimate'] is not None and primary['estimate']>=.6 and primary['interval_95'][0]>.5
 if not all_good:recommend='incompatible or runtime-failing on this panel; inspect classified failure evidence before reuse'
 elif gate:recommend='usable stronger opponent in the primary local matchup only'
 elif all(c['bo3']['estimate']<=.5 for c in cells.values()):recommend='weaker package on this panel'
 else:recommend='usable complementary opponent candidate; matchup estimates and intervals delimit evidence, no broad diversity benefit established'
 write(ROOT/'analysis.json',dict(at_utc=iso(),attempts=288,cells=cells,primary_gate=gate,all_correctness_runtime_pass=all_good,recommendation=recommend,bootstrap_seed=BOOT,bootstrap_replicates=20000,completed_UTC=True))
 with gzip.open(ROOT/'matches.jsonl.gz','wt') as f:
  for row in sorted(data,key=lambda x:x['match_id']):f.write(json.dumps(row)+'\n')

def run():
 try:os.setsid()
 except PermissionError:pass
 started=now();cpu=time.process_time();state='FAILED';reason=None
 exclusive(ROOT/'run-intent.json',dict(at_utc=iso(),pid=os.getpid(),process_group=os.getpgrp(),deadline_utc=DEADLINE))
 write(ROOT/'process.json',dict(pid=os.getpid(),process_group=os.getpgrp(),at_utc=iso()))
 try:
  m,r=verify();progress('COMPATIBILITY');report('RUNNING');mirror()
  if not (ROOT/'compatibility.json').exists():compatibility()
  assert read(ROOT/'compatibility.json')['status']=='PASSED'
  progress('SIX_OFFICIAL_CHECKS');report('COMPATIBILITY_PASSED');mirror()
  pool('official',read(ROOT/'official-schedule.json'),1)
  checks=rows('official');assert len(checks)==6
  if not all(x['status']=='DONE' and x['runtime_pass'] for x in checks):
   state='INCOMPATIBLE';raise RuntimeError('Six-check compatibility/runtime gate failed; no strength benchmark launched')
  # Clock-only admission: no strength outcomes or check wins enter the estimate.
  startup=max(x['metrics']['cold_candidate_startup_seconds'] for x in checks);worst=max(x['wall_seconds'] for x in checks);workers=r['workers'];estimate=288/workers*(worst+startup)*3+180
  remaining=(parse(DEADLINE)-now()).total_seconds();admit=estimate<remaining
  write(ROOT/'admission.json',dict(at_utc=iso(),admitted=admit,max_check_match_seconds=worst,max_cold_candidate_startup_seconds=startup,workers=workers,inflation=3,analysis_reserve_seconds=180,projected_remaining_seconds=estimate,remaining_budget_seconds=remaining,check_outcomes_used=False))
  if not admit:state='NOT_ADMITTED_BUDGET';raise RuntimeError('Fixed288-match benchmark cannot conservatively fit; sample not reduced')
  exclusive(ROOT/'strength-run-intent.json',dict(at_utc=iso(),attempts=288,schedule_sha256=sha(ROOT/'strength-schedule.json')))
  progress('STRENGTH_BENCHMARK',planned=288);report('RUNNING');mirror()
  pool('strength',read(ROOT/'strength-schedule.json'),workers)
  progress('ANALYSIS');analysis();verify();state='COMPLETE'
 except Exception as e:
  reason=type(e).__name__+': '+str(e);write(ROOT/'failure.json',dict(at_utc=iso(),state=state,reason=reason));(PRIVATE/'failure-trace.log').write_text(traceback.format_exc())
 finally:
  measured=resource.getrusage(resource.RUSAGE_CHILDREN);official=optional('official-completion.json');strength=optional('strength-completion.json');fixed=sum(x['process_CPU_seconds'] for x in official.get('workers',[]))+optional('compatibility.json').get('process_CPU_seconds',0);variable=sum(x['process_CPU_seconds'] for x in strength.get('workers',[]))
  write(ROOT/'costs.json',dict(at_utc=iso(),accounting_start_utc=START,deadline_utc=DEADLINE,task_elapsed_seconds=(now()-parse(START)).total_seconds(),runner_elapsed_seconds=(now()-started).total_seconds(),fixed_pipeline_measured_CPU_seconds=fixed,variable_agent_measured_CPU_seconds=variable,measured_CPU_core_hours=(fixed+variable+time.process_time()-cpu)/3600,runner_child_CPU_seconds=measured.ru_utime+measured.ru_stime,prelaunch_pipeline_CPU='Not fully metered; unavailable, not zero. Preparation included in task elapsed.',official_attempts=len(rows('official')),strength_attempts=len(rows('strength')),GPU_seconds=0,training_steps=0,uploads=0))
  progress(state,reason=reason,official_attempts=len(rows('official')),strength_attempts=len(rows('strength')));write(ROOT/'completion.json',dict(at_utc=iso(),state=state,pid=os.getpid(),reason=reason));report(state,reason);mirror()

if __name__=='__main__':
 {'register':register,'compatibility':compatibility,'run':run}[sys.argv[1]]()
