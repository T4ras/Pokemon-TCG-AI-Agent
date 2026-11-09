"""Registered round-effect audit and saved-workload arithmetic benchmark."""
from pathlib import Path
import datetime as dt,hashlib,json,os,time
ROOT=Path(__file__).resolve().parent.parent
RUN='astra-unown-round-effect-throughput-001';OUT=ROOT/'experiments'/RUN;PRIVATE=ROOT/'data/private'/RUN
SOURCE=ROOT/'experiments/astra-unown-corrected-hydrapple-ppo-6h-001';SAVED=ROOT/'data/private/astra-unown-corrected-hydrapple-ppo-6h-001'
REPORT=ROOT/'reports/astra-unown-round-effect-and-throughput.md'
SEED=904107;KERNEL='portfolio-owner/ptcg-unown-round-effect-throughput';DATASET=KERNEL+'-inputs'
IMAGE='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def parse(x):return dt.datetime.fromisoformat(x.replace('Z','+[removed]'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(v,indent=2,default=str));q.replace(p)
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,default=str)+'\n');f.flush();os.fsync(f.fileno())
def exclusive(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,default=str)
def deadline():return parse(read(OUT/'registration.json')['local_deadline_utc'])
def check_time(reserve=120):
 if (deadline()-now()).total_seconds()<reserve:raise TimeoutError('Registered four-hour local ceiling')
def progress(stage,**kw):write(OUT/'progress.json',dict(at_utc=iso(),stage=stage,pid=os.getpid(),**kw));print(iso(),stage,flush=True)
def setup():
 import shutil,random
 OUT.mkdir(exist_ok=False,parents=True);PRIVATE.mkdir(exist_ok=False,parents=True)
 inputs={}
 for label in ['initial','final']:p=SOURCE/f'evaluation-{label}/matches.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p)
 histories=sorted((SAVED/'histories').glob('*.gz'))
 for p in histories:inputs[str(p.relative_to(ROOT))]=sha(p)
 for name in ['initialization.pt','experimental-text-tables.npz']:shutil.copy2(SAVED/name,PRIVATE/name)
 shutil.copy2(SOURCE/'manifest.json',OUT/'manifest.json')
 write(OUT/'registration.json',dict(at_utc=iso(),local_deadline_utc=(now()+dt.timedelta(hours=4)).isoformat(),inputs=inputs,exploratory=True,bootstrap=dict(replicates=20000,seed=SEED,unit='Whole BO3,independently within checkpoint/initial-chooser strata; game1/game2 stay together',contrast='(final-initial)G1-(final-initial)G2',starting_roles='Descriptive actual learner first/second within each game; no causal interpretation'),
  audit=dict(histories='All28 preselected streams; legal-memory/feature checks every observation; both models replay at most768 preselected meaningful prefixes, round-robin file/game strata, fixedseed. Normal and cleared knowledge counterfactual on same prefixes; explicit off-policy labels',examples='First12 disagreements in deterministic sampled order, named physical options/card IDs and exact filename/line/prefix; observed vs counterfactual explicit',reuse='Prior broad feature/reset and optimized-math proofs reused,not rerun wholesale'),
  memory_gate='No unresolved interpretation-blocking correctness defect AND (exploratory interaction95% lower>0 or a specific reproducible retained-memory mechanism established). Save decision before games. Otherwise SKIP.',
  panel=dict(cells=['initial-normal','final-normal','initial-cleared','final-cleared'],BO3_per_cell=128,chooser_per_cell=[64,64],opponent='Frozen corrected initialization,greedy',learners='Initial/final frozen,greedy,original orderedHydrapple',memory='Both actors same treatment; clear ONLY cross-game lower bounds/provenance at game reset; within-game tracking,own deck unchanged',primary='(final-initial)BO3 cleared minus (final-initial)BO3 normal',bootstrap='20000 independent whole matches within checkpoint/treatment/chooser strata,equalchooser; paired round relationship preserved',failures='Consume attempts,stop on correctness failure,no replacements; first scheduled matches are integration checks',retention='First2 per cell/chooser,bothactors;512MiBlegalhistory/2GiBdetail caps'),
  numerical=dict(registered_before_results=True,reference='Immutable scalar source retained; historicalchosenLP1.69277e-5,TV4.42569e-6,KL1.34274e-7 motivates distribution-aware assessment',criteria=dict(legal_masks='exact',probability_max_abs=5e-5,TV_max=1e-4,forward_KL_max=1e-4,forward_KL_mean=1e-6,values_atol=1e-4,values_rtol=1e-4,loss_atol=1e-5,loss_rtol=1e-4,gradient_relative_L2=1e-3,gradient_max_abs=1e-4,optimizer_parameter_max_abs=2.1e-5,postupdate_TV_max=1e-3,postupdate_TV_mean=1e-4,greedy_changes='Only when scalar top2 logit margin <=2*measured maximum legal-logit perturbation; report all ties'),rationale='Float32batchedGEMM reduction differences evaluated over full legal distributions and gradient/update effects. A2LR per-parameter cap covers at most opposing nearzero freshAdam signs,does not excuse material distribution changes. Any failed criterion rejects revised path; no tuning after results.',likelihood='Never rebase old probabilities; report scalar-rollout/raw-trainer mismatch and consistent raw-collector/raw-trainer replay. No native batched collector adopted.'),
  workload=dict(phases=['first0','last47'],selection='Original cohort permutation(seed2026102701+phase): first512 decisions plus actual final remainder; prefix128 arithmetic comparisons,paddedfirst16 and synthetic17 remainder; no outcomes used',timing='Same hardware and restored model/optimizer/RNG,old/new complete transactions and fsync checkpoint cost.3 repeats,median; full gradient/update not forward-only',checkpoint='Every8 accepted updates and phase boundaries,future maximum7 lost acceptedupdates; in-memory full rollback eachstep; durable journal beforeupdate and no automatic replay. Benchmarks only,no candidates'),
  budgets=dict(new_native_attempts_max=512,local_elapsed_seconds=14400,GPU_session_seconds=1800,GPU_jobs_max=1,provider_timeout_seconds=1500,remote_process_seconds=1320,remote_benchmark_seconds=1100,no_replacement=True,no_training=True,private_upload_authorized=True,paid_model_API_requests=0),conservative_local_clock='Local four-hour ceiling includes remote waiting as extra conservatism; no automatic reset/extension'))
 rng=random.Random(SEED);jobs=[]
 for cell in ['initial-normal','final-normal','initial-cleared','final-cleared']:
  seats=[0]*64+[1]*64;rng.shuffle(seats);seen={0:0,1:0}
  for i,s in enumerate(seats):
   jobs.append(dict(id=f'memory-{cell}-{i:03d}',stage='memory-panel',cell=cell,learner_seat=s,learner_deck='Hydro',opponent_deck='Hydro',seed=SEED+len(jobs),retain_history=seen[s]<2));seen[s]+=1
 write(OUT/'memory-schedule.json',jobs)
 REPORT.write_text('# Corrected Hydrapple round effect and throughput\n\nState: **REGISTERED**. No new matches or GPU job launched. Historical evidence preserved.\n')
if __name__=='__main__':setup()
