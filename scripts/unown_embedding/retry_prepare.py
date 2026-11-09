"""Prepare the explicitly requested retry; never replace historical evidence."""
import shutil
from .common import *

def prepare():
 old_run='astra-unown-new-card-embedding-adaptation-001'
 old=ROOT/'experiments'/old_run;private=ROOT/'data/private'/old_run
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
 if (OUT/'registration.json').exists():raise RuntimeError('Retry already registered')
 assert read(old/'training-completion.json')['actor_changed'] is False
 assert read(old/'training-completion.json')['accepted_updates']==0
 assert read(old/'evaluation-initial-completion.json')['completed']==512
 assert read(old/'evaluation-initial-completion.json')['failures']==0
 assert read(old/'collector-failure.json')['started_attempts']==16
 shutil.copytree(private/'assets',PRIVATE/'assets',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ('initialization.pt','text-tables.npz','old-probes.pt','new-probes.pt'):
  shutil.copy2(private/name,PRIVATE/name)
 assert sha(PRIVATE/'initialization.pt')==INITIAL_FILE
 paths=read(private/'asset-paths.json')
 write(PRIVATE/'asset-paths.json',{k:v.replace(old_run,RUN) for k,v in paths.items()})
 for name in ('decks.json','trainable-rows.json','evaluation-initial-schedule.json','evaluation-final-schedule.json','initialization-provenance.json','mechanic-readiness.json','action-schema-check.json','probe-provenance.json','evaluation-initial-records.jsonl','evaluation-initial-completion.json'):
  shutil.copy2(old/name,OUT/name)
 schedule=copy.deepcopy(read(old/'training-schedule.json')[1:])
 assert len(schedule)==MAX_COHORTS
 for cohort in schedule:
  for job in cohort:job['id']=job['id'].replace('emb-train-','emb-retry-train-')
 write(OUT/'training-schedule.json',schedule)
 preserved=['registration.json','input-manifest.json','bundle.json','cold-start.json','collector-failure.json','training-failure.json','training-completion.json','GPU-accounting.json','completion.json','evaluation-initial-schedule.json','evaluation-initial-records.jsonl','evaluation-initial-completion.json']
 provenance=dict(at_utc=iso(),authority='User explicitly requested: fix whatever happened there and run again',
  prior_run=old_run,prior_files={str((old/name).relative_to(ROOT)):sha(old/name) for name in preserved},
  initial_evaluation_reused=True,initial_attempts_reused=512,new_initial_attempts=0,initial_evaluation_record_SHA256=sha(old/'evaluation-initial-records.jsonl'),
  prior_GPU_quota_seconds=read(old/'GPU-accounting.json')['quota_delta_seconds'],charged_prior_seconds=119,
  retry_session_cap_seconds=SESSION_CAP,aggregate_session_cap_seconds=21600,allocated_devices=2,
  prior_training_attempts=16,retry_training_attempts_cap=MAX_COHORTS*16,aggregate_training_attempts_cap=4096,
  unchanged='Initialization, sparse row scope, decks, PPO recipe, fresh optimizer, sampler/argmax, target500000, final512, statistics; no hyperparameter or checkpoint selection',
  schedule='Remaining original cohorts1..255 with unchanged deck/chooser/seed assignments; IDs gain retry prefix. No failed-cohort replacement or stale experience reuse.',
  repair='RemotePolicy declares diagnostic_forwards_allowed=False. Counterfactual logging makes no remote calls; actual action path and probabilities are unchanged. Pool checks learner requests equal returned rows.',
  regression_sampling='Before new diagnostic results: replay each of the two lexicographically sorted retained learner histories from failed cohort0, from match start through64 conditional decisions or EOF. No outcomes used for selection; old/unfixed flag versus repaired flag on same CPU packets/RNG. Zero new native games.',
  failed_report=str((old/'before-explicit-retry/reports/astra-unown-new-card-embedding-adaptation.md').relative_to(ROOT)),
  no_further_automatic_replacement=True,paid_model_APIs=False,competition_submission=False)
 write(OUT/'retry-provenance.json',provenance)
 reg=copy.deepcopy(read(old/'registration.json'))
 reg.update(at_utc=iso(),context=context().spec(),cohorts=schedule,retry=provenance)
 reg.pop('execution_freeze',None)
 reg['trainable_rows_sha256']=sha(OUT/'trainable-rows.json')
 reg['stops'].update(max_cohorts=MAX_COHORTS,max_training_attempts=MAX_COHORTS*16,session_seconds=SESSION_CAP,provider_seconds=PROVIDER_CAP,inner_seconds=PROCESS_CAP,absolute_seconds=ABSOLUTE_CAP,allocated_GPU_hours_max=SESSION_CAP*2/3600)
 reg['evaluation']['initial_before_training']='Reuse the exact completed512 initialization attempts from failed attempt; same weights/decks/decoders, no new baseline matches'
 reg['authority']='Explicit user repair-and-retry request; remaining original budgets; exact private retry upload approval before launch; no further automatic retries'
 write(OUT/'registration.json',reg)
 write(OUT/'repair-sampling-registration.json',dict(at_utc=iso(),rule=provenance['regression_sampling'],native_attempts=0,GPU_seconds=0))
 progress('RETRY_REGISTERED',reused_initial_attempts=512,new_native_attempts=0,GPU_seconds=0)

if __name__=='__main__':prepare()
