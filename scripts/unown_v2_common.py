"""Unown adaptation v2 frozen paths and durable bookkeeping; no network."""
from pathlib import Path
import datetime as dt, hashlib, json, os
REPO=Path(__file__).resolve().parent.parent
RUN='astra-unown-current-pool-adaptation-v2-001'
OUT=REPO/'experiments'/RUN
PRIVATE=REPO/'data/private'/RUN
PREVIOUS=REPO/'experiments/astra-unown-current-pool-adaptation-001'
PREVIOUS_PRIVATE=REPO/'data/private/astra-unown-current-pool-adaptation-001'
PARENT=REPO/'data/private/astra-unown-gradiant-benchmark-001'
UP=PARENT/'upstream/a176778d99da74a44b495bf19299b4795faf12f2'
SDK=REPO/'data/sdk/extracted/sample_submission/sample_submission/sample_submission'
SEED=904111
START='[removed]'
DEADLINE='[removed]'
SCHEMA='ug-current-v2-approx'
def now():return dt.datetime.now(dt.timezone.utc)
def iso():return now().isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def exclusive(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,default=str);f.flush();os.fsync(f.fileno())
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,default=str)+'\n');f.flush();os.fsync(f.fileno())
def progress(stage,**kw):
 write(OUT/'progress.json',dict(at_utc=iso(),pid=os.getpid(),stage=stage,**kw));print(iso(),stage,flush=True)
def assert_deadline():
 if now()>=dt.datetime.fromisoformat(DEADLINE):raise TimeoutError('Registered absolute local deadline')
def register():
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
 value=dict(at_utc=iso(),start_utc=START,deadline_utc=DEADLINE,schema=SCHEMA,seed=SEED,
  parent_manifest_sha256=sha(PREVIOUS/'parent-manifest.json'),
  mapping_sha256=sha(PREVIOUS/'identity-mapping.json'),
  embedding_payload_sha256=read(PREVIOUS/'embedding-request-manifest.json')['requests']['original-anchors-plus-new']['requests_sha256'],
  parent_checkpoint_sha256='959b220daf6777f45ad5d6aa14c35d3274bf3958ed502c7551679453e5f4aaea',
  migration='Reuse append-only v1 mapping; preserve every old learned row/buffer/text vector exactly. New learned rows named seed2026102303 normal(.0,.02); auxiliary output expands only. No candidate export until accepted text adaptation.',
  original_projection_search=dict(scope='One pass over pinned upstream tree, available isolated assets and public release assets; no contact, no waiting, no repeated acquisition',max_public_requests=1,timeout_seconds=30),
  text_mapping=dict(model='text-embedding-3-large',dimensions=3072,encoding='float',batch=256,
   spend_cap_USD=.10,rate_USD_per_million=.13,reserve_multiplier=1.25,max_attempts_per_request=2,
   grouping='NFKC lowercase punctuation/whitespace-normalized exact texts and char5 TFIDF cosine>=.92 connected components; all near/exact duplicates together',
   split='Group SHA256(seed:kind:canonical representative) into train70%, validation15%, test15%; freeze IDs before embeddings/fitting',
   lambdas=[.0001,.001,.01,.1,1.],method='Centered multioutput ridge via SVD, separate kinds, normalize predicted rows; validation mean cosine selects, larger lambda wins exact ties',
   baselines=['normalized training-target mean','fresh-embedding nearest train reference reduced vector'],
   criteria=dict(per_kind_mean_cosine_min=.85,median_cosine_min=.90,p10_cosine_min=.65,
    median_L2_max=.45,p90_L2_max=.85,least_similar_quartile_mean_cosine_min=.75,
    mean_cosine_gain_over_mean_min=.10,mean_cosine_gain_over_nearest_min=.02,
    policy_nontie_action_agreement_min=.97,policy_mean_TV_max=.02,policy_p95_TV_max=.10,legal_consistency=1.,minimum_policy_decisions=24),
   rationale='Conservative fidelity screen for new-only text coordinates, including novel-text tail and functional impact. Not a playing-strength criterion.',
   policy_inputs='First six retained Unown official histories in lexical path order; at most512 sequential decisions total. Replace held-out old rows only in diagnostic copy; full legal prefixes. Old top-logit gap<=1e-6 is a tie; report exact argmax and acceptable top-set agreement separately.',
   final_fit='Only after all criteria pass: refit selected lambdas on every eligible old reference. Use predictions exclusively on appended new/changed rows. Never replace old released vectors.'),
  inference=dict(dynamic='Per-decision capacities at least160 tokens/64 options/two tools; grow tokens on explicit overflow, include every option plus STOP, all tools. Exact submitted math under safe old layout.',
   attachment_fix='Official toolIndex/energyIndex resolve attached card, host and attachment pointers; preserve distinct masks. Old unsafe attachment-reference cases are expected fixes, separately counted.',
   information='actor current/select/legal logs only; opponent hand and unrelated opaque fields excluded. Game-local tracker resets; legal opponent physical-card lower bounds scoped by game and max across games; no scores or behavioral fitting.',
   effect_contract='Use current SDK metadata and verified old records; exact supported new text patterns only; retain explicit unsupported feature audit. Engine controls legality/outcomes.'),
  training=dict(backend='Python/PyTorch CPU reference; GPU not selected',
   optimizer='Fresh AdamW, lr.001 beta(.95,.999) eps1e-12; decay.01 only1D parameters,0 matrices. Deliberate reference-path difference from native Muon, not a resumed or numerically identical native recipe.',
   episode='One completed game for a fixed learner seat; reward+1/-1/0 from learner perspective, no negamax sign in learner-only transitions. BO3 opponent and weights pinned; legal deck facts persist between games.',
   gamma=1.,gae_lambda=.95,clip=.2,entropy_coefficient=.01,value_coefficient=2.,max_gradient_norm=1.5,
   privileged='Blind value drives GAE. Oracle value may regress same return with zero privileged inputs; hand/prize auxiliaries masked unless separate explicit truth targets available. No guessed truth targets.',
   smoke_updates=2,update_sample_cap=32,retained_probe_decision_cap=32,
   failures='Failed or truncated matches are errors, no invented rewards or bootstrap into restarted game; no replacement matches.'),
  verification=dict(parent_logits_atol=1e-6,parent_logits_rtol=1e-5,mask_and_actions='exact on faithful safe unchanged histories',
   same_backend_resume_atol=1e-6,same_backend_resume_rtol=1e-5,old_buffer_and_parameter_bytes='exact before disposable update',
   native_allocation='8 technical old-supported BO3 versus frozen Unown parent,4 per chooser; if text accepted then8 current-pool BO3 versus registered regression packages,4 per chooser. Total16 planned,16 unallocated and not automatic.',
   attempts_cap=32,prior_attempts=0,GPU_hours_cap=2,prior_GPU_hours=0,used_backend_GPU_hours=0,
   diagnostic_model='If credentials/text gate missing: deterministic nonsemantic random unit vectors only for isolated technical probe, explicitly not a candidate; allow old-supported smoke only. Current-pool readiness remains blocked.'),
  prohibited=['substantive BC/RL','competition submission','uploads','temperature/strength sweep','replacing old text rows','expanded-pool PCA'],
  cost='All preparation and smoke Pipeline/Fixed; substantive Agent/Variable0; measured CPU process/children, elapsed UTC, paid reservations and actual usage persisted')
 exclusive(OUT/'registration.json',value);write(OUT/'registration-sha256.json',dict(sha256=sha(OUT/'registration.json')))
 progress('REGISTERED')
if __name__=='__main__':register()
