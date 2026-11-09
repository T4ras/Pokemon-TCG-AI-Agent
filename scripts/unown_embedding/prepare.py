"""Register one unchanged scientific arm before any native attempts."""
import shutil,sys
from .common import *
from unown_runner.runtime import Runtime

def prepare():
 OUT.mkdir(exist_ok=True,parents=True);PRIVATE.mkdir(exist_ok=True,parents=True)
 if (OUT/'registration.json').exists():raise RuntimeError('Registration already exists; retain evidence rather than overwrite')
 copied=ROOT/'data/private/astra-unown-bucket64-compilation-repair-001/assets'
 shutil.copytree(copied,PRIVATE/'assets',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 original=ROOT/'data/private/astra-unown-local-init-learning-pilot-execution-repair-001'
 for source,target in [('initialization.pt','initialization.pt'),('experimental-text-tables.npz','text-tables.npz')]:shutil.copy2(original/source,PRIVATE/target)
 assert sha(PRIVATE/'initialization.pt')==INITIAL_FILE
 assets=dict(start_checkpoint=PRIVATE/'initialization.pt',opponent_checkpoint=PRIVATE/'initialization.pt',text_tables=PRIVATE/'text-tables.npz',
  identity_directory=PRIVATE/'assets/identity',current_tables=PRIVATE/'assets/current/current-tables.json',original_tables=PRIVATE/'assets/original/original-tables.json',released_submission=PRIVATE/'assets/released/submission',original_sdk=PRIVATE/'assets/original-sdk',environment=PRIVATE/'assets/environment')
 write(PRIVATE/'asset-paths.json',{k:str(v.relative_to(ROOT)) for k,v in assets.items()})
 inventory=read(ROOT/'experiments/top50-models-decks-novelty-904104-001/analysis.json');decks={}
 for name in ('D10','D04','D02','D19','D21','D01','D03'):
  item=next(d for d in inventory['decks'] if d['id']==name);cards=[c['card_id'] for c in item['cards'] for _ in range(c['count'])]
  assert hashlib.sha256(json.dumps(tuple(sorted(cards))).encode()).hexdigest()==item['sha256']
  decks[name]=dict(cards=cards,canonical_sha256=item['sha256'],source='Exact top50 frozen deck '+name)
 hydro=read(ROOT/'experiments/astra-unown-local-init-learning-pilot-001/manifest.json')['decks']['Hydro'];decks['Hydro']=hydro
 assert hashlib.sha256(json.dumps(hydro['cards'],separators=(',',':')).encode()).hexdigest()==hydro['ordered_deck_json_sha256']
 write(OUT/'decks.json',decks)
 ctx=context();runtime=Runtime(ctx)
 try:
  model=runtime.load_model(PRIVATE/'initialization.pt','cpu');assert runtime.state_hash(model)==INITIAL_TENSOR
  mp=read(assets['identity_directory']/'identity-mapping.json');cov=read(assets['identity_directory']/'identity-coverage.json')
  current=read(assets['current_tables']);cards={c['cardId']:c for c in current['cards']};attacks={a['attackId']:a for a in current['attacks']}
  own_ids=sorted(set(decks['D10']['cards']+decks['D04']['cards']));linked_attacks=sorted({aid for cid in own_ids for aid in cards[cid]['attacks']});linked_skills=sorted({sid for cid in own_ids for sid in mp['skills_by_card'][str(cid)]})
  rows=dict(cards=sorted({mp['cards'][str(cid)] for cid in own_ids if mp['cards'][str(cid)]>=cov['cards']['parent_rows']}),
   attacks=sorted({mp['attacks'][str(aid)] for aid in linked_attacks if mp['attacks'][str(aid)]>cov['attacks']['unknown']}),
   skills=sorted(sid for sid in linked_skills if sid>cov['skills']['unknown']))
  for kind in rows:assert cov[kind]['pad'] not in rows[kind] and cov[kind]['unknown'] not in rows[kind]
  sources=[dict(card_id=cid,name=cards[cid]['name'],row=mp['cards'][str(cid)],attacks=[dict(engine_id=a,row=mp['attacks'][str(a)],trainable=mp['attacks'][str(a)] in rows['attacks']) for a in cards[cid]['attacks']],skills=[dict(row=s,trainable=s in rows['skills']) for s in mp['skills_by_card'][str(cid)]],metadata=cards[cid],attack_metadata=[attacks[a] for a in cards[cid]['attacks']]) for cid in own_ids if mp['cards'][str(cid)] in rows['cards']]
  for item in sources:
   if item['metadata']['evolvesFrom']:assert any(c['name'].strip()==item['metadata']['evolvesFrom'].strip() for c in cards.values())
  write(OUT/'trainable-rows.json',dict(rows=rows,source_cards=sources,parent_row_counts={k:cov[k]['parent_rows'] for k in rows},frozen='All other identity/text rows, including old, pad, unknown, unused; shared actor and numerical buffers'))
  # Retained encoded probes are software/off-policy inputs only, never training.
  retained=torch.load(ROOT/'data/private/astra-unown-bucket64-compilation-repair-001/development.pt',map_location='cpu',weights_only=False)
  probes=retained['rows'][:64];torch.save(probes,PRIVATE/'old-probes.pt')
  write(OUT/'initialization-provenance.json',dict(tensor_hash=INITIAL_TENSOR,checkpoint_sha256=sha(PRIVATE/'initialization.pt'),text_tables_sha256=sha(PRIVATE/'text-tables.npz'),source_report='astra-unown-local-init-learning-pilot.md',starting_order='Same verified actual-firstPlayer feature and reset rules on both roles',optimizer='Fresh, not inherited from any later PPO checkpoint',probe_source='Frozen retained64 encoded Hydrapple decisions; off-policy software diagnostics only'))
 finally:runtime.close()
 training=[];pairings=[(a,b) for a in ('D10','D04') for b in ('Hydro','D02','D19','D21')]
 for phase in range(MAX_COHORTS):
  jobs=[dict(id=f'emb-train-{phase:03d}-{j:02d}-{seat}',stage='training',learner_seat=seat,learner_deck=a,opponent_deck=b,seed=SEED+100000+phase*100+j*2+seat,retain_history=phase<2 and j==0) for j,(a,b) in enumerate(pairings) for seat in (0,1)]
  random.Random(SEED+phase).shuffle(jobs);training.append(jobs)
 evaluations={}
 for label in ('initial','final'):
  jobs=[]
  for a in ('D10','D04'):
   for b in ('Hydro','D02','D01','D03'):
    for index in range(64):jobs.append(dict(id=f'emb-eval-{label}-{a}-{b}-{index:02d}',stage='evaluation-'+label,learner_seat=index%2,learner_deck=a,opponent_deck=b,seed=SEED+200000+(0 if label=='initial' else 10000)+len(jobs),retain_history=False))
  random.Random(SEED+(0 if label=='initial' else 1)).shuffle(jobs);assert len(jobs)==512;evaluations[label]=jobs
  write(OUT/f'evaluation-{label}-schedule.json',jobs)
 write(OUT/'training-schedule.json',training)
 registration=dict(at_utc=iso(),context=ctx.spec(),device='cuda',decks=decks,cohorts=training,
  scientific_question='Can adapting only used appended card/attack/skill identities and text corrections improve unfamiliar-card play with a frozen decision network?',
  initialization=read(OUT/'initialization-provenance.json'),trainable_rows_sha256=sha(OUT/'trainable-rows.json'),
  adaptation=dict(identity='Sparse zero-initialized additive row parameters, equivalent to changing eligible effective identity rows only',text='Zero corrections on fixed approximate vectors; all original vectors unchanged',value='Blind head alone; detached feature input; value coefficient2; no oracle or auxiliary objective',optimizer='Fresh AdamW(.95,.999),eps1e-8,weight_decay0,foreachFalse; only sparse rows and value parameters are members',embedding_lr='Cosine1e-4 to1e-5 indexed by collected fresh learner decisions at cohort end /500000; constant inside cohort',value_lr=1e-5),
  recipe=dict(FIXED_RECIPE,logical_minibatch=128,epochs=4,compilation=False,initialization_KL_stop=False,pre_KL=.02,post_KL=.03),
  stops=dict(fresh_decisions=TARGET,max_cohorts=256,max_training_attempts=4096,zero_accepted_actor_cohorts=3,session_seconds=SESSION_CAP,allocated_devices=2,allocated_GPU_hours_max=12,used_devices=1,provider_seconds=PROVIDER_CAP,inner_seconds=PROCESS_CAP,absolute_seconds=ABSOLUTE_CAP,export_reserve_seconds=EXPORT_RESERVE,no_replacement=True),
  evaluation=dict(attempts=1024,initial_before_training=512,final_after=512,per_cell=64,chooser=32,learners=['D10','D04'],opponents=['Hydro','D02','D01','D03'],decoders='both conditional argmax',primary='Final-minus-initial mean of8 cells and2 chooser strata; draws nonwins',interval='20000 independent whole-match bootstrap draws within checkpoint/learner-deck/opponent/chooser strata, equal weights; primary95% percentile; seed2026103001; failures and valid first games retained, no replacement',secondary=['each learner','seen Hydro/D02','held-out D01/D03','first-game outcomes','every cell W/D/L/attempts/completions/failures'],selection='One final committed checkpoint, no intermediate strength evaluation',no_actor_change='Skip redundant final512 if final effective actor is unchanged'),
  readiness='Local metadata/mapping/mechanic/row-isolation/zero-parity/value-detach/export/cache/old-policy tests before native/GPU; first planned training cohort contains live integration, first2cohorts verify refresh; no separate native diagnostics',
  authority='Current human request authorizes this single six-session-hour training job and1024 CPU attempts; exact private-upload review follows prepared payload; no paid model APIs or leaderboard submission',
  retained_original_reports={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'reports/astra-unown-local-init-learning-pilot.md',ROOT/'reports/astra-unown-starting-order-broad-validation.md',ROOT/'reports/astra-unown-bucket64-runner-integration.md',ROOT/'reports/astra-unown-bucket64-compilation-repair.md',ROOT/'reports/top50-models-decks-novelty.md']})
 write(OUT/'registration.json',registration)
 progress('REGISTERED_LOCAL_READINESS',rows={k:len(v) for k,v in rows.items()},native_attempts=0,GPU_seconds=0)

if __name__=='__main__':
 import traceback
 try:prepare()
 except BaseException as error:
  write(OUT/('prepare-failure-'+str(time.time_ns())+'.json'),dict(at_utc=iso(),error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),native_attempts=0,GPU_seconds=0));raise
