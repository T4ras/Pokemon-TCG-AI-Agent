"""Readable automatic report; inventories and raw evidence remain separate."""
import sys
from .common import *

def get(name):return read(OUT/name) if (OUT/name).exists() else {}
def pct(value):return 'pending' if value is None else f'{value*100:.1f}%'
def pp(value):return 'pending' if value is None else f'{value*100:+.1f} pp'
def link(name):return f'[{name}](../experiments/{RUN}/{name})'

def concise_reason(value):
 if 'probability_max_abs' in str(value) and 'precision_envelope' in str(value):
  return 'Stopped before cohort175 updates: collection/replay probability difference0.00005746 exceeded0.00005; the higher-precision diagnostic also exceeded its float32 tolerance. Full details remain in training-failure.json.'
 return value

def report(state='PREPARING',reason='Local preparation in progress'):
 training=get('training-completion.json');evaluation=get('evaluation-analysis.json');proof=get('parameter-verification.json');readiness=get('readiness.json');regression=get('runner-regression.json');scope=get('trainable-rows.json');account=get('GPU-accounting.json');cohorts=records(OUT/'adaptation-cohorts.jsonl');diagnostics=records(OUT/'policy-diagnostics.jsonl')
 rows=[]
 for p in proof.get('trainable_inventory',[]):
  if p['trainable']:rows.append(f"| `{p['name']}` | {' × '.join(map(str,p['shape']))} | {p['elements']:,} | {p['role']} |")
 cells=[]
 for stage in ('initial','final'):
  records_stage=records(OUT/f'evaluation-{stage}-records.jsonl')
  for deck in ('D10','D04'):
   for opponent in ('Hydro','D02','D01','D03'):
    selected=[r for r in records_stage if r['learner_deck']==deck and r['opponent_deck']==opponent];done=[r for r in selected if r['status']=='DONE'];first=[r for r in selected if r.get('games') and r['games'][0].get('completion_status')=='DONE'];W=sum(r['result']==r['learner_seat'] for r in done);D=sum(r['result']==2 for r in done);FW=sum(r['games'][0]['result']==r['learner_seat'] for r in first);FD=sum(r['games'][0]['result']==2 for r in first)
    item=next((c for c in evaluation.get('cells',[]) if c['checkpoint']==stage and c['deck']==deck and c['opponent']==opponent),{})
    def estimate(metric):
     data=item.get(metric,{})
     return f"{pct(data['rate'])} [{pct(data['interval95'][0])}, {pct(data['interval95'][1])}]" if data else 'pending'
    cells.append(f"| {stage} | {deck} | {opponent} | {len(selected)}/{len(done)}/{len(selected)-len(done)} | {W}/{D}/{len(done)-W-D} | {estimate('BO3')} | {FW}/{FD}/{len(first)-FW-FD} | {estimate('first_game')} | {len(first)} |")
 contrasts=[]
 for metric,views in evaluation.get('contrasts',{}).items():
  for name,v in views.items():
   if v['status']=='COMPLETE':contrasts.append(f"| {metric} | {name} | {pct(v['initial'])} | {pct(v['final'])} | {pp(v['final_minus_initial'])} | [{pp(v['interval95'][0])}, {pp(v['interval95'][1])}] |")
 journal=records(OUT/'journal.jsonl');collection=sum(e.get('metrics',{}).get('elapsed_seconds',0) for e in journal if e.get('kind')=='collected');update=sum(c.get('update_seconds',0) for c in cohorts);export=get('remote-completion.json')
 init_count=len(records(OUT/'evaluation-initial-records.jsonl'));final_count=len(records(OUT/'evaluation-final-records.jsonl'));training_attempts=len(list((OUT/'native-intents').glob('*.json'))) if (OUT/'native-intents').exists() else 0
 retry=get('retry-provenance.json')
 prior=retry.get('prior_GPU_quota_seconds',0)
 retry_note=(f"The first attempt failed before learning: counterfactual retained-history logging made remote inference requests without corresponding action rows. The repaired path disables those diagnostic-only queries and checks request/rollout counts. The original failed report and all evidence are preserved in `{retry.get('failed_report','')}`; see {link('retry-provenance.json')} and {link('collector-repair-verification.json')}. The exact512 completed initial matches are reused without new baseline matches or outcome-based selection. The failed attempt consumed16 training BO3 and {prior:.3f} GPU-quota seconds. This explicit retry allows at most {MAX_COHORTS*16:,} more training attempts and {SESSION_CAP:,} session seconds, keeping both attempts inside the original4,096-match/six-hour allocation. No further automatic retry is authorized.\n" if retry else '')
 benchmark=get('checkpoint-benchmark-registration.json')
 benchmark_note=(f"CPU benchmark separately authorized on{benchmark['at_utc'][:10]}: evaluate the exact retained checkpoint after174 completed cohorts using the original512 final BO3 schedule and original512 baseline. Both sides use conditional argmax. No weights, decks, decoder, training, GPU usage, sample size or statistical rules change. The training run remains FAILED at its pre-update numerical consistency check; evaluating this fixed partial-training checkpoint does not erase that failure or claim the500,000-decision target was reached. See {link('checkpoint-benchmark-registration.json')} and {link('checkpoint-benchmark-completion.json')}.\n" if benchmark else '')
 primary=evaluation.get('contrasts',{}).get('BO3',{}).get('primary',{})
 if training and training.get('actor_changed') is False:decision='No retained actor change. Final512 strength matches are skipped; independent-deal variation cannot demonstrate learning.'
 elif primary.get('status')=='COMPLETE':
  decision='The primary interval is entirely positive in this bounded panel.' if primary['interval95'][0]>0 else 'The primary interval is entirely negative in this bounded panel.' if primary['interval95'][1]<0 else 'The primary interval overlaps zero; this pilot does not establish a BO3-strength improvement.'
 else:decision='Strength comparison pending. Changed embeddings alone will not establish improved play.'
 failures=[]
 for name in ('training-failure.json','evaluation-initial-failure.json','evaluation-final-failure.json'):
  if get(name):failures.append(f"{name}: {concise_reason(get(name).get('error'))}")
 REPORT.parent.mkdir(exist_ok=True)
 REPORT.write_text(f'''# Unown new-card embedding adaptation

State: **{state}**. {concise_reason(reason)}

{retry_note}
{benchmark_note}
**{decision}** Original deployed policies remain unchanged. This experiment does not authorize a competition submission or another training run.

## Frozen comparison

One single-arm experiment starts from the experimental current-pool initialization, tensor SHA256 `{INITIAL_TENSOR}`, with the validated actual-starting-order adapter on both roles. File/text hashes are in {link('initialization-provenance.json')}. This initialization retains approximate local text mappings whose earlier vector-fidelity gate failed; it is not a later Hydrapple PPO checkpoint and its text tables are reused unchanged.

Learners: **D10** (Banette / Lillie’s Clefairy ex / Dudunsparce), **D04** (Metagross / Mega Excadrill ex / Genesect ex), equally weighted. Training opponents: original registered **Hydro**, **D02**, **D19**, **D21**, equally weighted. Exact60-card lists and hashes: {link('decks.json')}. These are local Unown policies, not their leaderboard source agents. Every16-BO3 cohort has all eight deck pairings twice, one match per initial chooser, with a frozen shuffled schedule. Learner sampling is conditional T1; the frozen opponent uses conditional argmax. Policies/decks/decoders are fixed within BO3; forced observations, legal knowledge and game/match resets use the existing corrected adapter. Native deals are independent and unpaired.

## What can learn

| Parameter | Shape | Elements | Role |
|---|---:|---:|---|
{chr(10).join(rows) or '| Inventory pending local checks | | | |'}

Eligible semantic rows: cards **{len(scope.get('rows',{}).get('cards',[]))}**, attacks **{len(scope.get('rows',{}).get('attacks',[]))}**, skills **{len(scope.get('rows',{}).get('skills',[]))}**. Exact row/card/link inventory: {link('trainable-rows.json')}. Only appended identities actually referenced by the two learner decks are selected; reused old attack/skill rows stay frozen. Sparse additive parameters start at zero, so the optimizer never owns old, padding, unknown or unused rows. All original identity/text buffers, numerical tables, shared projections, Transformer and action-scoring parameters are frozen. Text corrections change effective selected vectors without regenerating the base text file.

The blind value head has its own1e-5 learning rate and detached input features. Value-loss gradients cannot reach embeddings or the actor; oracle/auxiliary objectives are unused. The existing global1.5 norm clip remains unchanged across trainable parameters. Effective canonical export folds only eligible corrections into the corresponding rows; ordinary inference reloads the same effective vectors. Recovery retains sparse parameters, optimizer/RNG/cursor and row provenance.

## Readiness and scope limits

Focused local readiness: **{readiness.get('status','pending')}**. Existing runner regression exit: **{regression.get('returncode','pending')}**. Zero-correction parity, sparse-row/frozen-tensor identity, value-gradient isolation, repeated disposable updates, old-only actor equivalence, export/reload/cache refresh and exact rollback/resume evidence: {link('parameter-verification.json')}, {link('recovery-verification.json')}. Saved probes are explicitly off-policy software inputs and never count as training experience or strength matches.

The selected-card source review found no material missing observable preventing this scoped experiment. Hide-n-Sneak discard identities/counts, board identities, HP, attached/provided energy, prizes, hand/deck counts, stadium, legal choices and reference pointers are already visible. The existing threat helper omits conditional damage bonuses; no specialized Hide-n-Sneak counter or generic ability-used flag is added. Existing identity/count tokens and legal availability must carry that information. This is a concrete representation limitation, not an assumption that text corrections will solve the mechanic. Details and source lines: {link('mechanic-readiness.json')}. Live integration belongs to the first planned training cohort; policy refresh and old-only equivalence are checked across the first two and subsequent cohorts. No extra native diagnostic matches are added.

## Training

Eager bucket64 / concurrent16, compilation disabled; logical minibatch128 including actual remainders; four PPO epochs; float32/math SDPA and existing precision settings. Fresh AdamW(.95,.999),eps1e-8,decay0,foreachFalse. Embedding LR decays cosine1e-4→1e-5, indexed by collected fresh learner decisions at cohort end and held constant throughout updates; detached value LR1e-5. Terminal per-game rewards, gamma1,GAE.95,clip.2,entropy.01 and pre/post KL.02/.03 are retained. Collection probabilities/targets are immutable across epochs. Initialization KL is diagnostic only.

Target **500,000 fresh conditional decisions**, at most{MAX_COHORTS} retry cohorts /{MAX_COHORTS*16:,} attempted training BO3; stop after three consecutive cohorts with zero accepted actor updates or at resource limits. Collection starts only with a conservative reserve for a complete cohort/update, using measured current-experiment times after they exist. No prior throughput is claimed for this configuration. Retry GPU session cap{SESSION_CAP:,} seconds ({SESSION_CAP/3600:.4f}h), including startup/export; two allocated T4s, one used. The prior failed allocation counts toward the aggregate original6h/12 allocated-device-hour cap; no budget reset or further automatic replacement.

Training state **{training.get('state','not started')}**, reason **{concise_reason(training.get('reason','pending'))}**. Fresh decisions **{training.get('fresh_decisions',0):,}**; accepted updates **{training.get('accepted_updates',0):,}**, accepted actor updates **{training.get('accepted_actor_updates',0):,}**; accepted exposures **{training.get('accepted_exposures',0):,}**; effective PPO passes **{training.get('effective_PPO_passes',0):.3f}**. Completed cohort boundaries **{training.get('cohorts',0)}**. Final committed tensor hash **{training.get('final_tensor_hash','pending')}**. A resource-limited incomplete training target is preserved explicitly, including any active cohort/cursor; it is not relabeled as500k decisions.

All KL stops/rollbacks, losses, actor/value gradient norms, meaningful-action entropy, immutable targets and exposure counts appear in {link('journal.jsonl')}. New-card exposure by deck/physical identity appears in compact native records; row/correction norms, frozen hashes and before/after retained-policy drift appear in {link('adaptation-cohorts.jsonl')} and {link('policy-diagnostics.jsonl')}. Detailed rejected tensor examples are bounded to the first10 pairs per signature and2GiB total; every rejection keeps compact evidence and dropped-detail counts. Private checkpoints/rollouts stay outside Git.

## Fixed CPU evaluation

Exactly1,024 allocated attempts: initialization512 before training and final512 afterward, unless no actor change makes final evaluation redundant. Two learner decks × four opponents ×64, balanced32/32 by initial chooser. Both sides use conditional argmax. Hydro/D02 were seen in training; D01 Ogerpon/Mega Kangaskhan and D03 Dragapult/Blaziken are held-out decklists for this run with the same frozen policy family; their absence from released-agent pretraining is not established. This is not unseen-policy-family transfer or a reproduction of leaderboard agents.

| Checkpoint | Learner | Opponent | Attempt/done/fail | BO3 W/D/L | BO3 WR [95%] | First-game W/D/L | First WR [95%] | First games completed |
|---|---|---|---:|---:|---|---:|---|---:|
{chr(10).join(cells)}

| Metric | Aggregate | Initial | Final | Final−initial |95% interval |
|---|---|---:|---:|---:|---|
{chr(10).join(contrasts) or '| Pending | | | | | |'}

The primary endpoint equally weights all eight learner/opponent cells and both chooser strata.20,000 independent whole-match bootstrap draws remain within checkpoint/deck/opponent/chooser strata; deals are not paired. First-game outcomes use the same whole-match draw and retain valid first games after later failures. Draws are nonwins; failures, actual starters, completed denominators and cell details are explicit in {link('evaluation-analysis.json')}. Secondary intervals are pointwise. No intermediate checkpoint strength evaluation, replacement matches or outcome-based extension is permitted.

## Costs and disposition

CPU evaluation attempts: **{init_count} initial + {final_count} final**; retry training attempts **{training_attempts} /{MAX_COHORTS*16:,}**. GPU quota delta **{account.get('quota_delta_seconds','pending')} seconds**, session hours **{account.get('session_hours','pending')}**, allocated-device hours **{account.get('allocated_device_hours','pending')}**, used-device hours **{account.get('used_device_hours','pending')}**. These quantities have different accounting scopes. Prior failed quota use is {prior:.3f}s; total observed quota across attempts is {prior+account.get("quota_delta_seconds",0):.3f}s (retry quota pending until retrieved). Baseline CPU runtime is retained in the prior run; it is not spent again.

Measured collection **{collection:.3f}s**, recorded cohort update/validation **{update:.3f}s**; startup/profile/export and component counters are separate in {link('training-completion.json')}, {link('GPU-accounting.json')}, remote receipts and {link('local-costs.jsonl')}. Evaluation costs include UTC, awake time and per-match CPU time. Laptop power settings are preserved; pauses do not extend the GPU budget. Some detailed timers overlap and are not summed as disjoint wall time.

{chr(10).join(failures) if failures else 'No execution failure has been recorded.'}

A statistically flat result does not prove that embeddings are irrelevant or that full-model tuning would fail. Changed rows and policy drift are distinct from improved play. No automatic further experiment, paid model API, public upload, deployment or competition submission follows.

Reproduction: {link('registration.json')}, {link('source-freeze.json')}, {link('input-manifest.json')}, {link('decks.json')}, {link('trainable-rows.json')}, {link('readiness.json')}, {link('training-completion.json')}, {link('evaluation-analysis.json')}, {link('completion.json')}. Full inventories remain machine-readable instead of filling this narrative.
''',encoding='utf-8')
 path=ROOT/'docs/STATUS.md';text=path.read_text();heading='## New-card embedding adaptation'
 if heading in text:
  start=text.index(heading);end=text.find('\n## ',start+len(heading));text=text[:start]+(text[end+1:] if end>=0 else '')
 path.write_text(f'{heading}\n\n{iso()}: **{state}**. {reason} Fresh decisions {training.get("fresh_decisions",0):,}; CPU attempts {init_count}+{final_count}. [Report](../reports/astra-unown-new-card-embedding-adaptation.md). No competition submission or automatic follow-up.\n\n'+text)

if __name__=='__main__':report(*sys.argv[1:3])
