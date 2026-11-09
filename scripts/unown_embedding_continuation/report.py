"""Readable continuation receipt and costs; prior strength evidence stays frozen."""
import sys
from .common import *

def report(state='PREPARING',reason='Preparing saved-state continuation'):
 def get(name):return read(OUT/name) if (OUT/name).exists() else {}
 lineage=get('lineage.json');training=get('training-completion.json');account=get('GPU-accounting.json')
 events=records(OUT/'journal.jsonl');rejects=[e for e in events if e.get('kind')=='numerical-cohort-rejected']
 updates=[e for e in events if e.get('kind')=='update'];KL=[e for e in updates if not e.get('accepted') and e.get('reason') in ('phase_preupdate_drift_stop','phase_postupdate_drift_rollback_stop')]
 prior=lineage.get('prior_session_seconds',0);used=account.get('quota_delta_seconds')
 from .benchmark_summary import render_benchmark
 benchmark_section=render_benchmark()
 benchmark_allocated=(OUT/'checkpoint-benchmark-registration.json').exists()
 evaluation_note='User-authorized512 final CPU attempts; no new GPU or training' if benchmark_allocated else 'No additional strength evaluation'
 disposition=reason if len(reason)<500 else reason[:500]+' (full error retained in structured failure records)'
 REPORT.parent.mkdir(exist_ok=True)
 REPORT.write_text(f'''# Unown new-card embedding continuation

State: **{state}**. {disposition}

This is an explicitly requested continuation after the stopped checkpoint was benchmarked. The earlier training failure and benchmark results remain in [the adaptation report](astra-unown-new-card-embedding-adaptation.md). No earlier evidence is relabeled or overwritten.

The saved checkpoint resumes at update **14,132**, with the exact10 AdamW parameter states, momentum, optimizer learning rates, torch/numpy/python/CUDA RNG, immutable pending cohort174 and original schedule cursor. The global fresh-decision count starts at **444,018**; the target remains **500,000**. Original initialization, fixed text tables, eligible embedding rows, frozen actor, detached value head, decks, four PPO epochs, sampler, rewards and cosine schedule are unchanged.

## Registered handling of rejected updates

Pre-update KL0.02 ends the current cohort's updates. Post-update KL0.03 rolls back the offending update including optimizer/RNG state and ends that cohort's updates. Fresh balanced collection then continues. The global three-zero-actor-cohort stop is removed for this continuation. There is no initialization-KL stop.

Probability-consistency tolerances remain unchanged. A cohort that fails only the probability tolerance, while identity, finiteness, masks, value, KL and action checks pass and the double-precision paths agree within1e-10, is rejected **without training on it**. Its native matches and fresh decisions remain counted; probabilities are never rebased. Three consecutive numerical cohort rejections stop as an unresolved numerical constraint. Structural errors, nonfinite values, illegal actions, corrupt recovery state and hard resource limits remain fatal.

## Progress and resource accounting

Training state: **{training.get('state','not started')}**. Collected decisions: **{training.get('fresh_decisions',444018):,}**. Cumulative accepted updates: **{training.get('accepted_updates',14132):,}**. New accepted updates: **{training.get('accepted_updates',14132)-14132:,}**. Numerical cohorts rejected: **{len(rejects)}**. Cohort-local KL rejections recorded: **{len(KL)}**. New native attempts: **{training.get('training_attempts',0)} /1,280**, in addition to2,816 prior attempts.

One GPU job is capped at **5,063 seconds (1h24m23s)** including startup and export, using the remaining original six-hour allowance. Two allocated T4s are counted, with one used. Prior quota use: **{prior:.3f}s**. Continuation quota use: **{'pending' if used is None else f'{used:.3f}s'}**. Aggregate quota use: **{'pending' if used is None else f'{prior+used:.3f}s /21,600s'}**. Continuation allocated-device hours: **{account.get('allocated_device_hours','pending')}**. No automatic replacement job or budget extension.

{benchmark_section}

No paid model APIs, public upload, competition submission or deployment.

Registration and evidence: [registration.json](../experiments/{RUN}/registration.json), [lineage.json](../experiments/{RUN}/lineage.json), [continuation-verification.json](../experiments/{RUN}/continuation-verification.json), [bundle.json](../experiments/{RUN}/bundle.json), [recovery-adoption.json](../experiments/{RUN}/recovery-adoption.json), [training-completion.json](../experiments/{RUN}/training-completion.json), [GPU-accounting.json](../experiments/{RUN}/GPU-accounting.json), [local-costs.jsonl](../experiments/{RUN}/local-costs.jsonl). Private recovery artifacts are under `data/private/{RUN}/` and stay outside Git.
''',encoding='utf-8')
 path=ROOT/'docs/STATUS.md';text=path.read_text();heading='## New-card embedding continuation'
 if heading in text:
  start=text.index(heading);end=text.find('\n## ',start+len(heading));text=text[:start]+(text[end+1:] if end>=0 else '')
 path.write_text(f'{heading}\n\n{iso()}: **{state}**. {disposition} Saved optimizer retained; GPU allowance5,063s; {evaluation_note}. [Report](../reports/astra-unown-new-card-embedding-continuation.md).\n\n'+text)

if __name__=='__main__':report(*sys.argv[1:3])
