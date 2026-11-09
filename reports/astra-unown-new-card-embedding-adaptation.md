> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Unown new-card embedding adaptation

State: **BENCHMARK_COMPLETE**. All512 final CPU BO3 completed and analyzed against the original512 baseline. Training remains stopped at its recorded numerical-check failure.

The first attempt failed before learning: counterfactual retained-history logging made remote inference requests without corresponding action rows. The repaired path disables those diagnostic-only queries and checks request/rollout counts. The original failed report and all evidence are preserved in `experiments/astra-unown-new-card-embedding-adaptation-001/before-explicit-retry/reports/astra-unown-new-card-embedding-adaptation.md`; see retry-provenance.json (historical/private artifact omitted) and collector-repair-verification.json (historical/private artifact omitted). The exact512 completed initial matches are reused without new baseline matches or outcome-based selection. The failed attempt consumed16 training BO3 and 118.914 GPU-quota seconds. This explicit retry allows at most 4,080 more training attempts and 21,481 session seconds, keeping both attempts inside the original4,096-match/six-hour allocation. No further automatic retry is authorized.

CPU benchmark separately authorized onremoved-10-30: evaluate the exact retained checkpoint after174 completed cohorts using the original512 final BO3 schedule and original512 baseline. Both sides use conditional argmax. No weights, decks, decoder, training, GPU usage, sample size or statistical rules change. The training run remains FAILED at its pre-update numerical consistency check; evaluating this fixed partial-training checkpoint does not erase that failure or claim the500,000-decision target was reached. See checkpoint-benchmark-registration.json (historical/private artifact omitted) and checkpoint-benchmark-completion.json (historical/private artifact omitted).

**The primary interval overlaps zero; this pilot does not establish a BO3-strength improvement.** Original deployed policies remain unchanged. This experiment does not authorize a competition submission or another training run.

## Frozen comparison

One single-arm experiment starts from the experimental current-pool initialization, tensor SHA256 `[artifact fingerprint omitted]`, with the validated actual-starting-order adapter on both roles. File/text hashes are in initialization-provenance.json (historical/private artifact omitted). This initialization retains approximate local text mappings whose earlier vector-fidelity gate failed; it is not a later Hydrapple PPO checkpoint and its text tables are reused unchanged.

Learners: **D10** (Banette / Lillie’s Clefairy ex / Dudunsparce), **D04** (Metagross / Mega Excadrill ex / Genesect ex), equally weighted. Training opponents: original registered **Hydro**, **D02**, **D19**, **D21**, equally weighted. Exact60-card lists and hashes: decks.json (historical/private artifact omitted). These are local Unown policies, not their leaderboard source agents. Every16-BO3 cohort has all eight deck pairings twice, one match per initial chooser, with a frozen shuffled schedule. Learner sampling is conditional T1; the frozen opponent uses conditional argmax. Policies/decks/decoders are fixed within BO3; forced observations, legal knowledge and game/match resets use the existing corrected adapter. Native deals are independent and unpaired.

## What can learn

| Parameter | Shape | Elements | Role |
|---|---:|---:|---|
| `card_id_emb.parametrizations.weight.0.delta` | 12 × 64 | 768 | embedding/text |
| `attack_id_emb.parametrizations.weight.0.delta` | 12 × 48 | 576 | embedding/text |
| `skill_id_emb.parametrizations.weight.0.delta` | 5 × 48 | 240 | embedding/text |
| `value_mlp.0.weight` | 256 × 384 | 98,304 | detached value |
| `value_mlp.0.bias` | 256 | 256 | detached value |
| `value_mlp.2.weight` | 1 × 256 | 256 | detached value |
| `value_mlp.2.bias` | 1 | 1 | detached value |
| `parametrizations.card_text.0.delta` | 12 × 128 | 1,536 | embedding/text |
| `parametrizations.att_text.0.delta` | 12 × 64 | 768 | embedding/text |
| `parametrizations.skill_text.0.delta` | 5 × 64 | 320 | embedding/text |

Eligible semantic rows: cards **12**, attacks **12**, skills **5**. Exact row/card/link inventory: trainable-rows.json (historical/private artifact omitted). Only appended identities actually referenced by the two learner decks are selected; reused old attack/skill rows stay frozen. Sparse additive parameters start at zero, so the optimizer never owns old, padding, unknown or unused rows. All original identity/text buffers, numerical tables, shared projections, Transformer and action-scoring parameters are frozen. Text corrections change effective selected vectors without regenerating the base text file.

The blind value head has its own1e-5 learning rate and detached input features. Value-loss gradients cannot reach embeddings or the actor; oracle/auxiliary objectives are unused. The existing global1.5 norm clip remains unchanged across trainable parameters. Effective canonical export folds only eligible corrections into the corresponding rows; ordinary inference reloads the same effective vectors. Recovery retains sparse parameters, optimizer/RNG/cursor and row provenance.

## Readiness and scope limits

Focused local readiness: **PASS**. Existing runner regression exit: **0**. Zero-correction parity, sparse-row/frozen-tensor identity, value-gradient isolation, repeated disposable updates, old-only actor equivalence, export/reload/cache refresh and exact rollback/resume evidence: parameter-verification.json (historical/private artifact omitted), recovery-verification.json (historical/private artifact omitted). Saved probes are explicitly off-policy software inputs and never count as training experience or strength matches.

The selected-card source review found no material missing observable preventing this scoped experiment. Hide-n-Sneak discard identities/counts, board identities, HP, attached/provided energy, prizes, hand/deck counts, stadium, legal choices and reference pointers are already visible. The existing threat helper omits conditional damage bonuses; no specialized Hide-n-Sneak counter or generic ability-used flag is added. Existing identity/count tokens and legal availability must carry that information. This is a concrete representation limitation, not an assumption that text corrections will solve the mechanic. Details and source lines: mechanic-readiness.json (historical/private artifact omitted). Live integration belongs to the first planned training cohort; policy refresh and old-only equivalence are checked across the first two and subsequent cohorts. No extra native diagnostic matches are added.

## Training

Eager bucket64 / concurrent16, compilation disabled; logical minibatch128 including actual remainders; four PPO epochs; float32/math SDPA and existing precision settings. Fresh AdamW(.95,.999),eps1e-8,decay0,foreachFalse. Embedding LR decays cosine1e-4→1e-5, indexed by collected fresh learner decisions at cohort end and held constant throughout updates; detached value LR1e-5. Terminal per-game rewards, gamma1,GAE.95,clip.2,entropy.01 and pre/post KL.02/.03 are retained. Collection probabilities/targets are immutable across epochs. Initialization KL is diagnostic only.

Target **500,000 fresh conditional decisions**, at most255 retry cohorts /4,080 attempted training BO3; stop after three consecutive cohorts with zero accepted actor updates or at resource limits. Collection starts only with a conservative reserve for a complete cohort/update, using measured current-experiment times after they exist. No prior throughput is claimed for this configuration. Retry GPU session cap21,481 seconds (5.9669h), including startup/export; two allocated T4s, one used. The prior failed allocation counts toward the aggregate original6h/12 allocated-device-hour cap; no budget reset or further automatic replacement.

Training state **FAILED**, reason **Stopped before cohort175 updates: collection/replay probability difference0.00005746 exceeded0.00005; the higher-precision diagnostic also exceeded its float32 tolerance. Full details remain in training-failure.json.**. Fresh decisions **444,018**; accepted updates **14,132**, accepted actor updates **14,132**; accepted exposures **1,765,236**; effective PPO passes **3.976**. Completed cohort boundaries **174**. Final committed tensor hash **[artifact fingerprint omitted]**. A resource-limited incomplete training target is preserved explicitly, including any active cohort/cursor; it is not relabeled as500k decisions.

All KL stops/rollbacks, losses, actor/value gradient norms, meaningful-action entropy, immutable targets and exposure counts appear in journal.jsonl (historical/private artifact omitted). New-card exposure by deck/physical identity appears in compact native records; row/correction norms, frozen hashes and before/after retained-policy drift appear in adaptation-cohorts.jsonl (historical/private artifact omitted) and policy-diagnostics.jsonl (historical/private artifact omitted). Detailed rejected tensor examples are bounded to the first10 pairs per signature and2GiB total; every rejection keeps compact evidence and dropped-detail counts. Private checkpoints/rollouts stay outside Git.

## Fixed CPU evaluation

Exactly1,024 allocated attempts: initialization512 before training and final512 afterward, unless no actor change makes final evaluation redundant. Two learner decks × four opponents ×64, balanced32/32 by initial chooser. Both sides use conditional argmax. Hydro/D02 were seen in training; D01 Ogerpon/Mega Kangaskhan and D03 Dragapult/Blaziken are held-out decklists for this run with the same frozen policy family; their absence from released-agent pretraining is not established. This is not unseen-policy-family transfer or a reproduction of leaderboard agents.

| Checkpoint | Learner | Opponent | Attempt/done/fail | BO3 W/D/L | BO3 WR [95%] | First-game W/D/L | First WR [95%] | First games completed |
|---|---|---|---:|---:|---|---:|---|---:|
| initial | D10 | Hydro | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 2/0/62 | 3.1% [0.0%, 7.8%] | 64 |
| initial | D10 | D02 | 64/64/0 | 1/0/63 | 1.6% [0.0%, 4.7%] | 4/0/60 | 6.2% [1.6%, 12.5%] | 64 |
| initial | D10 | D01 | 64/64/0 | 2/0/62 | 3.1% [0.0%, 7.8%] | 5/0/59 | 7.8% [1.6%, 14.1%] | 64 |
| initial | D10 | D03 | 64/64/0 | 1/0/63 | 1.6% [0.0%, 4.7%] | 5/0/59 | 7.8% [1.6%, 14.1%] | 64 |
| initial | D04 | Hydro | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 0/0/64 | 0.0% [0.0%, 0.0%] | 64 |
| initial | D04 | D02 | 64/64/0 | 1/0/63 | 1.6% [0.0%, 4.7%] | 4/0/60 | 6.2% [1.6%, 12.5%] | 64 |
| initial | D04 | D01 | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 4/0/60 | 6.2% [1.6%, 12.5%] | 64 |
| initial | D04 | D03 | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 0/0/64 | 0.0% [0.0%, 0.0%] | 64 |
| final | D10 | Hydro | 64/64/0 | 1/0/63 | 1.6% [0.0%, 4.7%] | 1/0/63 | 1.6% [0.0%, 4.7%] | 64 |
| final | D10 | D02 | 64/64/0 | 4/0/60 | 6.2% [1.6%, 12.5%] | 7/0/57 | 10.9% [4.7%, 18.8%] | 64 |
| final | D10 | D01 | 64/64/0 | 4/0/60 | 6.2% [1.6%, 12.5%] | 7/0/57 | 10.9% [3.1%, 18.8%] | 64 |
| final | D10 | D03 | 64/64/0 | 3/0/61 | 4.7% [0.0%, 10.9%] | 4/0/60 | 6.2% [1.6%, 12.5%] | 64 |
| final | D04 | Hydro | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 1/0/63 | 1.6% [0.0%, 4.7%] | 64 |
| final | D04 | D02 | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 2/0/62 | 3.1% [0.0%, 7.8%] | 64 |
| final | D04 | D01 | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 2/0/62 | 3.1% [0.0%, 7.8%] | 64 |
| final | D04 | D03 | 64/64/0 | 0/0/64 | 0.0% [0.0%, 0.0%] | 1/0/63 | 1.6% [0.0%, 4.7%] | 64 |

| Metric | Aggregate | Initial | Final | Final−initial |95% interval |
|---|---|---:|---:|---:|---|
| BO3 | primary | 1.0% | 2.3% | +1.4 pp | [-0.2 pp, +2.9 pp] |
| BO3 | D10 | 1.6% | 4.7% | +3.1 pp | [+0.4 pp, +6.2 pp] |
| BO3 | D04 | 0.4% | 0.0% | -0.4 pp | [-1.2 pp, +0.0 pp] |
| BO3 | seen | 0.8% | 2.0% | +1.2 pp | [-0.8 pp, +3.1 pp] |
| BO3 | held_out | 1.2% | 2.7% | +1.6 pp | [-0.8 pp, +3.9 pp] |
| first_game | primary | 4.7% | 4.9% | +0.2 pp | [-2.3 pp, +2.7 pp] |
| first_game | D10 | 6.2% | 7.4% | +1.2 pp | [-3.1 pp, +5.5 pp] |
| first_game | D04 | 3.1% | 2.3% | -0.8 pp | [-3.5 pp, +2.0 pp] |
| first_game | seen | 3.9% | 4.3% | +0.4 pp | [-3.1 pp, +3.9 pp] |
| first_game | held_out | 5.5% | 5.5% | +0.0 pp | [-3.9 pp, +3.9 pp] |

The primary endpoint equally weights all eight learner/opponent cells and both chooser strata.20,000 independent whole-match bootstrap draws remain within checkpoint/deck/opponent/chooser strata; deals are not paired. First-game outcomes use the same whole-match draw and retain valid first games after later failures. Draws are nonwins; failures, actual starters, completed denominators and cell details are explicit in evaluation-analysis.json (historical/private artifact omitted). Secondary intervals are pointwise. No intermediate checkpoint strength evaluation, replacement matches or outcome-based extension is permitted.

## Costs and disposition

CPU evaluation attempts: **512 initial + 512 final**; retry training attempts **2800 /4,080**. GPU quota delta **16417.766000000003 seconds**, session hours **4.560490555555557**, allocated-device hours **9.120981111111114**, used-device hours **4.560490555555557**. These quantities have different accounting scopes. Prior failed quota use is 118.914s; total observed quota across attempts is 16536.680s (retry quota pending until retrieved). Baseline CPU runtime is retained in the prior run; it is not spent again.

Measured collection **6241.082s**, recorded cohort update/validation **9287.732s**; startup/profile/export and component counters are separate in training-completion.json (historical/private artifact omitted), GPU-accounting.json (historical/private artifact omitted), remote receipts and local-costs.jsonl (historical/private artifact omitted). Evaluation costs include UTC, awake time and per-match CPU time. Laptop power settings are preserved; pauses do not extend the GPU budget. Some detailed timers overlap and are not summed as disjoint wall time.

training-failure.json: Stopped before cohort175 updates: collection/replay probability difference0.00005746 exceeded0.00005; the higher-precision diagnostic also exceeded its float32 tolerance. Full details remain in training-failure.json.

A statistically flat result does not prove that embeddings are irrelevant or that full-model tuning would fail. Changed rows and policy drift are distinct from improved play. No automatic further experiment, paid model API, public upload, deployment or competition submission follows.

Reproduction: registration.json (historical/private artifact omitted), source-freeze.json (historical/private artifact omitted), input-manifest.json (historical/private artifact omitted), decks.json (historical/private artifact omitted), trainable-rows.json (historical/private artifact omitted), readiness.json (historical/private artifact omitted), training-completion.json (historical/private artifact omitted), evaluation-analysis.json (historical/private artifact omitted), completion.json (historical/private artifact omitted). Full inventories remain machine-readable instead of filling this narrative.
