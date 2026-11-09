> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Unown new-card embedding continuation

State: **BENCHMARK_COMPLETE**. All512 final CPU BO3 completed and analyzed against the original512 baseline. Completed500k-target training and earlier benchmark evidence remain preserved.

This is an explicitly requested continuation after the stopped checkpoint was benchmarked. The earlier training failure and benchmark results remain in [the adaptation report](astra-unown-new-card-embedding-adaptation.md). No earlier evidence is relabeled or overwritten.

The saved checkpoint resumes at update **14,132**, with the exact10 AdamW parameter states, momentum, optimizer learning rates, torch/numpy/python/CUDA RNG, immutable pending cohort174 and original schedule cursor. The global fresh-decision count starts at **444,018**; the target remains **500,000**. Original initialization, fixed text tables, eligible embedding rows, frozen actor, detached value head, decks, four PPO epochs, sampler, rewards and cosine schedule are unchanged.

## Registered handling of rejected updates

Pre-update KL0.02 ends the current cohort's updates. Post-update KL0.03 rolls back the offending update including optimizer/RNG state and ends that cohort's updates. Fresh balanced collection then continues. The global three-zero-actor-cohort stop is removed for this continuation. There is no initialization-KL stop.

Probability-consistency tolerances remain unchanged. A cohort that fails only the probability tolerance, while identity, finiteness, masks, value, KL and action checks pass and the double-precision paths agree within1e-10, is rejected **without training on it**. Its native matches and fresh decisions remain counted; probabilities are never rebased. Three consecutive numerical cohort rejections stop as an unresolved numerical constraint. Structural errors, nonfinite values, illegal actions, corrupt recovery state and hard resource limits remain fatal.

## Progress and resource accounting

Training state: **COMPLETE**. Collected decisions: **502,384**. Cumulative accepted updates: **15,996**. New accepted updates: **1,864**. Numerical cohorts rejected: **1**. Cohort-local KL rejections recorded: **0**. New native attempts: **368 /1,280**, in addition to2,816 prior attempts.

One GPU job is capped at **5,063 seconds (1h24m23s)** including startup and export, using the remaining original six-hour allowance. Two allocated T4s are counted, with one used. Prior quota use: **16536.680s**. Continuation quota use: **2268.887s**. Aggregate quota use: **18805.567s /21,600s**. Continuation allocated-device hours: **1.260492777777778**. No automatic replacement job or budget extension.

## Completed-checkpoint CPU benchmark

The primary95% interval overlaps zero; this benchmark does not establish an overall BO3 improvement.

The user separately authorized **512 fresh CPU BO3** after training reached502,384 collected decisions and15,996 accepted updates. The exact final checkpoint is frozen before the first match. The original512 initialization attempts are reused unchanged. This adds512 native attempts to the earlier1,024, for1,536 total; no new baseline matches are run. D10/D04 × Hydro/D02/D01/D03 ×64, balanced32/32 by initial chooser; both sides use conditional argmax. Forced-observation memory, game resets and legal cross-game knowledge use the existing adapter. No extra native diagnostics, replacement matches, outcome-based extension, checkpoint selection or new training.

Hydro/D02 are seen decks; D01/D03 are held-out training decklists with the same frozen policy family. These are local Unown policies, not leaderboard agents. Native deals are independent and unpaired. Reused schedule/inference seeds do not control native shuffles. The earlier stopped-checkpoint benchmark remains separate and is not pooled into this comparison.

This additional evaluation follows an earlier checkpoint evaluation. Intervals are pointwise; no family-wise or sequential error-control claim is made across the project's repeated looks. An empirical all-loss bootstrap interval is not proof of a zero true win probability.

Final attempts/completed/failures: **512/512/0**. Recorded evaluation elapsed seconds: **1683.205058066**. Additional GPU use and training updates: **zero**.

| Metric | Aggregate | Initialization | Completed checkpoint | Gain |95% interval |
|---|---|---:|---:|---:|---|
| BO3 | primary | 0.98% | 0.78% | -0.20 pp | [-1.37 pp, +0.98 pp] |
| BO3 | D10 | 1.56% | 1.56% | +0.00 pp | [-1.95 pp, +1.95 pp] |
| BO3 | D04 | 0.39% | 0.00% | -0.39 pp | [-1.17 pp, +0.00 pp] |
| BO3 | seen | 0.78% | 0.39% | -0.39 pp | [-1.56 pp, +0.78 pp] |
| BO3 | held_out | 1.17% | 1.17% | +0.00 pp | [-1.95 pp, +1.95 pp] |
| first_game | primary | 4.69% | 7.62% | +2.93 pp | [+0.00 pp, +5.86 pp] |
| first_game | D10 | 6.25% | 12.11% | +5.86 pp | [+1.17 pp, +10.94 pp] |
| first_game | D04 | 3.12% | 3.12% | +0.00 pp | [-3.12 pp, +3.12 pp] |
| first_game | seen | 3.91% | 6.64% | +2.73 pp | [-1.17 pp, +6.64 pp] |
| first_game | held_out | 5.47% | 8.59% | +3.12 pp | [-1.17 pp, +7.42 pp] |

| Checkpoint | Learner | Opponent | Attempts/done/fail | BO3 W/D/L | BO3 WR [95%] | First-game W/D/L | First WR [95%] |
|---|---|---|---:|---:|---|---:|---|
| initial | D10 | Hydro | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 2/0/62 | 3.12% [0.00%, 7.81%] |
| initial | D10 | D02 | 64/64/0 | 1/0/63 | 1.56% [0.00%, 4.69%] | 4/0/60 | 6.25% [1.56%, 12.50%] |
| initial | D10 | D01 | 64/64/0 | 2/0/62 | 3.12% [0.00%, 7.81%] | 5/0/59 | 7.81% [1.56%, 14.06%] |
| initial | D10 | D03 | 64/64/0 | 1/0/63 | 1.56% [0.00%, 4.69%] | 5/0/59 | 7.81% [1.56%, 14.06%] |
| initial | D04 | Hydro | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 0/0/64 | 0.00% [0.00%, 0.00%] |
| initial | D04 | D02 | 64/64/0 | 1/0/63 | 1.56% [0.00%, 4.69%] | 4/0/60 | 6.25% [1.56%, 12.50%] |
| initial | D04 | D01 | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 4/0/60 | 6.25% [1.56%, 12.50%] |
| initial | D04 | D03 | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 0/0/64 | 0.00% [0.00%, 0.00%] |
| final | D10 | Hydro | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 6/0/58 | 9.38% [3.12%, 17.19%] |
| final | D10 | D02 | 64/64/0 | 1/0/63 | 1.56% [0.00%, 4.69%] | 6/0/58 | 9.38% [3.12%, 17.19%] |
| final | D10 | D01 | 64/64/0 | 2/0/62 | 3.12% [0.00%, 7.81%] | 10/0/54 | 15.62% [7.81%, 25.00%] |
| final | D10 | D03 | 64/64/0 | 1/0/63 | 1.56% [0.00%, 4.69%] | 9/0/55 | 14.06% [6.25%, 23.44%] |
| final | D04 | Hydro | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 2/0/62 | 3.12% [0.00%, 7.81%] |
| final | D04 | D02 | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 3/0/61 | 4.69% [0.00%, 10.94%] |
| final | D04 | D01 | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 3/0/61 | 4.69% [0.00%, 10.94%] |
| final | D04 | D03 | 64/64/0 | 0/0/64 | 0.00% [0.00%, 0.00%] | 0/0/64 | 0.00% [0.00%, 0.00%] |

The primary contrast equally weights all8 cells and both chooser assignments.20,000 independent whole-match bootstrap draws stay within checkpoint/deck/opponent/chooser strata. First-game analysis keeps valid first games after later failures. Per-chooser W/D/L, actual starters, draws, failures and first-game denominators are in evaluation-analysis.json (historical/private artifact omitted). Registration (historical/private artifact omitted), compact final match records (historical/private artifact omitted), completion receipt (historical/private artifact omitted).


No paid model APIs, public upload, competition submission or deployment.

Registration and evidence: registration.json (historical/private artifact omitted), lineage.json (historical/private artifact omitted), continuation-verification.json (historical/private artifact omitted), bundle.json (historical/private artifact omitted), recovery-adoption.json (historical/private artifact omitted), training-completion.json (historical/private artifact omitted), GPU-accounting.json (historical/private artifact omitted), local-costs.jsonl (historical/private artifact omitted). Private recovery artifacts are under `data/private/astra-unown-new-card-embedding-continuation-001/` and stay outside Git.
