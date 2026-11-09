> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Pooled oracle-correction evaluation

State: **COMPLETE**. User-requested saved-data analysis, registered before recomputing pooled diagnostics on [removed]. This is explicitly post-hoc: both source results were already available.

Added all **196 completed BO3** from the stopped primary evaluation to the complete 1,440-match CPU replication: **1,636 completed / 1,639 attempted**, with three failed or interrupted attempts retained. No new matches, inference, training, GPU session, upload or submission occurred.

| Policy | Completed / attempted | Weighted BO3 win rate (95% interval) | Weighted first-game win rate (95% interval) |
|---|---:|---:|---:|
| Original Unown | 553/553 | 30.94% [26.19, 38.18] | 35.35% [29.64, 42.51] |
| Legal-belief correction | 543/545 | 33.26% [27.77, 40.84] | 37.50% [31.61, 45.03] |
| Private oracle correction | 540/541 | 30.20% [25.69, 37.41] | 39.43% [33.39, 46.41] |

| Contrast | BO3 difference (95% interval) | First-game difference (95% interval) |
|---|---:|---:|
| belief-original | +2.32 pp [-5.84, +10.48] | +2.15 pp [-6.54, +10.79] |
| oracle-original | -0.74 pp [-8.73, +7.08] | +4.08 pp [-4.66, +12.60] |
| oracle-belief | -3.06 pp [-11.36, +5.07] | +1.93 pp [-7.03, +10.63] |

All reported BO3 contrast intervals include zero. The legal-belief point gain over original is now +2.32 percentage points; the data do not establish a strength improvement. The oracle's first-game point advantage does not establish a BO3 advantage. Oracle remains a privileged offline diagnostic, ineligible for deployment.

## Records and weighting

| Policy | Primary completed / attempted | Replication completed / attempted | Pooled BO3 W/D/L |
|---|---:|---:|---:|
| Original Unown | 73/73 | 480/480 | 248/0/305 |
| Legal-belief correction | 63/65 | 480/480 | 223/0/320 |
| Private oracle correction | 60/61 | 480/480 | 218/0/322 |

The exact original, legal-belief and oracle model identities match across runs, as do the deck manifest, original Hydrapple deck, conditional argmax decoder and 30-opponent local-policy panel. Frozen model hashes are in the registration. Native deals are independent, not paired. The panel overlaps training and is not leaderboard validation.

Counts are pooled within policy/opponent/initial-chooser cells, then the original population n/82 opponent weights and equal chooser weights are applied. The incomplete primary contributed unequal counts across cells and policies, so raw total wins divided by total matches is not the reported population-weighted result. All 1,639 match IDs are unique, including across source runs; immutable input hashes were verified before and after analysis. Every completed source result is included exactly once.

Uncertainty uses the same 20,000 whole-match bootstrap draws within policy/opponent/chooser strata, independent resampling across policies and shared original-baseline draws across contrasts. Bootstrap seed: **904093**. Intervals retain the original percentile/Wilson-MOVER boundary envelope. Every opponent/chooser cell, actual first-game starter, all-game W/D/L and termination reasons is retained in analysis.json. Draws count as non-wins and are reported separately; there were no completed-match or first-game draws.

## Failures and limitations

The stopped primary contributed two legal-belief failures/interruption records and one oracle interruption. One legal-belief match hit `OperationLimit: Registered per-operation CPU guard`; two concurrent matches were interrupted without complete retained outcomes. None of these three records contains a usable completed first game, so first-game denominators also total 1,636. No result was imputed and no failed match was replaced. Full IDs and error strings are in count-audit.json.

This pooling was requested after results were viewed. The primary stopped on a runtime guard: its completed prefix and missing outcomes may be selected by runtime or game characteristics. The reported intervals condition on observed outcomes and do not correct informative stopping. This pooled analysis supplements the unchanged standalone 1,440-match result; it is not a new independent confirmation and does not erase the original execution failure.

## Reproduction and costs

Run `PYTHONPATH=scripts:src:. .venv/bin/python -m unown_oracle_pooled.run` from the canonical WSL checkout. Registered input hashes, disjoint IDs, exact record counts, valid outcomes and independent direct count-weighted point estimates all passed the saved-data checks. No policy behavior changed and no native integration matches were added.

Saved-data calculation elapsed: **2.093 seconds**; process CPU: **2.032 seconds**. These timings include input verification, resampling and evidence export, excluding interactive editing. GPU: **0**; fresh native BO3: **0**; paid model API requests: **0**. The original evaluations' costs remain in their source completion/accounting records and were not incurred again.

Registration (historical/private artifact omitted) · Source manifest (historical/private artifact omitted) · Count and failure audit (historical/private artifact omitted) · Full stratified analysis (historical/private artifact omitted) · Combined match evidence (historical/private artifact omitted) · Completion and costs (historical/private artifact omitted) · Standalone replication (historical/private artifact omitted) · Primary run (historical/private artifact omitted).
