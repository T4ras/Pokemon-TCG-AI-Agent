> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Learned Bayesian deck inference with baseline A allocation

State: **COMPLETE**. Updated [removed].

All296 out-of-fold BO3 predictions scored; no follow-up or held-out access.

This is an exploratory development benchmark on296 existing BO3 histories:37 exact lists × eight histories, balanced by initial chooser. The intended pool is30 passing lists representing82/100 snapshot entries; full-field reporting retains all37 lists and n/100 weights. Every list has60 cards. No new matches, held-out access, neural training, GPU, PPO, policy changes, uploads, submissions or paid model APIs are authorized. Historical evidence remains unchanged.

## Runtime repair and preserved failure

The original run stopped after 220 of 296 common-history replays completed. One allocation readout exceeded its 10 CPU-second guard; the other three active workers were cancelled. The historical `FAILED_SHARED` label grouped this runtime failure with correctness failures. It was not a model-quality result and did not exhaust the overall compute budget. The failed source, records and report are preserved; no comparative scores were produced.

The difficult observation contains 2,155 legal allocation layouts, 143 card queries and 4 compatible lists. Reusing component marginals and evaluating the query vector together reduced that readout from 34.882 to 0.356 CPU seconds (97.9×). Maximum feature difference was 0, checked against absolute and relative tolerances of 1e-10. The 10-second guard, constraints and probabilities are unchanged. Runtime failures now have their own status, and cancelled work is distinguished from budget exhaustion.

Before any fitting, the resumed runner checks all 220 completed historical caches against the repaired output, checks all 296 histories for true-list support in a separate diagnostic gate, and stops if either gate fails. The folds, queries, parameter grid and statistical rules are unchanged.

Historical-cache equivalence: 220/220 histories, 12,486 decisions; maximum absolute difference 0.

Budget is cumulative: 3017.41 CPU seconds are charged before this launch, leaving 25782.59 of the original 28,800. The original UTC deadline remains **[removed]**. Repair tests and profiling are accounted separately and conservatively deducted; no extra eight-hour allocation was created.

## Literature adaptation and model

Eger and Sauma Chacón’s Deck Archetype Prediction in Hearthstone (reference omitted), Section5.2, combines per-card observation probabilities with class prevalence under a conditional-independence approximation. This experiment adapts that idea to exact deck lists and accumulated legal revelations across BO3; it does not reproduce their first-turn Hearthstone classification experiment.

For each training BO3, the observation ledger counts legally established physical copies once per game and retains the maximum justified count across games. It does not sum overlapping prefixes or count the same recovered copy twice. The statistical fit sums one final ledger per training BO3. For deckd/cardc, theta=(training reveal count+alpha×deck copies/60)/(total training reveals+alpha). Candidate log weights are log(prevalence)+beta×sum(revealed lower bound×log(theta)); hard compatibility is identical to A and normalization uses stable log arithmetic. This is an approximate reveal-frequency classifier, not an exact history posterior or a policy-action likelihood model. Only legal card-copy lower bounds enter classifier inference.

The four outer folds each contain one chooser and one non-chooser BO3 per list. Each outer model fits six histories/list; three inner folds each validate two/list and train four/list. Alpha∈{30,120}, beta∈{.25,.5,1}. Selection minimizes BO3-averaged inner deck log loss under n/100 weighting. Ties within1e-12 prefer lower beta, then higher alpha. Outer labels/scores do not select parameters. Every evaluation prediction is out-of-fold. Fixed library/prevalence are shared; beta0 is baselineA.

## Shared observation and allocation correctness

The initial all-history gate failed25 histories. The isolated shared source was repaired using native SDK semantics: Metang86 attaches Metal Energy from Looking during its verified LookDeck4 segment; Dwebble344’s Ascension attack478 evolves from Deck. Both source fixes apply identically to A, B and the diagnostic. Failed gate records are retained. The repaired fact layer processes all296 histories, checks certified facts against separate labels, and passes deduplication/copy/reset checks. All17,029 registered selected points pass A’s ordering-support flag. Eleven histories retain PF-specific typed-compiler unsupported selection profiles; the Bayesian/A method does not use that particle/action-history compiler. These remain disclosed rather than being described as universal SDK support.

Focused prelaunch tests: **181 passed**. Exact physical-permutation references check allocation means and variances, Basic opening predicates, ambiguous known copies and hidden Basic slots; tests also cover the Bayesian formula, stable zeros, total variance, grouped balanced folds, deduplication and leakage boundaries.

At every selected point, common legal processing computes A’s conditional presence/count estimates once per supported deck. A mixes them using compatibility-filtered prevalence; B changes only those weights. Conditional allocation means preserve A’s existing assumptions. The feature output includes additional-unknown-copy hand presence, expected unknown hand/draw/prize counts and all three count variances. Total expected hand count is unknown mean plus the retained certified known-hand count. Mixtures use E[conditional variance]+variance of conditional means. No neural location model, particles or historical trajectory reconstruction are used.

A separately labelled known-deck diagnostic selects the true list’s already-computed conditional output in the scorer, after legal A/B predictions have been produced. Neither candidate weights nor allocation inference receives the true index or hidden labels. The diagnostic is not deployable and is not guaranteed to be an accuracy upper bound.

## Evaluation, uncertainty and limits

The existing queries and exact selected indices are preserved. Uncertain queries are fixed by A∈[.10,.90]. Aggregation is queries→decisions→games→BO3→equal chooser→n/82 or n/100. Paired20,000 whole-BO3 bootstrap draws use the same sampled histories across methods within list/chooser strata. Candidate-minus-A is reported directly: negative Brier/log-loss/count-error differences are better, positive accuracy/true-probability differences are better. Intervals condition on fitted CV models and do not capture full retraining uncertainty. The corpus was already used for development, so these are not untouched-test claims.

Remaining-composition error compares the inferred original-deck copies still unaccounted for by the certified reveal lower bound against the true list’s remaining copies, averaged across the frozen library vocabulary. It isolates deck composition under the same evidence. Game/stage summaries are descriptive; game-three participation depends on previous outcomes. Missing scores keep their expected denominators and population coverage. A shared allocation/correctness failure stops comparative scoring; no replacements are issued.

Execution caps: eight CPU-core hours cumulatively, including the failed run and charged repair work; the original eight-hour UTC deadline is preserved; four workers, one compute thread each. Workers have1GiB address-space limits,30s update and10s allocation/readout CPU guards; exact allocation enumeration retains4096-node bounds. Output is capped at2GiB and180s is reserved within both runtime caps for finalization/reporting. Budget exhaustion is explicit, with no extension or automatic follow-up. No power-plan change or sleep inhibition is required.

## Inner-selected parameters

| Outer fold | Alpha | Beta | Mean inner deck log loss | Training/excluded BO3 |
|---|---:|---:|---:|---:|
| 0 | 120.0 | 0.5 | 0.44679 | 222/74 |
| 1 | 120.0 | 0.5 | 0.46489 | 222/74 |
| 2 | 120.0 | 0.5 | 0.46591 | 222/74 |
| 3 | 120.0 | 0.5 | 0.46919 | 222/74 |

The full grid, training IDs, excluded IDs and private model hashes are retained in model-selection/.

## Frozen results

Expected/attempted/completed/scored BO3:296/296/296/296. Selected decisions scored: 17029/17029. Complete population strata: True.

### eligible: n/82

| Method | Uncertain Brier | All Brier | ECE | Hand/draw/prize MAE | Primary population coverage |
|---|---:|---:|---:|---|---:|
| A | 0.18374 | 0.02410 | 0.00161 | 0.05622/0.08096/0.04555 | 1.000 |
| B | 0.18384 | 0.02410 | 0.00162 | 0.05608/0.08003/0.04551 | 1.000 |
| known_deck_diagnostic | 0.18133 | 0.02369 | 0.00163 | 0.05516/0.07228/0.04523 | 1.000 |

| Legal method | Deck log loss | Multiclass Brier | True-deck probability | Top1 accuracy | Remaining-composition MAE |
|---|---:|---:|---:|---:|---:|
| A | 0.53460 | 0.26574 | 0.72897 | 0.81580 | 0.01809 |
| B | 0.51294 | 0.25731 | 0.73934 | 0.82968 | 0.01653 |

| Paired candidate-minus-A contrast | Point difference |95% interval |
|---|---:|---|
| B_minus_A/all_in/uncertain/prediction/brier | 0.00011 | [-0.00016457936272528116, 0.00044958613000394546] |
| B_minus_A/all_in/all/prediction/brier | 0.00001 | [-3.301764588989481e-05, 5.06941496570165e-05] |
| B_minus_A/deck/log_loss | -0.02167 | [-0.034501767180710054, -0.008733438297896676] |
| B_minus_A/deck/Brier | -0.00842 | [-0.017058986061633773, 0.0007059066822734529] |
| B_minus_A/deck/true_probability | 0.01037 | [0.0034197523805259756, 0.017203023776574705] |
| B_minus_A/deck/top1_accuracy | 0.01388 | [0.007505504742547403, 0.019742970867208887] |
| B_minus_A/deck/remaining_composition_MAE | -0.00156 | [-0.002230157008911788, -0.0009717998656068176] |
| known_deck_diagnostic_minus_A/all_in/uncertain/prediction/brier | -0.00240 | [-0.003043384136146714, -0.0018140859524663998] |

### full_field: n/100

| Method | Uncertain Brier | All Brier | ECE | Hand/draw/prize MAE | Primary population coverage |
|---|---:|---:|---:|---|---:|
| A | 0.18253 | 0.02350 | 0.00164 | 0.05434/0.08017/0.04594 | 1.000 |
| B | 0.18261 | 0.02350 | 0.00166 | 0.05423/0.07939/0.04591 | 1.000 |
| known_deck_diagnostic | 0.18049 | 0.02314 | 0.00166 | 0.05341/0.07126/0.04562 | 1.000 |

| Legal method | Deck log loss | Multiclass Brier | True-deck probability | Top1 accuracy | Remaining-composition MAE |
|---|---:|---:|---:|---:|---:|
| A | 0.47897 | 0.23800 | 0.75444 | 0.83708 | 0.01740 |
| B | 0.46077 | 0.23088 | 0.76369 | 0.84849 | 0.01608 |

| Paired candidate-minus-A contrast | Point difference |95% interval |
|---|---:|---|
| B_minus_A/all_in/uncertain/prediction/brier | 0.00008 | [-0.0001397884621976292, 0.0003640503145285348] |
| B_minus_A/all_in/all/prediction/brier | 0.00000 | [-2.804231877269409e-05, 4.0440664545315273e-05] |
| B_minus_A/deck/log_loss | -0.01819 | [-0.02885661631255091, -0.007411834815180449] |
| B_minus_A/deck/Brier | -0.00712 | [-0.014263323259903126, 0.00046433357624301407] |
| B_minus_A/deck/true_probability | 0.00925 | [0.003400952720915357, 0.014972600235387214] |
| B_minus_A/deck/top1_accuracy | 0.01141 | [0.006189236111110996, 0.016206814236111377] |
| B_minus_A/deck/remaining_composition_MAE | -0.00132 | [-0.0018789456575488128, -0.0008360612404269666] |
| known_deck_diagnostic_minus_A/all_in/uncertain/prediction/brier | -0.00204 | [-0.002569110686718353, -0.001549269100433581] |

### Game/stage breakdowns

| Group / population | A uncertain Brier | B uncertain Brier | Known-deck diagnostic | Scored histories | Primary population coverage |
|---|---:|---:|---:|---:|---:|
| by_game:1/full_field | 0.18365 | 0.18368 | 0.18042 | 296 | 1.000 |
| by_game:1/eligible | 0.18511 | 0.18516 | 0.18137 | 240 | 1.000 |
| by_game:2/full_field | 0.18103 | 0.18115 | 0.17984 | 296 | 1.000 |
| by_game:2/eligible | 0.18228 | 0.18242 | 0.18080 | 240 | 1.000 |
| by_game:3/full_field | 0.18348 | 0.18347 | 0.18281 | 125 | 0.427 |
| by_game:3/eligible | 0.18388 | 0.18388 | 0.18315 | 117 | 0.473 |
| by_stage:early/full_field | 0.17249 | 0.17246 | 0.17148 | 296 | 1.000 |
| by_stage:early/eligible | 0.17376 | 0.17373 | 0.17268 | 240 | 1.000 |
| by_stage:late/full_field | 0.18251 | 0.18257 | 0.18034 | 146 | 0.431 |
| by_stage:late/eligible | 0.17702 | 0.17709 | 0.17418 | 112 | 0.412 |
| by_stage:middle/full_field | 0.18888 | 0.18903 | 0.18628 | 296 | 1.000 |
| by_stage:middle/eligible | 0.19014 | 0.19032 | 0.18695 | 240 | 1.000 |
| by_stage:setup/full_field | 0.17966 | 0.17970 | 0.17079 | 216 | 0.607 |
| by_stage:setup/eligible | 0.18744 | 0.18749 | 0.17813 | 176 | 0.625 |

### Every deck

| List/family/group | Completed | A/B/diagnostic uncertain Brier | A/B deck log loss | A/B top1 accuracy |
|---|---:|---|---|---|
| 0330d443fb Hydrapple scaling (passing) | 8/8 | 0.17849 / 0.17848 / 0.17837 | 0.10628 / 0.11133 | 0.95883 / 0.95883 |
| 074abeace8 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19684 / 0.19667 / 0.19108 | 1.06388 / 1.04155 | 0.64323 / 0.64323 |
| 1d18ec49ba Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19078 / 0.19027 / 0.18674 | 2.29870 / 2.08004 | 0.26302 / 0.25521 |
| 2fd689f0eb Metal acceleration (rejected) | 8/8 | 0.18648 / 0.18643 / 0.18489 | 0.75072 / 0.86013 | 0.59115 / 0.59115 |
| 30ebd2289f Dragapult spread (passing) | 8/8 | 0.17584 / 0.17587 / 0.17541 | 0.43936 / 0.43343 | 0.89062 / 0.80990 |
| 35e119dc80 Metal acceleration (rejected) | 8/8 | 0.19085 / 0.19086 / 0.19122 | 0.29473 / 0.27658 | 0.98738 / 0.98738 |
| 45ff39d282 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18088 / 0.18028 / 0.17687 | 0.74007 / 0.63755 | 0.74045 / 0.73872 |
| 470d1ef975 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19384 / 0.19464 / 0.17708 | 2.78784 / 2.60300 | 0.18750 / 0.18750 |
| 48a1e257d1 Kangaskhan/Crustle disruption (passing) | 8/8 | 0.17364 / 0.17358 / 0.17319 | 0.05694 / 0.05201 | 0.99826 / 0.99826 |
| 5087f064a4 Kangaskhan/Crustle disruption (passing) | 8/8 | 0.16124 / 0.16124 / 0.16113 | 0.10721 / 0.10886 | 0.95139 / 0.95139 |
| 5e2b06c453 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19422 / 0.19373 / 0.18550 | 2.09855 / 2.05331 | 0.30642 / 0.30642 |
| 5e81fe946b Metal acceleration (rejected) | 8/8 | 0.17157 / 0.17151 / 0.17044 | 0.09198 / 0.09088 | 0.98090 / 0.97309 |
| 6b1df23695 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19210 / 0.19190 / 0.19128 | 0.17097 / 0.17131 | 0.95412 / 0.93589 |
| 6df51865a4 Festival Lead swarm (passing) | 8/8 | 0.18345 / 0.18354 / 0.18176 | 2.24908 / 2.42800 | 0.17882 / 0.04861 |
| 6fbde59950 Dragapult spread (passing) | 8/8 | 0.17237 / 0.17239 / 0.17172 | 0.32058 / 0.33186 | 0.87587 / 0.87587 |
| 778058128d Dragapult spread (passing) | 8/8 | 0.16786 / 0.16785 / 0.16736 | 1.08759 / 1.00159 | 0.33420 / 0.33767 |
| 7e314fd667 Dragapult spread (passing) | 8/8 | 0.16584 / 0.16597 / 0.16520 | 0.12072 / 0.13818 | 0.98438 / 0.97569 |
| 7e5ce58ba3 Dragapult spread (passing) | 8/8 | 0.17840 / 0.17804 / 0.17449 | 0.62294 / 0.58961 | 0.83941 / 0.75781 |
| 83d6a55e94 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18116 / 0.18096 / 0.18023 | 0.71102 / 0.61656 | 0.94010 / 0.91753 |
| 8584c5d67f Festival Lead swarm (passing) | 8/8 | 0.18621 / 0.18621 / 0.18618 | 0.11916 / 0.10110 | 0.99542 / 0.99542 |
| 8b173f2c9c Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18901 / 0.18892 / 0.18798 | 0.19341 / 0.18266 | 0.91319 / 0.95486 |
| 8d9c5c270d Dragapult spread (passing) | 8/8 | 0.17785 / 0.17787 / 0.17891 | 0.88563 / 0.67350 | 0.36024 / 0.90451 |
| 900f4714b9 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19541 / 0.19696 / 0.19199 | 0.62964 / 0.78757 | 0.42101 / 0.42101 |
| 9fe2ae85a4 Hydrapple scaling (anchor) | 8/8 | 0.16745 / 0.16739 / 0.16674 | 0.05788 / 0.05248 | 0.98958 / 0.98958 |
| aeab90a8bd Banette disruption (rejected) | 8/8 | 0.17599 / 0.17599 / 0.17600 | 0.09471 / 0.09573 | 0.99306 / 0.99306 |
| b8ff5a5d7a Dragapult spread (passing) | 8/8 | 0.16678 / 0.16673 / 0.16657 | 0.42922 / 0.38462 | 0.71374 / 0.71374 |
| bb9d240b1b Banette disruption (rejected) | 8/8 | 0.17361 / 0.17357 / 0.17327 | 0.74653 / 0.73956 | 0.61198 / 0.61198 |
| c139b38f9a Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18014 / 0.17977 / 0.17821 | 0.65124 / 0.51374 | 0.73611 / 0.78299 |
| d8576027b5 Ogerpon/Kangaskhan energy-switch (passing) | 8/8 | 0.17254 / 0.17262 / 0.17084 | 0.08709 / 0.09192 | 0.96094 / 0.96094 |
| d997ab90d8 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18775 / 0.18711 / 0.17774 | 2.66058 / 2.56084 | 0.20486 / 0.20486 |
| dce830b5ca Metal acceleration (rejected) | 8/8 | 0.17452 / 0.17450 / 0.17298 | 0.52835 / 0.47872 | 0.76562 / 0.77604 |
| ee5ad71b52 Dragapult spread (passing) | 8/8 | 0.16905 / 0.16907 / 0.16777 | 0.38788 / 0.40358 | 0.79427 / 0.80729 |
| f1d283ea09 Dragapult spread (passing) | 8/8 | 0.18499 / 0.18501 / 0.18467 | 0.16196 / 0.16146 | 0.93056 / 0.94097 |
| f468acbbe8 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.18412 / 0.18440 / 0.18127 | 0.32159 / 0.30577 | 1.00000 / 1.00000 |
| fbbc3d6bc7 Dragapult spread (passing) | 8/8 | 0.17966 / 0.17960 / 0.18078 | 0.51057 / 0.48864 | 0.68837 / 0.75694 |
| ff2c32d9b5 Alakazam/Dudunsparce draw (passing) | 8/8 | 0.19614 / 0.19595 / 0.18841 | 0.79532 / 0.75979 | 0.59115 / 0.59115 |
| ffbfcd1291 Festival Lead swarm (passing) | 8/8 | 0.19320 / 0.19295 / 0.18994 | 1.38741 / 1.33907 | 0.10243 / 0.51997 |

## Costs

Statistical fitting/inner validation: 12.38 CPU s. Common observer/allocation: 165.61 CPU s. A weight/readout work: 0.88 CPU s; B weight/readout work: 1.08 CPU s. Scoring/export: 807.97 CPU s. These are components of one offline experiment, not three separate deployed runs.

Complete A system: 0.562 CPU s/BO3; complete B: 0.563 CPU s/BO3, each including the same common pass. Model fitting is reported separately as an upfront cost. Recurring overhead is not compared with A’s complete cost. Worker peak RSS: 58.7MiB; this is not aggregate simultaneous memory.

- common_update_CPU_p50_p95 (seconds): [0.0, 0.004365030750001608].
- common_allocation_CPU_p50_p95 (seconds): [0.00435402800000162, 0.017453810399999624].
- A_weight_and_mixture_CPU_p50_p95 (seconds): [0.0, 0.0].
- B_weight_and_mixture_CPU_p50_p95 (seconds): [0.0, 0.0].

## Decision

Outcome2: deck identification improves, but improved hidden-card prediction is not established.

Primary eligible-pool Brier difference: +0.000108; practical improvement magnitude -0.000108. A0.005 reduction is a descriptive practical reference, not a substituted statistical gate. Pair it with the reported confidence interval and recurring CPU overhead.

Known-deck diagnostic minus A: -0.00240. Knowing the exact list does not show a large average downstream gain here; conditional allocation assumptions/calibration may matter more than further classifier complexity. This is a scorer-only diagnostic, not proof of an achievable upper bound. No automatic PPO, held-out evaluation or additional data collection follows.

Execution before final report: 0.257 UTC hours and 0.291 CPU-core hours; final totals in completion.json. Common attempted/completed/failed: 296/296/0. Private output 18.38MiB.

Preparation elapsed at freeze: 7.3 minutes. Measured shared-gate CPU and focused test timings are retained separately; unmetered interactive work is not assigned invented compute hours.

Evidence: registration (historical/private artifact omitted), folds (historical/private artifact omitted), input hashes (historical/private artifact omitted), shared validation (historical/private artifact omitted), source hashes (historical/private artifact omitted), literature notes (historical/private artifact omitted), analysis (historical/private artifact omitted), completion/cost (historical/private artifact omitted).
