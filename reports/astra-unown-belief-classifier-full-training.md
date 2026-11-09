> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief classifiers trained on all available histories

State: **COMPLETE**. Updated [removed].

7/7 fits complete; all completed fits scored on the fixed592 validation BO3; locked test remains sealed.

This separate experiment uses **all2,334 valid training BO3**, the unchanged592 validation BO3 and zero locked-test evaluations. New fits complete: **7/7**. No new native matches, replacements, actor changes, GPU, paid model APIs, uploads, PPO or submission. Historical experiments and four missing histories remain unchanged.

## Frozen population and objective

The committed archive has2,334 training,592 validation and592 locked-test records. Whole match IDs and non-test legal histories are unique and partitions are disjoint. All2,926 non-test feature/target/metadata caches were checked by hash. No additional training features were needed. The locked-test payload is never parsed by this benchmark; integrity checks hash opaque committed files only. There is no downsampling.

Training composition: {'new': 1970, 'retained-screen-screening': 60, 'legacy-belief': 296, 'retained-screen-confirmation': 8}. Deck totals range 59–68; deck/chooser cells range 28–35. The full point has unequal counts, unlike the balanced296/592 subsets. full-training-manifest.json (historical/private artifact omitted) freezes every training ID, origin and weight.

The intended objective is sum over decks of n/100 prevalence × one-half per chooser × mean BO3 loss in that stratum, followed by the unchanged game/decision/query/zone averages. Neural epochs uniformly permute all2,334 records once, with per-record loss factor2,334×p(deck)/(2×stratum count); weights average1. There is no weighted sampling. Trees retain32 stages with4 records per deck/chooser and the existing37×p(deck) batch loss weight; no second inverse-count correction is applied. Bayesian fitting uses one final reveal ledger per BO3, equal chooser influence within each class via n(deck)/(2×stratum count), and the unchanged prevalence prior; alpha smoothing retains the actual per-deck effective sample count.

| Deck / family | Chooser0 | Chooser1 | Total |
|---|---:|---:|---:|
| 0330d443fb / Hydrapple scaling | 30 | 32 | 62 |
| 074abeace8 / Alakazam/Dudunsparce draw | 30 | 34 | 64 |
| 1d18ec49ba / Alakazam/Dudunsparce draw | 33 | 33 | 66 |
| 2fd689f0eb / Metal acceleration | 29 | 33 | 62 |
| 30ebd2289f / Dragapult spread | 31 | 31 | 62 |
| 35e119dc80 / Metal acceleration | 32 | 32 | 64 |
| 45ff39d282 / Alakazam/Dudunsparce draw | 35 | 30 | 65 |
| 470d1ef975 / Alakazam/Dudunsparce draw | 34 | 31 | 65 |
| 48a1e257d1 / Kangaskhan/Crustle disruption | 32 | 29 | 61 |
| 5087f064a4 / Kangaskhan/Crustle disruption | 33 | 33 | 66 |
| 5e2b06c453 / Alakazam/Dudunsparce draw | 32 | 32 | 64 |
| 5e81fe946b / Metal acceleration | 32 | 34 | 66 |
| 6b1df23695 / Alakazam/Dudunsparce draw | 31 | 31 | 62 |
| 6df51865a4 / Festival Lead swarm | 30 | 30 | 60 |
| 6fbde59950 / Dragapult spread | 35 | 32 | 67 |
| 778058128d / Dragapult spread | 32 | 32 | 64 |
| 7e314fd667 / Dragapult spread | 28 | 33 | 61 |
| 7e5ce58ba3 / Dragapult spread | 33 | 30 | 63 |
| 83d6a55e94 / Alakazam/Dudunsparce draw | 31 | 31 | 62 |
| 8584c5d67f / Festival Lead swarm | 30 | 30 | 60 |
| 8b173f2c9c / Alakazam/Dudunsparce draw | 29 | 31 | 60 |
| 8d9c5c270d / Dragapult spread | 30 | 33 | 63 |
| 900f4714b9 / Alakazam/Dudunsparce draw | 32 | 33 | 65 |
| 9fe2ae85a4 / Hydrapple scaling | 33 | 32 | 65 |
| aeab90a8bd / Banette disruption | 30 | 30 | 60 |
| b8ff5a5d7a / Dragapult spread | 29 | 32 | 61 |
| bb9d240b1b / Banette disruption | 32 | 34 | 66 |
| c139b38f9a / Alakazam/Dudunsparce draw | 32 | 33 | 65 |
| d8576027b5 / Ogerpon/Kangaskhan energy-switch | 30 | 29 | 59 |
| d997ab90d8 / Alakazam/Dudunsparce draw | 33 | 31 | 64 |
| dce830b5ca / Metal acceleration | 34 | 34 | 68 |
| ee5ad71b52 / Dragapult spread | 34 | 30 | 64 |
| f1d283ea09 / Dragapult spread | 31 | 30 | 61 |
| f468acbbe8 / Alakazam/Dudunsparce draw | 30 | 30 | 60 |
| fbbc3d6bc7 / Dragapult spread | 30 | 33 | 63 |
| ff2c32d9b5 / Alakazam/Dudunsparce draw | 32 | 32 | 64 |
| ffbfcd1291 / Festival Lead swarm | 29 | 31 | 60 |

## Unchanged methods and coverage

Architectures, residual logits around A, legal masks, count support0–17, legal observer, exact A marginals, query universe and full-prefix GRU resets match005. Trees and MLP receive the same information; the GRU retains every required callback, including forced observations. No deck-head repair, count-total projection, history change or new representation was introduced. Neural AdamW0.001/weight decay0.0001, clip1, six maximum epochs, patience3 and minimum validation-NLL improvement1e-5 are unchanged. Both seeds11/29 are retained. Bayesian alpha30/120 and beta0.25/0.5/1 selection still uses validation deck log loss, ties to lower beta then higher alpha.

Trees seed[removed] stages, 2334 unique training BO3, 9472 total exposures, per-history exposure range 3–5. The original coverage rule holds; no schedule amendment. tree-schedule-s11.json (historical/private artifact omitted).

Trees seed29:32 stages, 2334 unique training BO3, 9472 total exposures, per-history exposure range 3–5. The original coverage rule holds; no schedule amendment. tree-schedule-s29.json (historical/private artifact omitted).

More data at the same epoch count permits more optimizer updates. This is a population-and-compute scaling comparison, not a controlled equal-update comparison. Recovery checkpoints cannot substitute for incomplete fits.

## Validation learning curve

The592 validation BO3 and original A-defined uncertain-query mask are unchanged. These are model-selection results, not final test evidence. Nominal95% intervals use20,000 paired whole-BO3 resamples within deck/chooser strata, identical across all methods/sizes/seeds. Seeds and queries are not independent matches; intervals condition on fitted models and omit retraining uncertainty. Seed means average metrics inside each BO3 and are not evaluated ensembles.

!Learning curves (historical/private artifact omitted)

### Primary30-list n/82

| Model / training BO3 | Uncertain Brier | Minus A [95%] | Deck accuracy | Deck NLL | Count NLL | Combined NLL |
|---|---:|---|---:|---:|---:|---:|
| A / 0 | 0.18471 | — | 0.79906 | 0.60411 | 0.10030 | 0.70441 |
| Bayes-n16-s0 / 592 | 0.18454 | -0.000165 [-0.000455, +0.000114] | 0.81685 | 0.55878 | 0.09954 | 0.65833 |
| Bayes-n8-s0 / 296 | 0.18452 | -0.000187 [-0.000462, +0.000085] | 0.81640 | 0.56362 | 0.09954 | 0.66316 |
| Bayes-nfull-s0 / 2334 | 0.18443 | -0.000281 [-0.000515, -0.000061] | 0.81590 | 0.55120 | 0.09949 | 0.65069 |
| GRU-n16-s11 / 592 | 0.18185 | -0.002860 [-0.003508, -0.002238] | 0.80551 | 0.64389 | 0.09947 | 0.74336 |
| GRU-n16-s29 / 592 | 0.18211 | -0.002598 [-0.003138, -0.002094] | 0.64032 | 0.82371 | 0.09959 | 0.92330 |
| GRU-n16-seedmean / 592 | 0.18198 | -0.002729 [-0.003317, -0.002178] | 0.72292 | 0.73380 | 0.09953 | 0.83333 |
| GRU-n8-s11 / 296 | 0.18237 | -0.002337 [-0.002855, -0.001847] | 0.81072 | 0.60955 | 0.09968 | 0.70923 |
| GRU-n8-s29 / 296 | 0.18202 | -0.002687 [-0.003400, -0.002015] | 0.81054 | 0.63535 | 0.09963 | 0.73498 |
| GRU-n8-seedmean / 296 | 0.18220 | -0.002512 [-0.003123, -0.001934] | 0.81063 | 0.62245 | 0.09965 | 0.72211 |
| GRU-nfull-s11 / 2334 | 0.18043 | -0.004275 [-0.005294, -0.003302] | 0.61629 | 0.88437 | 0.09906 | 0.98343 |
| GRU-nfull-s29 / 2334 | 0.17878 | -0.005927 [-0.006877, -0.005006] | 0.62631 | 0.84703 | 0.09881 | 0.94584 |
| GRU-nfull-seedmean / 2334 | 0.17961 | -0.005101 [-0.006065, -0.004174] | 0.62130 | 0.86570 | 0.09893 | 0.96464 |
| MLP-n16-s11 / 592 | 0.18141 | -0.003298 [-0.003932, -0.002688] | 0.80798 | 0.64645 | 0.09940 | 0.74585 |
| MLP-n16-s29 / 592 | 0.18113 | -0.003582 [-0.004316, -0.002886] | 0.66952 | 0.76454 | 0.09940 | 0.86394 |
| MLP-n16-seedmean / 592 | 0.18127 | -0.003440 [-0.004114, -0.002797] | 0.73875 | 0.70549 | 0.09940 | 0.80490 |
| MLP-n8-s11 / 296 | 0.18212 | -0.002591 [-0.003167, -0.002052] | 0.81225 | 0.60259 | 0.09957 | 0.70217 |
| MLP-n8-s29 / 296 | 0.18253 | -0.002181 [-0.002560, -0.001823] | 0.81226 | 0.61673 | 0.09975 | 0.71649 |
| MLP-n8-seedmean / 296 | 0.18232 | -0.002386 [-0.002863, -0.001940] | 0.81226 | 0.60966 | 0.09966 | 0.70933 |
| MLP-nfull-s11 / 2334 | 0.18053 | -0.004177 [-0.005137, -0.003277] | 0.61753 | 0.90200 | 0.09909 | 1.00109 |
| MLP-nfull-s29 / 2334 | 0.17812 | -0.006587 [-0.007554, -0.005639] | 0.62554 | 0.93174 | 0.09883 | 1.03057 |
| MLP-nfull-seedmean / 2334 | 0.17933 | -0.005382 [-0.006313, -0.004496] | 0.62154 | 0.91687 | 0.09896 | 1.01583 |
| Trees-n16-s11 / 592 | 0.18423 | -0.000476 [-0.000623, -0.000347] | 0.81047 | 0.59556 | 0.10013 | 0.69569 |
| Trees-n16-s29 / 592 | 0.18415 | -0.000563 [-0.000710, -0.000432] | 0.81126 | 0.59782 | 0.10012 | 0.69793 |
| Trees-n16-seedmean / 592 | 0.18419 | -0.000519 [-0.000665, -0.000391] | 0.81086 | 0.59669 | 0.10013 | 0.69681 |
| Trees-n8-s11 / 296 | 0.18423 | -0.000481 [-0.000615, -0.000362] | 0.80697 | 0.59563 | 0.10013 | 0.69576 |
| Trees-n8-s29 / 296 | 0.18429 | -0.000417 [-0.000530, -0.000316] | 0.80843 | 0.59273 | 0.10014 | 0.69287 |
| Trees-n8-seedmean / 296 | 0.18426 | -0.000449 [-0.000572, -0.000339] | 0.80770 | 0.59418 | 0.10013 | 0.69432 |
| Trees-nfull-s11 / 2334 | 0.18410 | -0.000611 [-0.000747, -0.000487] | 0.81213 | 0.58816 | 0.10010 | 0.68826 |
| Trees-nfull-s29 / 2334 | 0.18426 | -0.000453 [-0.000558, -0.000354] | 0.81255 | 0.58920 | 0.10011 | 0.68931 |
| Trees-nfull-seedmean / 2334 | 0.18418 | -0.000532 [-0.000651, -0.000424] | 0.81234 | 0.58868 | 0.10011 | 0.68879 |

| Full model | Brier minus same family/seed at592 [95%] | Deck NLL change [95%] | Combined NLL change [95%] |
|---|---|---|---|
| Bayes-nfull-s0 | -0.000116 [-0.000238, -0.000003] | -0.007583 [-0.016676, +0.001372] | -0.007637 [-0.016796, +0.001386] |
| GRU-nfull-s11 | -0.001415 [-0.001878, -0.000955] | +0.240481 [+0.189825, +0.284645] | +0.240071 [+0.189459, +0.284210] |
| GRU-nfull-s29 | -0.003329 [-0.003918, -0.002746] | +0.023322 [+0.007318, +0.040175] | +0.022539 [+0.006560, +0.039347] |
| GRU-nfull-seedmean | -0.002372 [-0.002881, -0.001869] | +0.131901 [+0.103316, +0.157068] | +0.131305 [+0.102782, +0.156462] |
| MLP-nfull-s11 | -0.000879 [-0.001255, -0.000520] | +0.255546 [+0.204832, +0.299892] | +0.255239 [+0.204591, +0.299553] |
| MLP-nfull-s29 | -0.003004 [-0.003538, -0.002470] | +0.167200 [+0.137751, +0.195043] | +0.166634 [+0.137191, +0.194460] |
| MLP-nfull-seedmean | -0.001942 [-0.002321, -0.001558] | +0.211373 [+0.173281, +0.244547] | +0.210936 [+0.172875, +0.244092] |
| Trees-nfull-s11 | -0.000135 [-0.000183, -0.000089] | -0.007395 [-0.014745, -0.000591] | -0.007431 [-0.014779, -0.000630] |
| Trees-nfull-s29 | +0.000111 [+0.000049, +0.000175] | -0.008617 [-0.017357, -0.001324] | -0.008619 [-0.017355, -0.001325] |
| Trees-nfull-seedmean | -0.000012 [-0.000058, +0.000033] | -0.008006 [-0.015476, -0.001601] | -0.008025 [-0.015500, -0.001619] |

| Full model | All-query Brier | ECE | H/D/P count MAE | Deck multiclass Brier | Impossible mass | Coverage |
|---|---:|---:|---|---:|---:|---:|
| Bayes-nfull-s0 | 0.02464 | 0.00200 | 0.05728 / 0.08230 / 0.04628 | 0.27279 | 0.00000000 | 1.000 |
| GRU-nfull-s11 | 0.02416 | 0.00205 | 0.05697 / 0.08440 / 0.04632 | 0.42717 | 0.00000000 | 1.000 |
| GRU-nfull-s29 | 0.02395 | 0.00179 | 0.05686 / 0.08476 / 0.04559 | 0.41681 | 0.00000000 | 1.000 |
| GRU-nfull-seedmean | 0.02405 | 0.00187 | 0.05691 / 0.08458 / 0.04595 | 0.42199 | 0.00000000 | 1.000 |
| MLP-nfull-s11 | 0.02417 | 0.00224 | 0.05686 / 0.08453 / 0.04600 | 0.43449 | 0.00000000 | 1.000 |
| MLP-nfull-s29 | 0.02387 | 0.00181 | 0.05658 / 0.08484 / 0.04539 | 0.44334 | 0.00000000 | 1.000 |
| MLP-nfull-seedmean | 0.02402 | 0.00193 | 0.05672 / 0.08468 / 0.04569 | 0.43891 | 0.00000000 | 1.000 |
| Trees-nfull-s11 | 0.02462 | 0.00134 | 0.05730 / 0.08408 / 0.04612 | 0.28894 | 0.00000000 | 1.000 |
| Trees-nfull-s29 | 0.02464 | 0.00129 | 0.05737 / 0.08412 / 0.04617 | 0.29035 | 0.00000000 | 1.000 |
| Trees-nfull-seedmean | 0.02463 | 0.00130 | 0.05734 / 0.08410 / 0.04614 | 0.28965 | 0.00000000 | 1.000 |

| Full model | H/D/P expected-total MAE | Card-total violation MAE / fraction |
|---|---|---|
| Bayes-nfull-s0 | 0.00000 / 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| GRU-nfull-s11 | 0.17421 / 0.28968 / 0.12419 | 0.00149 / 0.04664 |
| GRU-nfull-s29 | 0.24119 / 0.39623 / 0.17279 | 0.00176 / 0.05459 |
| GRU-nfull-seedmean | 0.20770 / 0.34296 / 0.14849 | 0.00163 / 0.05061 |
| MLP-nfull-s11 | 0.21696 / 0.29580 / 0.12808 | 0.00110 / 0.03900 |
| MLP-nfull-s29 | 0.21245 / 0.43948 / 0.19586 | 0.00172 / 0.05466 |
| MLP-nfull-seedmean | 0.21470 / 0.36764 / 0.16197 | 0.00141 / 0.04683 |
| Trees-nfull-s11 | 0.12622 / 0.06594 / 0.09584 | 0.00001 / 0.00398 |
| Trees-nfull-s29 | 0.11732 / 0.07809 / 0.08811 | 0.00002 / 0.01146 |
| Trees-nfull-seedmean | 0.12177 / 0.07202 / 0.09198 | 0.00002 / 0.00772 |

### Secondary37-list n/100

| Model / training BO3 | Uncertain Brier | Minus A [95%] | Deck accuracy | Deck NLL | Count NLL | Combined NLL |
|---|---:|---|---:|---:|---:|---:|
| A / 0 | 0.18307 | — | 0.82082 | 0.53908 | 0.09852 | 0.63760 |
| Bayes-n16-s0 / 592 | 0.18295 | -0.000127 [-0.000364, +0.000102] | 0.83525 | 0.50119 | 0.09791 | 0.59909 |
| Bayes-n8-s0 / 296 | 0.18293 | -0.000144 [-0.000370, +0.000079] | 0.83526 | 0.50515 | 0.09790 | 0.60305 |
| Bayes-nfull-s0 / 2334 | 0.18285 | -0.000222 [-0.000412, -0.000043] | 0.83453 | 0.49557 | 0.09787 | 0.59344 |
| GRU-n16-s11 / 592 | 0.18004 | -0.003029 [-0.003596, -0.002478] | 0.82713 | 0.56798 | 0.09769 | 0.66568 |
| GRU-n16-s29 / 592 | 0.18045 | -0.002624 [-0.003086, -0.002186] | 0.69099 | 0.71797 | 0.09784 | 0.81580 |
| GRU-n16-seedmean / 592 | 0.18025 | -0.002826 [-0.003337, -0.002341] | 0.75906 | 0.64298 | 0.09776 | 0.74074 |
| GRU-n8-s11 / 296 | 0.18085 | -0.002225 [-0.002662, -0.001807] | 0.83046 | 0.54251 | 0.09794 | 0.64045 |
| GRU-n8-s29 / 296 | 0.18035 | -0.002718 [-0.003325, -0.002140] | 0.83037 | 0.56265 | 0.09787 | 0.66052 |
| GRU-n8-seedmean / 296 | 0.18060 | -0.002471 [-0.002990, -0.001977] | 0.83042 | 0.55258 | 0.09790 | 0.65048 |
| GRU-nfull-s11 / 2334 | 0.17862 | -0.004456 [-0.005337, -0.003599] | 0.67142 | 0.76588 | 0.09727 | 0.86314 |
| GRU-nfull-s29 / 2334 | 0.17675 | -0.006324 [-0.007165, -0.005513] | 0.67983 | 0.73518 | 0.09694 | 0.83212 |
| GRU-nfull-seedmean / 2334 | 0.17768 | -0.005390 [-0.006233, -0.004573] | 0.67562 | 0.75053 | 0.09710 | 0.84763 |
| MLP-n16-s11 / 592 | 0.17961 | -0.003463 [-0.004030, -0.002912] | 0.82879 | 0.57068 | 0.09763 | 0.66831 |
| MLP-n16-s29 / 592 | 0.17945 | -0.003619 [-0.004260, -0.003005] | 0.71506 | 0.66832 | 0.09765 | 0.76597 |
| MLP-n16-seedmean / 592 | 0.17953 | -0.003541 [-0.004140, -0.002967] | 0.77192 | 0.61950 | 0.09764 | 0.71714 |
| MLP-n8-s11 / 296 | 0.18060 | -0.002467 [-0.002952, -0.002006] | 0.83177 | 0.53755 | 0.09784 | 0.63540 |
| MLP-n8-s29 / 296 | 0.18092 | -0.002151 [-0.002476, -0.001842] | 0.83178 | 0.54806 | 0.09799 | 0.64605 |
| MLP-n8-seedmean / 296 | 0.18076 | -0.002309 [-0.002712, -0.001927] | 0.83177 | 0.54281 | 0.09792 | 0.64073 |
| MLP-nfull-s11 / 2334 | 0.17863 | -0.004440 [-0.005278, -0.003642] | 0.67230 | 0.78020 | 0.09727 | 0.87747 |
| MLP-nfull-s29 / 2334 | 0.17599 | -0.007086 [-0.007938, -0.006245] | 0.67920 | 0.80462 | 0.09697 | 0.90158 |
| MLP-nfull-seedmean / 2334 | 0.17731 | -0.005763 [-0.006572, -0.004976] | 0.67575 | 0.79241 | 0.09712 | 0.88953 |
| Trees-n16-s11 / 592 | 0.18250 | -0.000568 [-0.000697, -0.000450] | 0.83031 | 0.53180 | 0.09836 | 0.63016 |
| Trees-n16-s29 / 592 | 0.18243 | -0.000642 [-0.000774, -0.000520] | 0.83096 | 0.53377 | 0.09834 | 0.63211 |
| Trees-n16-seedmean / 592 | 0.18247 | -0.000605 [-0.000733, -0.000486] | 0.83063 | 0.53278 | 0.09835 | 0.63113 |
| Trees-n8-s11 / 296 | 0.18249 | -0.000587 [-0.000712, -0.000471] | 0.82744 | 0.53180 | 0.09836 | 0.63016 |
| Trees-n8-s29 / 296 | 0.18254 | -0.000530 [-0.000640, -0.000427] | 0.82864 | 0.52938 | 0.09836 | 0.62774 |
| Trees-n8-seedmean / 296 | 0.18251 | -0.000558 [-0.000676, -0.000449] | 0.82804 | 0.53059 | 0.09836 | 0.62895 |
| Trees-nfull-s11 / 2334 | 0.18243 | -0.000645 [-0.000761, -0.000537] | 0.83167 | 0.52569 | 0.09833 | 0.62402 |
| Trees-nfull-s29 / 2334 | 0.18256 | -0.000511 [-0.000606, -0.000421] | 0.83201 | 0.52652 | 0.09834 | 0.62486 |
| Trees-nfull-seedmean / 2334 | 0.18249 | -0.000578 [-0.000682, -0.000480] | 0.83184 | 0.52610 | 0.09834 | 0.62444 |

| Full model | Brier minus same family/seed at592 [95%] | Deck NLL change [95%] | Combined NLL change [95%] |
|---|---|---|---|
| Bayes-nfull-s0 | -0.000095 [-0.000196, -0.000003] | -0.005622 [-0.013197, +0.001838] | -0.005660 [-0.013284, +0.001850] |
| GRU-nfull-s11 | -0.001427 [-0.001826, -0.001028] | +0.197891 [+0.156309, +0.234193] | +0.197468 [+0.155932, +0.233712] |
| GRU-nfull-s29 | -0.003700 [-0.004242, -0.003161] | +0.017211 [+0.003855, +0.031105] | +0.016314 [+0.003014, +0.030206] |
| GRU-nfull-seedmean | -0.002564 [-0.003009, -0.002117] | +0.107551 [+0.084191, +0.128284] | +0.106891 [+0.083541, +0.127615] |
| MLP-nfull-s11 | -0.000977 [-0.001300, -0.000665] | +0.209524 [+0.168076, +0.245874] | +0.209159 [+0.167766, +0.245469] |
| MLP-nfull-s29 | -0.003467 [-0.003971, -0.002960] | +0.136293 [+0.112043, +0.159195] | +0.135611 [+0.111378, +0.158511] |
| MLP-nfull-seedmean | -0.002222 [-0.002566, -0.001871] | +0.172909 [+0.141635, +0.200048] | +0.172385 [+0.141124, +0.199495] |
| Trees-nfull-s11 | -0.000077 [-0.000122, -0.000033] | -0.006109 [-0.012137, -0.000539] | -0.006137 [-0.012165, -0.000567] |
| Trees-nfull-s29 | +0.000131 [+0.000075, +0.000188] | -0.007247 [-0.014414, -0.001269] | -0.007248 [-0.014420, -0.001273] |
| Trees-nfull-seedmean | +0.000027 [-0.000015, +0.000069] | -0.006678 [-0.012821, -0.001419] | -0.006693 [-0.012841, -0.001432] |

| Full model | All-query Brier | ECE | H/D/P count MAE | Deck multiclass Brier | Impossible mass | Coverage |
|---|---:|---:|---|---:|---:|---:|
| Bayes-nfull-s0 | 0.02394 | 0.00208 | 0.05519 / 0.08171 / 0.04643 | 0.24554 | 0.00000000 | 1.000 |
| GRU-nfull-s11 | 0.02346 | 0.00162 | 0.05483 / 0.08345 / 0.04652 | 0.37119 | 0.00000000 | 1.000 |
| GRU-nfull-s29 | 0.02324 | 0.00128 | 0.05464 / 0.08363 / 0.04584 | 0.36273 | 0.00000000 | 1.000 |
| GRU-nfull-seedmean | 0.02335 | 0.00143 | 0.05474 / 0.08354 / 0.04618 | 0.36696 | 0.00000000 | 1.000 |
| MLP-nfull-s11 | 0.02346 | 0.00179 | 0.05473 / 0.08354 / 0.04622 | 0.37719 | 0.00000000 | 1.000 |
| MLP-nfull-s29 | 0.02315 | 0.00135 | 0.05440 / 0.08373 / 0.04567 | 0.38448 | 0.00000000 | 1.000 |
| MLP-nfull-seedmean | 0.02331 | 0.00151 | 0.05457 / 0.08363 / 0.04594 | 0.38084 | 0.00000000 | 1.000 |
| Trees-nfull-s11 | 0.02391 | 0.00136 | 0.05515 / 0.08317 / 0.04628 | 0.25864 | 0.00000000 | 1.000 |
| Trees-nfull-s29 | 0.02393 | 0.00128 | 0.05522 / 0.08321 / 0.04634 | 0.25979 | 0.00000000 | 1.000 |
| Trees-nfull-seedmean | 0.02392 | 0.00132 | 0.05519 / 0.08319 / 0.04631 | 0.25921 | 0.00000000 | 1.000 |

| Full model | H/D/P expected-total MAE | Card-total violation MAE / fraction |
|---|---|---|
| Bayes-nfull-s0 | 0.00000 / 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| GRU-nfull-s11 | 0.19132 / 0.30300 / 0.12455 | 0.00138 / 0.04610 |
| GRU-nfull-s29 | 0.23370 / 0.38845 / 0.16487 | 0.00169 / 0.05400 |
| GRU-nfull-seedmean | 0.21251 / 0.34573 / 0.14471 | 0.00153 / 0.05005 |
| MLP-nfull-s11 | 0.22768 / 0.30513 / 0.12177 | 0.00104 / 0.03862 |
| MLP-nfull-s29 | 0.21052 / 0.41895 / 0.18872 | 0.00165 / 0.05395 |
| MLP-nfull-seedmean | 0.21910 / 0.36204 / 0.15525 | 0.00134 / 0.04628 |
| Trees-nfull-s11 | 0.12317 / 0.06220 / 0.09171 | 0.00001 / 0.00392 |
| Trees-nfull-s29 | 0.11391 / 0.07366 / 0.08363 | 0.00002 / 0.01124 |
| Trees-nfull-seedmean | 0.11854 / 0.06793 / 0.08767 | 0.00002 / 0.00758 |

## Interpretation and candidate

Bayes: a validation improvement supported by the paired interval; Brier change versus592 is -0.000116 [-0.000238, -0.000003].

Trees: an uncertain change at this resolution; Brier change versus592 is -0.000012 [-0.000058, +0.000033].

MLP: a validation improvement supported by the paired interval; Brier change versus592 is -0.001942 [-0.002321, -0.001558].

GRU: a validation improvement supported by the paired interval; Brier change versus592 is -0.002372 [-0.002881, -0.001869].

MLP has the smallest full-data family-mean point Brier. This ranking alone does not establish statistically significant superiority over the other families.

MLP deck/count/combined NLL: full 0.91687/0.09896/1.01583;592 0.70549/0.09940/0.80490;A 0.60411/0.10030/0.70441. Deck accuracy full 0.62154 versus592 0.73875 and A 0.79906. A lower hand Brier does not conceal a worse deck head or inconsistent count totals.

GRU deck/count/combined NLL: full 0.86570/0.09893/0.96464;592 0.73380/0.09953/0.83333;A 0.60411/0.10030/0.70441. Deck accuracy full 0.62130 versus592 0.72292 and A 0.79906. A lower hand Brier does not conceal a worse deck head or inconsistent count totals.

The registered selection rule nominates **MLP-nfull-s11** for a separately authorized locked-test confirmation of belief prediction. The family/size uses the smallest qualifying validation point Brier; seed11 was fixed in advance, not chosen as the better seed. Tradeoffs above remain relevant. No test evaluation or deployment follows automatically.

## Per-deck and game/stage results

| Deck / family | Bayes | Trees seed mean | MLP seed mean | GRU seed mean |
|---|---:|---:|---:|---:|
| 0330d443fb / Hydrapple scaling | 0.17829 | 0.17733 | 0.17087 | 0.17030 |
| 074abeace8 / Alakazam/Dudunsparce draw | 0.19578 | 0.19393 | 0.18721 | 0.18837 |
| 1d18ec49ba / Alakazam/Dudunsparce draw | 0.19161 | 0.19203 | 0.18762 | 0.18637 |
| 2fd689f0eb / Metal acceleration | 0.18222 | 0.18042 | 0.16819 | 0.17039 |
| 30ebd2289f / Dragapult spread | 0.16738 | 0.16757 | 0.16321 | 0.16360 |
| 35e119dc80 / Metal acceleration | 0.18179 | 0.17998 | 0.17153 | 0.17433 |
| 45ff39d282 / Alakazam/Dudunsparce draw | 0.18208 | 0.18424 | 0.17790 | 0.17755 |
| 470d1ef975 / Alakazam/Dudunsparce draw | 0.19495 | 0.19514 | 0.19257 | 0.19256 |
| 48a1e257d1 / Kangaskhan/Crustle disruption | 0.16155 | 0.16188 | 0.16064 | 0.16062 |
| 5087f064a4 / Kangaskhan/Crustle disruption | 0.16062 | 0.16079 | 0.16183 | 0.16169 |
| 5e2b06c453 / Alakazam/Dudunsparce draw | 0.19216 | 0.19098 | 0.18259 | 0.18256 |
| 5e81fe946b / Metal acceleration | 0.16441 | 0.16388 | 0.16149 | 0.16111 |
| 6b1df23695 / Alakazam/Dudunsparce draw | 0.19521 | 0.19468 | 0.19116 | 0.19128 |
| 6df51865a4 / Festival Lead swarm | 0.18243 | 0.18141 | 0.17481 | 0.17520 |
| 6fbde59950 / Dragapult spread | 0.18035 | 0.18028 | 0.17528 | 0.17548 |
| 778058128d / Dragapult spread | 0.17003 | 0.17016 | 0.16595 | 0.16645 |
| 7e314fd667 / Dragapult spread | 0.16558 | 0.16521 | 0.16353 | 0.16349 |
| 7e5ce58ba3 / Dragapult spread | 0.17607 | 0.17600 | 0.17239 | 0.17247 |
| 83d6a55e94 / Alakazam/Dudunsparce draw | 0.18543 | 0.18483 | 0.18154 | 0.18216 |
| 8584c5d67f / Festival Lead swarm | 0.17918 | 0.17864 | 0.17092 | 0.17206 |
| 8b173f2c9c / Alakazam/Dudunsparce draw | 0.19085 | 0.19043 | 0.18524 | 0.18468 |
| 8d9c5c270d / Dragapult spread | 0.17801 | 0.17776 | 0.17457 | 0.17402 |
| 900f4714b9 / Alakazam/Dudunsparce draw | 0.18811 | 0.18755 | 0.18393 | 0.18408 |
| 9fe2ae85a4 / Hydrapple scaling | 0.16887 | 0.16843 | 0.16422 | 0.16395 |
| aeab90a8bd / Banette disruption | 0.18159 | 0.18137 | 0.17455 | 0.17488 |
| b8ff5a5d7a / Dragapult spread | 0.17054 | 0.17024 | 0.16712 | 0.16768 |
| bb9d240b1b / Banette disruption | 0.17268 | 0.17215 | 0.16521 | 0.16543 |
| c139b38f9a / Alakazam/Dudunsparce draw | 0.19118 | 0.19170 | 0.18680 | 0.18645 |
| d8576027b5 / Ogerpon/Kangaskhan energy-switch | 0.17289 | 0.17252 | 0.16949 | 0.16883 |
| d997ab90d8 / Alakazam/Dudunsparce draw | 0.18909 | 0.18997 | 0.18550 | 0.18439 |
| dce830b5ca / Metal acceleration | 0.16303 | 0.16141 | 0.15114 | 0.15186 |
| ee5ad71b52 / Dragapult spread | 0.17379 | 0.17324 | 0.16685 | 0.16755 |
| f1d283ea09 / Dragapult spread | 0.16567 | 0.16544 | 0.16122 | 0.16160 |
| f468acbbe8 / Alakazam/Dudunsparce draw | 0.19603 | 0.19575 | 0.19055 | 0.19092 |
| fbbc3d6bc7 / Dragapult spread | 0.17666 | 0.17681 | 0.17225 | 0.17203 |
| ff2c32d9b5 / Alakazam/Dudunsparce draw | 0.19232 | 0.19259 | 0.18734 | 0.18695 |
| ffbfcd1291 / Festival Lead swarm | 0.18120 | 0.18109 | 0.17538 | 0.17608 |

| Game/stage; primary population | Bayes | Trees | MLP | GRU |
|---|---|---|---|---|
| by_game:1 | 0.18296 / coverage 1.00 | 0.18268 / coverage 1.00 | 0.17745 / coverage 1.00 | 0.17761 / coverage 1.00 |
| by_game:2 | 0.18474 / coverage 1.00 | 0.18436 / coverage 1.00 | 0.17956 / coverage 1.00 | 0.17990 / coverage 1.00 |
| by_game:3 | 0.18562 / coverage 0.51 | 0.18547 / coverage 0.51 | 0.18043 / coverage 0.51 | 0.18073 / coverage 0.51 |
| by_stage:early | 0.17432 / coverage 1.00 | 0.17381 / coverage 1.00 | 0.16615 / coverage 1.00 | 0.16641 / coverage 1.00 |
| by_stage:late | 0.17530 / coverage 0.35 | 0.17467 / coverage 0.35 | 0.17007 / coverage 0.35 | 0.17001 / coverage 0.35 |
| by_stage:middle | 0.18875 / coverage 1.00 | 0.18868 / coverage 1.00 | 0.18595 / coverage 1.00 | 0.18627 / coverage 1.00 |
| by_stage:setup | 0.18029 / coverage 0.69 | 0.18033 / coverage 0.69 | 0.18022 / coverage 0.69 | 0.18086 / coverage 0.69 |

Later-game/stage participation is outcome-dependent. Full seed-specific per-deck and game/stage metrics remain in analysis.json (historical/private artifact omitted).

## Compute, artifacts and online cost

Caps:20 additional CPU-core hours,24 UTC elapsed hours from launch,10 fitting core hours, four compute threads,8GiB additional private artifacts. Individual fit caps:Bayes10min;trees45min each;MLP90min each;GRU120min each. Private raw caches are referenced in place. New common online features are measured once on the same16 frozen validation histories; incremental model timing processes all legal callbacks and genuine readouts. There is zero training/timing overlap and timing is not predictive evaluation.

| Fit | State | CPU / elapsed min | Epochs/stages / selected epoch | Unique BO3 / updates | Cache-load CPU min | Parameters |
|---|---|---|---|---|---:|---|
| Bayes-nfull-s0 (historical/private artifact omitted) | COMPLETE | 1.03 / 0.99 | 1 / None | 2334 / 0 | 0.83 | 5291 theta values |
| GRU-nfull-s11 (historical/private artifact omitted) | COMPLETE | 39.76 / 12.30 | 5 / 2 | 2334 / 11670 | 5.61 | 177755 trainable |
| GRU-nfull-s29 (historical/private artifact omitted) | COMPLETE | 49.77 / 15.14 | 6 / 6 | 2334 / 14004 | 7.00 | 177755 trainable |
| MLP-nfull-s11 (historical/private artifact omitted) | COMPLETE | 19.97 / 6.29 | 5 / 2 | 2334 / 11670 | 6.05 | 57691 trainable |
| MLP-nfull-s29 (historical/private artifact omitted) | COMPLETE | 25.35 / 7.84 | 6 / 6 | 2334 / 14004 | 7.67 | 57691 trainable |
| Trees-nfull-s11 (historical/private artifact omitted) | COMPLETE | 6.82 / 6.26 | 32 / None | 2334 / 0 | 1.25 | 64 trees; 13768 numeric values/thresholds |
| Trees-nfull-s29 (historical/private artifact omitted) | COMPLETE | 7.61 / 6.86 | 32 / None | 2334 / 0 | 1.41 | 64 trees; 13958 numeric values/thresholds |

Actual tree exposure counts, parameter definitions, recovery artifacts and full SHA256s for each model are in fit-results (historical/private artifact omitted) and the per-fit records. Trees report split thresholds and leaf-value scalars rather than neural trainable weights. Bayesian theta values and neural trainable parameter counts are explicit.

| Full model | Observer/A CPU s per BO3 | Incremental model CPU s | Complete online CPU s | Validation/export CPU min | Peak process MiB |
|---|---:|---:|---:|---:|---:|
| Bayes-nfull-s0 | 8.466 | 2.936 | 11.402 | 1.37 | 1082.1 |
| GRU-nfull-s11 | 8.466 | 0.262 | 8.728 | 0.95 | 1082.1 |
| GRU-nfull-s29 | 8.466 | 0.290 | 8.756 | 1.13 | 1082.1 |
| MLP-nfull-s11 | 8.466 | 0.273 | 8.739 | 0.82 | 1082.1 |
| MLP-nfull-s29 | 8.466 | 0.346 | 8.812 | 1.00 | 1082.1 |
| Trees-nfull-s11 | 8.466 | 0.878 | 9.344 | 0.76 | 1082.1 |
| Trees-nfull-s29 | 8.466 | 1.165 | 9.631 | 1.01 | 1082.1 |

Final controller-accounted CPU: 2.885 core hours; measured execution 2.718; measured preparation 0.032. The preparation reserve is a conservative budget charge, not invented actual compute. Fit subprocess CPU including startup/export: 2.505 hours. Elapsed 1.222 hours. Added private artifacts 0.189GiB. Native/GPU/paidAPI/test scores all0.

Evidence: registration.json (historical/private artifact omitted), manifest-audit.json (historical/private artifact omitted), full-training-manifest.json (historical/private artifact omitted), stratum-counts.json (historical/private artifact omitted), model-config.json (historical/private artifact omitted), historical-results-manifest.json (historical/private artifact omitted), preprocessing-cost.json (historical/private artifact omitted), baseline-current-timing.json (historical/private artifact omitted), source-manifest.json (historical/private artifact omitted), analysis.json (historical/private artifact omitted), completion.json (historical/private artifact omitted). Historical reports and results were not rewritten.
