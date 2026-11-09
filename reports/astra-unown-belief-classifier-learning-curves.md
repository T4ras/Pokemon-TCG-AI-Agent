> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief classifier learning curves

State: **COMPLETE_ELIGIBLE_CURVES**. Updated [removed].

All14 authorized fits and15 fixed validation evaluations at8/16 completed. Original5920-record plan remains incomplete;32/64/128 unavailable because four interrupted histories were not replaced. Locked test unscored; no automatic follow-up.

Original full dataset complete: **False**. Eligible subsets verified: **True**. Finished authorized supervised fits: **14/14**. Original35-fit plan remains incomplete. Locked-test evaluations: **0**. Model scores below are fixed-validation/model-selection evidence, not final test performance. An incomplete fit is not a scientific negative result.

## Frozen comparison

A is the unchanged legal conditional-allocation baseline, extended to exact count marginals and checked against its existing moments. Bayesian weighting uses the same six alpha/beta settings and selects validation deck log loss, with one final reveal ledger per training BO3. Trees, the two-layer width-128 feedforward network, and the single width-128 GRU use seeds 11 and 29 at each fixed training size. Both seeds are retained. Each neural model is below one million parameters.

Trees and the feedforward network receive identical fixed current public/history-summary and per-card feature matrices. Card identity is an eight-bit exact query index; fixed signed sketches summarize legal copy/event counts. No text embedding API or new text vectors are used. Both models have the same A probabilities/count distributions. The GRU adds the full ordered encoded current-game callback sequence, including forced observations; state resets to zero at every new game. Only permitted legal deck facts cross game boundaries. No early prefix is discarded. The event encoding is lossy; a null GRU result would not rule out all possible history encodings.

The tree method is 32-stage stochastic gradient boosting of multi-output depth-three regression trees on proper masked multiclass negative gradients. Each stage uses a cyclic balanced eight BO3/list batch; all training BO3 are covered at least twice at every eligible size; the original32/64/128 sizes remain unavailable. All registered labelled queries participate; legally deterministic count cells have zero gradient. No count class is rebalanced. Neural models use AdamW at 0.001, six maximum epochs, validation-NLL patience three and fixed loss weights. All learned models predict residual logits around A; the legal count support includes every frozen multiplicity up to 17. Recovery checkpoints are diagnostic, not retrospectively selected validation winners; neural best-epoch retention follows the frozen early-stop rule.

The common legal support mask and normalization apply to every method. No joint zone-total projection is applied. The report retains remaining expected-zone-total errors and known-fact/support violations; marginal distributions need not define a coherent complete hidden world. Proper deck and count log losses are averaged by query→decision→game→BO3→chooser/population. Primary uncertain-hand Brier uses the same A-defined [0.10,0.90] subset for all models. Primary weights are the 30-list n/82 pool; secondary weights cover all 37 lists at n/100.

The 592 validation BO3 are fixed, with eight per initial chooser/list. Paired 20,000 whole-match bootstrap samples use identical resampling across methods, sizes and seeds within list/chooser strata. Seeds are not independent matches: both individual scores and within-BO3 mean metrics are reported. Intervals condition on fitted models, omit full retraining uncertainty, and are nominal descriptive intervals across multiple learning-curve comparisons. Later-game participation is outcome-dependent and game/stage breakdowns are descriptive.

## Validation learning curves

!Validation learning curves (historical/private artifact omitted)

### Primary n/82

| Model / independent training BO3 | Uncertain Brier | Difference vs A [paired 95% interval] | Deck accuracy | Deck log loss | ECE | Hand/draw/prize MAE |
|---|---:|---|---:|---:|---:|---|
| A | 0.18471 | — | 0.79906 | 0.60411 | 0.00199 | 0.05761 / 0.08423 / 0.04639 |
| Bayes-n16-s0 / 592 | 0.18454 | -0.000165 [-0.000455, +0.000114] | 0.81685 | 0.55878 | 0.00198 | 0.05730 / 0.08231 / 0.04629 |
| Bayes-n8-s0 / 296 | 0.18452 | -0.000187 [-0.000462, +0.000085] | 0.81640 | 0.56362 | 0.00200 | 0.05729 / 0.08231 / 0.04628 |
| GRU-n16-s11 / 592 | 0.18185 | -0.002860 [-0.003508, -0.002238] | 0.80551 | 0.64389 | 0.00173 | 0.05680 / 0.08414 / 0.04623 |
| GRU-n16-s29 / 592 | 0.18211 | -0.002598 [-0.003138, -0.002094] | 0.64032 | 0.82371 | 0.00140 | 0.05726 / 0.08404 / 0.04607 |
| GRU-n16-seedmean / 592 | 0.18198 | -0.002729 [-0.003317, -0.002178] | 0.72292 | 0.73380 | 0.00148 | 0.05703 / 0.08409 / 0.04615 |
| GRU-n8-s11 / 296 | 0.18237 | -0.002337 [-0.002855, -0.001847] | 0.81072 | 0.60955 | 0.00180 | 0.05664 / 0.08405 / 0.04715 |
| GRU-n8-s29 / 296 | 0.18202 | -0.002687 [-0.003400, -0.002015] | 0.81054 | 0.63535 | 0.00158 | 0.05759 / 0.08399 / 0.04593 |
| GRU-n8-seedmean / 296 | 0.18220 | -0.002512 [-0.003123, -0.001934] | 0.81063 | 0.62245 | 0.00153 | 0.05711 / 0.08402 / 0.04654 |
| MLP-n16-s11 / 592 | 0.18141 | -0.003298 [-0.003932, -0.002688] | 0.80798 | 0.64645 | 0.00202 | 0.05681 / 0.08397 / 0.04640 |
| MLP-n16-s29 / 592 | 0.18113 | -0.003582 [-0.004316, -0.002886] | 0.66952 | 0.76454 | 0.00159 | 0.05727 / 0.08401 / 0.04628 |
| MLP-n16-seedmean / 592 | 0.18127 | -0.003440 [-0.004114, -0.002797] | 0.73875 | 0.70549 | 0.00166 | 0.05704 / 0.08399 / 0.04634 |
| MLP-n8-s11 / 296 | 0.18212 | -0.002591 [-0.003167, -0.002052] | 0.81225 | 0.60259 | 0.00159 | 0.05699 / 0.08393 / 0.04617 |
| MLP-n8-s29 / 296 | 0.18253 | -0.002181 [-0.002560, -0.001823] | 0.81226 | 0.61673 | 0.00149 | 0.05741 / 0.08409 / 0.04673 |
| MLP-n8-seedmean / 296 | 0.18232 | -0.002386 [-0.002863, -0.001940] | 0.81226 | 0.60966 | 0.00140 | 0.05720 / 0.08401 / 0.04645 |
| Trees-n16-s11 / 592 | 0.18423 | -0.000476 [-0.000623, -0.000347] | 0.81047 | 0.59556 | 0.00118 | 0.05733 / 0.08415 / 0.04604 |
| Trees-n16-s29 / 592 | 0.18415 | -0.000563 [-0.000710, -0.000432] | 0.81126 | 0.59782 | 0.00121 | 0.05732 / 0.08417 / 0.04611 |
| Trees-n16-seedmean / 592 | 0.18419 | -0.000519 [-0.000665, -0.000391] | 0.81086 | 0.59669 | 0.00119 | 0.05732 / 0.08416 / 0.04607 |
| Trees-n8-s11 / 296 | 0.18423 | -0.000481 [-0.000615, -0.000362] | 0.80697 | 0.59563 | 0.00120 | 0.05735 / 0.08413 / 0.04607 |
| Trees-n8-s29 / 296 | 0.18429 | -0.000417 [-0.000530, -0.000316] | 0.80843 | 0.59273 | 0.00127 | 0.05736 / 0.08416 / 0.04613 |
| Trees-n8-seedmean / 296 | 0.18426 | -0.000449 [-0.000572, -0.000339] | 0.80770 | 0.59418 | 0.00124 | 0.05736 / 0.08414 / 0.04610 |

### Secondary n/100

| Model / independent training BO3 | Uncertain Brier | Difference vs A [paired 95% interval] | Deck accuracy | Deck log loss | ECE | Hand/draw/prize MAE |
|---|---:|---|---:|---:|---:|---|
| A | 0.18307 | — | 0.82082 | 0.53908 | 0.00188 | 0.05546 / 0.08329 / 0.04652 |
| Bayes-n16-s0 / 592 | 0.18295 | -0.000127 [-0.000364, +0.000102] | 0.83525 | 0.50119 | 0.00202 | 0.05520 / 0.08169 / 0.04644 |
| Bayes-n8-s0 / 296 | 0.18293 | -0.000144 [-0.000370, +0.000079] | 0.83526 | 0.50515 | 0.00209 | 0.05519 / 0.08168 / 0.04643 |
| GRU-n16-s11 / 592 | 0.18004 | -0.003029 [-0.003596, -0.002478] | 0.82713 | 0.56798 | 0.00135 | 0.05466 / 0.08324 / 0.04642 |
| GRU-n16-s29 / 592 | 0.18045 | -0.002624 [-0.003086, -0.002186] | 0.69099 | 0.71797 | 0.00098 | 0.05515 / 0.08317 / 0.04626 |
| GRU-n16-seedmean / 592 | 0.18025 | -0.002826 [-0.003337, -0.002341] | 0.75906 | 0.64298 | 0.00105 | 0.05490 / 0.08321 / 0.04634 |
| GRU-n8-s11 / 296 | 0.18085 | -0.002225 [-0.002662, -0.001807] | 0.83046 | 0.54251 | 0.00164 | 0.05453 / 0.08321 / 0.04729 |
| GRU-n8-s29 / 296 | 0.18035 | -0.002718 [-0.003325, -0.002140] | 0.83037 | 0.56265 | 0.00119 | 0.05548 / 0.08310 / 0.04613 |
| GRU-n8-seedmean / 296 | 0.18060 | -0.002471 [-0.002990, -0.001977] | 0.83042 | 0.55258 | 0.00110 | 0.05501 / 0.08315 / 0.04671 |
| MLP-n16-s11 / 592 | 0.17961 | -0.003463 [-0.004030, -0.002912] | 0.82879 | 0.57068 | 0.00153 | 0.05469 / 0.08306 / 0.04662 |
| MLP-n16-s29 / 592 | 0.17945 | -0.003619 [-0.004260, -0.003005] | 0.71506 | 0.66832 | 0.00113 | 0.05516 / 0.08313 / 0.04648 |
| MLP-n16-seedmean / 592 | 0.17953 | -0.003541 [-0.004140, -0.002967] | 0.77192 | 0.61950 | 0.00119 | 0.05492 / 0.08309 / 0.04655 |
| MLP-n8-s11 / 296 | 0.18060 | -0.002467 [-0.002952, -0.002006] | 0.83177 | 0.53755 | 0.00124 | 0.05489 / 0.08311 / 0.04636 |
| MLP-n8-s29 / 296 | 0.18092 | -0.002151 [-0.002476, -0.001842] | 0.83178 | 0.54806 | 0.00121 | 0.05527 / 0.08320 / 0.04690 |
| MLP-n8-seedmean / 296 | 0.18076 | -0.002309 [-0.002712, -0.001927] | 0.83177 | 0.54281 | 0.00099 | 0.05508 / 0.08315 / 0.04663 |
| Trees-n16-s11 / 592 | 0.18250 | -0.000568 [-0.000697, -0.000450] | 0.83031 | 0.53180 | 0.00121 | 0.05518 / 0.08323 / 0.04621 |
| Trees-n16-s29 / 592 | 0.18243 | -0.000642 [-0.000774, -0.000520] | 0.83096 | 0.53377 | 0.00122 | 0.05517 / 0.08325 / 0.04629 |
| Trees-n16-seedmean / 592 | 0.18247 | -0.000605 [-0.000733, -0.000486] | 0.83063 | 0.53278 | 0.00122 | 0.05518 / 0.08324 / 0.04625 |
| Trees-n8-s11 / 296 | 0.18249 | -0.000587 [-0.000712, -0.000471] | 0.82744 | 0.53180 | 0.00123 | 0.05520 / 0.08321 / 0.04624 |
| Trees-n8-s29 / 296 | 0.18254 | -0.000530 [-0.000640, -0.000427] | 0.82864 | 0.52938 | 0.00130 | 0.05521 / 0.08324 / 0.04630 |
| Trees-n8-seedmean / 296 | 0.18251 | -0.000558 [-0.000676, -0.000449] | 0.82804 | 0.53059 | 0.00126 | 0.05520 / 0.08322 / 0.04627 |

### Every opponent list at the largest completed sizes

All per-deck, game/stage, count-consistency, coverage and seed-specific values are retained in analysis.json (historical/private artifact omitted). Missing metrics retain explicit population coverage; partial fits do not substitute another sample.

Largest fully matched learning-curve size: **16 BO3/list**. Cells below show uncertain-hand Brier / deck accuracy.

| Opponent list / family | A | Bayes-n16-s0 | Trees-n16-seedmean | MLP-n16-seedmean | GRU-n16-seedmean |
|---|---|---|---|---|---|
| 0330d443fb Hydrapple scaling | 0.17827 / 0.939 | 0.17829 / 0.939 | 0.17694 / 0.939 | 0.17178 / 0.939 | 0.17272 / 0.939 |
| 074abeace8 Alakazam/Dudunsparce draw | 0.19492 / 0.369 | 0.19588 / 0.369 | 0.19380 / 0.369 | 0.19217 / 0.391 | 0.19208 / 0.404 |
| 1d18ec49ba Alakazam/Dudunsparce draw | 0.19282 / 0.098 | 0.19124 / 0.095 | 0.19198 / 0.091 | 0.18841 / 0.091 | 0.18826 / 0.094 |
| 2fd689f0eb Metal acceleration | 0.18222 / 0.438 | 0.18234 / 0.419 | 0.18038 / 0.419 | 0.17420 / 0.419 | 0.17681 / 0.419 |
| 30ebd2289f Dragapult spread | 0.16760 / 0.875 | 0.16739 / 0.875 | 0.16758 / 0.828 | 0.16518 / 0.843 | 0.16547 / 0.812 |
| 35e119dc80 Metal acceleration | 0.18157 / 0.978 | 0.18177 / 0.969 | 0.17959 / 0.978 | 0.17739 / 0.978 | 0.17849 / 0.978 |
| 45ff39d282 Alakazam/Dudunsparce draw | 0.18504 / 0.602 | 0.18151 / 0.747 | 0.18415 / 0.598 | 0.18085 / 0.710 | 0.18130 / 0.726 |
| 470d1ef975 Alakazam/Dudunsparce draw | 0.19622 / 0.092 | 0.19589 / 0.110 | 0.19534 / 0.110 | 0.19334 / 0.450 | 0.19407 / 0.469 |
| 48a1e257d1 Kangaskhan/Crustle disruption | 0.16160 / 0.993 | 0.16152 / 0.990 | 0.16198 / 0.993 | 0.16146 / 0.993 | 0.16115 / 0.993 |
| 5087f064a4 Kangaskhan/Crustle disruption | 0.16057 / 0.880 | 0.16062 / 0.880 | 0.16087 / 0.880 | 0.16191 / 0.880 | 0.16129 / 0.880 |
| 5e2b06c453 Alakazam/Dudunsparce draw | 0.19183 / 0.202 | 0.19285 / 0.212 | 0.19099 / 0.253 | 0.18593 / 0.255 | 0.18704 / 0.228 |
| 5e81fe946b Metal acceleration | 0.16436 / 0.979 | 0.16447 / 0.974 | 0.16370 / 0.979 | 0.16258 / 0.979 | 0.16297 / 0.979 |
| 6b1df23695 Alakazam/Dudunsparce draw | 0.19518 / 0.905 | 0.19524 / 0.903 | 0.19450 / 0.891 | 0.19237 / 0.891 | 0.19254 / 0.891 |
| 6df51865a4 Festival Lead swarm | 0.18227 / 0.301 | 0.18247 / 0.130 | 0.18149 / 0.130 | 0.17715 / 0.130 | 0.17828 / 0.130 |
| 6fbde59950 Dragapult spread | 0.18041 / 0.865 | 0.18039 / 0.865 | 0.18021 / 0.865 | 0.17705 / 0.865 | 0.17760 / 0.869 |
| 778058128d Dragapult spread | 0.17021 / 0.415 | 0.17006 / 0.533 | 0.17001 / 0.410 | 0.16815 / 0.479 | 0.16861 / 0.561 |
| 7e314fd667 Dragapult spread | 0.16555 / 0.982 | 0.16559 / 0.977 | 0.16525 / 0.982 | 0.16293 / 0.966 | 0.16330 / 0.965 |
| 7e5ce58ba3 Dragapult spread | 0.17597 / 0.830 | 0.17604 / 0.730 | 0.17592 / 0.717 | 0.17379 / 0.772 | 0.17426 / 0.770 |
| 83d6a55e94 Alakazam/Dudunsparce draw | 0.18543 / 0.916 | 0.18551 / 0.894 | 0.18496 / 0.916 | 0.18342 / 0.896 | 0.18424 / 0.881 |
| 8584c5d67f Festival Lead swarm | 0.17917 / 0.992 | 0.17916 / 0.992 | 0.17879 / 0.992 | 0.17516 / 0.999 | 0.17622 / 0.996 |
| 8b173f2c9c Alakazam/Dudunsparce draw | 0.19120 / 0.917 | 0.19085 / 0.964 | 0.19042 / 0.954 | 0.18607 / 0.957 | 0.18627 / 0.959 |
| 8d9c5c270d Dragapult spread | 0.17790 / 0.465 | 0.17804 / 0.961 | 0.17768 / 0.899 | 0.17513 / 0.970 | 0.17567 / 0.980 |
| 900f4714b9 Alakazam/Dudunsparce draw | 0.18813 / 0.351 | 0.18813 / 0.360 | 0.18745 / 0.351 | 0.18642 / 0.351 | 0.18695 / 0.347 |
| 9fe2ae85a4 Hydrapple scaling | 0.16893 / 0.984 | 0.16887 / 0.984 | 0.16831 / 0.984 | 0.16579 / 0.989 | 0.16588 / 0.992 |
| aeab90a8bd Banette disruption | 0.18155 / 0.979 | 0.18158 / 0.979 | 0.18116 / 0.979 | 0.17775 / 0.982 | 0.17837 / 0.981 |
| b8ff5a5d7a Dragapult spread | 0.17055 / 0.587 | 0.17053 / 0.603 | 0.17032 / 0.587 | 0.16876 / 0.587 | 0.16902 / 0.587 |
| bb9d240b1b Banette disruption | 0.17249 / 0.643 | 0.17268 / 0.643 | 0.17204 / 0.643 | 0.16900 / 0.643 | 0.16973 / 0.643 |
| c139b38f9a Alakazam/Dudunsparce draw | 0.19253 / 0.780 | 0.19099 / 0.831 | 0.19168 / 0.790 | 0.18765 / 0.792 | 0.18806 / 0.825 |
| d8576027b5 Ogerpon/Kangaskhan energy-switch | 0.17281 / 0.958 | 0.17289 / 0.968 | 0.17233 / 0.958 | 0.16986 / 0.958 | 0.17043 / 0.958 |
| d997ab90d8 Alakazam/Dudunsparce draw | 0.19057 / 0.117 | 0.18865 / 0.117 | 0.18978 / 0.117 | 0.18539 / 0.269 | 0.18564 / 0.266 |
| dce830b5ca Metal acceleration | 0.16321 / 0.775 | 0.16304 / 0.819 | 0.16123 / 0.806 | 0.15584 / 0.806 | 0.15740 / 0.810 |
| ee5ad71b52 Dragapult spread | 0.17345 / 0.749 | 0.17371 / 0.757 | 0.17317 / 0.749 | 0.16907 / 0.748 | 0.16985 / 0.758 |
| f1d283ea09 Dragapult spread | 0.16560 / 0.915 | 0.16565 / 0.934 | 0.16542 / 0.942 | 0.16267 / 0.914 | 0.16304 / 0.898 |
| f468acbbe8 Alakazam/Dudunsparce draw | 0.19649 / 1.000 | 0.19659 / 0.999 | 0.19581 / 0.992 | 0.19199 / 0.672 | 0.19318 / 0.601 |
| fbbc3d6bc7 Dragapult spread | 0.17692 / 0.548 | 0.17670 / 0.754 | 0.17676 / 0.851 | 0.17360 / 0.724 | 0.17400 / 0.729 |
| ff2c32d9b5 Alakazam/Dudunsparce draw | 0.19334 / 0.618 | 0.19185 / 0.620 | 0.19238 / 0.618 | 0.18957 / 0.646 | 0.18986 / 0.689 |
| ffbfcd1291 Festival Lead swarm | 0.18123 / 0.070 | 0.18127 / 0.269 | 0.18117 / 0.269 | 0.17791 / 0.269 | 0.17874 / 0.269 |

| Game or stage (primary pool) | A | Bayes-n16-s0 | Trees-n16-seedmean | MLP-n16-seedmean | GRU-n16-seedmean |
|---|---|---|---|---|---|
| by_game:1 | 0.18326 (coverage 1.00) | 0.18309 (coverage 1.00) | 0.18274 (coverage 1.00) | 0.17917 (coverage 1.00) | 0.18002 (coverage 1.00) |
| by_game:2 | 0.18491 (coverage 1.00) | 0.18488 (coverage 1.00) | 0.18435 (coverage 1.00) | 0.18172 (coverage 1.00) | 0.18237 (coverage 1.00) |
| by_game:3 | 0.18590 (coverage 0.51) | 0.18576 (coverage 0.51) | 0.18540 (coverage 0.51) | 0.18279 (coverage 0.51) | 0.18348 (coverage 0.51) |
| by_stage:early | 0.17445 (coverage 1.00) | 0.17435 (coverage 1.00) | 0.17379 (coverage 1.00) | 0.16935 (coverage 1.00) | 0.17059 (coverage 1.00) |
| by_stage:late | 0.17499 (coverage 0.35) | 0.17549 (coverage 0.35) | 0.17473 (coverage 0.35) | 0.17150 (coverage 0.35) | 0.17157 (coverage 0.35) |
| by_stage:middle | 0.18916 (coverage 1.00) | 0.18891 (coverage 1.00) | 0.18870 (coverage 1.00) | 0.18720 (coverage 1.00) | 0.18746 (coverage 1.00) |
| by_stage:setup | 0.18021 (coverage 0.69) | 0.18031 (coverage 0.69) | 0.18041 (coverage 0.69) | 0.18022 (coverage 0.69) | 0.18045 (coverage 0.69) |

| Model | H/D/P expected-total MAE | Card-total violation MAE / fraction | Impossible-count mass |
|---|---|---|---:|
| A | 0.00000 / 0.00000 / 0.00000 | 0.00000 / 0.00000 | 0.00000000 |
| Bayes-n16-s0 | 0.00000 / 0.00000 / 0.00000 | 0.00000 / 0.00000 | 0.00000000 |
| Trees-n16-seedmean | 0.13413 / 0.07079 / 0.11355 | 0.00002 / 0.00878 | 0.00000000 |
| MLP-n16-seedmean | 0.19829 / 0.27060 / 0.08595 | 0.00118 / 0.04339 | 0.00000000 |
| GRU-n16-seedmean | 0.16708 / 0.24522 / 0.07364 | 0.00083 / 0.03432 | 0.00000000 |

## Decision

MLP at 16 BO3/list has the strongest eligible-pool point reduction among families with a negative paired interval upper bound. This is a validation-selected hypothesis. A separate authorized confirmation could use its frozen seed-11 representative (the seed was predetermined, not chosen as the better seed); Bayesian fitting is deterministic. The two-seed mean shown is a mean of validation metrics, not an evaluated model ensemble. No locked-test access or deployment follows automatically.

The largest eligible curve has592 independent training BO3,16/list; it does not establish performance at the original larger training sizes.

The32/64/128 curves are unavailable because the user chose not to replace interrupted histories. Every reported family/seed uses the same original nested subset and fixed validation population.

## Costs and reproducibility

Historical audit/conversion, new collection, feature extraction, fit process startup/CPU, validation inference, analysis, memory and storage are tracked separately. Complete recurring inference is timed separately on 16 fixed validation histories spanning all eight archetypes and both chooser assignments. It processes every legal callback and every genuine decision readout, including GRU updates and A/event features. Sampled validation-point inference cost is labelled separately and is not presented as full online cost. Complete fit/evaluation costs and model parameter counts are in per-fit and validation-result records. Reconciliation consists only of legal support masking/normalization and is included in inference timing. Offline label scoring/export time is not misrepresented as deployed inference.

| Frozen model | Fit CPU min | Full online CPU s/BO3 | Incremental model CPU s/BO3 | Validation/scoring/export CPU min | Peak process MiB |
|---|---:|---:|---:|---:|---:|
| A | 0.00 | 8.872 | 0.013 | 0.32 | 1314.8 |
| Bayes-n16-s0 | 0.84 | 11.907 | 3.048 | 1.37 | 1462.4 |
| Bayes-n8-s0 | 0.79 | 11.625 | 2.766 | 1.33 | 1314.8 |
| GRU-n16-s11 | 15.88 | 9.129 | 0.269 | 1.03 | 1473.1 |
| GRU-n16-s29 | 12.37 | 9.175 | 0.316 | 1.09 | 1481.7 |
| GRU-n8-s11 | 7.01 | 9.210 | 0.350 | 1.17 | 1440.2 |
| GRU-n8-s29 | 9.10 | 9.188 | 0.329 | 1.10 | 1451.9 |
| MLP-n16-s11 | 8.53 | 9.104 | 0.245 | 0.83 | 1467.5 |
| MLP-n16-s29 | 7.45 | 9.116 | 0.257 | 0.82 | 1492.4 |
| MLP-n8-s11 | 4.25 | 9.156 | 0.296 | 0.86 | 1424.6 |
| MLP-n8-s29 | 4.21 | 9.117 | 0.258 | 0.83 | 1447.3 |
| Trees-n16-s11 | 6.98 | 10.004 | 1.144 | 1.10 | 1477.5 |
| Trees-n16-s29 | 6.69 | 9.705 | 0.846 | 0.84 | 1487.2 |
| Trees-n8-s11 | 6.83 | 9.734 | 0.875 | 0.84 | 1369.1 |
| Trees-n8-s29 | 6.96 | 9.809 | 0.950 | 0.84 | 1457.1 |

Full online timing is an actual streaming pass on all 16 fixed histories; per-callback recurrent-update and per-readout p50/p95 CPU latency are retained for every model. Peak memory is per process, not a fabricated aggregate simultaneous peak.

Configuration and evidence: model-config.json (historical/private artifact omitted), fit-schedule.json (historical/private artifact omitted), schema.json (historical/private artifact omitted), training-subsets.json (historical/private artifact omitted), probe.json (historical/private artifact omitted), feasibility-forecast.json (historical/private artifact omitted), analysis.json (historical/private artifact omitted), completion.json (historical/private artifact omitted). The raw split, model source, SDK/policy/deck hashes and fitted-artifact checksums are retained. Test remains sealed.
