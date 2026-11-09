> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief observer development repair

State: **COMPLETE**. Updated [removed].

All296 development native completions retained; repaired analyses and96 stability replays attempted. Analysis failures, if any, remain explicit. Held-out untouched; no automatic follow-up.

Historical run001 used zero native matches. Run002’s six native BO3 completions, five scored records and area14 analysis failure are unchanged. Repair003 retains those six and authorizes only the 290 previously unattempted development IDs. New collection records: 290; new native completions: 290; new aligned truth records: 290. **Held-out collection, inspection and scoring: zero.** No match replacement or rerun is authorized.

## Verified native semantics

The official C++ SDK’s Types.h assigns 13=Playing and 14=DeckBottom. State::pushCardRef sends Deck to push_back (native top) and DeckBottom to push_front (native bottom); CardMove::MoveCard keeps 14 in the movement log but normalizes the stored area to Deck. DeckBottom is a destination alias, not a separate pile or a valid source. EffectInstant::ToDeckBottomClose shuffles the target list, then moves each card with openType2. CardImpl’s Metang86 chain is LookDeck4 → selected Metal attachments → ToDeckBottomClose. Thus the failed four-card return shuffles only that returned block, preserves the rest of the deck, and fixes the block at the bottom. This proof uses native implementation, not card text or area-pair guessing. Source/binary hashes and exact symbols are in native-area-semantics.json.

The old Python sample enum omitted both 13 and 14. The audit also distinguishes storage areas from All and selector/reference/internal values 15–24; unexpected physical use is explicit unsupported semantics. Temporary source 24 is normalized to the card’s original area by native MoveCard before logging. The new compiler, physical particles, certified-fact observer and A/B baseline use the same destination semantics. Known bottom copies remain protected until top draws reach their suffix. Ambiguous conditional suffix removal remains an explicit baseline failure rather than invented order.

## Fallback diagnosis and bounded repair

Run002 stopped the entire aggregate on the first depleted compatible stratum and then skipped compilation for the rest of the BO3. Its single degraded flag survived new-game boundaries. This was a conservative global readout latch, not proof the deck was impossible. Failed strata had material estimated masses, including one 100% case. Old replay diagnostics reproduce the original callback counts exactly.

| Retained match | First callback/game/turn | Cause and hypothesis | Pre-event mass | Compatible allocation evidence |
|---|---|---|---:|---|
| belief-51295ec93e0da133637f | 15/1/3 | FINITE_PARTICLE_DEPLETION 900f4714b9 | 0.06259 | PROVED_BY_VALID_REPLAY_WITNESS |
| belief-5881bf146536d670ff40 | — | No fallback | — | Full original support |
| belief-6e454e8d52f31dfce58d | 29/1/8 | FINITE_PARTICLE_DEPLETION 35e119dc80 | 0.17898 | PROVED_BY_VALID_REPLAY_WITNESS |
| belief-b1a8a85c8f5fba029e2c | 44/1/5 | UNSUPPORTED_SEMANTICS Concealed deck selection without resolved effect 1182 | — | No numerical witness search applies to this cause |
| belief-3e5bef53c16eccaf0ccf | 25/1/4 | FINITE_PARTICLE_DEPLETION 9fe2ae85a4 | 1.00000 | PROVED_BY_VALID_REPLAY_WITNESS |
| belief-6b857ec3737f16689bbf | 39/1/5 | FINITE_PARTICLE_DEPLETION 5e2b06c453 | 0.05531 | PROVED_BY_VALID_REPLAY_WITNESS |

Each numerical cause retains its exact typed event, legal context, original recovery failure histogram and normalized pre-event masses in legacy-first-fallbacks.json. The separate diagnostic uses one 2048-state, seed 11 replay per first-depletion prefix; it does not change the primary 512 population. A successful witness proves compatibility under the saved typed history; failure is unresolved, never proof of impossibility. Unsupported search semantics cannot be adjudicated by a numerical witness alone.

The first failed events were 1407 hand→playing, 1219 hand→playing, 920 hand→public board, and 1097 hand→playing. Original 256-state recovery returned zero survivors: availability contradictions dominated; 1219 and 920 also failed joint chance-location plans, and 1097 failed a known-copy hand location. The b1a history instead stopped at an unresolved concealed deck→hand selection while a stale 1182 effect remained in context; no particle increase could fix that compiler omission. All four numerical first-failure prefixes now have valid 2048-state replay witnesses. This establishes finite reconstruction/proposal failure rather than an impossible list for these examples. Exact event payloads and per-cause counts remain in the linked diagnosis file.

The repair separates true history-order pins from auxiliary proposal pins. Auxiliary pins expire after the guided chance segment; true known/latent top-bottom history constraints survive until consumed or explicitly shuffled. Verified native unrevealed deck-search profiles use positive availability and a declared uniform latent selection, not a random-draw likelihood. These repairs are tested against small exact cases and the six retained histories.

Numerically unresolved hypotheses keep their identities and evidence status; healthy strata continue updating. Full readout remains fallback while any compatible stratum lacks complete-prefix evidence. Recovery can resume after later callbacks or a new game only by reconstructing the complete missing prefix and multiplying the original prior by cached completed-game likelihood estimates and the current-game estimate exactly once. No prior mass is restored or failed list silently dropped. A hard public composition contradiction may remove a list; finite depletion may not. Unsupported past semantics still block a justified full posterior across later games.

Primary population 512, floor 4, auxiliary recovery 256 remain fixed; the stability population 2048 uses auxiliary 1024. Retries are deterministic: at most four/game, at least 16 callbacks apart, with a 2,000,000 proposal-state-step BO3 cap (8,000,000 at2048). Completed-game caches store likelihood estimates, not hand/order state. Log evidence is retained; a -700 display/allocation exponent floor prevents floating-point disappearance of compatible strata. Normalized SMC remains approximate.

### Six-history development diagnostics

| Match | Original supported/fallback callbacks | Repaired supported/fallback callbacks | Repaired selected coverage | Scoring | CPU s |
|---|---:|---:|---:|---|---:|
| belief-51295ec93e0da133637f | 15/66 | 58/23 | 57.1% | COMPLETE | 7.95 |
| belief-5881bf146536d670ff40 | 154/0 | 154/0 | 100.0% | COMPLETE | 21.89 |
| belief-6e454e8d52f31dfce58d | unscored/unscored | 75/0 | 100.0% | COMPLETE | 11.36 |
| belief-b1a8a85c8f5fba029e2c | 44/231 | 44/231 | 19.4% | COMPLETE | 43.83 |
| belief-3e5bef53c16eccaf0ccf | 25/160 | 50/135 | 30.6% | COMPLETE | 14.35 |
| belief-6b857ec3737f16689bbf | 39/251 | 47/243 | 20.8% | COMPLETE | 27.63 |

Callback coverage and selected-decision coverage have different denominators. These reused histories informed implementation; original-versus-repaired differences are development diagnostics, not independent evidence of predictive improvement. Detailed repaired analyses, conservation/copy assertions, timing and failure examples remain separate from run002.

## Verification before collection

72 focused tests passed using synthetic inputs and retained histories, with zero new native diagnostics. The actual area14 history replays and scores completely. The independent isolation check compared 75 callbacks after changing forbidden hidden-hand/diagnostic/scheduled-list fields and global RNG draws, checked 37,286 live particle states, verified duplicate callbacks, particle conservation, physical serial facts, fixed weights, all frozen policy files and all six original corpus/label hashes. Exact-reference tests cover top/bottom order and completed-game evidence without reset or double counting. The pinned CPU runtime, tests and history checks are linked in the run directory.

## Collection and analysis contract

Both actors retain submission 57834515, the existing adapter, fixed deck and conditional argmax for each BO3. Initial chooser rights retain the original 4/4 allocation per list; actors choose actual starting order. Forced observations still update policy memory. Tracker features, diagnostic truth and scheduled opponent identity never enter actor input; observer construction receives the full 37-list library/prior and its own RNG, not the true scheduled list. Native deals are independent and policy seeds do not control native randomness.

Collection, truth alignment, observer analysis and scoring have separate status fields. Observer exceptions are grouped offline and cannot erase a native game or issue a replacement. Actor errors, policy mutation, invalid chooser handling, information leakage, unusable corpus/truth alignment or resource exhaustion stop collection. An append-only attempt ledger and exclusive worker intents prohibit duplicate or ambiguous reissues. Legal streams and separate truth labels remain private; sampling retains the original lowest-hash 24 genuine decisions/game.

Scores retain query→decision→game→BO3→equal chooser→n/100 or passing-pool n/82 aggregation, fixed tactical categories and A∈[.10,.90] uncertain queries. 20,000 paired whole-BO3 bootstrap draws use shared indices within list/chooser. Failed analyses retain explicit expected-point and population coverage; metrics on available scores are labeled conditional. Missing strata/matches never establish complete-field improvement, and no missing prediction is fabricated. Supported-only and Full-plus-A-fallback/B-plus-A-fallback are separate.

## Complete development analysis

Native corpus: 296/296; scoring failures: 103; complete-field scoring valid: False. These results are **exploratory development**, with implementation selected using this corpus.

### full_field

| Query/scope | Method | Brier | Hand MAE | Deck MAE | Prize MAE | ECE | Score coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| all_in/all | A | 0.02387 | 0.05590 | 0.08085 | 0.04517 | 0.00200 | 79.0% |
| all_in/all | B_plus_fallback | 0.02409 | 0.05606 | 0.08071 | 0.04514 | 0.00235 | 79.0% |
| all_in/all | Full_plus_fallback | 0.02650 | 0.05683 | 0.08228 | 0.04491 | 0.00831 | 79.0% |
| all_in/uncertain | A | 0.18457 | 0.42805 | 0.57035 | 0.34860 | 0.01537 | 79.0% |
| all_in/uncertain | B_plus_fallback | 0.18636 | 0.42963 | 0.57128 | 0.34851 | 0.02012 | 79.0% |
| all_in/uncertain | Full_plus_fallback | 0.20428 | 0.43481 | 0.58263 | 0.34672 | 0.06510 | 79.0% |
| supported_only/all | A | 0.02439 | 0.05694 | 0.09978 | 0.05049 | 0.00197 | 76.5% |
| supported_only/all | B_plus_fallback | 0.02466 | 0.05715 | 0.09943 | 0.05045 | 0.00222 | 76.5% |
| supported_only/all | Full_plus_fallback | 0.02971 | 0.05840 | 0.10272 | 0.05033 | 0.01478 | 76.5% |
| supported_only/uncertain | A | 0.17660 | 0.41365 | 0.62251 | 0.37323 | 0.01535 | 76.5% |
| supported_only/uncertain | B_plus_fallback | 0.17879 | 0.41561 | 0.62239 | 0.37301 | 0.01662 | 76.5% |
| supported_only/uncertain | Full_plus_fallback | 0.21626 | 0.42449 | 0.64687 | 0.37087 | 0.11019 | 76.5% |

- all_in/all/A_minus_Full_brier: -0.00263, paired95% interval [-0.0030820279957600298, -0.0021400007636241873].
- all_in/all/B_plus_fallback_minus_Full_brier: -0.00241, paired95% interval [-0.00279154625026033, -0.0019973358550317275].
- all_in/uncertain/A_minus_Full_brier: -0.01971, paired95% interval [-0.023252416345055434, -0.016005471938219194].
- all_in/uncertain/B_plus_fallback_minus_Full_brier: -0.01792, paired95% interval [-0.020949795910316052, -0.01479688324227683].
- supported_only/all/A_minus_Full_brier: -0.00532, paired95% interval [-0.006531210639628629, -0.004227197237375603].
- supported_only/all/B_plus_fallback_minus_Full_brier: -0.00504, paired95% interval [-0.006324395881257831, -0.0039723292024672315].
- supported_only/uncertain/A_minus_Full_brier: -0.03966, paired95% interval [-0.049648782763874875, -0.030737167852549332].
- supported_only/uncertain/B_plus_fallback_minus_Full_brier: -0.03747, paired95% interval [-0.04790463176147265, -0.028639120464700556].

### eligible

| Query/scope | Method | Brier | Hand MAE | Deck MAE | Prize MAE | ECE | Score coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| all_in/all | A | 0.02459 | 0.05823 | 0.08180 | 0.04457 | 0.00207 | 77.7% |
| all_in/all | B_plus_fallback | 0.02487 | 0.05843 | 0.08169 | 0.04454 | 0.00218 | 77.7% |
| all_in/all | Full_plus_fallback | 0.02744 | 0.05937 | 0.08352 | 0.04427 | 0.00846 | 77.7% |
| all_in/uncertain | A | 0.18676 | 0.43597 | 0.57109 | 0.33464 | 0.01377 | 77.7% |
| all_in/uncertain | B_plus_fallback | 0.18900 | 0.43796 | 0.57264 | 0.33453 | 0.01887 | 77.7% |
| all_in/uncertain | Full_plus_fallback | 0.20802 | 0.44490 | 0.58657 | 0.33290 | 0.06570 | 77.7% |
| supported_only/all | A | 0.02420 | 0.05740 | 0.09091 | 0.04962 | 0.00200 | 75.5% |
| supported_only/all | B_plus_fallback | 0.02454 | 0.05766 | 0.09055 | 0.04957 | 0.00164 | 75.5% |
| supported_only/all | Full_plus_fallback | 0.02931 | 0.05905 | 0.09401 | 0.04914 | 0.01383 | 75.5% |
| supported_only/uncertain | A | 0.17965 | 0.42331 | 0.60198 | 0.36759 | 0.01167 | 75.5% |
| supported_only/uncertain | B_plus_fallback | 0.18239 | 0.42587 | 0.60316 | 0.36741 | 0.01488 | 75.5% |
| supported_only/uncertain | Full_plus_fallback | 0.21767 | 0.43605 | 0.62981 | 0.36427 | 0.10601 | 75.5% |

- all_in/all/A_minus_Full_brier: -0.00285, paired95% interval [-0.0033925238723646494, -0.0022624072524602052].
- all_in/all/B_plus_fallback_minus_Full_brier: -0.00258, paired95% interval [-0.003029142857385647, -0.002092433537998395].
- all_in/uncertain/A_minus_Full_brier: -0.02126, paired95% interval [-0.025451771321097386, -0.016875026523902138].
- all_in/uncertain/B_plus_fallback_minus_Full_brier: -0.01903, paired95% interval [-0.022559621762256683, -0.015440213035949682].
- supported_only/all/A_minus_Full_brier: -0.00512, paired95% interval [-0.006515448601746563, -0.003936562593382975].
- supported_only/all/B_plus_fallback_minus_Full_brier: -0.00478, paired95% interval [-0.0062723339879390515, -0.0036421588488973356].
- supported_only/uncertain/A_minus_Full_brier: -0.03803, paired95% interval [-0.04938980365357324, -0.028619859393184562].
- supported_only/uncertain/B_plus_fallback_minus_Full_brier: -0.03529, paired95% interval [-0.047458436745449493, -0.026076608887986328].

### Every frozen list

| Hash/family | Native/scored | BO3 W/D/L | First W/D/L | Learner actually first, game1 | Supported callbacks | Supported selected decisions |
|---|---:|---|---|---:|---:|---:|
| 0330d443fb Hydrapple scaling | 8/8 | 4/0/4 | 5/0/3 | 4 | 607/1314 | 209/453 |
| 074abeace8 Alakazam/Dudunsparce draw | 8/8 | 2/0/6 | 2/0/6 | 4 | 1146/1342 | 304/365 |
| 1d18ec49ba Alakazam/Dudunsparce draw | 8/8 | 1/0/7 | 2/0/6 | 4 | 853/1614 | 228/432 |
| 2fd689f0eb Metal acceleration | 8/3 | 8/0/0 | 8/0/0 | 4 | 437/746 | 229/384 |
| 30ebd2289f Dragapult spread | 8/0 | 6/0/2 | 5/0/3 | 4 | 154/1315 | 58/432 |
| 35e119dc80 Metal acceleration | 8/4 | 8/0/0 | 8/0/0 | 4 | 334/679 | 188/373 |
| 45ff39d282 Alakazam/Dudunsparce draw | 8/8 | 4/0/4 | 1/0/7 | 4 | 1204/1890 | 319/528 |
| 470d1ef975 Alakazam/Dudunsparce draw | 8/8 | 2/0/6 | 5/0/3 | 4 | 1286/1940 | 350/528 |
| 48a1e257d1 Kangaskhan/Crustle disruption | 8/1 | 7/0/1 | 6/0/2 | 4 | 297/1898 | 73/480 |
| 5087f064a4 Kangaskhan/Crustle disruption | 8/0 | 6/0/2 | 4/0/4 | 4 | 281/1759 | 81/480 |
| 5e2b06c453 Alakazam/Dudunsparce draw | 8/8 | 4/0/4 | 3/0/5 | 4 | 1405/1757 | 368/456 |
| 5e81fe946b Metal acceleration | 8/8 | 8/0/0 | 7/0/1 | 4 | 179/1208 | 62/408 |
| 6b1df23695 Alakazam/Dudunsparce draw | 8/8 | 1/0/7 | 1/0/7 | 4 | 1229/1628 | 354/453 |
| 6df51865a4 Festival Lead swarm | 8/8 | 2/0/6 | 2/0/6 | 4 | 293/1554 | 99/456 |
| 6fbde59950 Dragapult spread | 8/0 | 5/0/3 | 4/0/4 | 4 | 148/1423 | 58/480 |
| 778058128d Dragapult spread | 8/1 | 7/0/1 | 5/0/3 | 4 | 190/1351 | 71/459 |
| 7e314fd667 Dragapult spread | 8/0 | 4/0/4 | 5/0/3 | 4 | 184/1628 | 63/497 |
| 7e5ce58ba3 Dragapult spread | 8/0 | 7/0/1 | 6/0/2 | 4 | 134/1626 | 32/504 |
| 83d6a55e94 Alakazam/Dudunsparce draw | 8/8 | 3/0/5 | 4/0/4 | 4 | 1003/1610 | 299/474 |
| 8584c5d67f Festival Lead swarm | 8/8 | 2/0/6 | 2/0/6 | 4 | 492/1600 | 166/461 |
| 8b173f2c9c Alakazam/Dudunsparce draw | 8/8 | 1/0/7 | 1/0/7 | 4 | 1611/1901 | 406/480 |
| 8d9c5c270d Dragapult spread | 8/1 | 4/0/4 | 5/0/3 | 4 | 170/1611 | 57/516 |
| 900f4714b9 Alakazam/Dudunsparce draw | 8/8 | 2/0/6 | 2/0/6 | 4 | 1063/1632 | 305/456 |
| 9fe2ae85a4 Hydrapple scaling | 8/8 | 5/0/3 | 3/0/5 | 4 | 315/1462 | 115/504 |
| aeab90a8bd Banette disruption | 8/8 | 8/0/0 | 8/0/0 | 4 | 203/1074 | 84/408 |
| b8ff5a5d7a Dragapult spread | 8/0 | 4/0/4 | 2/0/6 | 4 | 164/1509 | 53/469 |
| bb9d240b1b Banette disruption | 8/8 | 8/0/0 | 8/0/0 | 4 | 199/908 | 88/384 |
| c139b38f9a Alakazam/Dudunsparce draw | 8/8 | 1/0/7 | 2/0/6 | 4 | 1246/1538 | 368/456 |
| d8576027b5 Ogerpon/Kangaskhan energy-switch | 8/8 | 6/0/2 | 3/0/5 | 4 | 537/1331 | 180/480 |
| d997ab90d8 Alakazam/Dudunsparce draw | 8/8 | 0/0/8 | 0/0/8 | 4 | 1013/1573 | 296/456 |
| dce830b5ca Metal acceleration | 8/7 | 8/0/0 | 8/0/0 | 4 | 584/768 | 315/408 |
| ee5ad71b52 Dragapult spread | 8/0 | 4/0/4 | 4/0/4 | 4 | 112/1285 | 41/411 |
| f1d283ea09 Dragapult spread | 8/0 | 3/0/5 | 5/0/3 | 4 | 133/1601 | 47/525 |
| f468acbbe8 Alakazam/Dudunsparce draw | 8/8 | 0/0/8 | 2/0/6 | 4 | 1036/1672 | 302/456 |
| fbbc3d6bc7 Dragapult spread | 8/0 | 6/0/2 | 4/0/4 | 4 | 123/1541 | 42/504 |
| ff2c32d9b5 Alakazam/Dudunsparce draw | 8/8 | 0/0/8 | 1/0/7 | 4 | 1464/1722 | 410/482 |
| ffbfcd1291 Festival Lead swarm | 8/8 | 1/0/7 | 2/0/6 | 4 | 523/1866 | 152/531 |

BO3 and first-game win-rate uncertainty is retained with the same whole-match bootstrap in development-analysis.json; draws remain denominators. Later games and actual starting order are descriptive. Recorded per-game chooser, starter, outcomes and termination labels remain in compact native records.

### Coverage by game/stage and archetype

| Group | Supported callbacks / expected | Supported selected / expected |
|---|---:|---:|
| family:Alakazam/Dudunsparce draw | 15559/21819 | 4309/6022 |
| family:Banette disruption | 402/1982 | 172/792 |
| family:Dragapult spread | 1512/14890 | 522/4797 |
| family:Festival Lead swarm | 1308/5020 | 417/1448 |
| family:Hydrapple scaling | 922/2776 | 324/957 |
| family:Kangaskhan/Crustle disruption | 578/3657 | 154/960 |
| family:Metal acceleration | 1534/3401 | 794/1573 |
| family:Ogerpon/Kangaskhan energy-switch | 537/1331 | 180/480 |
| game1/early | 6102/7519 | 2046/2508 |
| game1/late | 301/1330 | 76/338 |
| game1/middle | 5475/12927 | 1657/3991 |
| game1/setup | 687/687 | 180/180 |
| game2/early | 2871/7709 | 898/2568 |
| game2/late | 165/1235 | 55/337 |
| game2/middle | 4067/12894 | 1207/3955 |
| game2/setup | 261/663 | 73/183 |
| game3/early | 1027/3387 | 283/1050 |
| game3/late | 75/605 | 20/132 |
| game3/middle | 1240/5656 | 351/1712 |
| game3/setup | 81/264 | 26/75 |

First-fallback causes: {'FINITE_PARTICLE_DEPLETION': 172, 'UNSUPPORTED_SEMANTICS': 56}. Later selected decisions affected: 5713. Analysis/scoring failures are separate from full-filter unsupported/depletion and from native/truth failures. Each match’s first event, unresolved lists, masses and resumed-readout transitions remain in analysis records/private diagnostic artifacts.

## Offline stability

96 replays on 16 frozen development IDs, budgets 512/2048 and seeds 11/29/47; no extra native matches. All-in mean absolute probability discrepancy 0.01152; paired-point coverage 86.4%. Shared-full differences, support disagreement, Brier and failures are reported by pair. Larger budget is a convergence comparison, not ground truth; agreement caused by A fallback is explicitly distinguished.

## Analysis failures

{'InvariantFailure: Certified-fact native movement/count mismatch': 96, 'InvariantFailure: Certified-fact movement has negative source count': 7} Detailed examples stay private and replayable. Valid native games remain in the corpus.

## Decision and costs

**Recommendation:** Repair the remaining exact-fact/observer-analysis failures recorded by signature before freezing for held-out evaluation.

Execution: 1.246 UTC hours; worker CPU 4.786 core-hours; controller CPU 19.18s; private storage 24.17MiB. Preparation/replay readiness costs are recorded separately. Fresh deadline: [removed]. At most four local workers and one Torch thread. GPU 0, training 0, uploads 0, submissions 0, paid model API calls 0; no power-plan changes or sleep inhibition.

Preparation elapsed at freeze: 45.5 minutes; metered replay/witness/isolation CPU subtotal 166.8s. Earlier inspection, legacy reproduction and setup CPU were not all metered, so this is a lower bound rather than total preparation CPU. Expected execution 1–3 hours; conservative forecast 6.25 hours, using complete retained workloads, four workers, a 3× margin and startup/analysis allowance. The fixed deadline includes preparation and is not extended by sleep.

Evidence: registration (historical/private artifact omitted), ledger (historical/private artifact omitted), native-area semantics (historical/private artifact omitted), first-fallback diagnoses (historical/private artifact omitted), compatibility witnesses (historical/private artifact omitted), source manifest (historical/private artifact omitted), development analysis (historical/private artifact omitted), completion/cost (historical/private artifact omitted). Historical run002 files remain unchanged; a guard manifest verifies this.
