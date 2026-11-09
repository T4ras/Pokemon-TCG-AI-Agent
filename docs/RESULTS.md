# Results and unsuccessful hypotheses

Historical outcomes; no experiments rerun for publication.

The author reports leaderboard-score improvement after fine-tuning. Inconclusive strength comparisons below refer to their particular local BO3 panels, not a blanket claim that leaderboard scores did not increase.

| Experiment | Evidence | Conclusion |
|---|---|---|
| Original BC/PPO | 3,105,473-parameter pilot; [report](../reports/ml-phase-one.md) | Pipeline implementation; small pilot gameplay samples do not establish leaderboard strength. |
| Corrected Unown PPO | 768 training BO3, 144,612 decisions, 1,147 accepted updates; initial/final fixed 256-BO3 wins both 52.34%; [report](../reports/astra-unown-corrected-hydrapple-ppo-6h.md) | No established BO3 gain. |
| Sparse adaptation | Initial failure and partial-retry numerical guard retained; continuation 502,384 decisions / 15,996 updates; [failure/retry](../reports/astra-unown-new-card-embedding-adaptation.md), [continuation](../reports/astra-unown-new-card-embedding-continuation.md) | Completion/software correctness are distinct from strength. |
| Bayesian development | 296 matches, 37 lists, 17,029 points; [report](../reports/astra-unown-belief-bayesian-deck-inference.md) | Reveal-frequency classifier is approximate, not an exact action-history posterior. |
| PF/Rao–Blackwell | 16 selected histories; uncertain Brier A 0.17641, PF512 0.19294, PF2048 0.18769, RB256 0.19265, RB512 0.19321; [report](../reports/astra-unown-belief-rao-blackwellization.md) | Lower is better; no improvement over baseline. Support/fallback rates and failures retained. |
| Full-data classifiers | 2,334 training BO3, 592 validation; seven fits; [report](../reports/astra-unown-belief-classifier-full-training.md) | GRU, MLP, Bayes and trees evaluated separately; actor unchanged. |
| Frozen zone test | 592/592 test BO3; all-card hand-presence Brier A 0.024900 vs full-data MLP seed29 0.024047; [full tables/intervals](../reports/astra-unown-belief-hidden-zone-brier-test.md) | Illustrative fixed-model result, not a multiplicity-adjusted winner claim. Test access disclosed. |
| Belief policy correction | 1,636 complete / 1,639 attempted; original 30.94%, legal belief 33.26%; difference +2.32pp [−5.84,+10.48]; [report](../reports/astra-unown-belief-correction-pooled-evaluation.md) | No established BO3 advantage; post-hoc pooling and runtime-selected prefix are limitations. |

Native randomness was not exposed as seed-controllable. Chooser rights and actual turn order differ. Raw wins, weighted rates, first-game and BO3 outcomes are not interchangeable. Protocol and failure coverage are given in the retained reports.
