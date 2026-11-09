# Resume claim → evidence

| Statement | Evidence / qualification |
|---|---|
| End-to-end imitation learning and PPO with a 3.1M Transformer–GRU | [Original pilot](../reports/ml-phase-one.md), [policy](../src/ptcg_agent/learning/model.py), [training](../src/ptcg_agent/learning/training.py). Recorded historical parameter count: 3,105,473. Later schema/vocabulary changes can alter it. |
| Adapted and fine-tuned Unown | [Compatibility](../reports/astra-unown-gradiant-benchmark.md), [PPO](../reports/astra-unown-corrected-hydrapple-ppo-6h.md), [continuation](../reports/astra-unown-new-card-embedding-continuation.md), [attribution](../THIRD_PARTY_NOTICES.md). Released architecture/pretrained weights are upstream work. |
| Bayesian, particle and neural inference over 37 deck configurations | [Bayesian benchmark](../reports/astra-unown-belief-bayesian-deck-inference.md), [particle benchmark](../reports/astra-unown-belief-rao-blackwellization.md), [classifiers](../reports/astra-unown-belief-classifier-full-training.md). A fixed 37-list catalogue is not arbitrary open-world support. |
| Concealed presence/composition from legal histories | [Frozen zone test](../reports/astra-unown-belief-hidden-zone-brier-test.md), [scorer](../scripts/unown_belief_zone_test/evaluate.py), [legal compiler](../scripts/unown_belief_bayes_repair/adapter.py). Privileged truth supplies offline labels only. |
| GPU usage | [Retained-record audit](COMPUTE_USAGE.md): about 97 GPU-session hours / about 194 allocated T4 device-hours for the project through [removed]. 72 allocated T4 device-hours for the final model..  |

The original 3.1M policy, released Unown adaptation and belief predictors are separate stages. Do not imply that Unown has the 3.1M Transformer–GRU architecture or that a particular stage caused the reported leaderboard peak without matching evidence.

## Evidence-backed wording
- Built an imitation-learning and PPO pipeline around a 3.1M-parameter Transformer–GRU policy with recurrent memory and masked autoregressive action decoding; later adapted and fine-tuned the MIT-licensed Unown policy.
- Developed and benchmarked Bayesian, particle-filtering and neural hidden-state inference over 37 opponent deck configurations, predicting concealed card presence and copy counts from legal observation histories.

Keep a historical leaderboard achievement separate until a dated percentile snapshot is available. Omit the exact compute total unless its scope/accounting are reconciled.
