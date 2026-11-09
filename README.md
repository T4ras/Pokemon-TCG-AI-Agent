# Pokémon TCG AI: recurrent policies and hidden-state inference

A source-only research portfolio for the Pokémon TCG AI Battle Challenge Playground. It follows an original imitation-learning/PPO policy, adaptation and fine-tuning of the released **Unown** policy, and Bayesian, particle-filtering and neural methods for inferring concealed cards from legal observation histories.

**Start with [the resume evidence map](docs/RESUME_EVIDENCE.md)**, [the results](docs/RESULTS.md) and [the source guide](docs/SOURCE_GUIDE.md). The [experiment index](experiments/index.json) connects historical registrations, implementations and reports.

## Project progression

| Stage | Work | Implementation |
|---|---|---|
| Original policy | 3.1M-parameter Transformer–GRU; imitation learning followed by PPO; legal deck memory and masked autoregressive pointer decoding | [Policy](src/ptcg_agent/learning/model.py), [observations](src/ptcg_agent/learning/observations.py), [training](src/ptcg_agent/learning/training.py), [batched PPO](src/ptcg_agent/learning/ppo_batched.py) |
| Switch to Unown | Compatibility, current identity mapping, legal-information tracking, starting-order correction and live action-prefix assembly | [Model and inference assembly](models/unown/), [runtime](scripts/unown_runner/) |
| Unown fine-tuning | Corrected full-policy PPO; sparse new-card/attack/skill parameters; frozen shared actor; detached critic; continuation | [PPO](scripts/unown_corrected_ppo_core.py), [sparse adaptation](scripts/unown_embedding/model.py), [continuation](scripts/unown_embedding_continuation/) |
| Bayesian inference | Compatibility-filtered priors, smoothed/tempered reveal-frequency classification, exact conditional allocation, grouped cross-validation | [Classifier](scripts/unown_belief_bayes_repair/deck.py), [allocation](scripts/unown_belief_bayes_repair/allocation.py), [cross-validation](scripts/unown_belief_bayes_repair/cv.py) |
| Particle methods | Physical-copy/serial-aware latent state, history likelihoods, guided recovery and limited Rao–Blackwellization | [State kernel](scripts/unown_belief_rb/state.py), [filter](scripts/unown_belief_rb/filter.py), [collapsed state](scripts/unown_belief_rb/collapsed.py) |
| Neural concealed-card prediction | GRU/MLP classifiers and proper-loss boosted trees over 37 deck configurations; presence/count distributions in hand, draw pile and prizes | [Models](scripts/unown_belief_full_training/models.py), [features](scripts/unown_belief_full_training/features.py), [scoring](scripts/unown_belief_zone_test/evaluate.py) |

The original Transformer–GRU and Unown are **different models**. Unown was released by **Dipam Chakraborty** under MIT: four Transformer layers, width 256, no neural recurrent layer. Its external deck-information tracker is distinct from a GRU. Project contributions around Unown concern compatibility/adaptation, training/evaluation and belief inference; the released architecture and original pretrained weights are upstream work. See [attribution](THIRD_PARTY_NOTICES.md).

## Selected historical evidence

- Original BC pilot: **3,105,473 parameters** for its historical vocabulary/schema.
- Corrected Unown PPO: **768 training BO3**, **144,612 conditional decisions**, **1,147 accepted updates**. Initial/final fixed 256-BO3 evaluations both measured **52.34% BO3 wins**; AVG 900 -> 1000 score on the leaderboard.
- Sparse adaptation continuation: **502,384 collected decisions**, **15,996 accepted updates**. Earlier failures and rejected numerical cohorts remain disclosed.
- Bayesian development: **296 BO3 = 37 lists × 8 histories**, **17,029 selected decisions**, nested grouped folds.
- Belief classifier training: **2,334 valid training BO3**, **592 validation BO3**; a separate frozen test scored **592/592 BO3**. Presence and multiclass count Brier are distinct.
- Particle/Rao–Blackwell variants were implemented and benchmarked; they did **not** outperform the baseline on the reported selected 16-history panel.
- A historical **top-10% leaderboard peak is achieved**.
- [Project compute audit](docs/COMPUTE_USAGE.md): **about 97 Kaggle GPU-session hours**, including failed runs and GPU benchmarks; For the final tuned model **about 72 T4 device-hours was spent**.

The author reports a leaderboard-score gain after fine-tuning. That is separate from statistical conclusions on a particular fixed local BO3 panel.

These are different populations and dates, not one combined performance result. Native shuffles/coins were not controlled by policy seeds. Better belief predictions do not imply better gameplay: [policy-conditioning analysis](reports/astra-unown-belief-correction-pooled-evaluation.md) did not establish a BO3 gain.

## Portable checks

Python 3.12+. After dependencies are installed, checks need no credentials, weights or network services.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[review]"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python examples/concealed_cards.py
python tools/verify_export.py
python tools/check_temporal_data.py
~~~

The default suite uses fictional cards, synthetic observations and exact tiny-world references. It checks action likelihood including STOP, restricted deck memory, hidden-field exclusion, Bayesian weights, particle transitions, conditional marginalization and scoring. It does not access private histories or run native BO3.

Historical tests remain for inspection outside the default tests/review selection; many require omitted private assets. Historical scripts retain experiment-specific layouts/deadlines and are not a turnkey public training service. See [reproduction limits](docs/REPRODUCIBILITY.md).

## Contents and exclusions

Included: relevant source and dependencies, scientific registration excerpts, sanitized aggregate reports, synthetic checks and upstream license notices.

Excluded: all trained weights, optimizer/RNG checkpoints, tensor embedding assets, private SDK/native binaries, card/deck catalogues, raw demonstrations, legal or hidden-state trajectories, per-match logs, credentials, account-specific notebook identifiers and Git history.

This curated folder does not replace the private research archive. See [verification](docs/EXPORT_AUDIT.md), [origins](source-provenance.json) and [file manifest](publication-manifest.json).
