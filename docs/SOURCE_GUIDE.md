# Source guide

## Original model
- src/ptcg_agent/learning/model.py: PolicyNetwork, Transformer, game/pick GRUs, masked decoder and complete-action likelihood.
- observations.py: allowlisted inputs and deduplicated legal deck-knowledge lower bounds.
- replays.py, full_bc.py, recovery_bc.py: demonstrations, behavioral cloning and numerical recovery.
- training.py, ppo_batched.py, rl_stage3.py: PPO, recurrent chunks, checkpointing and fixed-policy evaluation.
- live.py: BO3 runtime; the native SDK is external.

## Released Unown and project adaptation
- models/unown/ptcg/rl/model.py: upstream PTCGTransformer.
- models/unown/frozen_unown/runtime.py and units/: model construction, current-format mapping, prefix length, starting order and conditional argmax.
- scripts/unown_runner/: shared runtime and dependent numerical/collection modules.
- scripts/unown_corrected_ppo_core.py and train.py: corrected rollout/update path.
- scripts/unown_embedding/model.py: sparse additive new-identity parameters; detached value head.
- scripts/unown_embedding_continuation/: fixed continuation, runtime and statistical analysis.

## Hidden-state inference
- scripts/unown_belief_provenance/: legal events and physical-copy provenance.
- scripts/unown_belief_bayes_repair/baseline.py and allocation.py: exact conditional allocation under baseline assumptions.
- deck.py: approximate reveal-frequency classifier and stable deck-mixture weights.
- cv.py, evaluate.py, analysis.py: grouped folds, common queries and whole-match uncertainty.
- scripts/unown_belief_rb/state.py, filter.py and proposal.py: latent physical copies, history constraints and guided recovery.
- collapsed.py, candidate.py and candidate_kernel.py: limited one-bit Rao–Blackwellization, not full analytic integration.
- scripts/unown_belief_learning_source_repair/: whole-match dataset, feature/target boundaries and learning curves.
- scripts/unown_belief_full_training/: Bayes, GRU, MLP and proper-loss boosted trees; copy-count distributions and fitting.
- scripts/unown_belief_zone_test/evaluate.py: concealed-card presence/count metrics and game/match aggregation.
- scripts/unown_belief_population_sampling/: sampling versus prevalence-weighted loss.
- scripts/unown_belief_correction/: belief-conditioned residual policy, separate from prediction scoring.

Earlier scalar/failed/repaired versions remain where they support comparisons or exact regression checks. Imported orchestration/support modules are included to preserve dependencies. Their inclusion is not authorization to execute remote jobs.

"Oracle" or "truth" modules are offline training-label/scoring/diagnostic paths. The deployed policy uses blind legal observations, not privileged labels.
