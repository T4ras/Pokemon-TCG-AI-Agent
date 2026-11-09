> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Game-reward lambda / entropy: stage three

Stage: COMPLETE. Updated UTC: [removed].

Locked pair: lambda1, entropy coefficient0.001. G continues its selected round10 checkpoint to round30 with model/optimizer/RNG restored. R1 and R2 start from identical BC epoch34 tensors with fresh optimizers and seeds2026101411/904094. All train against frozen BC34 OceanMix using identical OceanMix decks. Game-terminal rewards, gamma1, batch16, constant shared/actor LR0.00003 and value-head LR0.0001. Both sample at temperature1.

Three separate GPU jobs, maximum five hours each. Aggregate target eleven GPU hours including startup, plus one15-minute finishing allowance.80 new rounds/5,120 training BO3. No performance pruning. Every round has a recoverable checkpoint. Continuation lifetime exposure is reported separately from new stage-three training.

Progress:200 CPU BO3 at round20 for G, rounds10/20 for R1/R2. Final:600 fresh CPU BO3 at the fixed round30 checkpoint for each run;2,800 total. Chooser rights exactly half per block; legal policy chooses first/second. Report BO3 and first-game outcomes with pointwise equal-chooser Wilson-MOVER95% intervals. Native shuffle/coin randomness is unseeded.

Qualification: each run reaches at least60% BO3 wins and a95% lower bound above50%. Report the two fresh replicates separately from the selected G continuation. No pooled confidence interval across training runs. Final checkpoint and inference settings are fixed before training.

Verified completed GPU jobs: 3/3. Charged aggregate budget: 9.83/11.00 hours.

CPU worker: COMPLETE; checkpoint None.

| Checkpoint | Matches | BO3 WR (95% CI) | First-game WR (95% CI) | Failures |
|---|---:|---|---|---:|
| G-round-020 | 200 | 55.5% (48.6%–62.1%) | 59.5% (52.6%–65.9%) | 0 |
| G-round-030 | 600 | 60.2% (56.2%–64.0%) | 54.8% (50.8%–58.7%) | 0 |
| R1-round-010 | 200 | 55.5% (48.6%–62.1%) | 51.0% (44.2%–57.8%) | 0 |
| R1-round-020 | 200 | 53.5% (46.7%–60.2%) | 57.0% (50.1%–63.5%) | 0 |
| R1-round-030 | 600 | 59.7% (55.7%–63.5%) | 55.8% (51.8%–59.7%) | 0 |
| R2-round-010 | 200 | 56.0% (49.1%–62.6%) | 51.0% (44.2%–57.8%) | 0 |
| R2-round-020 | 200 | 58.0% (51.1%–64.5%) | 57.0% (50.1%–63.6%) | 0 |
| R2-round-030 | 600 | 62.8% (58.9%–66.6%) | 55.2% (51.2%–59.1%) | 0 |

Final qualification: NOT_QUALIFIED. Fresh replicates qualified: False. Selected continuation qualified: True.

!Stage-three learning curves (historical/private artifact omitted)

Completed-job checkpoints are downloaded to the laptop. A separate local CPU worker evaluates them while the next predefined GPU job trains. Keep the laptop awake and online for downloads, CPU checks and subsequent launches. Kaggle training continues independently once submitted. No notification automation or leaderboard submission configured.

## Training cost audit

G used approximately **3 hours 41 minutes of Kaggle GPU-session quota** in total: 1 hour 13 minutes for rounds1-10 in stage two, followed by 2 hours28 minutes for rounds11-30 in stage three. These are account-quota deltas, including startup; script time was3 hours40 minutes36 seconds. G trained on1,920 BO3 and416,311 learner decisions. About2 hours35 minutes went to collection (including neural inference),55 minutes to PPO, with the remainder recovery/logging/startup/checkpoints.

All three lambda/entropy stages together used approximately **21 GPU hours** of script/session time across search candidates and confirmation runs. This excludes the earlier BC pretraining, older RL pilots and CPU evaluation. G did not inherit training updates from the other candidates or replicates.

G's round10 exploratory result was59.0% over200 BO3 (95% CI52.1-65.5%); its fixed round30 confirmation was60.17% over600 BO3 (56.18-63.97%). Different match samples and overlapping uncertainty do not establish an improvement from extending G from round10 to30. The independent final runs support a modest advantage over the frozen sampled BC34 OceanMix opponent, not broad superiority against other decks or greedy policies. The search does not establish lambda1/entropy0.001 as a unique optimum.
