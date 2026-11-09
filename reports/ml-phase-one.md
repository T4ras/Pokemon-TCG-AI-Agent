> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Phase one implementation — [removed]

The four-demonstrator, one-learner pipeline is implemented. Kaggle private kernel version 2 completed. All five trained bot checkpoint hashes are verified and artifacts are downloaded. Version 1 stopped before training because Kaggle automatically expanded the uploaded archive. That failure is retained.

## Data and bots

The frozen acquisition contains 92 unique public Playground BO3 episodes and 18,738 teacher decisions. Episodes containing multiple selected teachers supply both legal perspectives but remain in the same split. The audit rejected zero episodes. About 1.14 GB of raw files is retained locally (including downloads from an interrupted first acquisition); processed audit shards total 308 MB. The GPU input includes only the 92 registered episodes, about 958 MB raw or 61 MB compressed with code.

| Bot | Source team | Submission ID | Copied deck family |
| --- | --- | --- | --- |
| Demonstrator 0 | sue124 | 56785813 | Sinistcha / Banette |
| Demonstrator 1 | JerryChen | 56790226 | Hydrapple / Meganium |
| Demonstrator 2 | Leon Liu | 56719905 | Dragapult |
| Demonstrator 3 | YumeNeko | 56792100 | Mega Excadrill |
| Learner 0 | Initialized from demonstrator 0 | Our checkpoint | Sinistcha / Banette |

These are learned approximations to the listed submission versions. Source ratings and leaderboard positions are acquisition-time snapshots, not claims about present ranking or clone strength.

## Implemented behavior

- A shared 3,105,473-parameter BC policy followed by four separate deck specialists.
- Four pre-normalized transformer layers, width 192, six attention heads; learned card IDs plus static card properties; legal state tokens; option/state cross-attention.
- A 192-unit within-game GRU and autoregressive pointer decoder. Every pick, including optional STOP, contributes to the complete action likelihood. Masks enforce minimum/maximum cardinality and distinct indices; options are never silently truncated.
- Explicit cross-game opponent deck lower bounds from legally observed card serials. Repeated observations are deduplicated within each game and counts use the maximum across games. Recurrent state resets at every game and all knowledge resets at every match.
- The same visible-observation encoder for demonstrations and live play. Opaque search data, full replay visuals and match score are excluded. Scalar game outcomes are extracted separately from replay metadata for labels.
- Whole-BO3 hash splits, with the first metadata-only split seed that covers train/validation/test for all four teachers. No outcome or action information is used to choose that seed. Test labels are not used in model selection.
- BC action NLL plus 0.1 outcome MSE; deck-balanced shared training; specialist fine-tuning. Recurrent windows train 32 decisions with up to eight burn-in observations. This is truncated recurrent training, not an exact reconstruction of every full history.
- One learner initialized from specialist 0. Initial PPO cohort: 24 BO3 against BC opponents and eight against a frozen current copy (75%/25%). Each specific opponent is paired across initial chooser seats. Later cohorts support 50% BC, 25% current and 25% historical copies, with at most four historical versions.
- Networks remain fixed during complete BO3 collection. PPO uses only learner-side fresh actions and their joint behavior likelihoods. Game-terminal rewards are +1/-1/0; gamma=1, GAE=0.95, clip=0.2, two passes, entropy=0.005, value coefficient=0.5, BC-reference conditional-prefix KL=0.01, learning rate=3e-5. Recurrent PPO chunks initialize from stored behavior hidden states with burn-in; this is an approximation after parameter updates.
- First-game and BO3 win/loss/draw counts, Wilson 95% intervals for win probability, chooser allocation, actual starting order, all-game counts, failures and timing. Individual games from one BO3 are not assigned falsely independent confidence intervals.
- Compact per-match JSONL outcomes; no successful full replay retention in the native training runner. Capped visible failure snapshots (ten per exception type), a 32,768-decision rollout bound and bounded match time/decision limits.
- Atomic checkpoints include model, optimizer, Python/PyTorch RNG and CUDA RNG. Recovery compares exact action likelihoods and the next optimizer update. Bot manifests bind copied decks and source versions to checkpoint hashes.

## Pilot limits and evidence

Twenty local tests passed, including two real neural BO3 matches, persistent native engine reuse, hidden-field invariance, restricted deck counts, multi-select likelihoods, PPO gradients and checkpoint recovery. A four-decision BC CPU smoke also completed. Actual preprocessing, BC specialization, collection and PPO training are scheduled in Kaggle's GPU runtime. The simulator and parsing naturally execute on its CPU; neural operations use its GPU.

The pilot requests three shared BC passes, one specialist pass each, and one complete PPO cohort. Stage deadlines may truncate passes, which are reported explicitly. The script has a 3,000-second budget and the provider has a 3,600-second timeout. Kaggle initially reported 45 GPU quota hours remaining, resetting [removed]. The completed runtime allocated two Tesla T4 devices; this single-device learner used cuda:0 with PyTorch 2.11.0+cu128. Training script wall time was 780.6 seconds. Kaggle reported 0.23 GPU quota hours used in total, including the startup failure, and 44.77 hours remaining.

Evaluation is only two BO3 per learned opponent before and after PPO, plus two technical gameplay matches per clone against random using its copied deck. These small samples check mechanics and reliability. They do not qualify clones as leaderboard-strength opponents, establish improvement, or replace the proposed 40-match screen and 200-match confirmation. Native game randomness remains uncontrolled by policy seeds.

## Private Kaggle run

- Kernel: [private resource omitted]
- Input dataset: [private resource omitted]
- Only authored code and public episode data are uploaded, following explicit user approval. Private SDK assets and credentials are excluded. The runtime uses Kaggle's existing competition input and verifies the native binary hash before training.
- Results expected: four demonstrator checkpoints, BC learner checkpoint, PPO learner checkpoint, optimizer/RNG states, agents.json, data-manifest.json, report.json, training.jsonl, matches.jsonl and retained failures.

## Reproduce in the WSL checkout

> Historical execution commands omitted. See [reproduction limits](../docs/REPRODUCIBILITY.md).

The installed Kaggle CLI 2.2.4 parses a version suffix but omits it from output requests. The collector uses the SDK version_label field with the required v-prefixed label and verifies all five checkpoint hashes.

A reused acquisition output preserves its frozen source manifest. Register a fresh experiment and verify quota before another run; retrieve the exact executed kernel version rather than an unrelated latest output. Competition submission remains a separate approval step.

## Completed pilot results

All three shared BC passes and all four specialist passes finished without deadline truncation. Shared validation action NLL decreased from 1.683 after pass one to 1.601 after pass three. The first PPO cohort contained 24 matches against the BC pool and eight against the frozen current copy; 4,903 learner decisions received two optimization passes. Approximate old-policy KL was 0.00373 and 0.00467, below the registered stopping threshold. Frozen opponents and exact checkpoint recovery were verified.

The GPU run completed 56 BO3 matches / 135 games / 20,163 total agent decisions, with zero execution failures. The learner's eight-match validation screen went from 2/8 BO3 wins before PPO to 7/8 after; first-game wins went from 2/8 to 6/8. These tiny, stochastic screens do not establish an improvement.

| Copied-deck clone | BO3 wins vs random, out of 2 | First-game wins, out of 2 |
| --- | ---: | ---: |
| Sinistcha / Banette | 1 | 2 |
| Hydrapple / Meganium | 0 | 0 |
| Dragapult | 0 | 0 |
| Mega Excadrill | 1 | 1 |

The clones won only 2/8 aggregate BO3 smoke matches against random. They are mechanically usable frozen models, but have not qualified as strong substitutes for their source submissions. More demonstrations, stronger BC fitting, and a larger fixed gameplay qualification are required before using this pool for expensive RL. Do not interpret beating these clones as leaderboard strength.

Retrieved outputs occupy about 425 MB, including resumable optimizer checkpoints. Source experiment records are in experiments/ml-phase-one-001; model files remain ignored under artifacts/ml-phase-one-001/v2/phase-one-results. Version 2's compact records lacked explicit opponent fields, so the audit reconstructs its bootstrap matchup/seat schedule from the saved Python RNG state; the current runner writes those fields directly. The uploaded dataset version remains frozen to the executed code; rebuild and create a new private dataset version before a later run.

Checkpoint deployment audit also loaded all five downloaded models under local CPU PyTorch and completed three native BO3 matches without failures. Its reconstructed PPO schedule confirms each BC opponent received three learner-chooser and three non-chooser matches; the frozen current copy received four of each. GPU collection took 165.5 seconds for 20,163 decisions (about 121.8 decisions/second including encoding/inference). This single-process pilot has not demonstrated million-game throughput.
