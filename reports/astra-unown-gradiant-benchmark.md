> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

> Long raw diagnostic payloads were omitted from this public copy; original evidence remains in the private archive.
# Unown Gradiant frozen-package benchmark

Updated [removed]. State: **COMPLETE**.

This is an inference/compatibility experiment comparing complete deck-policy packages. Previous third place does not establish current Playground strength. No training, weight adaptation, deck search, uploads, submissions or GPU usage.

Upstream: [dipamc/kaggle-ptcg-ai-battle](https://github.com/dipamc/kaggle-ptcg-ai-battle/tree/a176778d99da74a44b495bf19299b4795faf12f2), commit `a176778d99da74a44b495bf19299b4795faf12f2`. MIT License, copyright [removed] Dipam Chakraborty; notice retained with the isolated private source. Exact submitted package 55559155, original Hydrapple/Meganium deck, heads8, temperature0, cfg=None; original rejection guard and time valve retained. The newer top-level runtime is excluded from inference.

## Frozen identities

Candidate/checkpoint/deck/table/dependency hashes: manifest.json and upstream-manifest.json. Current official environment: `f994f2050d58d84f649329c1e35ce09ada7007c5`, version2, BO3. Native deals are independent and not seedable by our inference seed.

| Package | Checkpoint SHA256 | Deck SHA256 |
|---|---|---|
| BC34-Hydrapple | `[artifact fingerprint omitted]` | `[artifact fingerprint omitted]` |
| League32-Hydrapple | `[artifact fingerprint omitted]` | `[artifact fingerprint omitted]` |
| G30-OceanMix | `[artifact fingerprint omitted]` | `[artifact fingerprint omitted]` |
| Unown-Gradiant | `[artifact fingerprint omitted]` | `[artifact fingerprint omitted]` |

## Compatibility and adaptation

The release has 1,267 original cards and 1,556 attacks; Playground has 1,431 cards and 1,755 attacks. Model tables have 1,284 card rows and 1,573 attack rows, including reserved slack, which is not a learned representation of the 164 new cards or 199 new attacks. Fixed-pool support is checked by identity, not merely numeric ID.

The old official library already installed in kaggle-environments supplies original static metadata without starting games. The original competition download endpoint returned403; this alternative is verified against all checkpoint static and text buffers. API bindings are from the locally held official SDK; the legacy metadata-only binding sets AllCard/AllAttack return types. Current engine execution is a separate process/library namespace and always uses Playground v2.

Candidate inference runs in an isolated child using the unmodified submitted Python main.py and ptcg runtime. The adapter resets the original tracker, forced-run counter and rejection state at every game/match; it invents no cross-game knowledge. Our comparators retain their original legal deck-memory rules and forced-observation recurrent updates. Only actor-visible observations cross the pipe. The privileged oracle block is zero; the actor has no path from it. Original static table/PCA buffers and all weights remain fixed.

There are 63 scored option slots plus STOP and 160 state-token slots. Non-forced option overflow, token overflow and unverified card/attack identities raise an explicit compatibility error rather than silently dropping options or clamping unknown identities. These guards leave supported decisions unchanged. Original sequential pick/STOP, equivalence-class deduplication, forced tracking, rejection recovery and panic valve are retained. Panic/fallback/errors disqualify the runtime gate.

Compatibility evidence: detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

## Official checks and admission forecast

Official attempts recorded: 6/6. Strength attempts recorded: 288/288. No replacements and no native diagnostics outside the294-attempt allocation.

Admission: `{"at_utc": "[removed]", "admitted": true, "max_check_match_seconds": 7.928565, "max_cold_candidate_startup_seconds": 1.4909777920000238, "workers": 4, "inflation": 3, "analysis_reserve_seconds": 180, "projected_remaining_seconds": 2214.6212430720047, "remaining_budget_seconds": 5249.288906, "check_outcomes_used": false}`.

| Check | Opponent | Candidate chooser | Status | BO3 result | Match seconds |
|---|---|---:|---|---:|---:|
| U-check-000 | BC34-Hydrapple | True | DONE | 0 | 7.93 |
| U-check-001 | G30-OceanMix | True | DONE | 0 | 3.36 |
| U-check-002 | League32-Hydrapple | True | DONE | 0 | 3.97 |
| U-check-003 | G30-OceanMix | False | DONE | 1 | 3.59 |
| U-check-004 | League32-Hydrapple | False | DONE | 1 | 4.23 |
| U-check-005 | BC34-Hydrapple | False | DONE | 1 | 4.33 |

## Strength results

Primary: released package versus final league32 Hydrapple. Secondary: BC34 Hydrapple and G30 OceanMix. For each pairing,96 independent fresh BO3,48 per initial chooser stratum. Draws are nonwins. Twenty thousand whole-match bootstrap replicates independently within pairing/initial-chooser strata; the same resampled match indices produce BO3 and first-game intervals. Later-game and actual-starter summaries are descriptive. Failed later rounds retain any valid first game. Missing outcomes are shown rather than replaced. No checkpoint/temperature selection or outcome-based extensions.

| Opponent | BO3 W/L/D/failed | BO3 win rate,95% interval | First-game W/L/D/failed | First-game win rate,95% interval |
|---|---|---|---|---|
| BC34-Hydrapple | {'win': 95, 'loss': 1, 'draw': 0, 'failure': 0} | 98.96% [96.88,100.00] | {'win': 93, 'loss': 3, 'draw': 0, 'failure': 0} | 96.88% [92.71,100.00] |
| League32-Hydrapple | {'win': 91, 'loss': 5, 'draw': 0, 'failure': 0} | 94.79% [89.58,98.96] | {'win': 90, 'loss': 6, 'draw': 0, 'failure': 0} | 93.75% [88.54,97.92] |
| G30-OceanMix | {'win': 93, 'loss': 3, 'draw': 0, 'failure': 0} | 96.88% [92.71,100.00] | {'win': 90, 'loss': 6, 'draw': 0, 'failure': 0} | 93.75% [88.54,97.92] |

Recommendation: **usable stronger opponent in the primary local matchup only.**
Primary gate: True. Requires at least60% wins,95% lower bound above50%, and no correctness/runtime failure. Passing establishes only the requested local matchup, not architecture/deck causality or broad diversity benefit.

Compact complete outcomes, all opponent/chooser cells, actual starters, draws, finish reasons and failures are in official/ and strength/ and matches.jsonl.gz when complete. Detailed successful histories are retained only for the six official checks; bounded first failure examples stay private. Official checks are excluded from strength statistics.

## Runtime, limits and costs

Current official wrapper timing behavior is recorded rather than assumed. Cold startup/model load, per-decision and match inference, IPC time and parent/child RSS are measured. In addition to wrapper checks, require full BO3 candidate inference plus cold startup below570 seconds, respecting a conservative shared600-second allowance and the original30-second valve. Only two CPUs per match worker; candidate keeps the submitted two Torch threads. Exact hosted container digest is unavailable; this is a local official-environment test, not hosted validation.

Costs: `{"at_utc": "[removed]", "accounting_start_utc": "[removed]", "deadline_utc": "[removed]", "task_elapsed_seconds": 2618.037809, "runner_elapsed_seconds": 702.47991, "fixed_pipeline_measured_CPU_seconds": 47.395280421, "variable_agent_measured_CPU_seconds": 3666.109810168, "measured_CPU_core_hours": 1.0321785822444445, "runner_child_CPU_seconds": 3713.5515480000004, "prelaunch_pipeline_CPU": "Not fully metered; unavailable, not zero. Preparation included in task elapsed.", "official_attempts": 6, "strength_attempts": 288, "GPU_seconds": 0, "training_steps": 0, "uploads": 0}`. Fixed/Pipeline costs cover acquisition, metadata/adapter compatibility and the six checks. Variable/Agent costs cover only the288-match benchmark. CPU core-hours are measured process CPU seconds divided by3600; wall time includes preparation from [removed]. Absolute deadline [removed]. GPU zero, training zero, uploads/submissions zero.

Failure/blocker: none currently recorded.

The ordinary local runner enforces deadlines and generates this report without model calls. Existing policies, checkpoints, active submissions and observations are preserved. No further training or experiment is launched automatically.
