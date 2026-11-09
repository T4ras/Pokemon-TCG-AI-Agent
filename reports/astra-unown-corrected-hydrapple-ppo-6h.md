> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

> Long raw diagnostic payloads were omitted from this public copy; original evidence remains in the private archive.
# Corrected Hydrapple PPO six-hour experiment

Updated [removed]. State: **COMPLETE**. Original fixed evaluation completed; CPU recovery explicitly authorized [removed] after historical proxy failure.

**Decision: No demonstrated improvement at the registered resolution; report uncertainty and actual experience, no automatic sweep.** Changed weights or elapsed GPU time do not establish competence. No current global ranking is asserted. No deployment, competition submission, added seed, opponent, feature mapping, LR sweep or longer training is launched.

## Frozen initialization and protocol

Initial checkpoint SHA256 `[artifact fingerprint omitted]`,tensor hash `[artifact fingerprint omitted]`. Fresh AdamW state, no earlier PPO continuation. Both actors use the exact broad-validated starting-order implementation SHA256 `[artifact fingerprint omitted]`: physical seat while public starting order is unknown; then0 actual first/1 actual second; reset each game. Physical seats, legal observations/actions and reward bookkeeping stay unchanged. Collection stores the same corrected packets used for sampling, without changing recorded old action probabilities/log probabilities. PPO recomputation, the frozen greedy corrected initialization opponent, evaluation and diagnostics all consume corrected encoding.

Original ordered Hydrapple deck hash `[artifact fingerprint omitted]`; one fixed opponent only. Learner training uses the existing temperature1 legal conditional sampler; evaluated policies and opponent use existing conditional argmax. Text/static tables fixed; oracle/unavailable auxiliary objectives disabled. Recipe is constant AdamW1e-5,betas(.95,.999),epsilon1e-8,weight decay0,no scheduler; one PPO epoch,logical128/remainder actual-size averaging,gamma1,GAE.95,clip.2,blind-value coefficient2,entropy coefficient.01,global gradient clip1.5. Existing per-game learner-relative terminal±1/0 rewards and terminal bootstrap masks are retained; whole BO3 policies remain fixed. No anchor loss.

Registration/manifests/source hashes: experiments/astra-unown-corrected-hydrapple-ppo-6h-001. Exact unchanged optimized core SHA256 `[artifact fingerprint omitted]`; reused CPU/CUDA arithmetic proof is in reused-numerics.json. The new storage/stop wiring has focused regressions; fresh2-BO3 CPU compatibility cohort was disposable and excluded from learning. CPU parity/checkpoint-resume and the exact extracted upload archive are checked without checkout fallback. GPU startup checks128 corrected compatibility packets against scalar inference; every actual CUDA training cohort passes the unchanged chosen-LP1e-5/old-policy-KL1e-6 bounds before learning. No probability rebasing, truncation or relaxed tolerance. Kernel/payload SHA, pinned SDK/image, new public-first semantics and fixed schedules are retained separately.

## Revised stopping and actual learning

Phase forward categorical KL(old phase||current)>.02 before a minibatch ends optimization of that rollout. Post-update phase KL>.03 persists rejected diagnostics, restores model/optimizer/gradients/mode/execution and all relevant RNG exactly, and ends that rollout. Earlier accepted updates remain. Remaining optimization work is discarded; next cohort is fresh on-policy. Three consecutive complete fresh cohorts with zero accepted updates stop the run; numerical/mask/nonfinite correctness failures stop with preserved evidence. Initialization KL is diagnostic only—no0.10 stop and no hidden replacement threshold. Startup near-zero reference check is a correctness check on identical initialization, not a cumulative drift gate.

Training status/stop reason: `COMPLETE` / `registered_time_endpoint`. Completed BO3 768,optimized-cohort BO3 768,complete cohorts 48,learner conditional decisions 144612. Retained accepted updates 1147,post-update rejections 0,rollout early stops `{'phase_preupdate_drift_stop': 3}`,zero-update streak 0. Incomplete final cohort: `None`. Each complete16-match cohort has8/8 shuffled initial choosers. Incomplete experience is saved/reported and never optimized as a complete cohort.

Final tensor hash `[artifact fingerprint omitted]`,checkpoint SHA256 `[artifact fingerprint omitted]`. Recoverable latest/phase-input/current+first rollout/rejected snapshots/hourly/final states are private; exact checksums and paths are in recovery-manifest.json. No intermediate checkpoint was evaluated for winner selection.

Loss/gradient windows (unweighted raw component values; value coefficient2 and entropy coefficient-.01 applied in the verified loss): detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

Fixed corrected diagnostic set: detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

Local preparation repair: the released main silently set Torch CPU threads to2 after the registered1-thread setup. A disposable resume check detected different subsequent arithmetic despite exact restored parameters/optimizer/RNG. The new loader enforces the registered backend after construction; focused regression and bitwise uninterrupted/resumed update checks pass. Earlier local check failures and component differences are retained, no tolerances relaxed. The2 original native compatibility matches were replayed from both retained learner histories with exact sampler actions; repair added0 native matches. Probe weights/optimizer/experience were excluded from training. Details: preparation-repair.json and CPU-resume-details.json.

## Fixed fresh evaluation

Evaluation status: `COMPLETE`. Exactly256 fresh CPU BO3 per fixed checkpoint,128/128 initial chooser assignments,against frozen greedy corrected initialization. Both evaluated actors greedy. Native deals independent,not paired by inference seeds. No previous-test pooling,no intermediate selection,no outcome-based extensions. Strength evaluation is skipped for no accepted surviving change/identical tensors; correctness-failed runs do not receive a misleading strength conclusion.

| Checkpoint | Attempted/completed/failed | Equal-chooser BO3 rate | First-game rate |
|---|---:|---:|---:|
| initial | 256/256/0 | 52.34% | 48.83% |
| final | 256/256/0 | 52.34% | 57.42% |

Primary final-minus-initial equal-chooser BO3 difference: detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

This is one sustained time-bounded arm,not a controlled comparison with another training recipe. Any observed gain applies to these frozen corrected Hydrapple policies/opponent,not leaderboard/cross-deck superiority. A negative/inconclusive result remains valid; accepted updates and weight change are not learning-strength evidence by themselves. No follow-up is authorized automatically. A positive result warrants a separately registered broader fixed-opponent confirmation before investment; a negative/uncertain result warrants stopping and interpreting the reported actual experience/diagnostics,not automatically increasing time or weakening safeguards.

Per-game/chooser/actual-starter WDL and first-game conversion/recovery (with whole-match denominators and intervals): detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

## Costs and execution

GPU record: detailed per-step diagnostic/quota payload omitted from the public copy; aggregate outcomes are reported in the surrounding sections.

Measured phase timings: `{'collection_seconds': 8305.446730914004, 'checkpoint_seconds': 1099.3454122619896, 'diagnostic_seconds': 5702.309847180039, 'update_seconds': 3772.012329473985, 'transaction_snapshot_rollback_overhead_seconds': 34.252661001972285}`. Optimizer timing separates gradient/step from phase diagnostics; snapshot/rollback overhead is fixed pipeline cost. Actual remote script elapsed 19074.631189851s; completed useful BO3 per script hour `144.94644601417306` including recorded startup/checks/checkpoint/diagnostics (provider provision/export additionally in receipt). Pipeline/Fixed covers preparation,tests,cold extraction,transfers,startup,checks,diagnostics,snapshots/checkpoints/export,fixed evaluation and analysis; Agent/Variable covers native collection and PPO gradient/optimizer updates. Full accounting: costs.json,local-costs.jsonl,controller-costs.json,gpu-costs.json and remote receipt.

Measured local CPU 1.9279 core-hours,remote trainer CPU 5.237724470000001,controller CPU 0.0006987847222222222. Original registration-to-report wall time includes the failed monitor and idle gap; it is not the renewed recovery budget. The separately authorized CPU recovery has a14400-second elapsed ceiling; exact costs/deadline are recorded in cpu-evaluation-recovery and costs.json. Preparation implementation/read CPU before measured stages is not fully metered; missing measurements are unknown. Native attempts `{'compatibility': 2, 'training': 768, 'evaluation-initial': 256, 'evaluation-final': 256}`. Private files are outside Git; histories512MiB/recovery2GiB caps and compressed export bound storage. GPU paid-model API requests0/cost$0,model credential searches0,public publication0,competition submissions0.


## [removed] CPU-only recovery

The user explicitly authorized launching the previously unstarted CPU evaluation after a proxy failure. This recovery used the exact frozen512-match schedules, policies, decoders, inference seeds and original statistical rules. No intermediate checkpoint selection, replacement match, new training, GPU job or upload. New recovery window: [removed] through [removed]. Costs: {'at_utc': '[removed]', 'elapsed_seconds': 6354.467343236, 'CPU_core_hours': 1.8455171600625002, 'peak_RSS_kib': 731740, 'GPU_seconds': 0, 'paid_model_API_requests': 0}. Historical failed controller, report and accounting preserved in cpu-evaluation-recovery/preserved-local.
