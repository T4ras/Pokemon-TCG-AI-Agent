# Project GPU usage audit — [removed]

Estimated project total: **96.82 Kaggle GPU-session hours** (round to **about 97**). Under two allocated T4s per session, approximately **193.64 allocated T4 device-hours** (about **194**).

Hybrid of per-job/provider quota deltas and early measured script time. Approximate, not hardware utilization telemetry.

A GPU-session hour is one hour that a GPU notebook occupies its allocation, including CPU simulation, inference, updates, startup/export where captured. One GPU-hour in a device-count accounting convention is one accelerator allocated for one hour. Most project learners used one GPU, even where two T4s were allocated. Neither total measures time at 100% hardware utilization.

## Breakdown

| Work | GPU-session hours |
|---|---:|
| Original BC/PPO, search and representation pilot | 42.06 |
| Hydrapple league/control GPU training | 10.72 |
| Unown adaptation, fine-tuning and GPU engineering | 25.76 |
| Belief-conditioned PPO, including failures/retries | 10.16 |
| Oracle diagnostic PPO, including failed startup | 8.12 |
| **Total** | **96.82** |

## Scope and uncertainty

- Includes failed startups, failed numerical runs, repairs, GPU timing/compilation experiments and retained training attempts.
- Provider quota deltas are preferred over reserved time, approved caps, conservative forecasting reserves and launch-to-download wall time. Early runs with missing exact quota deltas use measured script durations; small provider overheads can therefore be missing.
- Provider quota is account-wide and may have delayed accounting. This is a best estimate from retained project records, not a complete provider invoice or formal error interval.
- Exact two-device receipts are not present for every early job. The 194 allocated-device-hour result is an estimate under the documented standard two-T4 session assumption.
- CPU-only Bayesian/PF/neural-classifier learning and evaluation are not counted. The later belief-conditioned and oracle policy PPO did use GPU and is included.
- Original third-party Unown pretraining is excluded. This total concerns the project adaptation/research, not training Unown from scratch.
- Through the end of lambda/entropy confirmation (before Stage B), the estimate is **41.84 GPU-session hours**. The older approximately-40-hour figure is plausible for that earlier scope, not the whole project through [removed].

## Avoided double counting

- The old compute-ledger scalar was stale; the job history and receipts were used rather than treating it as a full project total.
- The three lambda/entropy stages use one quota aggregate. G lineage totals, per-trial histories and the aggregate are not summed together.
- Hydrapple preflight/control/league uses one aggregate that already includes all three jobs.
- Per-attempt Unown and belief-correction charges exclude cumulative prior-attempt fields. Oracle uses one two-execution aggregate.
- The wrong-job monitor timeout is reconciled to the actual 21.952-second failed job, not two hours of GPU use.

## Leaderboard versus local BO3 evidence

The author reports a leaderboard-score improvement after fine-tuning. That can coexist with an inconclusive comparison on a particular fixed local BO3 panel. The earlier general statement about fine-tuning should be read as specific to those local experiments, not as a denial of the leaderboard observation. Saved snapshots record the trained adapted Unown at 1001.2 and the earlier project policy at 428.2 at an earlier check; these are different policies/times, not an isolated causal fine-tuning estimate.


Private source receipt paths and raw account records are omitted from this public summary. No weights, credentials or job-access tokens are included.
