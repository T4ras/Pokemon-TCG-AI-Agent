> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Shared BC learning-rate search

Search status: COMPLETE. GPU script wall time: 5.94 hours.

One shared network across four fixed decks; specialist fine-tuning and RL are outside this run.

| Trial | LR | Status | Completed epochs | Updates | Best macro validation NLL |
|---|---:|---|---:|---:|---:|
| initial-0 | 1e-05 | COMPLETE | 6 | 1158 | 1.440297 |
| initial-1 | 0.0001 | COMPLETE | 6 | 1158 | 1.248544 |
| initial-2 | 0.001 | COMPLETE | 6 | 1158 | 1.257279 |
| initial-3 | 0.003 | COMPLETE | 6 | 1158 | 1.187832 |
| initial-4 | 0.01 | FAILED_NUMERICAL | 2 | 437 | 1.541637 |
| refine-0 | 0.0015 | COMPLETE | 8 | 451 | 1.187832 |
| refine-1 | 0.003 | COMPLETE | 8 | 451 | 1.187832 |
| refine-2 | 0.006 | COMPLETE | 8 | 451 | 1.187832 |

Selected: initial-3, target LR 0.003, validation NLL 1.187832.
Checkpoint: `artifacts/bc-lr-search-001/v1/bc-lr-search-results/selected.pt`; SHA256 `[artifact fingerprint omitted]`.

## CPU benchmark

4800 attempted BO3; 0 execution failures; 2.46 hours wall time.

| Deck / teacher | Opponent | First-game wins | First-game 95% CI | BO3 wins | BO3 95% CI |
|---|---|---:|---|---:|---|
| sue124 | random | 72.8% | 68.1–76.8% | 82.0% | 77.8–85.3% |
| sue124 | heuristic-v1 | 63.7% | 58.9–68.2% | 68.0% | 63.2–72.3% |
| sue124 | heuristic-v2 | 67.3% | 62.5–71.6% | 69.5% | 64.8–73.7% |
| JerryChen | random | 83.2% | 79.2–86.5% | 92.5% | 89.4–94.6% |
| JerryChen | heuristic-v1 | 55.0% | 50.1–59.7% | 51.8% | 46.9–56.6% |
| JerryChen | heuristic-v2 | 58.2% | 53.4–62.9% | 61.7% | 56.9–66.3% |
| Leon Liu | random | 29.2% | 25.1–33.9% | 22.5% | 18.8–26.9% |
| Leon Liu | heuristic-v1 | 30.5% | 26.3–35.2% | 25.5% | 21.6–30.0% |
| Leon Liu | heuristic-v2 | 31.0% | 26.8–35.7% | 21.5% | 17.9–25.9% |
| YumeNeko | random | 74.2% | 69.7–78.2% | 81.5% | 77.3–84.9% |
| YumeNeko | heuristic-v1 | 53.8% | 48.9–58.5% | 52.5% | 47.6–57.3% |
| YumeNeko | heuristic-v2 | 51.5% | 46.6–56.3% | 53.2% | 48.4–58.0% |

Untouched test-set macro action NLL: 1.195551.

Intervals are pointwise, equally weighted across chooser strata; draws count as nonwins. Native shuffles/coins are not seeded. Mirror decks isolate policy quality; this does not establish leaderboard-teacher equivalence.
