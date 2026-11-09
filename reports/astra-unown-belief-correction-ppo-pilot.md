> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief-conditioned policy-correction PPO pilot

State: **COMPLETE**. Updated [removed].

Both frozen targets and the fixed1,440fresh CPU matches completed. Retrieval timing-file collision repaired; all prior failure evidence retained. No automatic follow-up or submission.

The **state-only arm is complete and frozen**:251,042 fresh decisions,1,280 completed BO3 and8,032 accepted updates. The belief continuation starts from **171,258 updated decisions and5,467 accepted updates**, with the exact saved AdamW and Python/NumPy/Torch/CUDA RNG state. Its endpoint is the first complete cohort reaching at least250,000 cumulative fresh decisions; a small whole-cohort overshoot is reported. No completed cohort/update is repeated.

## Current authorization and stopping rule

The user explicitly instructed: “continue the run to reach250,000decisions” and withdrew the budget constraints. This supersedes the study’s nine-hour aggregate GPU cap,5.5-hour belief cap,14CPU-hour cap,36-hour elapsed cap and conservative throughput admission gate. The prior gates and failures remain historical evidence. This run **does not stop because a forecast exceeds an old time allowance**. It stops at its fixed decision target or an actual correctness, resource or provider boundary.

Kaggle’s [official notebook documentation](https://www.kaggle.com/docs/notebooks) lists a12-hour CPU/GPU session limit, verified [removed]. The job uses that technical window and reserves time for checkpoint/export; it ends as soon as its target is complete. GPU time and both allocated T4s remain accounted. No paid model APIs, competition submission, public upload, new seed, reward change, classifier change or state-arm retraining is authorized.

## Recovery and shuffle repair

Execution008 retained171,258trained decisions and5,467accepted updates, then stopped during original cohort56 on `Conditional order unsupported`. Its memory guard repair worked: maximum sampled proportional charge17.80GiB, three safe worker recycles. The failing legal observation contains two explicit opponent deck shuffles. The observer cleared top/bottom caches but never cleared its unsupported-order flag. The run-local repair restores support at the legal opponent shuffle while keeping physical-copy domains, count bounds and unsupported checks for later events. An own-player shuffle does not reset opponent support. The frozen baseline conditions on certified facts; it does not estimate likelihoods of opponent decisions.

The original full observer history was not retained. Replaying the actual legal failure delta on explicitly synthetic starting facts reproduces the stale-flag behavior; this is not an exact reconstruction of the original ledger. Four preregistered retained histories and40fixed legal-fact states check supported-case feature/vector parity. The first real BO3 integration is part of the registered training cohort; no extra native diagnostic matches are added. Future observer failures retain a bounded legal-fact snapshot.

The exact step5,467model, AdamW and RNG boundary is adopted. Optimization resumes at phase54, original schedule cohort56. The five completed interrupted-cohort rollouts (618decisions) remain private, but381of420original inference batches lack peer inputs. They are not a validated complete cohort. All16IDs receive labelled retries, and all prior960attempts remain counted; at most1,088new attempts preserve the2,048attempt ceiling. No completed update is repeated.

Shared-aware24GiB memory and actual-pressure guards remain, with worker recycling only after four complete drained cohorts. The exact-length baseline learner and strict numerical gates remain. No tolerance, probability reference, reward, model architecture, deck, classifier or PPO setting changes. The state-only arm remains frozen. Historical failures and their charges remain in the earlier execution directories. Repair registration (historical/private artifact omitted), diagnostics (historical/private artifact omitted), checkpoint provenance (historical/private artifact omitted).

Focused checks: **25 passed**, covering strict reference behavior, memory-accounting/pressure failures, safe worker recycling, exact model/optimizer/RNG transplantation, original next-phase shuffle and same-backend update/resume/rollback. No new native diagnostic matches.

## Preserved scientific comparison

Original policy: trained embedding-final Unown Hydrapple, submission57834515. Classifier: frozen population-sampling MLP, seed11/update14,004. The original Hydrapple deck and conditional argmax deployment decoder stay fixed. The state correction receives a zero belief channel; the belief correction receives the complete frozen1,479-value channel. This tests the channel as a whole, not the MLP in isolation from legal-history information.

Actor: detached384-state and256-candidate features plus a128-dimensional belief projection feed768→256→1 GELU scores added to the frozen original scores. Critic: independent128-dimensional projection and detached state feed512→256→1, added to the existing value head. Trainable counts:386,561 actor,321,025 new critic,98,817 existing value head; **806,403 total**. Only the final scalar output layers start at zero. Embeddings, text tables, Transformer, original actor and classifier remain frozen. Critic gradients cannot enter the actor.

The learner samples conditional actions at temperature1. Opponents are unchanged frozen local policies on30 passing lists, weighted by top-100 prevalence n/82; the classifier keeps its full37-list prior. These are local policy/deck packages, not the leaderboard authors’ agents. Each16-match cohort has shuffled8/8 chooser assignments and fixed versions/decks within BO3. Native deals are independent.

Target:250,000 validated fresh decisions per arm, finishing complete cohorts and reporting overshoot. Endpoint, opponent/chooser order and inference seeds stay frozen. Throughput forecasts are recorded, never used to restore a withdrawn budget stop.

PPO remains four passes, logical128 and actual remainder, revised concurrent16 with exact-length baseline learner, gamma1, lambda0.95, clip0.2, entropy0.01, value coefficient2, gradclip1.5; terminal per-game rewards+1/−1/0. AdamW(.95,.999),eps1e−8,decay0,foreachFalse. New branch rates use cosine1e−4→1e−5 by fresh decisions; the existing value head stays1e−5. Pre-KL0.02 stops the cohort; post-KL0.03 restores full model/optimizer/RNG. No base-policy KL penalty. Collection probabilities, values, GAE and belief inputs remain immutable.

## Training endpoints

| Arm | State | Fresh decisions | BO3 done/attempted | Accepted updates | Effective passes |
|---|---|---:|---:|---:|---:|
| state | COMPLETE | 251042 | 1280/1280 | 8032 | 4.000 |
| belief | COMPLETE | 250958 | 400/400 | 8007 | 4.000 |

Prior960belief attempts include896trained completed matches, nine completed but untrained matches, and55failed/cancelled attempts. Only complete validated cohorts contributed updates. Saved finals are fixed endpoints, not retrospective winners. The training journals retain actor/value losses, meaningful entropy, old-policy KL, stops/rollbacks, full cycle timings and a fixed bounded sample of original-policy KL, action disagreement, correction magnitudes and critic explained variance.

## Fixed CPU evaluation

Once the belief continuation reaches its target without an unresolved correctness failure, the saved original, state and belief policies receive the unchanged1,440fresh attempts:30opponents ×16matches ×3agents, eight per chooser in every cell. All use conditional argmax. No sample increase or replacement is made after results. Intervals use20,000 independent-agent whole-BO3 bootstrap draws within deck/chooser strata, frozen population weights and a Wilson/MOVER envelope for boundary cells. Shared original draws remain shared across contrasts.

| Agent | Attempts/done/fail | Population BO3 win [95%] | First-game win [95%] |
|---|---:|---:|---:|
| original | 480/480/0 | 32.16% [26.07%, 39.89%] | 39.94% [33.31%, 47.20%] |
| state | 480/480/0 | 26.45% [21.80%, 34.20%] | 35.29% [28.96%, 42.94%] |
| belief | 480/480/0 | 36.20% [29.80%, 43.89%] | 43.37% [36.28%, 50.61%] |

| Contrast | BO3 gain [95%] | First-game gain [95%] |
|---|---:|---:|
| belief-state | 9.76% [0.82%, 18.14%] | 8.08% [-1.83%, 17.76%] |
| belief-original | 4.04% [-5.18%, 13.34%] | 3.43% [-6.40%, 13.26%] |
| state-original | -5.72% [-14.00%, 3.07%] | -4.65% [-14.10%, 4.88%] |

| Deck/family | Original BO3; game1 W/D/L | State BO3; game1 W/D/L | Belief BO3; game1 W/D/L |
|---|---|---|---|
| 0330d443fb Hydrapple scaling | 7/0/9; 7/0/9 | 4/0/12; 4/0/12 | 8/0/8; 7/0/9 |
| 074abeace8 Alakazam/Dudunsparce draw | 5/0/11; 9/0/7 | 3/0/13; 5/0/11 | 5/0/11; 5/0/11 |
| 1d18ec49ba Alakazam/Dudunsparce draw | 1/0/15; 1/0/15 | 4/0/12; 3/0/13 | 0/0/16; 2/0/14 |
| 30ebd2289f Dragapult spread | 8/0/8; 7/0/9 | 9/0/7; 8/0/8 | 11/0/5; 11/0/5 |
| 45ff39d282 Alakazam/Dudunsparce draw | 5/0/11; 5/0/11 | 5/0/11; 7/0/9 | 5/0/11; 6/0/10 |
| 470d1ef975 Alakazam/Dudunsparce draw | 3/0/13; 3/0/13 | 4/0/12; 4/0/12 | 2/0/14; 2/0/14 |
| 48a1e257d1 Kangaskhan/Crustle disruption | 10/0/6; 9/0/7 | 12/0/4; 11/0/5 | 9/0/7; 11/0/5 |
| 5087f064a4 Kangaskhan/Crustle disruption | 8/0/8; 10/0/6 | 10/0/6; 12/0/4 | 8/0/8; 9/0/7 |
| 5e2b06c453 Alakazam/Dudunsparce draw | 3/0/13; 4/0/12 | 2/0/14; 2/0/14 | 2/0/14; 4/0/12 |
| 6b1df23695 Alakazam/Dudunsparce draw | 7/0/9; 7/0/9 | 2/0/14; 6/0/10 | 3/0/13; 6/0/10 |
| 6df51865a4 Festival Lead swarm | 5/0/11; 6/0/10 | 4/0/12; 5/0/11 | 4/0/12; 6/0/10 |
| 6fbde59950 Dragapult spread | 10/0/6; 10/0/6 | 12/0/4; 9/0/7 | 7/0/9; 9/0/7 |
| 778058128d Dragapult spread | 7/0/9; 8/0/8 | 10/0/6; 9/0/7 | 12/0/4; 9/0/7 |
| 7e314fd667 Dragapult spread | 11/0/5; 11/0/5 | 9/0/7; 5/0/11 | 11/0/5; 8/0/8 |
| 7e5ce58ba3 Dragapult spread | 10/0/6; 10/0/6 | 12/0/4; 10/0/6 | 7/0/9; 7/0/9 |
| 83d6a55e94 Alakazam/Dudunsparce draw | 7/0/9; 5/0/11 | 6/0/10; 10/0/6 | 6/0/10; 8/0/8 |
| 8584c5d67f Festival Lead swarm | 5/0/11; 10/0/6 | 3/0/13; 6/0/10 | 6/0/10; 7/0/9 |
| 8b173f2c9c Alakazam/Dudunsparce draw | 6/0/10; 5/0/11 | 3/0/13; 7/0/9 | 5/0/11; 6/0/10 |
| 8d9c5c270d Dragapult spread | 7/0/9; 9/0/7 | 8/0/8; 8/0/8 | 9/0/7; 7/0/9 |
| 900f4714b9 Alakazam/Dudunsparce draw | 3/0/13; 6/0/10 | 2/0/14; 5/0/11 | 7/0/9; 9/0/7 |
| b8ff5a5d7a Dragapult spread | 8/0/8; 7/0/9 | 11/0/5; 10/0/6 | 9/0/7; 9/0/7 |
| c139b38f9a Alakazam/Dudunsparce draw | 2/0/14; 2/0/14 | 1/0/15; 4/0/12 | 1/0/15; 4/0/12 |
| d8576027b5 Ogerpon/Kangaskhan energy-switch | 13/0/3; 10/0/6 | 12/0/4; 11/0/5 | 13/0/3; 10/0/6 |
| d997ab90d8 Alakazam/Dudunsparce draw | 4/0/12; 6/0/10 | 1/0/15; 2/0/14 | 3/0/13; 6/0/10 |
| ee5ad71b52 Dragapult spread | 13/0/3; 12/0/4 | 10/0/6; 11/0/5 | 10/0/6; 9/0/7 |
| f1d283ea09 Dragapult spread | 8/0/8; 8/0/8 | 8/0/8; 8/0/8 | 14/0/2; 12/0/4 |
| f468acbbe8 Alakazam/Dudunsparce draw | 3/0/13; 4/0/12 | 1/0/15; 3/0/13 | 3/0/13; 5/0/11 |
| fbbc3d6bc7 Dragapult spread | 8/0/8; 8/0/8 | 10/0/6; 11/0/5 | 10/0/6; 8/0/8 |
| ff2c32d9b5 Alakazam/Dudunsparce draw | 1/0/15; 3/0/13 | 1/0/15; 2/0/14 | 4/0/12; 7/0/9 |
| ffbfcd1291 Festival Lead swarm | 2/0/14; 4/0/12 | 5/0/11; 3/0/13 | 3/0/13; 4/0/12 |

All chooser cells, draws, actual starters, all-game totals, termination reasons and failure denominators are retained in analysis (historical/private artifact omitted) and compact matches (historical/private artifact omitted).

This development panel overlaps training. Any gain remains a single-pilot signal, not a leaderboard upgrade or proof of a classifier-specific benefit. Uncertain differences are not equivalence, and no automatic follow-up or deployment follows.

## Costs and handoff

Prior conservative GPU charges total27,744seconds (7h42m24s), including failed attempts; both allocated-device time and provider quota deltas are separate evidence. The new session is charged in full, including startup, checks, training and export. The original study budget caps are historical and no longer govern this continuation. No paid model API has been used.

Recovery session including export: 2.447h; aggregate charged session time: 10.156h.

Accounted local CPU including prior work/preparation reserve: 12.615core-hours; evaluation state COMPLETE.

Private recovery job (historical/private artifact omitted), version1. Receipt (historical/private artifact omitted).

The ordinary owner retrieves verified artifacts, runs eligible CPU evaluation, and updates this report/status without model calls. Private rollouts and checkpoints remain outside Git. No public upload, paid model API, competition submission or automatic further replacement is included.

Evidence: recovery registration (historical/private artifact omitted), checkpoint provenance (historical/private artifact omitted), source/input manifest (historical/private artifact omitted), cold check (historical/private artifact omitted), owner state (historical/private artifact omitted).


The separately authorized Kaggle CPU replication is complete: 1,440/1,440 BO3, zero failures. It did not reproduce the primary sample's positive interval for belief versus state: +0.91 pp [-7.92, +9.83]. Belief versus original was -0.61 pp [-9.91, +8.46]. Neither sample establishes an upgrade over the original frozen Unown policy. Results remain separate. See the [completed replication report](astra-unown-belief-correction-cpu-replication.md).
