> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Offline Rao–Blackwellized belief-filter experiment

State: **COMPLETE**. Updated [removed].

Frozen allocation finished; failed analyses are retained, no replacements or automatic follow-up.

This is a frozen offline development comparison, using exactly the16 BO3 histories selected by repair003’s stability schedule, including its three shared-analysis failures. No new native matches, held-out access, policy changes, training, GPU work, uploads, submissions or paid model APIs are used. These selected histories are not a representative full-field sample. Historical implementations, failures and results remain unchanged.

## What is integrated out

Each outer particle samples a uniform initial physical-copy permutation modulo one fixed transposition. For each library deck, choose the first physical copy of the most numerous card type and the first copy of the most numerous different type, breaking count ties by card ID. Canonicalize their two positions, then retain both assignments with probability1/2. The unknown binary assignment of those two anonymous copies is integrated exactly. The rest of the permutation, subsequent chance/proposal randomness, strategic matching-copy choices and deck strata remain sampled. This is a deliberately limited one-bit collapse, not analytic integration of the whole hand/deck/prize allocation.

The conditional component is two complete serial-aware states and their normalized history weights. For branchs, letα_s be its conditional mass andf_s the unchanged explicit transition’s likelihood/proposal correction. Update the outer weight byZ=Σ_s α_s f_s and the branch masses byα_s f_s/Z. For an observed play/search ofX, f_s=1{X available} under the existing positive-availability target, followed by its existing uniform matching-copy transition; thereforeZ is the exact conditional availability probability. This retains the history event rather than resetting to baselineA. AABB with two initial hand cards followed by playingA leavesP(A still in hand)=1/5, whereas a current-allocation baseline gives1/3. Exact enumeration verifies this difference and the unequal-list posterior3/8.

All existing supported events run through the same explicit kernel inside each branch: anonymous draws, revealed chance identities and their chance factors, prize/looking/resolving movement, Basic opening predicates, known physical copies, selected plays/searches, complete returns, shuffles, snapshots and order constraints. Historical known top/bottom pins are retained per branch; only explicit shuffles clear them. No conditional exchangeability is assumed across history-constrained zones, and no new hypergeometric approximation is introduced. The existing guided-recovery combinatorics and proposal corrections remain unchanged.

Subsequent randomness is a sampled common MT word tape for both orientations. Each branch starts at the same RNG state; the outer RNG advances past the maximum number of words consumed by either branch, including rejected branches. This couples the two valid explicit transition kernels without reusing future randomness. It does not analytically integrate later random choices. Conditional orientation weights are carried through the whole game. Between games, game-specific locations/serials/order reset; deck evidence uses original prior × completed-game likelihoods × current-prefix likelihood exactly once. Unsupported or numerically unresolved compatible deck strata still block a full posterior.

Queries sum branch probabilities for unknown-copy presence and branch count moments, then outer-state and deck masses. Count variance usesE[Var(X|component)]+Var(E[X|component]); component standard deviations are never averaged. Exact conditional materialization is implemented by drawing a branch from its current weights and cloning its complete serial/order state. No automatic materialization is needed for this two-branch representation; there is no later recollapse that would erase history.

## Shared repair and correctness gate

The old stability results contain18 failed stochastic replays across three histories. Native CardImpl1198 (Crispin) selects a Basic Energy from the deck and invokes effectAttachFromEach; CardMove::AttachProc logs ATTACH and suppresses a separate movement log while moving the card from its actual source. The old observers assumed hand source. The isolated shared version tracks Crispin’s pending attachment until it occurs or the deck shuffles, and uses deck source for that one attachment. Native Drakloak120’s LookDeck2 profile is also resolved explicitly. Both changes are applied to A, ordinary PF and the candidate, and all16 shared fact/compiler histories now pass. This is a source repair, not a card-count snapshot overwrite or a new likelihood model. SDK source/binary hashes and the original/repaired traces remain in the manifest.

91 focused tests passed before benchmark launch. Exact references cover random allocation among hand/deck/prizes, observed identities, positive-availability plays/searches, opening restrictions, ordered top/bottom returns, duplicate callbacks, BO3 evidence/reset behavior, conditional materialization and total variance. The test helper’s initial failure is preserved separately. No real-history comparative scores were examined before source/configuration freeze.

## Frozen comparison and limits

A is deterministic on16 histories. Ordinary PF512, PF2048, RB256 and RB512 each replay all16 with inference seeds11/29/47:192 scheduled stochastic replays. A runs first to retain the fixed uncertain-query definition; its predictions and separate labels are consumed only after tracker inference. The stochastic order is frozen and interleaves all four configurations in history/seed blocks, using cyclic rotations and a fixed schedule seed. Actors are never loaded.

Both ordinary configurations retain repair003’s four-state deck floor, primary recovery256 at512/1024 at2048, four retries/game,16-callback retry spacing and2M/8M proposal-state-step caps. The candidate uses the same four-outer-state floor,256 recovery components, four retries and16-callback spacing at both budgets; its2M component-step cap permits at most4M explicit branch transitions. A512 candidate component can contain two explicit states, so outer particle counts are not an equal-cost unit. Candidate256 deliberately retains512’s recovery settings rather than accidentally receiving zero recovery budget.

There are at most two branches/component and60 physical copies/branch; no exact-state expansion is allowed. Each worker has a1GiB address-space limit. Update CPU is capped at30s and readout CPU at10s, equally across methods; exceeding a safe full-filter operation limit yields explicit unsupported/fallback or a retained analysis failure if certified facts cannot be trusted. Aggregate CPU is measured from the isolated process group plus the Windows owner, including controller and analysis. Limits are8 CPU-core hours and8 UTC elapsed hours from launch, at most four workers and one compute thread each. A180s finalization reserve is inside both caps. Budget termination saves partial results as BUDGET_EXHAUSTED; no retries, additional seeds or automatic follow-up occur. No power setting or sleep inhibition is required.

Primary accuracy is uncertain-hand Brier using the frozen A∈[.10,.90] query rule. Queries→decisions→games→BO3 aggregation is retained. Seeds average inside each BO3 before equal-history aggregation and20,000 paired whole-BO3 bootstrap draws. Common-support comparisons require the same selected points supported by all12 stochastic runs of that history. Report their reduced denominators. Available-score estimates are conditional if any replay fails. A operational fallback is included explicitly, with its actual readout work charged to each stochastic replay.

Practical screens against each reference: Brier improvement≥.005 at no higher CPU cost, OR at least2× lower CPU cost with Brier deterioration≤.002; non-fallback coverage may decline by at most.02. The thresholds are exploratory screens, not significance tests. A gain versus a costly PF does not establish usefulness if A remains preferable.

## Results

| Configuration | Attempted/completed/failed/scored replays | Fully seed-averaged histories | Uncertain Brier | All Brier | ECE | CPU s/BO3 | Non-fallback coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 16/16/0/16 | 16/16 | 0.17641 | 0.02307 | 0.00330 | 5.76 | 1.000 |
| PF512 | 48/48/0/48 | 16/16 | 0.19294 | 0.02527 | 0.00661 | 36.31 | 0.526 |
| PF2048 | 48/48/0/48 | 16/16 | 0.18769 | 0.02455 | 0.00631 | 89.82 | 0.588 |
| RB256 | 48/48/0/48 | 16/16 | 0.19265 | 0.02527 | 0.00679 | 79.10 | 0.444 |
| RB512 | 48/48/0/48 | 16/16 | 0.19321 | 0.02519 | 0.00616 | 83.98 | 0.431 |

| Configuration | Scored/expected points | Common-support uncertain Brier | Hand/deck/prize MAE | Across-seed probability difference | Depletions / recovery successes/attempts / unsupported | Update/readout p95 CPU ms | Peak MiB | Inclusive integration CPU s |
|---|---|---:|---|---:|---|---|---:|---:|
| A | 943/943 | 0.16919 | 0.05395/0.08110/0.04768 | — | 0 / 0/0 / 0 | 4.01 / 376.00 | 52.7 | 0.00 |
| PF512 | 2829/2829 | 0.19135 | 0.05437/0.08271/0.04780 | 0.01168 | 175 / [removed]/2615 / 0 | 489.37 / 135.27 | 77.2 | 0.00 |
| PF2048 | 2829/2829 | 0.18820 | 0.05388/0.08158/0.04714 | 0.01063 | 65 / 1160/1408 / 8 | 795.92 / 379.64 | 77.2 | 0.00 |
| RB256 | 2829/2829 | 0.19563 | 0.05434/0.08316/0.04824 | 0.01141 | 68 / 1747/1968 / 17 | 358.48 / 292.25 | 77.2 | 3327.64 |
| RB512 | 2829/2829 | 0.20108 | 0.05456/0.08325/0.04800 | 0.01061 | 64 / 1509/1756 / 16 | 785.56 / 308.05 | 77.2 | 3538.61 |

Integration CPU includes the two branch transitions and analytic readout; it is not merely the final weighted-sum instruction. Update and readout latency are separate. Seed variability includes operational A fallback.

| Candidate / reference | Brier gain (positive is better) | Paired95% interval | CPU speedup | Coverage difference | Screen |
|---|---:|---|---:|---:|---|
| RB256/A | -0.01624 | [-0.02392694445276537, -0.009469929030980108] | 0.07× | -0.556 | FAIL (16/16 histories) |
| RB256/PF512 | +0.00029 | [-0.004971478416553689, 0.005716210276489561] | 0.46× | -0.082 | FAIL (16/16 histories) |
| RB256/PF2048 | -0.00496 | [-0.010442093713779615, 3.777905552857059e-05] | 1.14× | -0.144 | FAIL (16/16 histories) |
| RB512/A | -0.01680 | [-0.026017932758410356, -0.008853956170525467] | 0.07× | -0.569 | FAIL (16/16 histories) |
| RB512/PF512 | -0.00026 | [-0.004574657888379721, 0.003887866715924369] | 0.43× | -0.095 | FAIL (16/16 histories) |
| RB512/PF2048 | -0.00552 | [-0.012797568993552402, 0.0013928102616088687] | 1.07× | -0.157 | FAIL (16/16 histories) |

### Every retained BO3

| History / descriptive family | Common supported / expected | A / PF512 / PF2048 / RB256 / RB512 uncertain Brier |
|---|---:|---|
| belief-1141bf015b746fc8b464 / Alakazam/Dudunsparce draw | 6/72 | 0.18249 / 0.18572 / 0.18571 / 0.18151 / 0.18191 |
| belief-69a6c2ea02e9367767d6 / Kangaskhan/Crustle disruption | 16/72 | 0.16063 / 0.16971 / 0.17737 / 0.17505 / 0.16396 |
| belief-ec015ac43bc922150c10 / Alakazam/Dudunsparce draw | 0/48 | 0.18876 / 0.18924 / 0.18876 / 0.18876 / 0.18876 |
| belief-81963acabc634d030a18 / Dragapult spread | 10/72 | 0.17790 / 0.18333 / 0.19316 / 0.19339 / 0.17662 |
| belief-1cd45c6c7abd8f4a1169 / Festival Lead swarm | 20/72 | 0.16921 / 0.20289 / 0.17390 / 0.19005 / 0.19678 |
| belief-a53eda37795e40183a6d / Hydrapple scaling | 5/72 | 0.16363 / 0.16357 / 0.16862 / 0.17449 / 0.17588 |
| belief-0742b0a253f99a9f5a2b / Alakazam/Dudunsparce draw | 24/72 | 0.18466 / 0.20481 / 0.21750 / 0.20683 / 0.21717 |
| belief-fc25f4cec64beff4f38a / Alakazam/Dudunsparce draw | 38/48 | 0.19488 / 0.23925 / 0.22182 / 0.25007 / 0.25539 |
| belief-279e10f8b5687a090654 / Alakazam/Dudunsparce draw | 12/72 | 0.17890 / 0.18437 / 0.17976 / 0.18544 / 0.18685 |
| belief-c92f708284cd53394f7e / Dragapult spread | 14/48 | 0.14890 / 0.15181 / 0.15523 / 0.15486 / 0.16187 |
| belief-e978106aebf9649acd00 / Alakazam/Dudunsparce draw | 17/48 | 0.16730 / 0.18300 / 0.17668 / 0.17610 / 0.17901 |
| belief-6fcc25a5483bc603edc1 / Alakazam/Dudunsparce draw | 27/48 | 0.18060 / 0.24200 / 0.18925 / 0.21522 / 0.22337 |
| belief-e986d726b9f63fc935db / Banette disruption | 19/48 | 0.16456 / 0.17029 / 0.18343 / 0.19155 / 0.17394 |
| belief-71ac6e625765a78d11bf / Festival Lead swarm | 24/55 | 0.19953 / 0.23897 / 0.21795 / 0.23209 / 0.23645 |
| belief-cad59001a907d81e2c64 / Dragapult spread | 2/48 | 0.19116 / 0.19301 / 0.19745 / 0.19055 / 0.19145 |
| belief-c0d5643180c62f9971a0 / Alakazam/Dudunsparce draw | 6/48 | 0.16946 / 0.18510 / 0.17648 / 0.17649 / 0.18189 |

### Descriptive family breakdown

| Family | A / PF512 / PF2048 / RB256 / RB512 uncertain Brier |
|---|---|
| Alakazam/Dudunsparce draw | 0.18088 / 0.20169 / 0.19199 / 0.19755 / 0.20179 |
| Banette disruption | 0.16456 / 0.17029 / 0.18343 / 0.19155 / 0.17394 |
| Dragapult spread | 0.17266 / 0.17605 / 0.18194 / 0.17960 / 0.17665 |
| Festival Lead swarm | 0.18437 / 0.22093 / 0.19592 / 0.21107 / 0.21662 |
| Hydrapple scaling | 0.16363 / 0.16357 / 0.16862 / 0.17449 / 0.17588 |
| Kangaskhan/Crustle disruption | 0.16063 / 0.16971 / 0.17737 / 0.17505 / 0.16396 |

Failures: none. Concentrated regressions (> .01 worse Brier or > .10 coverage loss within a history) appear in 6 reference comparisons; every affected ID and value is retained in concentrated-regressions.json. No difficult history is removed.

## Decision

Stop this tested one-bit collapse: it does not establish a useful accuracy/CPU advantage over both the ordinary alternatives and A. This does not rule out other, more substantial Rao–Blackwellizations. No automatic follow-up.

The tables distinguish accuracy, cost, depletion/coverage and baseline competitiveness. Any improvement applies only to this frozen limited representation on sixteen selected development histories.

## Costs

Execution before report finalization: 0.995 UTC hours and3.937 CPU-core hours. Completed 208/208 scheduled replays; unresolved intents 0. Private output 40.27MiB. Final metered totals are in completion.json. Native0; held-out0; GPU0; training0; uploads/submissions0; paid model API calls0.

Preparation: 19.9 minutes elapsed at freeze; tests and shared diagnostics CPU/timing evidence retained separately. Unmetered interactive work is not assigned invented compute hours.

Evidence: registration (historical/private artifact omitted), frozen method (historical/private artifact omitted), schedule (historical/private artifact omitted), input hashes (historical/private artifact omitted), source hashes (historical/private artifact omitted), correctness gate (historical/private artifact omitted), progress (historical/private artifact omitted), analysis (historical/private artifact omitted), cost/completion (historical/private artifact omitted). No recurring monitoring or follow-up experiment is created.
