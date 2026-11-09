# Experiment specifications

Agreed [removed] (Europe/Moscow). Canonical project:
`/path/to/ptcg-agent` in Ubuntu WSL.

These requirements govern future training and evaluation experiments. They supplement
PROMPT.md and take precedence over its earlier guidance where starting-order balance,
first-game reporting, match memory, or log retention differs. Historical experiment
records remain unchanged. This document is a specification, not a claim that the
current arena implements the requirements below.

## 1. Match format and outcomes

- Continue running official best-of-three (BO3) matches, including the official
  early stopping and draw behavior. Keep decks fixed for the whole match.
- Report both BO3 match win/draw/loss and first-game win/draw/loss, with attempted,
  completed, and failed denominators. First-game win rate is wins divided by
  completed first games, with draws explicitly counted in that denominator.
- Every attempted match contributes one first-game attempt; never select only
  matches that reach game three or successfully finish all rounds. A valid first
  game remains usable for the first-game metric even if a later round fails;
  identify these cases separately. Record failures rather than silently excluding
  or replacing them. If retries are needed, keep the original attempt and label
  the retry; do not count retries as additional independent planned trials.
- Save each game's index, outcome, finish reason, chooser identity and decision,
  actual first player, agent versions, deck references, and completion status in a compact record. Report all-game
  totals as a secondary metric, separate from first-game and BO3 results.
- For fixed evaluation agents, preregister the number of matches per matchup and
  the uncertainty method. Report 95% intervals for first-game and BO3 win rates.
  First-game analysis must account for its balanced chooser allocation; report
  chooser/non-chooser strata and their equally weighted overall estimate. Also
  report actual first/second-player breakdowns descriptively, without forcing
  equal counts or treating the policy's starting-order choice as randomized. Use
  a stated stratified method or explicitly justify a binomial approximation.
  When using all games, account for starting-order differences, match memory,
  and BO3 early stopping; resampling whole matches is an available method.
  Training win rates across changing checkpoints are learning curves, not one
  stationary matchup estimate. Evaluate a fixed checkpoint for comparisons.

## 2. Balance game-one chooser allocation

- Preserve the game's rule that one player gets to decide who goes first in game
  one. Balance the right to choose, not the actual first/second turn order.
- For each evaluated matchup, deck pair, and fixed checkpoint comparison, allocate
  half the BO3 matches with the candidate as game-one chooser and half with the
  opponent as chooser. When both players are trainable, each is chooser in half
  their matches. Balance training within declared rollout cohorts; record any
  unmatched remainder and complete its counterpart rather than hiding an imbalance.
- Randomly shuffle the balanced chooser assignments in each block/cohort so the
  sequence is not predictable from match number. This maintains a fair 50/50
  allocation while ensuring exact balance, rather than relying on independent
  coin flips that only balance in expectation. Record the scheduler seed.
- The designated chooser's unchanged policy decides whether to go first or
  second. Do not force the actual starter, override its setup-selection answer,
  or target equal numbers of actual first/second starts. A policy that always
  chooses second is allowed to produce an imbalance in actual turn order.
- Player seat, the right to choose, and actual turn order are different quantities.
  The pinned local engine initially gives player seat 0 the choice. Assigning
  each model to that seat in half its matches implements chooser balance for
  this adapter. Verify the actual chooser from the selection observation; do
  not assume the same seat mapping for a future engine version.
- Record assigned and observed chooser, its chosen first/second answer, and the
  actual starter from game state. A chooser-allocation mismatch is a protocol
  violation; preserve the diagnostic and report it separately. A first-player
  imbalance caused by legal agent choices is not a protocol violation.
- Later games follow the official BO3 setup/selection rules with the same agent
  policy. Record their chooser and actual starter, without selectively retaining
  games or manipulating choices to balance their turn order.
- Policy seeds do not control native shuffles/coins. Record this limitation; a
  seat swap is not a paired identical deal.

## 3. Fixed policy and restricted memory within a BO3

- Each player uses the same neural network architecture, weights/checkpoint,
  inference settings, decision rules, and deck throughout one BO3. Different
  players can use different networks, but each player's version is fixed.
- No optimizer updates, checkpoint swaps, behavioral opponent fitting, or strategic
  mode changes between games of that match. Training updates may occur between
  matches; ongoing matches must retain their pinned versions. Evaluation pins
  both players' versions for the entire evaluation run.
- Preserve knowledge of legally revealed opponent deck information between games.
  The same fixed policy may condition its choices on that knowledge; different
  actions caused by new information do not mean the network changed.
- Persist only deck facts and deck-composition beliefs supported by the acting
  player's legal observations. Store evidence/provenance for these facts. Preserve
  known own-deck composition as well. Do not persist inferred opponent behavioral
  tendencies, previous outcomes, or match score as strategic adaptation features.
  They can still be recorded as experiment metadata.
- Reset hands, card locations, deck order, damage, resources, and within-game
  recurrent state at a new game. Previously seen cards are not assumed to remain
  in the opponent's new hand. Only the restricted deck-knowledge representation
  crosses the game boundary; reset it completely for a new match/opponent.
- Deduplicate repeated reveal events within a game. Across games, observed copies
  provide deck-count constraints, not new cards to add together: seeing two copies
  in each game does not prove four copies exist. Physical card identifiers must
  be scoped to their game; record lower bounds or justified beliefs, not invented
  exact counts. Absence of a reveal is not proof a card is absent from the deck.
- Full-state replay visualizations, private opponent hands, face-down prizes,
  hidden deck order, and opaque simulator fields must never populate policy input
  or persistent knowledge. Logs may contain them for private debugging only,
  through a separate diagnostic path.
- Training and inference must use the same memory updates and reset boundaries.
  Keep whole matches together in dataset splits. Validate reveal deduplication,
  count constraints, game/match resets, hidden-information exclusion, checkpoint
  immutability, and actual game-one chooser allocation before promotion.

## 4. Storage for large simulation runs

Do not retain a full serialized replay or one directory per match by default for
million-match training. Keep experiment evidence without retaining every full
visualization. Existing Phase A/B artifacts are preserved.

| Data | Future retention requirement |
| --- | --- |
| Run registration, configuration, code/SDK/card hashes, checkpoint and deck manifests | Keep for every run; store shared provenance once per run/version |
| BO3 and per-game outcomes, chooser allocation/decisions, actual starters, failure counters, throughput and resource summaries | Keep compact records in compressed shards; aggregate for reporting |
| Training observations, legal options, actions, rewards and restricted memory state | Keep in bounded rollout buffers for the algorithm's required lifetime; release after use unless the algorithm needs replay/checkpoint recovery |
| Full successful match replays | Keep a bounded reproducible sample, with outcomes/opponents covered; record selection method and inclusion rates |
| Failures and unusual behavior | Keep compact records/counts for every occurrence, plus bounded detailed examples for diagnosis |
| Long-lived imitation datasets | Save only deliberately curated, versioned legal-observation shards with grouped splits and manifests |

Initial large-run diagnostic defaults: a uniform reservoir of at most 100 successful
BO3 replays per run; detailed examples for the first 10 occurrences of each failure
signature; a combined 2 GiB cap for detailed replays and diagnostics per run. Both
count and byte caps apply. Within that cap prioritize new failure signatures over
successful examples. Preserve occurrence counts and an explicit dropped-detail
count when caps are reached. Register overrides before launching a run. Curated
training datasets have their own declared storage budget and are not constrained
by the diagnostic cap.

Avoid materializing full visualization JSON for matches that will not be retained.
The training adapter must also avoid unbounded simulator/framework history in RAM;
disabling disk writes alone is not enough. Compress/rotate metric shards rather
than creating millions of small result/request/log files. Capture ordinary worker
output at run/shard level and cap repetitive error output. Checkpoints preserve
whatever bounded buffer/rollout state is required for meaningful recovery.

Measure serialized bytes per match, disk write throughput, rollout-buffer size and
peak memory on a small run before scaling. Record estimated storage for the planned
run and enforce the declared caps. Historical data must not be deleted as part of
implementing this future retention policy.

## 5. Implementation status and evidence

As of [removed], the current Phase B arena:

- Runs BO3, swaps seats, pins handwritten policy versions, and records BO3 plus
  aggregate individual-game totals. The pinned engine initially gives seat 0
  chooser rights, so the existing seat swaps already balance initial choosers.
  Replay audit confirms V2 was chooser in 12/24 matches against each opponent.
- Saves a full replay.json, request.json, result.json and worker.log for every
  match, plus run metadata, metrics.jsonl, summary.json and report.md. Failures can
  also save partial replays and failing observations. Replays contain event logs
  and visualization data; retain them as private diagnostic assets.
- Does not yet report first-game outcomes/rates or chooser strata as separate
  metrics, explicitly validate/log chooser balance in compact records, randomly
  shuffle the balanced chooser schedule, maintain restricted cross-game deck
  knowledge, or implement the bounded storage profile above. Actual turn order
  remains an agent choice and is not required to balance. No forced-start override
  has been implemented; the earlier specification of one was corrected.

Measured existing pilot: 48 full replay files total 164,746,752 bytes, averaging
3,432,224 bytes (3.43 MB decimal) per BO3, with a maximum of 8,729,606 bytes.
At the same average, one million full BO3 replays would require about 3.43 TB
uncompressed, before request/result/log files. This is an extrapolation from one
sample deck, not a guarantee about other decks or a compression benchmark.

Next implementation work for the arena/training adapter is to satisfy sections
1-4, with focused checks and real BO3 integration evidence. Update this status
only after verifying the behavior; specification changes alone do not implement it.
