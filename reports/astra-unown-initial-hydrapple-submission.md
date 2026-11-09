> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Untrained adapted Unown: original Hydrapple submission

State: **PASSED**. Updated [removed].

Submission **57837942**. Hosted status **COMPLETE**; validation **PASSED**. Error: none reported.
Receipt time UTC: [removed]. Official validation completion UTC: [removed].
[Kaggle Playground submissions](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/submissions).

## Exact candidate

The user explicitly selected the saved **untrained adapted initialization**, with the exact original released Hydrapple deck and conditional argmax. This is the current-format experimental initialization used before the new-card embedding learning experiment. Original learned network weights/old embedding rows are preserved; new identity rows use the previously frozen deterministic random initialization and new text rows use the previously frozen local text mapping. This is **not an UNK-only variant**, the untouched old-format runtime, or the trained embedding-final checkpoint. No optimizer step, new mapping fit, weight change, deck edit, search, sampler, or decoder tuning was performed for this submission.
The local text mapper failed its original vector-fidelity criteria; its use remains experimental. Technical compatibility and hosted validation do not establish original embedding quality or competitive superiority.
Checkpoint: `data/private/astra-unown-new-card-embedding-continuation-001/initialization.pt`. SHA256 `[artifact fingerprint omitted]`.
Model tensor SHA256: `[artifact fingerprint omitted]`.
Original ordered deck SHA256: `[artifact fingerprint omitted]`.
Archive: `artifacts/astra-unown-initial-hydrapple-submission-001/unown-initial-adapted-hydrapple-argmax.tar.gz` (18421400 bytes). SHA256 `[artifact fingerprint omitted]`.
The archive contains inference weights, original deck.csv, main.py, model/encoder/tracker source and pinned identity/card metadata. It excludes optimizer/RNG training state, trajectories, credentials and native SDK binaries. The existing starting-order correction, conditional legal-action prefixes, hidden-information filtering and legal deck-knowledge/reset rules are unchanged. The Unown policy has a memoryless Transformer forward; forced observations still update its legal-information tracker and forced-run count.

## Verification

Exactly **6 registered CPU official-environment BO3**, failures **0**, across three fixed opponent deck families. The shuffled schedule balances initial chooser rights3/3, scheduler seed202610311001. Native deals are independent; inference seeds do not seed native randomness. These six are packaging/runtime checks, not a strength comparison.
Source/package parity: **575 exact action matches**, **576 exact input-packet/conditional-forward comparisons**, maximum logit difference **0.0** on the same CPU backend. Complete legal prefixes include game boundaries and forced observations. Greedy inference consumed no RNG after startup. Checkpoint tensors and deck bytes remain identical.
Focused pytest result: **2 passed**. Checks cover every exported tensor, original deck bytes, exclusion of injected hidden information through a full legal history, forced observations, game boundaries and complete match reset. No extra native diagnostic matches were generated.

| Check | Opponent | Initial chooser | BO3 game wins seat0/seat1/draw | Games | Decisions | Runtime seconds | Peak RSS MiB |
|---|---|---|---|---:|---:|---:|---:|
| package-0 | BC-Sinistcha-Banette | candidate | 2/0/0 | 2 | 120 | 7.39 | 577.3 |
| package-1 | BC-Sinistcha-Banette | opponent | 0/2/0 | 2 | 93 | 5.24 | 549.2 |
| package-2 | BC-Mega-Excadrill | candidate | 2/0/0 | 2 | 89 | 5.48 | 529.2 |
| package-3 | BC-Mega-Excadrill | opponent | 0/2/0 | 2 | 79 | 5.18 | 556.1 |
| package-4 | heuristic-v2-Abomasnow | candidate | 2/1/0 | 3 | 120 | 4.32 | 533.8 |
| package-5 | heuristic-v2-Abomasnow | opponent | 0/2/0 | 2 | 74 | 3.73 | 500.4 |

Compact records include every game result, available terminal reason, observed chooser/choice and actual starter. Initial chooser was verified from the selection observation. Fixed weights/deck/decoder were retained for the entire BO3; legal cross-game deck lower bounds carry over while game-specific tracker state resets. These results are not pooled into earlier strength benchmarks.

## Allowance, slots and official environment

Immediately before upload: 4 submissions allowed today; active IDs [57834515, 57535360]. The new initial model preserves trained embedding-final **57834515**, while older league Hydrapple **57535360** leaves the two latest tracked slots. Original G and BC34 were already inactive. Observed active IDs at the latest validation check: **[57837942, 57834515]**.
Fresh authenticated official API checks confirm the competition is open, entry accepted, five daily submissions and two tracked latest slots. Fresh official wrapper/config/native-SDK downloads match the reviewed versions. The container recipe still uses the same base default and includes the reviewed CA-certificate security patch. Official published limits are197.7MiB archive,12.2GiB RAM,11.8GiB disk and2vCPUs. The exact hosted image/Torch build is not exposed; local checks do not claim to reproduce it. [Official requirements](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description).
An immutable intent binds the archive checksum and destination to one upload call. An ambiguous response is reconciled through submission records before any action; no blind duplicate is allowed. Hosted failure retrieves logs without automatic changes or resubmission. The trained and initial models may now receive field matches, but adaptive matchmaking and dated scores do not constitute a controlled learning-effect comparison. No recurring monitoring or new strength experiment is created.

## Evidence and cost

Local verification owner elapsed: 62.593912358 seconds. Sum of six worker runtimes (including load and packaged replay): 31.34 seconds. Pytest process elapsed: 5.7334715849999895 seconds. These are separate scopes, not additive all-in cost. Full lifecycle wall accounting begins at the first retained live check [removed] and is finalized in lifecycle-costs.json. **Zero GPU use, zero training updates, zero paid model API requests; exactly six new local BO3.**
Registration, manifests, schedule, official results, source-parity proofs, tests, source/SDK verification, preflight, immutable upload intent, receipt and hosted result are under `experiments/astra-unown-initial-hydrapple-submission-001/`. Private models, legal histories, logs and extracted packages remain outside Git under ignored `data/private/` and `artifacts/` paths. Previous submissions and training/benchmark evidence remain unchanged.
