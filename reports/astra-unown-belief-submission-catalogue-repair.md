> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief submission: catalogue-support runtime repair

State: **PASSED HOSTED VALIDATION**. Submission **58133803**, validated **[removed]**. Uploaded exactly once. Active entries are **58133803** and **57837942** (initial adapted Unown). No recurring monitor is running.

## What failed

The previous submission **58069431** passed self-validation on [removed], then Kaggle invalidated it during public play with **“Invalidated due to high error rate.”** This is a later failure; the preserved initial validation receipt was accurate at its time.

Four selected public agent logs—latest three matches with missing candidate reward plus the earliest such match—show the same exception: `UnknownProvenance: OUTSIDE_LIBRARY_OR_UNMODELED_PROVENANCE`. Legally revealed cards **4, 721 and 1141** were outside the belief tracker's frozen 37-list catalogue. The observer deliberately rejected unsupported attribution to a starting deck; the deployment wrapper let that exception terminate the agent. The local strength panel and self-play validation did not cover this field-distribution failure.

The retrieved listing contains 47 public episodes, of which 34 omit the candidate reward. Missing reward alone is not an exhaustive runtime classification; the concrete diagnosis is verified by four downloaded error logs and one reproduced failed hosted replay (episode **126190446**). Full logs and replay remain private. The older worker-thread signal failure in submission 58069189 is a separate, already repaired issue.

## Minimal repair

The new deployment wrapper treats explicit unsupported belief evidence or the existing belief resource guard as an unavailable correction. It latches that state for the rest of the BO3 and uses the **exact frozen embedding-trained base policy's action logits**. It resets only for a new match. It does not merely pass a zero vector into the residual head: that would still apply a learned state correction.

Caught support/resource conditions are `UnsupportedEvent` (including `UnknownProvenance`), `OutOfLibrary`, `IncompatibleAllocation` and `OperationLimit`. Numerical, invariant and unexpected software errors still raise errors. The original observer's rejection rules and numerical tolerances remain unchanged. Partially updated unsupported facts never enter a prediction after fallback. The normal base tracker continues processing all legal observations, including forced selections and game resets.

Checkpoint, classifier, original Hydrapple/Meganium deck, text/encoder assets, legal masks, conditional action prefixes and argmax decoder are byte-identical to the previous package. All base action parameters also match the previously submitted embedding-trained base package. The only inference-source changes are the new deployment guard and its runtime wiring. Historical training source, datasets, failed packages and receipts are preserved.

This is an explicit deployment behavior change on unsupported inputs. It does not expand the classifier's deck catalogue, establish new belief accuracy, or turn earlier closed-panel strength estimates into evidence of field strength for the repaired combined policy.

## Verification

- **16 focused tests passed**, including supported-path behavior, each expected fallback condition, match latching/reset, strict numerical errors, immutable private assets, legal masking and bypass of the entire residual.
- The original package reproduces the actual hosted `card 4` failure. The repaired package processes that same legal observation history and returns a legal action.
- Ten retained supported histories replay in real worker threads: **1,793 identical actions**, **1,801 conditional forwards**, identical packets/belief vectors and **zero logit difference** from the frozen source traces.
- Exactly **six new official-environment CPU BO3** completed with **zero failures**: existing OceanMix, Abomasnow and Hydrapple lists, each with both initial chooser assignments. These are runtime checks, not strength samples. Candidate deck/policy configuration stayed fixed; native randomness is independent.
- All six legal histories replay exactly. Across the two unsupported deck families, **555 fallback actions** match the frozen base policy and **561 conditional forwards** retain exact base logits. Supported Hydrapple games did not trigger fallback. Game resets, forced-observation handling and next-match reset were checked.
- Peak recorded process RSS was **0.636 GiB**; the lowest retained official time bank was **592.90 seconds**. Every individual runtime/chooser/starter record remains linked below.

Current official `cabt.py`, `cabt.json` and native SDK hashes match the pinned runtime. The current Dockerfile matches the [removed] reviewed version. Current authenticated competition pages and upload allowance were checked before upload. The official daily limit is five submissions with the latest two successful entries active; five submissions were available before this upload. Exact hosted image identity remains unavailable; the actual hosted validation below is the compatibility evidence.

## Submission and result

Archive: **unown-belief-final-hydrapple-supported-argmax.tar.gz**, **21,290,223 bytes**.

SHA256: `[artifact fingerprint omitted]`.

[Playground submissions](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/submissions). Submission receipt UTC: **[removed]**. Hosted validation episode **126310427** completed with submission status **COMPLETE**, without a reported error. Both hosted agent logs were retrieved: **305 callbacks, zero tracebacks**. The observed active pair is **58133803 / 57837942**; invalidated 58069431 remains in history. Original G and embedding-trained baseline artifacts/history are preserved; they were already inactive before this repair.

One validation-status read hit a network timeout. The first bounded reader also exposed an SDK serialization detail: the default PENDING status is omitted from `to_dict()`. The reader was fixed to use the SDK enum directly and restarted within its original time window. Both were read-only tooling issues; no archive change, second upload or duplicate submission occurred. Those receipts are retained separately.

A final read-only snapshot at [removed] still showed COMPLETE with no reported error; no public episodes were available yet. Immediate health snapshot (historical/private artifact omitted). Initial hosted validation does not establish sustained public-play health or a strength improvement. No tuning, training, extra submission or recurring monitoring follows automatically.

## Cost and evidence

GPU **0**; training updates **0**; paid model API requests **0**; local native BO3 **6**; new strength samples **0**; uploads **1**. Local verification suite wall time: **155.08 seconds**. Recorded supported-history replay CPU: **174.54 seconds**; recorded native-check CPU: **54.95 seconds**. These CPU subsets exclude other replay/tests, editing and controller/network work. The total wall time from registration through this report was **19.8 minutes**, including hosted waiting.

Registration (historical/private artifact omitted) · Repair plan (historical/private artifact omitted) · Log selection (historical/private artifact omitted) · Hosted diagnosis (historical/private artifact omitted) · Verification (historical/private artifact omitted) · Six-match schedule (historical/private artifact omitted) · Package manifest (historical/private artifact omitted) · Source manifest (historical/private artifact omitted) · Upload preflight (historical/private artifact omitted) · Receipt (historical/private artifact omitted) · Validation (historical/private artifact omitted) · Costs (historical/private artifact omitted) · Previous repair report (historical/private artifact omitted).
