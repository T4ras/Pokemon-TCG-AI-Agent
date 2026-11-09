> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Unown adaptation: fully local text mapping

Updated [removed]. State: **PARTIAL**.

Local lexical mapping did not pass the frozen gate. Independent inference/training interfaces remain verified; no usable current-pool checkpoint was exported.

**Current authorization:** separately billed embedding, Codex and other model APIs are forbidden. The old US$0.10 allowance is revoked. No credentials are sought. Interactive Sol/Astra work uses the user subscription; experiments use local or explicitly allocated Kaggle compute. This run is CPU-only and offline.

## Requests and preserved evidence

**Paid model requests sent across this adaptation:0. Actual API cost:$0.00.** The15 prepared batches were never sent. The initial Windows worker failed before HTTP/credential lookup; the later offline check found no key. There is no spend ledger or successful external-response cache. Historical records and checksums are preserved; their allowance/credential requirements are superseded, not current instructions.

Accounting: `{"at_utc": "[removed]", "paid_model_API_requests_total": 0, "actual_cost_USD": 0, "current_allowance_USD": 0, "credential_lookup_performed_this_run": false, "evidence_sha256": {"experiments/astra-unown-current-pool-adaptation-v2-001/embedding-startup-failure.json": "[artifact fingerprint omitted]", "experiments/astra-unown-current-pool-adaptation-v2-001/embedding-worker-repair.json": "[artifact fingerprint omitted]", "experiments/astra-unown-current-pool-adaptation-v2-001/embedding-status.json": "[artifact fingerprint omitted]"}, "explanation": "15 prepared batches were never sent. Initial failure preceded HTTP and credential lookup; subsequent check found no key. No paid ledger or successful response cache exists."}`. configs/compute-usage-policy.json and AGENTS.md prohibit new paid requests and credential searches. The old embedding/supervisor/credential-check entry points are retired, with originals archived privately and SHA256 recorded in the revocation manifest. No Codex/other model API was called by any runner. Earlier public GitHub metadata retrieval is distinct from model-API use; this continuation performs zero external requests.

## Available assets and local method

The already completed bounded search found no release assets and only reduced embed_pca arrays plus builders in the pinned tree. Local array keys confirm there are no saved means/loadings/whitening scales or raw vectors in those available assets. available-assets.json preserves the evidence; acquisition and benchmarks were not repeated.

The single registered local feature family concatenates word unigram/bigram and char_wb3–5gram TFIDF,8192 maximum features per channel, sublinear term frequencies and equal1/sqrt(2) channel weights. Vocabulary and IDF fit only on training references during validation/test. Separate centered ridge regressions predict released card128/attack64/skill64 coordinates, followed by L2 normalization. There is no PCA refit, model download, remote embedding or assumption of equivalent semantic quality.

The frozen v2 semantic/text groups and70/15/15 assignment are reused unchanged. Duplicates/near-duplicates share groups. Regularization remains0.0001,.001,.01,.1,1; select on validation mean cosine, larger value on exact ties. Test predictions are made once after selection. The mapping is compared against normalized training-target mean and nearest local-feature train reference; local lexical geometry is explicitly different from the original embedding geometry.

Acceptance is unchanged: per kind mean cosine≥.85, median≥.90,p10≥.65; median L2≤.45,p90≤.85; least train-similar test quartile mean cosine≥.75; mean-cosine gains≥.10 over training mean and≥.02 over nearest reference. Predicted real semantic rows must be finite and nonzero. Conservative thresholds seek fidelity before new-row use, not evidence of playing strength. No threshold/feature search is expanded after results.

| Kind | Lambda | Test mean/median/p10 cosine | Median/p90 L2 | Least-similar quartile mean cosine | Mean / nearest baselines | Gate |
|---|---:|---|---|---:|---|---|
| cards | 0.01 | 0.6074/0.6196/0.3883 | 0.8722/1.1061 | 0.5289 | -0.0416/0.3139 | False |
| attacks | 0.1 | 0.8579/0.8739/0.6965 | 0.5021/0.7791 | 0.7616 | -0.0368/0.5865 | False |
| skills | 0.01 | 0.6859/0.7064/0.5011 | 0.7663/0.9989 | 0.6204 | -0.0523/0.4075 | False |

Every held-out identity, nearest train-reference identity/similarity, cosine/L2 error, failed gate and least-similar-quartile membership is retained in vector-results.json and {kind}-heldout-identities.json. Tails are not removed from the metrics. Original text reconstruction may differ from historical text bytes; local word/character features also have limited semantic generalization, so this approach is not presumed to match the released text channel.

## Frozen-policy impact

The same preregistered first six retained official Unown histories are replayed in lexical order with full tracking prefixes, original game resets and forced-observation updates. Only held-out old text rows are replaced in a diagnostic copy; the actual released model and all its old vectors stay unchanged. At most512 eligible conditional decisions are compared. Native deals are not regenerated.

Required:≥24 non-tie decisions,≥97% non-tie greedy agreement, mean probability TV≤.02,p95 TV≤.10,100% legal-mask consistency. Original top-logit gap≤1e−6 is a deterministic tie; exact argmax and acceptable top-set agreement are both reported. Policy replay is off-policy diagnostic input, not a strength estimate. Policy impact runs even if vector fidelity fails and cannot rescue a failed primary fidelity gate.

Policy-impact result: `{"n": 512, "nontie_n": 484, "ties": 28, "nontie_action_agreement": 0.9917355371900827, "exact_action_agreement": 0.990234375, "tie_acceptable_agreement": 0.990234375, "mean_TV": 0.009612973034919405, "p95_TV": 0.04761656876653433, "legal_consistency": true, "label": "Fixed retained histories are off-policy diagnostic inputs; not a strength result", "passed": true}`. Detailed decisions and TV/JS changes are in text-policy-impact.json when complete. Missing results are NOT_RUN, never passes.

## Preservation, inference and training interfaces

Original submitted checkpoint/runtime/deck are immutable. The append-only identity mappings, regenerated static rows, dynamic encoder/model/options/tokens/tool handling, legal attached-card references, current effects audit, actor/critic isolation and fresh Python optimizer/recovery path from v2 are retained. No BC restart or substantive RL is performed.

Reuse verified evidence:12 synthetic/retained tests;8 completed old-supported CPU BO3 against the frozen Unown parent; two32-decision disposable updates plus a resumed reference with max parameter difference0. Existing old-row preservation and inference/training checks remain valid for the unchanged architecture. Those tests do not establish new-card text quality. No completed benchmark or old-supported match is repeated here.

New local mapper tests: `{"exit_code": 0, "native_attempts": 0, "network_requests": 0, "source_sha256": "[artifact fingerprint omitted]"}`. Candidate disposition: `{"exported": false, "reason": "Local mapping did not pass unchanged fidelity/policy gates"}`. If any vector or policy gate fails, no final all-reference refit, no current-pool checkpoint export and no conditional native checks occur. Provisional diagnostic predictions are not model text replacements.

Only after all gates pass may the selected local models be refit on all eligible old references and used solely for appended new/changed semantic rows. Export checks every old learned/table/text prefix byte-for-byte. This is an approximate local adaptation, not original text-channel recovery. If accepted, only the eight previously frozen unused current-pool native checks run; total native attempts remain≤32, failures count, and no replacements are added.

Unresolved temporal/ability features and parent representation limits remain documented in the v2 effect/interface evidence. The official engine supplies legality/outcomes; no tactical rule, search, new reward, temperature tuning or architecture/strength sweep is introduced. A technically accepted local mapping is not proof of playing strength.

## Costs and disposition

Costs: `{"start_utc": "[removed]", "end_utc": "[removed]", "runner_wall_seconds": 12.977126575999996, "total_elapsed_with_preparation_seconds": 688.208343, "measured_CPU_seconds": 20.737394433, "measured_CPU_core_hours": 0.005760387342499999, "average_used_CPU_cores": 1.597995851512453, "peak_parent_RSS_kib": 889856, "peak_child_RSS_kib": 338860, "new_native_attempts": 0, "prior_native_attempts": 8, "total_native_attempts": 8, "new_optimizer_updates": 0, "GPU_session_hours": 0, "paid_model_requests": 0, "actual_model_API_cost_USD": 0, "network_requests": 0, "credential_lookups": 0, "uploads": 0, "submissions": 0, "pipeline_fixed": "All work", "agent_variable_substantive": 0, "limitations": "WSL runner and children CPU measured; Windows supervisor and implementation CPU not fully metered; prior v2 costs separate"}`. All local text transfer, fidelity/policy checks and optional compatibility checks are Pipeline/Fixed. Agent/Variable substantive training0. Paid API cost0, GPU0, uploads/submissions0. Prior measured v2 CPU core-hours0.026742 are separate; current wall includes preparation. Windows supervisor and implementation CPU are not fully metered.

A failed local mapping closes this particular registered lexical approach at these thresholds. It does not prove every local semantic model must fail, and it does not justify paid APIs or automatic further sweeps. Independent inference/training work remains available; readiness depends on an accepted semantic representation and remaining compatibility checks. No training forecast is fabricated without an accepted candidate.

The ordinary persistent runner owns exclusive intent, PID, network prohibition, deadline, diagnostics, hashes, costs, report and terminal state. There are no routine model calls or automatic retries. No deployed/competition policy is changed.
