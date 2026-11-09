> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Belief observer provenance repair

State: **SOFTWARE_CHECKS_PASSED**. Updated [removed].

The separate `unown_belief_provenance` package adds SDK-verified physical replacements for Palafin106 (deck), Ogre’s Mask1090 (discard) and Transformation Tome1415 (discard). The native `TransformProc` exchanges two same-owner physical card references; it does not create a new card or rename an existing physical token. The event compiler and independent fact observer now emit the corresponding movements. PF and RB reuse these shared changes. Old experiments, sources and predictions remain unchanged; the playing policy is unchanged.

For deck replacements, the departing board copy is placed at the deck top exactly as native push_back specifies; the later explicit shuffle clears that order constraint. For discard replacements, both copies retain their identities and serials. Physical-card conservation, known locations, duplicate callbacks and deck copy lower bounds are checked.

Unknown effect profiles, contradictory or missing source provenance, same-serial identity changes and card IDs outside the37-list universe raise explicit unsupported-provenance/library errors. In particular, an unfamiliar ID is not silently counted as an original-deck copy and is not automatically labeled generated. A genuine unseen starting deck is another possible explanation.

This is **not general generated/foreign-card support**. No verified native card-generation event was established by this focused inspection. Foreign cards with familiar IDs also require explicit provenance; the ID guard alone cannot detect those. When the shared fact model cannot establish origin, stopping that analysis is safer than inventing a posterior. No claim of safe fallback is made for such cases.

Verification: **99 tests passed**, including SDK-derived synthetic replacements, identity/serial conservation, known top ordering, duplicate events, explicit unknown-origin guards, the existing16 saved legal-history fact/compiler regressions, and previous focused tests. The newly supported replacement cases have not been exercised in fresh native games. This repair is a separately versioned implementation for later validation, not a modified result of the frozen RB experiment.

## Current-hand probability versus full history

For a known remaining populationN withc copies ofX and an exchangeable unknown hand of sizeh, the exact probability is `1 - C(N-c,h)/C(N,h)`. One copy among40 candidates and five hand slots gives12.5%. Known locations, ordered cards, hidden prizes and historical constraints can change the conditional allocation law. BaselineA already performs current-allocation combinatorics under its declared assumptions; the full PF instead attempts to preserve the full history-conditioned target.

The old reconstruction samples full initial arrangements, replays the legal prefix, and guides chance-draw blocks using exact valid/total combinatorial corrections. Finite reconstruction can fail even when an allocation exists. Those failures do not show that a simpler current-allocation calculation is infeasible. At a new game, fresh locations can be initialized without replaying old locations when the required deck-probability summary is available; the old unresolved-state coupling is an implementation limitation.

Costs: test suite18.92s elapsed (XML retains the exact timing); setup/source inspection CPU was not fully metered. Zero new native matches, predictive replay experiments, held-out access, GPU, training, uploads, submissions or paid model API calls. No automatic follow-up.

Evidence: registration (historical/private artifact omitted), native semantics (historical/private artifact omitted), source manifest (historical/private artifact omitted), completion (historical/private artifact omitted).
