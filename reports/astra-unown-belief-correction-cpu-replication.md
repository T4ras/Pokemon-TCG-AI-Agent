> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# Additional Kaggle CPU belief-correction replication

State: **COMPLETE**. Updated [removed].

The additional1,440fresh CPU BO3 and separate analysis completed; laptop primary results remain separate. No training or GPU allocation.

The laptop primary evaluation also completed all 1,440 matches with zero failures and is reported separately. This is a separately registered additional1,440fresh BO3: original/state/belief finals ×30unchanged local deck-policy opponents ×16matches,8per chooser in each cell. All use conditional argmax; no training, tuning, checkpoint selection or GPU allocation. Native deals and inference seeds are independent of the laptop sample. Results remain separate; neither sample replaces the other. The panel overlaps training.

The checkpoints and required runtime assets are reused from pinned existing private Kaggle dataset/output versions. Only notebook source/configuration is sent; no new model/data upload. Standard CPU execution has a documented12-hour session limit. Three single-threaded workers; reserve300seconds for export. If the fixed allocation cannot finish, retain it as incomplete, without shrinking, replacement matches or extra jobs. The Kaggle image is pinned to the training image; its Python/Torch CPU backend can differ from the laptop. Hosted software checks compare the reference and replication paths on the same backend.

Private CPU notebook (historical/private artifact omitted), version1. CPU-only metadata explicitly disablesGPUandTPU.

| Agent | Completed/attempted | BO3 win rate | First-game win rate |
|---|---:|---:|---:|
| original | 480/480 | 34.37% [28.43, 42.02] | 33.23% [27.36, 40.77] |
| state | 480/480 | 32.85% [27.21, 40.44] | 41.16% [33.92, 48.7] |
| belief | 480/480 | 33.77% [27.44, 41.38] | 34.45% [27.52, 42.08] |

All opponent/chooser cells, firstgames, draws, starters, failures and confidence intervals are in the linked analysis. Independent wholeBO3 bootstrap20,000draws within agent/deck/chooser strata; frozen n/82deck weights/equal chooser and Wilson/MOVER envelope.

Evidence: registration (historical/private artifact omitted), frozen manifests (historical/private artifact omitted), cold check (historical/private artifact omitted), analysis (historical/private artifact omitted), compact matches (historical/private artifact omitted), CPU accounting (historical/private artifact omitted). No paid model API, public upload or competition submission.


Completed interpretation, [removed]. Both independent samples finished all 1,440 BO3 with zero failures. These are population-weighted development results; the 30 local opponent packages overlap training.

| BO3 contrast, 95% interval | Laptop primary | Kaggle replication |
|---|---:|---:|
| belief-state | +9.76 pp [+0.82, +18.14] | +0.91 pp [-7.92, +9.83] |
| belief-original | +4.04 pp [-5.18, +13.34] | -0.61 pp [-9.91, +8.46] |
| state-original | -5.72 pp [-14.00, +3.07] | -1.52 pp [-10.17, +7.12] |

The laptop sample supported a belief-versus-state difference, but that positive interval did not replicate on Kaggle. The replication estimates a small positive difference with substantial uncertainty. Neither sample establishes improvement of belief over the original policy. This pilot does not establish a reliable strength upgrade, and it does not prove that belief conditioning cannot help. The samples remain separate; no pooling or selection of the more favorable result was performed. The pinned Kaggle Python/Torch CPU backend differs from the laptop, so this is also a cross-platform replication. Same-backend inference-path parity passed on retained probes; cross-platform bitwise identity on all native decisions was not demonstrated.

Kaggle CPU session including export: 4.138 hours (about 4h08m); completed evaluation: 4.122 hours; zero GPU usage and no training. All 1,476 exported files were checksum-verified. No further experiment, training, deployment or submission was launched.
