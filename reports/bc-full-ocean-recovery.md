> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# OceanMix BC numerical recovery

Stage: COMPLETE. Updated UTC: [removed].

Four teachers: sue124, JerryChen, OceanMix and YumeNeko. Resume epoch seven with optimizer, RNG and data order restored. CUDA finite-loss/gradient gate precedes training. LR0.0003 without warmup, cosine to 0.00003; initially finish epoch32, adjust after two resumed epochs using measured throughput. Remaining original budget about4h23m including startup and CUDA gate, plus15min finishing margin. Permanent current-state snapshots every six epochs, rolling recovery every epoch, best shared/all-four validation and best per-deck weights.

CPU benchmarks begin after GPU completion: 100 BO3 per deck/baseline for snapshots and actual final checkpoint, then 400 fresh BO3 per pairing for validation-selected shared weights. Both players use the evaluated deck. Shuffled equal chooser rights, first-game and BO3 equal-chooser Wilson-MOVER95% intervals; native shuffles/coins unseeded.

GPU: COMPLETE, 34 completed epochs, 6562 updates, 4.11 script hours. Best validation loss: 1.[removed].

Budget adjustment: {"kind": "budget_adjustment", "epoch": 9, "previous_epochs": 32, "planned_epochs": 34, "mean_epoch_seconds": 550.8788440774999, "remaining_seconds": 14275.503049911, "cosine_anchor": {"update": 1737, "lr": 0.0002957569774353757}}

CPU epoch-012: COMPLETE; 1,200/1,200, 0 failures.

CPU epoch-018: COMPLETE; 1,200/1,200, 0 failures.

CPU epoch-024: COMPLETE; 1,200/1,200, 0 failures.

CPU epoch-030: COMPLETE; 1,200/1,200, 0 failures.

CPU epoch-034-final: COMPLETE; 1,200/1,200, 0 failures.

CPU selected-final: COMPLETE; 4,800/4,800, 0 failures.

| Checkpoint | Deck | Opponent | First-game WR (95% CI) | BO3 WR (95% CI) |
|---|---|---|---|---|
| epoch-012 | sue124 | random | 85.0% (76.3%–90.2%) | 89.0% (80.9%–93.3%) |
| epoch-012 | sue124 | heuristic-v1 | 75.0% (65.7%–81.9%) | 85.0% (76.6%–90.2%) |
| epoch-012 | sue124 | heuristic-v2 | 72.0% (62.3%–79.4%) | 77.0% (67.6%–83.7%) |
| epoch-012 | JerryChen | random | 98.0% (92.0%–99.2%) | 100.0% (95.0%–100.0%) |
| epoch-012 | JerryChen | heuristic-v1 | 75.0% (65.5%–82.0%) | 81.0% (71.9%–87.0%) |
| epoch-012 | JerryChen | heuristic-v2 | 71.0% (61.3%–78.6%) | 77.0% (67.6%–83.7%) |
| epoch-012 | OceanMix | random | 87.0% (78.5%–91.8%) | 98.0% (92.0%–99.2%) |
| epoch-012 | OceanMix | heuristic-v1 | 83.0% (74.2%–88.6%) | 96.0% (89.3%–98.0%) |
| epoch-012 | OceanMix | heuristic-v2 | 78.0% (68.7%–84.5%) | 86.0% (77.5%–91.0%) |
| epoch-012 | YumeNeko | random | 86.0% (77.4%–91.0%) | 96.0% (89.4%–98.1%) |
| epoch-012 | YumeNeko | heuristic-v1 | 71.0% (61.3%–78.6%) | 83.0% (74.1%–88.6%) |
| epoch-012 | YumeNeko | heuristic-v2 | 69.0% (59.3%–76.8%) | 74.0% (64.5%–81.1%) |
| epoch-018 | sue124 | random | 84.0% (75.5%–89.4%) | 91.0% (83.2%–94.7%) |
| epoch-018 | sue124 | heuristic-v1 | 76.0% (66.6%–82.8%) | 87.0% (78.5%–91.8%) |
| epoch-018 | sue124 | heuristic-v2 | 83.0% (74.1%–88.6%) | 90.0% (82.0%–94.0%) |
| epoch-018 | JerryChen | random | 96.0% (89.4%–98.1%) | 98.0% (92.0%–99.2%) |
| epoch-018 | JerryChen | heuristic-v1 | 75.0% (65.5%–82.0%) | 87.0% (78.6%–91.8%) |
| epoch-018 | JerryChen | heuristic-v2 | 81.0% (71.9%–87.0%) | 88.0% (79.9%–92.6%) |
| epoch-018 | OceanMix | random | 92.0% (84.3%–95.4%) | 96.0% (89.3%–98.0%) |
| epoch-018 | OceanMix | heuristic-v1 | 78.0% (68.8%–84.5%) | 87.0% (78.5%–91.8%) |
| epoch-018 | OceanMix | heuristic-v2 | 76.0% (66.7%–82.8%) | 86.0% (77.4%–91.0%) |
| epoch-018 | YumeNeko | random | 90.0% (82.0%–94.0%) | 95.0% (88.0%–97.4%) |
| epoch-018 | YumeNeko | heuristic-v1 | 76.0% (66.5%–82.9%) | 80.0% (70.8%–86.2%) |
| epoch-018 | YumeNeko | heuristic-v2 | 66.0% (56.3%–74.1%) | 82.0% (73.0%–87.8%) |
| epoch-024 | sue124 | random | 75.0% (65.5%–82.0%) | 92.0% (84.4%–95.5%) |
| epoch-024 | sue124 | heuristic-v1 | 69.0% (59.3%–76.8%) | 83.0% (74.1%–88.6%) |
| epoch-024 | sue124 | heuristic-v2 | 70.0% (60.3%–77.7%) | 86.0% (77.5%–91.0%) |
| epoch-024 | JerryChen | random | 97.0% (90.6%–98.7%) | 99.0% (93.5%–99.8%) |
| epoch-024 | JerryChen | heuristic-v1 | 80.0% (70.8%–86.2%) | 90.0% (82.0%–94.0%) |
| epoch-024 | JerryChen | heuristic-v2 | 74.0% (64.5%–81.1%) | 89.0% (80.8%–93.3%) |
| epoch-024 | OceanMix | random | 82.0% (73.0%–87.8%) | 97.0% (90.8%–99.0%) |
| epoch-024 | OceanMix | heuristic-v1 | 76.0% (66.6%–82.8%) | 92.0% (84.4%–95.5%) |
| epoch-024 | OceanMix | heuristic-v2 | 76.0% (66.6%–82.8%) | 86.0% (77.4%–91.0%) |
| epoch-024 | YumeNeko | random | 89.0% (80.8%–93.3%) | 93.0% (85.6%–96.1%) |
| epoch-024 | YumeNeko | heuristic-v1 | 75.0% (65.7%–81.9%) | 80.0% (70.8%–86.2%) |
| epoch-024 | YumeNeko | heuristic-v2 | 69.0% (59.4%–76.8%) | 75.0% (65.5%–82.0%) |
| epoch-030 | sue124 | random | 84.0% (75.2%–89.4%) | 92.0% (84.4%–95.5%) |
| epoch-030 | sue124 | heuristic-v1 | 75.0% (65.5%–82.0%) | 83.0% (74.1%–88.6%) |
| epoch-030 | sue124 | heuristic-v2 | 71.0% (61.3%–78.6%) | 79.0% (69.7%–85.4%) |
| epoch-030 | JerryChen | random | 93.0% (85.6%–96.1%) | 99.0% (93.5%–99.8%) |
| epoch-030 | JerryChen | heuristic-v1 | 84.0% (75.3%–89.4%) | 91.0% (83.3%–94.8%) |
| epoch-030 | JerryChen | heuristic-v2 | 83.0% (74.1%–88.6%) | 92.0% (84.4%–95.4%) |
| epoch-030 | OceanMix | random | 92.0% (84.4%–95.4%) | 98.0% (92.1%–99.4%) |
| epoch-030 | OceanMix | heuristic-v1 | 79.0% (69.7%–85.4%) | 85.0% (76.3%–90.2%) |
| epoch-030 | OceanMix | heuristic-v2 | 85.0% (76.3%–90.2%) | 96.0% (89.3%–98.0%) |
| epoch-030 | YumeNeko | random | 90.0% (82.0%–94.0%) | 94.0% (86.9%–96.9%) |
| epoch-030 | YumeNeko | heuristic-v1 | 65.0% (55.2%–73.3%) | 70.0% (60.4%–77.6%) |
| epoch-030 | YumeNeko | heuristic-v2 | 77.0% (67.6%–83.7%) | 79.0% (69.7%–85.4%) |
| epoch-034-final | sue124 | random | 83.0% (74.1%–88.6%) | 89.0% (80.8%–93.3%) |
| epoch-034-final | sue124 | heuristic-v1 | 74.0% (64.5%–81.1%) | 87.0% (78.5%–91.8%) |
| epoch-034-final | sue124 | heuristic-v2 | 70.0% (60.3%–77.7%) | 84.0% (75.2%–89.4%) |
| epoch-034-final | JerryChen | random | 90.0% (82.0%–94.0%) | 99.0% (93.5%–99.8%) |
| epoch-034-final | JerryChen | heuristic-v1 | 85.0% (76.4%–90.2%) | 94.0% (86.8%–96.8%) |
| epoch-034-final | JerryChen | heuristic-v2 | 81.0% (71.9%–87.0%) | 87.0% (78.5%–91.8%) |
| epoch-034-final | OceanMix | random | 86.0% (77.4%–91.0%) | 98.0% (92.0%–99.2%) |
| epoch-034-final | OceanMix | heuristic-v1 | 79.0% (69.9%–85.3%) | 93.0% (85.6%–96.1%) |
| epoch-034-final | OceanMix | heuristic-v2 | 86.0% (77.6%–91.0%) | 93.0% (85.6%–96.2%) |
| epoch-034-final | YumeNeko | random | 88.0% (79.8%–92.6%) | 98.0% (92.0%–99.2%) |
| epoch-034-final | YumeNeko | heuristic-v1 | 75.0% (65.5%–82.0%) | 78.0% (68.6%–84.5%) |
| epoch-034-final | YumeNeko | heuristic-v2 | 67.0% (57.2%–75.1%) | 85.0% (76.3%–90.2%) |
| selected-final | sue124 | random | 75.5% (71.0%–79.3%) | 87.2% (83.5%–90.0%) |
| selected-final | sue124 | heuristic-v1 | 72.0% (67.4%–76.1%) | 82.0% (77.8%–85.3%) |
| selected-final | sue124 | heuristic-v2 | 71.0% (66.3%–75.1%) | 79.5% (75.2%–83.0%) |
| selected-final | JerryChen | random | 87.0% (83.2%–89.8%) | 94.2% (91.3%–96.0%) |
| selected-final | JerryChen | heuristic-v1 | 67.2% (62.5%–71.6%) | 68.8% (64.0%–73.0%) |
| selected-final | JerryChen | heuristic-v2 | 68.0% (63.2%–72.3%) | 71.5% (66.8%–75.6%) |
| selected-final | OceanMix | random | 78.2% (73.9%–81.9%) | 86.8% (82.9%–89.6%) |
| selected-final | OceanMix | heuristic-v1 | 69.0% (64.3%–73.2%) | 77.0% (72.5%–80.7%) |
| selected-final | OceanMix | heuristic-v2 | 71.0% (66.4%–75.0%) | 83.8% (79.7%–86.9%) |
| selected-final | YumeNeko | random | 75.2% (70.7%–79.1%) | 79.5% (75.2%–83.0%) |
| selected-final | YumeNeko | heuristic-v1 | 56.2% (51.4%–61.0%) | 53.0% (48.1%–57.8%) |
| selected-final | YumeNeko | heuristic-v2 | 48.5% (43.7%–53.4%) | 53.8% (48.9%–58.5%) |

| Deck | Opponent | Prior six-epoch BO3 | Final selected BO3 | Difference (points) |
|---|---|---|---|---:|
| sue124 | random | 89.0% | 87.2% | -1.7 |
| sue124 | heuristic-v1 | 81.2% | 82.0% | +0.8 |
| sue124 | heuristic-v2 | 81.8% | 79.5% | -2.2 |
| JerryChen | random | 87.8% | 94.2% | +6.5 |
| JerryChen | heuristic-v1 | 43.5% | 68.8% | +25.2 |
| JerryChen | heuristic-v2 | 48.3% | 71.5% | +23.3 |
| OceanMix | random | 87.0% | 86.8% | -0.3 |
| OceanMix | heuristic-v1 | 81.0% | 77.0% | -4.0 |
| OceanMix | heuristic-v2 | 78.8% | 83.8% | +5.0 |
| YumeNeko | random | 82.2% | 79.5% | -2.7 |
| YumeNeko | heuristic-v1 | 67.5% | 53.0% | -14.5 |
| YumeNeko | heuristic-v2 | 65.0% | 53.8% | -11.2 |

!training curves (historical/private artifact omitted)

!gameplay curves (historical/private artifact omitted)

One training seed; confidence intervals describe fixed-checkpoint match uncertainty. Prior and current runs use different training seeds/schedules. Snapshot gameplay is diagnostic; the final checkpoint is selected by all-four validation, not gameplay. Keep laptop awake for retrieval and CPU work. Supervisor updates this report; no notification automation configured.
