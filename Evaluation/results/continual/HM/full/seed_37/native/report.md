# Hawkes Memory Tree CL Evaluation

- Data root: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2`
- Benchmark manifest: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2/benchmark_manifest.json`
- Persistent-law averages follow `persistent_regimes`; diagnostic transient anchors are reported in the OOD section.
- Checkpoints: `/home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_37/checkpoint`
- Checkpoint tasks: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Variants: `['frozen/full', 'fast_adapt/full', 'online_write/full']`
- Task-test protocol: checkpoint `task_k_best` is evaluated on `D_k^test`; `D_{k+1}^test` is also evaluated before learning when available.
- Frozen anchors: `enabled`.

## Checkpoint topology

| checkpoint | nodes | leaves | max depth | memory rows |
|---:|---:|---:|---:|---:|
| task_00_best | 1 | 1 | 0 | 12 |
| task_01_best | 1 | 1 | 0 | 36 |
| task_02_best | 1 | 1 | 0 | 60 |
| task_03_best | 1 | 1 | 0 | 108 |
| task_04_best | 1 | 1 | 0 | 12 |
| task_05_best | 1 | 1 | 0 | 48 |
| task_06_best | 1 | 1 | 0 | 60 |
| task_07_best | 1 | 1 | 0 | 72 |
| task_08_best | 1 | 1 | 0 | 72 |
| task_09_best | 1 | 1 | 0 | 72 |

## Current-task test quality

| checkpoint | variant | NLL/event | accuracy | time MAE |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.054314 | 0.3243 | 1.2009 |
| task_00_best | fast_adapt/full | 3.054222 | 0.3248 | 1.2007 |
| task_00_best | online_write/full | 3.054227 | 0.3248 | 1.3551 |
| task_01_best | frozen/full | 3.296787 | 0.2476 | 1.2691 |
| task_01_best | fast_adapt/full | 3.296680 | 0.2462 | 1.2690 |
| task_01_best | online_write/full | 3.296680 | 0.2462 | 1.3342 |
| task_02_best | frozen/full | 2.554551 | 0.3950 | 0.8923 |
| task_02_best | fast_adapt/full | 2.554661 | 0.3940 | 0.8923 |
| task_02_best | online_write/full | 2.554661 | 0.3940 | 0.9645 |
| task_03_best | frozen/full | 3.020899 | 0.3269 | 1.1654 |
| task_03_best | fast_adapt/full | 3.021064 | 0.3278 | 1.1654 |
| task_03_best | online_write/full | 3.021064 | 0.3278 | 1.2591 |
| task_04_best | frozen/full | 2.910834 | 0.2665 | 1.0781 |
| task_04_best | fast_adapt/full | 2.910845 | 0.2660 | 1.0781 |
| task_04_best | online_write/full | 2.910845 | 0.2660 | 1.1333 |
| task_05_best | frozen/full | 2.987961 | 0.3827 | 1.1688 |
| task_05_best | fast_adapt/full | 2.988048 | 0.3832 | 1.1687 |
| task_05_best | online_write/full | 2.988048 | 0.3832 | 1.2727 |
| task_06_best | frozen/full | 2.777778 | 0.3433 | 0.9659 |
| task_06_best | fast_adapt/full | 2.777778 | 0.3423 | 0.9659 |
| task_06_best | online_write/full | 2.777783 | 0.3423 | 1.0411 |
| task_07_best | frozen/full | 3.065891 | 0.3035 | 1.1429 |
| task_07_best | fast_adapt/full | 3.065952 | 0.3010 | 1.1429 |
| task_07_best | online_write/full | 3.065956 | 0.3010 | 1.2417 |
| task_08_best | frozen/full | 3.055616 | 0.3174 | 1.1218 |
| task_08_best | fast_adapt/full | 3.055690 | 0.3165 | 1.1218 |
| task_08_best | online_write/full | 3.055688 | 0.3165 | 1.1952 |
| task_09_best | frozen/full | 3.211081 | 0.1914 | 1.1218 |
| task_09_best | fast_adapt/full | 3.210887 | 0.1933 | 1.1217 |
| task_09_best | online_write/full | 3.210882 | 0.1933 | 1.1938 |

## Continual retention and anchors

CLNLL averages only anchor laws whose first occurrence is no later than the checkpoint. Forgetting is current NLL minus the best NLL since that law was first seen.

| checkpoint | variant | CLNLL | avg forgetting | seen laws |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.024039 | 0.000000 | 1 |
| task_01_best | frozen/full | 3.274803 | 0.175557 | 2 |
| task_02_best | frozen/full | 3.147222 | 0.232434 | 3 |
| task_03_best | frozen/full | 3.170271 | 0.272130 | 3 |
| task_04_best | frozen/full | 3.102344 | 0.190064 | 4 |
| task_05_best | frozen/full | 3.155577 | 0.219107 | 5 |
| task_06_best | frozen/full | 3.187972 | 0.251502 | 5 |
| task_07_best | frozen/full | 3.115032 | 0.168091 | 6 |
| task_08_best | frozen/full | 3.111395 | 0.165818 | 6 |
| task_09_best | frozen/full | 3.106279 | 0.140624 | 7 |

## Stage-level plasticity

`adaptation_gain_nll = pre_nll - post_nll`; positive means the current task improved after training.

| task | shift type | variant | pre NLL | post NLL | adaptation gain |
|---:|---|---|---:|---:|---:|
| task_00 | initial | frozen/full | NA | 3.054314 | NA |
| task_01 | novel | frozen/full | 3.880081 | 3.296787 | 0.583294 |
| task_02 | novel | frozen/full | 3.090546 | 2.554551 | 0.535996 |
| task_03 | exact_recurrence | frozen/full | 3.443289 | 3.020899 | 0.422391 |
| task_04 | near_recurrence | frozen/full | 3.456868 | 2.910834 | 0.546035 |
| task_05 | specialization | frozen/full | 3.296202 | 2.987961 | 0.308241 |
| task_06 | transient_anomaly | frozen/full | 3.198350 | 2.777778 | 0.420572 |
| task_07 | merge | frozen/full | 3.251478 | 3.065891 | 0.185588 |
| task_08 | long_gap_recurrence | frozen/full | 3.062923 | 3.055616 | 0.007307 |
| task_09 | mixture | frozen/full | 3.389053 | 3.211081 | 0.177972 |

## Transfer and adaptation contract

FWT compares the same task test set from the fixed C_init and the pre-task checkpoint. Only genuinely unseen persistent-law tasks enter the average; recurrence tasks remain diagnostics.

- Average FWT: `0.017451` (available).

| protocol | task | K min | K max | adaptation AUC | status |
|---|---:|---:|---:|---:|---|
| fast_adapt | 1 | 0 | 32 | -0.001197 | available |
| fast_adapt | 2 | 0 | 32 | -0.003635 | available |
| fast_adapt | 3 | 0 | 32 | -0.009907 | available |
| fast_adapt | 4 | 0 | 32 | -0.002613 | available |
| fast_adapt | 5 | 0 | 32 | -0.004671 | available |
| fast_adapt | 6 | 0 | 32 | -0.006727 | available |
| fast_adapt | 7 | 0 | 32 | -0.008954 | available |
| fast_adapt | 8 | 0 | 32 | -0.003504 | available |
| fast_adapt | 9 | 0 | 32 | -0.001560 | available |
| online_write | 1 | 0 | 32 | -0.001206 | available |
| online_write | 2 | 0 | 32 | -0.003635 | available |
| online_write | 3 | 0 | 32 | -0.009907 | available |
| online_write | 4 | 0 | 32 | -0.002613 | available |
| online_write | 5 | 0 | 32 | -0.004656 | available |
| online_write | 6 | 0 | 32 | -0.006727 | available |
| online_write | 7 | 0 | 32 | -0.008954 | available |
| online_write | 8 | 0 | 32 | -0.003503 | available |
| online_write | 9 | 0 | 32 | -0.001559 | available |

| returned law | first task | return task | shift | RRR | status |
|---|---:|---:|---|---:|---|
| A_1 | 0 | 3 | exact_recurrence | NA | not_available_missing_first_gain |
| A_1 | 0 | 8 | long_gap_recurrence | NA | not_available_missing_first_gain |

## HM-specific state

Topology action counts come from committed transaction events; no leaf-count difference is inferred.

| task | nodes | leaves | episodic rows | episodic bytes | semantic bytes | split | merge | prune | NISE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 12 | 195019 | 1692 | 0 | 0 | 0 | 0.169515 |
| 1 | 1 | 1 | 36 | 402962 | 1692 | 2 | 2 | 0 | 0.444552 |
| 2 | 1 | 1 | 60 | 604444 | 1692 | 3 | 3 | 0 | 0.619832 |
| 3 | 1 | 1 | 108 | 1053308 | 1692 | 6 | 5 | 0 | 0.467394 |
| 4 | 1 | 1 | 12 | 191697 | 1692 | 7 | 6 | 0 | 0.339609 |
| 5 | 1 | 1 | 48 | 503068 | 1692 | 8 | 7 | 0 | 0.399306 |
| 6 | 1 | 1 | 60 | 596704 | 1692 | 8 | 7 | 0 | 0.482029 |
| 7 | 1 | 1 | 72 | 705064 | 1692 | 8 | 7 | 0 | 0.330204 |
| 8 | 1 | 1 | 72 | 712808 | 1692 | 9 | 7 | 0 | 0.321525 |
| 9 | 1 | 1 | 72 | 721384 | 1692 | 10 | 8 | 0 | 0.329655 |

## Schedule-driven diagnostics

For NLL differences, positive values mean the right-hand condition has lower NLL; definitions come from task shift_type and paired controls in the protocol.

| metric | formula | value | expected |
|---|---|---:|---|
| A_1_retention_before_task3 | `L_2,A_1 - L_0,A_1` | 0.390792 | near_zero_or_negative |
| A_1_exact_recurrence_recovery_task3 | `L_2,A_1 - L_3,A_1` | 0.440734 | positive |
| B_1_near_recurrence_impact_task5 | `L_5,B_1 - L_4,B_1` | 0.280538 | near_zero_or_negative |
| B_prime_1_near_recurrence_gain_task5 | `L_4,B_prime_1 - L_5,B_prime_1` | -0.274789 | positive |
| A_1_specialization_impact_task6 | `L_6,A_1 - L_5,A_1` | 0.261562 | near_zero_or_negative |
| A_2_specialization_gain_task6 | `L_5,A_2 - L_6,A_2` | -0.250365 | positive |
| controls_task_06_no_transient_transient_control_adaptation_task6 | `L_{5,6}^control - L_{6,6}^control` | 0.447762 | positive |
| controls_task_06_no_transient_transient_excess_adaptation_task6 | `(L_{5,6} - L_{6,6}) - (L_{5,6}^control - L_{6,6}^control)` | -0.027190 | near_zero_or_negative |
| A_1_long_gap_reference_task8 | `L_7,A_1 - L_0,A_1` | -0.017946 | near_zero_or_negative |
| A_1_long_gap_recurrence_recovery_task8 | `L_7,A_1 - L_8,A_1` | 0.009200 | positive |
| E_1_B_1_mixture_adaptation_task9 | `P_9^pre - P_9^post` | 0.177972 | positive |
| A_1_rrr_task3 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |
| A_1_rrr_task8 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |

## Hawkes law recovery

NISE compares causal total and representative event-type intensity curves against `ground_truth/regimes.npz`.

| checkpoint | variant | regime | NISE | sequences |
|---:|---|---|---:|---:|
| task_00_best | frozen/full | A_1 | 0.169515 | 64 |
| task_01_best | frozen/full | A_1 | 0.589404 | 64 |
| task_01_best | frozen/full | B_1 | 0.299701 | 64 |
| task_02_best | frozen/full | A_1 | 1.045759 | 64 |
| task_02_best | frozen/full | B_1 | 0.780951 | 64 |
| task_02_best | frozen/full | C_1 | 0.032788 | 64 |
| task_03_best | frozen/full | A_1 | 0.090185 | 64 |
| task_03_best | frozen/full | B_1 | 0.742973 | 64 |
| task_03_best | frozen/full | C_1 | 0.569024 | 64 |
| task_04_best | frozen/full | A_1 | 0.516952 | 64 |
| task_04_best | frozen/full | B_1 | 0.111903 | 64 |
| task_04_best | frozen/full | B_prime_1 | 0.120331 | 64 |
| task_04_best | frozen/full | C_1 | 0.609252 | 64 |
| task_05_best | frozen/full | A_1 | 0.023949 | 64 |
| task_05_best | frozen/full | A_2 | 0.019252 | 64 |
| task_05_best | frozen/full | B_1 | 0.578254 | 64 |
| task_05_best | frozen/full | B_prime_1 | 0.610682 | 64 |
| task_05_best | frozen/full | C_1 | 0.764391 | 64 |
| task_06_best | frozen/full | A_1 | 0.493306 | 64 |
| task_06_best | frozen/full | A_2 | 0.510715 | 64 |
| task_06_best | frozen/full | B_1 | 0.656007 | 64 |
| task_06_best | frozen/full | B_prime_1 | 0.657678 | 64 |
| task_06_best | frozen/full | C_1 | 0.092440 | 64 |
| task_07_best | frozen/full | A_1 | 0.142511 | 64 |
| task_07_best | frozen/full | A_2 | 0.137942 | 64 |
| task_07_best | frozen/full | A_merge | 0.138788 | 64 |
| task_07_best | frozen/full | B_1 | 0.576399 | 64 |
| task_07_best | frozen/full | B_prime_1 | 0.597112 | 64 |
| task_07_best | frozen/full | C_1 | 0.388471 | 64 |
| task_08_best | frozen/full | A_1 | 0.119331 | 64 |
| task_08_best | frozen/full | A_2 | 0.118208 | 64 |
| task_08_best | frozen/full | A_merge | 0.116989 | 64 |
| task_08_best | frozen/full | B_1 | 0.578506 | 64 |
| task_08_best | frozen/full | B_prime_1 | 0.601393 | 64 |
| task_08_best | frozen/full | C_1 | 0.394723 | 64 |
| task_09_best | frozen/full | A_1 | 0.229844 | 64 |
| task_09_best | frozen/full | A_2 | 0.237594 | 64 |
| task_09_best | frozen/full | A_merge | 0.223916 | 64 |
| task_09_best | frozen/full | B_1 | 0.368982 | 64 |
| task_09_best | frozen/full | B_prime_1 | 0.388441 | 64 |
| task_09_best | frozen/full | C_1 | 0.400215 | 64 |
| task_09_best | frozen/full | E_1 | 0.458591 | 64 |

## Unseen/OOD novelty control

Transient/unseen anchors are reported separately and never enter CLNLL, average forgetting, or average seen-task NLL.

| checkpoint | variant | regime | NLL/event | accuracy | time MAE |
|---:|---|---|---:|---:|---:|
| task_00_best | frozen/full | B_1 | 3.846862 | 0.0737 | 1.4604 |
| task_00_best | frozen/full | C_1 | 3.256636 | 0.1632 | 1.0735 |
| task_00_best | frozen/full | B_prime_1 | 3.800119 | 0.0791 | 1.4590 |
| task_00_best | frozen/full | A_2 | 3.153937 | 0.3028 | 1.2113 |
| task_00_best | frozen/full | X_transient | 3.174220 | 0.2527 | 1.1272 |
| task_00_best | frozen/full | A_merge | 3.012378 | 0.3217 | 1.1441 |
| task_00_best | frozen/full | E_1 | 3.420736 | 0.1498 | 1.1016 |
| task_01_best | frozen/full | C_1 | 3.101315 | 0.1894 | 0.9976 |
| task_01_best | frozen/full | B_prime_1 | 3.155639 | 0.2451 | 1.2017 |
| task_01_best | frozen/full | A_2 | 3.466744 | 0.1148 | 1.2254 |
| task_01_best | frozen/full | X_transient | 3.387509 | 0.1292 | 1.1228 |
| task_01_best | frozen/full | A_merge | 3.362078 | 0.1494 | 1.1647 |
| task_01_best | frozen/full | E_1 | 3.306436 | 0.1344 | 1.0442 |
| task_02_best | frozen/full | B_prime_1 | 3.435737 | 0.0803 | 1.1881 |
| task_02_best | frozen/full | A_2 | 3.511362 | 0.0935 | 1.2108 |
| task_02_best | frozen/full | X_transient | 3.441483 | 0.0757 | 1.1040 |
| task_02_best | frozen/full | A_merge | 3.407669 | 0.0912 | 1.1319 |
| task_02_best | frozen/full | E_1 | 3.407292 | 0.0901 | 1.0027 |
| task_03_best | frozen/full | B_prime_1 | 3.540253 | 0.0829 | 1.3143 |
| task_03_best | frozen/full | A_2 | 3.100233 | 0.3080 | 1.1725 |
| task_03_best | frozen/full | X_transient | 3.115732 | 0.2460 | 1.0844 |
| task_03_best | frozen/full | A_merge | 2.965778 | 0.3222 | 1.1068 |
| task_03_best | frozen/full | E_1 | 3.306833 | 0.1476 | 1.0229 |
| task_04_best | frozen/full | A_2 | 3.358905 | 0.2272 | 1.2583 |
| task_04_best | frozen/full | X_transient | 3.312899 | 0.2148 | 1.1596 |
| task_04_best | frozen/full | A_merge | 3.250887 | 0.2488 | 1.1956 |
| task_04_best | frozen/full | E_1 | 3.298347 | 0.1564 | 1.0563 |
| task_05_best | frozen/full | X_transient | 3.084800 | 0.3010 | 1.0848 |
| task_05_best | frozen/full | A_merge | 2.941450 | 0.3583 | 1.1082 |
| task_05_best | frozen/full | E_1 | 3.331336 | 0.1023 | 1.0162 |
| task_06_best | frozen/full | X_transient | 3.241600 | 0.1447 | 1.0925 |
| task_06_best | frozen/full | A_merge | 3.196583 | 0.1482 | 1.1227 |
| task_06_best | frozen/full | E_1 | 3.334131 | 0.0950 | 1.0083 |
| task_07_best | frozen/full | X_transient | 3.095899 | 0.2693 | 1.0876 |
| task_07_best | frozen/full | E_1 | 3.263423 | 0.1419 | 1.0196 |
| task_08_best | frozen/full | X_transient | 3.096126 | 0.2731 | 1.0857 |
| task_08_best | frozen/full | E_1 | 3.272070 | 0.1356 | 1.0153 |
| task_09_best | frozen/full | X_transient | 3.110520 | 0.2686 | 1.0764 |

The full per-law anchor table contains `42` rows; see `law_metrics.csv` for start/current/best NLL, forgetting, and BWT.

## Summary plots

### Current Task Quality

![Current Task Quality](plots/current_task_quality.png)

### Continual Learning

![Continual Learning](plots/continual_learning.png)

### Anchor Nll Heatmap

![Anchor Nll Heatmap](plots/anchor_nll_heatmap.png)

### Stage Adaptation Gain

![Stage Adaptation Gain](plots/stage_adaptation_gain.png)

### Hawkes Law Recovery

![Hawkes Law Recovery](plots/hawkes_law_recovery.png)

### Topology And Memory Growth

![Topology And Memory Growth](plots/topology_and_memory_growth.png)

### Special Case Metrics

![Special Case Metrics](plots/special_case_metrics.png)


## Output files

- `task_metrics.csv`: checkpoint × task-test × variant metrics.
- `control_metrics.csv`: checkpoint × manifest-declared matched-control metrics.
- `anchor_metrics.csv`: checkpoint × frozen-anchor × variant metrics.
- `continual_summary.csv`: current quality, CLNLL, forgetting, and checkpoint topology.
- `law_metrics.csv`: per-law CLNLL support, forgetting, and BWT terms.
- `stage_metrics.csv`: pre/post task-test adaptation gains.
- `fwt_metrics.csv`: protocol-scoped forward transfer with fixed scratch baseline when supplied.
- `adaptation_points.csv` / `adaptation_summary.csv`: fixed-query K-indexed adaptation curves and normalized AUC.
- `rrr_metrics.csv`: protocol-driven exact/long-gap recurrence retention ratios.
- `hm_state.csv`: HM-only memory, topology transaction counts, and NISE.
- `cl_metrics.json`: canonical CL metric contract shared with baseline runners.
- `anchor_nll_matrix.csv`: paper-style wide checkpoint × regime NLL matrix.
- `special_case_metrics.csv`: schedule-driven recurrence, near-recurrence, specialization, mixture, and transient diagnostics.
- `intensity_metrics.csv` / `intensity_summary.csv`: causal intensity-curve NISE and checkpoint summaries.
- `ood_metrics.csv`: transient/unseen-anchor novelty control, excluded from CL averages.
- `intensity_curves/`: optional total-plus-representative-type GT/prediction plots.
- `plots/`: current quality, CLNLL/forgetting, anchor heatmap, adaptation, NISE, and topology figures.
- `checkpoint_tree.csv`: leaf/node counts and checkpoint memory sizes.
- `summary.json`: machine-readable copy of the complete evaluation manifest.
- `event_predictions.csv`: optional rows selected by `--event-prediction-scope` (default `none`; legacy `--save-event-predictions` means `all`).
- `protocol_comparison.csv`: one comparison table across the selected protocols.
- `frozen/`: strict frozen anchor matrix, CLNLL, forgetting, BWT, and law metrics.
- `fast_adapt/`: official fixed-query adaptation curve plus event-exposure diagnostics; it is excluded from CL aggregates.
- `online_write/`: independent fixed-query write curve plus event-exposure diagnostics; each eval set starts from a fresh checkpoint load.
