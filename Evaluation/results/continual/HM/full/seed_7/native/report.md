# Hawkes Memory Tree CL Evaluation

- Data root: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2`
- Benchmark manifest: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2/benchmark_manifest.json`
- Persistent-law averages follow `persistent_regimes`; diagnostic transient anchors are reported in the OOD section.
- Checkpoints: `/home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_7/checkpoint`
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
| task_03_best | 1 | 1 | 0 | 76 |
| task_04_best | 1 | 1 | 0 | 120 |
| task_05_best | 1 | 1 | 0 | 144 |
| task_06_best | 1 | 1 | 0 | 36 |
| task_07_best | 5 | 3 | 2 | 60 |
| task_08_best | 5 | 3 | 2 | 60 |
| task_09_best | 3 | 2 | 1 | 108 |

## Current-task test quality

| checkpoint | variant | NLL/event | accuracy | time MAE |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.054394 | 0.3253 | 1.2030 |
| task_00_best | fast_adapt/full | 3.054288 | 0.3243 | 1.2028 |
| task_00_best | online_write/full | 3.054296 | 0.3243 | 1.3597 |
| task_01_best | frozen/full | 3.335791 | 0.2193 | 1.2932 |
| task_01_best | fast_adapt/full | 3.335606 | 0.2179 | 1.2930 |
| task_01_best | online_write/full | 3.335606 | 0.2179 | 1.3493 |
| task_02_best | frozen/full | 2.555381 | 0.3915 | 0.8919 |
| task_02_best | fast_adapt/full | 2.555501 | 0.3910 | 0.8919 |
| task_02_best | online_write/full | 2.555501 | 0.3910 | 0.9630 |
| task_03_best | frozen/full | 3.086482 | 0.2201 | 1.1765 |
| task_03_best | fast_adapt/full | 3.086561 | 0.2206 | 1.1765 |
| task_03_best | online_write/full | 3.086707 | 0.2206 | 1.2498 |
| task_04_best | frozen/full | 2.901127 | 0.2734 | 1.0801 |
| task_04_best | fast_adapt/full | 2.901145 | 0.2725 | 1.0801 |
| task_04_best | online_write/full | 2.901161 | 0.2725 | 1.1501 |
| task_05_best | frozen/full | 3.049265 | 0.3254 | 1.1711 |
| task_05_best | fast_adapt/full | 3.049361 | 0.3254 | 1.1711 |
| task_05_best | online_write/full | 3.049361 | 0.3254 | 1.2375 |
| task_06_best | frozen/full | 2.746166 | 0.3659 | 0.9655 |
| task_06_best | fast_adapt/full | 2.746120 | 0.3659 | 0.9655 |
| task_06_best | online_write/full | 2.746120 | 0.3659 | 1.0471 |
| task_07_best | frozen/full | 3.104168 | 0.2694 | 1.1506 |
| task_07_best | fast_adapt/full | 3.104136 | 0.2694 | 1.1505 |
| task_07_best | online_write/full | 3.104154 | 0.2694 | 1.2575 |
| task_08_best | frozen/full | 3.081628 | 0.2773 | 1.1242 |
| task_08_best | fast_adapt/full | 3.081663 | 0.2764 | 1.1242 |
| task_08_best | online_write/full | 3.081680 | 0.2764 | 1.1986 |
| task_09_best | frozen/full | 3.057942 | 0.3000 | 1.1106 |
| task_09_best | fast_adapt/full | 3.057946 | 0.3000 | 1.1105 |
| task_09_best | online_write/full | 3.057964 | 0.3000 | 1.1798 |

## Continual retention and anchors

CLNLL averages only anchor laws whose first occurrence is no later than the checkpoint. Forgetting is current NLL minus the best NLL since that law was first seen.

| checkpoint | variant | CLNLL | avg forgetting | seen laws |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.024283 | 0.000000 | 1 |
| task_01_best | frozen/full | 3.292404 | 0.169919 | 2 |
| task_02_best | frozen/full | 3.151699 | 0.220655 | 3 |
| task_03_best | frozen/full | 3.159353 | 0.228309 | 3 |
| task_04_best | frozen/full | 3.067487 | 0.147149 | 4 |
| task_05_best | frozen/full | 3.089359 | 0.131974 | 5 |
| task_06_best | frozen/full | 3.309252 | 0.354219 | 5 |
| task_07_best | frozen/full | 3.144193 | 0.174532 | 6 |
| task_08_best | frozen/full | 3.142211 | 0.176765 | 6 |
| task_09_best | frozen/full | 3.221296 | 0.269450 | 7 |

## Stage-level plasticity

`adaptation_gain_nll = pre_nll - post_nll`; positive means the current task improved after training.

| task | shift type | variant | pre NLL | post NLL | adaptation gain |
|---:|---|---|---:|---:|---:|
| task_00 | initial | frozen/full | NA | 3.054394 | NA |
| task_01 | novel | frozen/full | 3.871806 | 3.335791 | 0.536015 |
| task_02 | novel | frozen/full | 3.061029 | 2.555381 | 0.505649 |
| task_03 | exact_recurrence | frozen/full | 3.451349 | 3.086482 | 0.364866 |
| task_04 | near_recurrence | frozen/full | 3.442471 | 2.901127 | 0.541345 |
| task_05 | specialization | frozen/full | 3.250513 | 3.049265 | 0.201247 |
| task_06 | transient_anomaly | frozen/full | 3.057909 | 2.746166 | 0.311742 |
| task_07 | merge | frozen/full | 3.435616 | 3.104168 | 0.331448 |
| task_08 | long_gap_recurrence | frozen/full | 3.104069 | 3.081628 | 0.022441 |
| task_09 | mixture | frozen/full | 3.415455 | 3.057942 | 0.357514 |

## Transfer and adaptation contract

FWT compares the same task test set from the fixed C_init and the pre-task checkpoint. Only genuinely unseen persistent-law tasks enter the average; recurrence tasks remain diagnostics.

- Average FWT: `-0.001327` (available).

| protocol | task | K min | K max | adaptation AUC | status |
|---|---:|---:|---:|---:|---|
| fast_adapt | 1 | 0 | 32 | -0.001189 | available |
| fast_adapt | 2 | 0 | 32 | -0.004325 | available |
| fast_adapt | 3 | 0 | 32 | -0.010418 | available |
| fast_adapt | 4 | 0 | 32 | -0.004979 | available |
| fast_adapt | 5 | 0 | 32 | -0.004053 | available |
| fast_adapt | 6 | 0 | 32 | -0.008321 | available |
| fast_adapt | 7 | 0 | 32 | -0.008720 | available |
| fast_adapt | 8 | 0 | 32 | -0.003003 | available |
| fast_adapt | 9 | 0 | 32 | -0.000788 | available |
| online_write | 1 | 0 | 32 | -0.001197 | available |
| online_write | 2 | 0 | 32 | -0.004325 | available |
| online_write | 3 | 0 | 32 | -0.010418 | available |
| online_write | 4 | 0 | 32 | -0.004959 | available |
| online_write | 5 | 0 | 32 | -0.004049 | available |
| online_write | 6 | 0 | 32 | -0.008321 | available |
| online_write | 7 | 0 | 32 | -0.008720 | available |
| online_write | 8 | 0 | 32 | -0.002999 | available |
| online_write | 9 | 0 | 32 | -0.000793 | available |

| returned law | first task | return task | shift | RRR | status |
|---|---:|---:|---|---:|---|
| A_1 | 0 | 3 | exact_recurrence | NA | not_available_missing_first_gain |
| A_1 | 0 | 8 | long_gap_recurrence | NA | not_available_missing_first_gain |

## HM-specific state

Topology action counts come from committed transaction events; no leaf-count difference is inferred.

| task | nodes | leaves | episodic rows | episodic bytes | semantic bytes | split | merge | prune | NISE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 12 | 190539 | 1692 | 0 | 0 | 0 | 0.173882 |
| 1 | 1 | 1 | 36 | 418450 | 1692 | 2 | 2 | 0 | 0.475876 |
| 2 | 1 | 1 | 60 | 634076 | 1692 | 3 | 3 | 0 | 0.640328 |
| 3 | 1 | 1 | 76 | 767484 | 1692 | 6 | 4 | 0 | 0.469822 |
| 4 | 1 | 1 | 120 | 1171844 | 1692 | 9 | 4 | 1 | 0.308497 |
| 5 | 1 | 1 | 144 | 1348692 | 1692 | 12 | 6 | 1 | 0.302259 |
| 6 | 1 | 1 | 36 | 383632 | 1692 | 13 | 7 | 1 | 0.738112 |
| 7 | 5 | 3 | 60 | 610584 | 3868 | 16 | 8 | 1 | 0.370741 |
| 8 | 5 | 3 | 60 | 636696 | 3868 | 18 | 10 | 1 | 0.355587 |
| 9 | 3 | 2 | 108 | 1058876 | 2780 | 21 | 12 | 3 | 0.565089 |

## Schedule-driven diagnostics

For NLL differences, positive values mean the right-hand condition has lower NLL; definitions come from task shift_type and paired controls in the protocol.

| metric | formula | value | expected |
|---|---|---:|---|
| A_1_retention_before_task3 | `L_2,A_1 - L_0,A_1` | 0.398300 | near_zero_or_negative |
| A_1_exact_recurrence_recovery_task3 | `L_2,A_1 - L_3,A_1` | 0.375431 | positive |
| B_1_near_recurrence_impact_task5 | `L_5,B_1 - L_4,B_1` | 0.148904 | near_zero_or_negative |
| B_prime_1_near_recurrence_gain_task5 | `L_4,B_prime_1 - L_5,B_prime_1` | -0.141699 | positive |
| A_1_specialization_impact_task6 | `L_6,A_1 - L_5,A_1` | 0.403987 | near_zero_or_negative |
| A_2_specialization_gain_task6 | `L_5,A_2 - L_6,A_2` | -0.373944 | positive |
| controls_task_06_no_transient_transient_control_adaptation_task6 | `L_{5,6}^control - L_{6,6}^control` | 0.349682 | positive |
| controls_task_06_no_transient_transient_excess_adaptation_task6 | `(L_{5,6} - L_{6,6}) - (L_{5,6}^control - L_{6,6}^control)` | -0.037940 | near_zero_or_negative |
| A_1_long_gap_reference_task8 | `L_7,A_1 - L_0,A_1` | 0.026316 | near_zero_or_negative |
| A_1_long_gap_recurrence_recovery_task8 | `L_7,A_1 - L_8,A_1` | 0.025544 | positive |
| E_1_B_1_mixture_adaptation_task9 | `P_9^pre - P_9^post` | 0.357514 | positive |
| A_1_rrr_task3 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |
| A_1_rrr_task8 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |

## Hawkes law recovery

NISE compares causal total and representative event-type intensity curves against `ground_truth/regimes.npz`.

| checkpoint | variant | regime | NISE | sequences |
|---:|---|---|---:|---:|
| task_00_best | frozen/full | A_1 | 0.173882 | 64 |
| task_01_best | frozen/full | A_1 | 0.576262 | 64 |
| task_01_best | frozen/full | B_1 | 0.375490 | 64 |
| task_02_best | frozen/full | A_1 | 1.099924 | 64 |
| task_02_best | frozen/full | B_1 | 0.782250 | 64 |
| task_02_best | frozen/full | C_1 | 0.038810 | 64 |
| task_03_best | frozen/full | A_1 | 0.219212 | 64 |
| task_03_best | frozen/full | B_1 | 0.732410 | 64 |
| task_03_best | frozen/full | C_1 | 0.457845 | 64 |
| task_04_best | frozen/full | A_1 | 0.464179 | 64 |
| task_04_best | frozen/full | B_1 | 0.099220 | 64 |
| task_04_best | frozen/full | B_prime_1 | 0.111967 | 64 |
| task_04_best | frozen/full | C_1 | 0.558621 | 64 |
| task_05_best | frozen/full | A_1 | 0.127474 | 64 |
| task_05_best | frozen/full | A_2 | 0.128955 | 64 |
| task_05_best | frozen/full | B_1 | 0.346129 | 64 |
| task_05_best | frozen/full | B_prime_1 | 0.375106 | 64 |
| task_05_best | frozen/full | C_1 | 0.533632 | 64 |
| task_06_best | frozen/full | A_1 | 0.949300 | 64 |
| task_06_best | frozen/full | A_2 | 0.962513 | 64 |
| task_06_best | frozen/full | B_1 | 0.882875 | 64 |
| task_06_best | frozen/full | B_prime_1 | 0.872668 | 64 |
| task_06_best | frozen/full | C_1 | 0.023204 | 64 |
| task_07_best | frozen/full | A_1 | 0.224303 | 64 |
| task_07_best | frozen/full | A_2 | 0.217011 | 64 |
| task_07_best | frozen/full | A_merge | 0.219699 | 64 |
| task_07_best | frozen/full | B_1 | 0.597171 | 64 |
| task_07_best | frozen/full | B_prime_1 | 0.612522 | 64 |
| task_07_best | frozen/full | C_1 | 0.353740 | 64 |
| task_08_best | frozen/full | A_1 | 0.167290 | 64 |
| task_08_best | frozen/full | A_2 | 0.165334 | 64 |
| task_08_best | frozen/full | A_merge | 0.164279 | 64 |
| task_08_best | frozen/full | B_1 | 0.619944 | 64 |
| task_08_best | frozen/full | B_prime_1 | 0.638474 | 64 |
| task_08_best | frozen/full | C_1 | 0.378201 | 64 |
| task_09_best | frozen/full | A_1 | 0.810654 | 64 |
| task_09_best | frozen/full | A_2 | 0.826730 | 64 |
| task_09_best | frozen/full | A_merge | 0.824335 | 64 |
| task_09_best | frozen/full | B_1 | 0.272750 | 64 |
| task_09_best | frozen/full | B_prime_1 | 0.306436 | 64 |
| task_09_best | frozen/full | C_1 | 0.795565 | 64 |
| task_09_best | frozen/full | E_1 | 0.119152 | 64 |

## Unseen/OOD novelty control

Transient/unseen anchors are reported separately and never enter CLNLL, average forgetting, or average seen-task NLL.

| checkpoint | variant | regime | NLL/event | accuracy | time MAE |
|---:|---|---|---:|---:|---:|
| task_00_best | frozen/full | B_1 | 3.838401 | 0.0739 | 1.4658 |
| task_00_best | frozen/full | C_1 | 3.254494 | 0.1496 | 1.0781 |
| task_00_best | frozen/full | B_prime_1 | 3.791924 | 0.0789 | 1.4640 |
| task_00_best | frozen/full | A_2 | 3.153496 | 0.3040 | 1.2139 |
| task_00_best | frozen/full | X_transient | 3.173158 | 0.2532 | 1.1306 |
| task_00_best | frozen/full | A_merge | 3.012785 | 0.3250 | 1.1466 |
| task_00_best | frozen/full | E_1 | 3.416092 | 0.1493 | 1.1064 |
| task_01_best | frozen/full | C_1 | 3.073029 | 0.2456 | 1.0062 |
| task_01_best | frozen/full | B_prime_1 | 3.210828 | 0.2194 | 1.2320 |
| task_01_best | frozen/full | A_2 | 3.463700 | 0.1136 | 1.2103 |
| task_01_best | frozen/full | X_transient | 3.376492 | 0.1287 | 1.1085 |
| task_01_best | frozen/full | A_merge | 3.349236 | 0.1474 | 1.1498 |
| task_01_best | frozen/full | E_1 | 3.293178 | 0.1214 | 1.0414 |
| task_02_best | frozen/full | B_prime_1 | 3.441685 | 0.0829 | 1.1926 |
| task_02_best | frozen/full | A_2 | 3.518604 | 0.0947 | 1.2098 |
| task_02_best | frozen/full | X_transient | 3.451806 | 0.0735 | 1.1039 |
| task_02_best | frozen/full | A_merge | 3.415537 | 0.0927 | 1.1311 |
| task_02_best | frozen/full | E_1 | 3.416061 | 0.0888 | 1.0020 |
| task_03_best | frozen/full | B_prime_1 | 3.526844 | 0.0744 | 1.2524 |
| task_03_best | frozen/full | A_2 | 3.165653 | 0.2187 | 1.1814 |
| task_03_best | frozen/full | X_transient | 3.149321 | 0.1713 | 1.0887 |
| task_03_best | frozen/full | A_merge | 3.034607 | 0.2336 | 1.1150 |
| task_03_best | frozen/full | E_1 | 3.284431 | 0.2132 | 1.0254 |
| task_04_best | frozen/full | A_2 | 3.315024 | 0.1916 | 1.2508 |
| task_04_best | frozen/full | X_transient | 3.270185 | 0.1599 | 1.1548 |
| task_04_best | frozen/full | A_merge | 3.204713 | 0.2148 | 1.1843 |
| task_04_best | frozen/full | E_1 | 3.254186 | 0.1591 | 1.0589 |
| task_05_best | frozen/full | X_transient | 3.102742 | 0.2624 | 1.0847 |
| task_05_best | frozen/full | A_merge | 3.000082 | 0.3212 | 1.1136 |
| task_05_best | frozen/full | E_1 | 3.225762 | 0.1273 | 1.0021 |
| task_06_best | frozen/full | X_transient | 3.370149 | 0.1154 | 1.0940 |
| task_06_best | frozen/full | A_merge | 3.403363 | 0.0917 | 1.1265 |
| task_06_best | frozen/full | E_1 | 3.411682 | 0.0874 | 1.0098 |
| task_07_best | frozen/full | X_transient | 3.110483 | 0.2598 | 1.0968 |
| task_07_best | frozen/full | E_1 | 3.267800 | 0.1586 | 1.0296 |
| task_08_best | frozen/full | X_transient | 3.100877 | 0.2596 | 1.0887 |
| task_08_best | frozen/full | E_1 | 3.274736 | 0.1571 | 1.0187 |
| task_09_best | frozen/full | X_transient | 3.344701 | 0.1078 | 1.0675 |

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
