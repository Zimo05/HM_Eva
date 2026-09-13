# Hawkes Memory Tree CL Evaluation

- Data root: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2`
- Benchmark manifest: `/home/xinye/Benchmark/Datasets/CL/hm_continual_v2/benchmark_manifest.json`
- Persistent-law averages follow `persistent_regimes`; diagnostic transient anchors are reported in the OOD section.
- Checkpoints: `/home/xinye/Benchmark/Evaluation/results/continual/HM/full/seed_17/checkpoint`
- Checkpoint tasks: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Variants: `['frozen/full', 'fast_adapt/full', 'online_write/full']`
- Task-test protocol: checkpoint `task_k_best` is evaluated on `D_k^test`; `D_{k+1}^test` is also evaluated before learning when available.
- Frozen anchors: `enabled`.

## Checkpoint topology

| checkpoint | nodes | leaves | max depth | memory rows |
|---:|---:|---:|---:|---:|
| task_00_best | 1 | 1 | 0 | 12 |
| task_01_best | 1 | 1 | 0 | 36 |
| task_02_best | 3 | 2 | 1 | 72 |
| task_03_best | 3 | 2 | 1 | 96 |
| task_04_best | 5 | 3 | 2 | 96 |
| task_05_best | 3 | 2 | 1 | 108 |
| task_06_best | 3 | 2 | 1 | 132 |
| task_07_best | 3 | 2 | 1 | 132 |
| task_08_best | 3 | 2 | 1 | 132 |
| task_09_best | 3 | 2 | 1 | 144 |

## Current-task test quality

| checkpoint | variant | NLL/event | accuracy | time MAE |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.054703 | 0.3262 | 1.2020 |
| task_00_best | fast_adapt/full | 3.054608 | 0.3267 | 1.2018 |
| task_00_best | online_write/full | 3.054570 | 0.3267 | 1.3575 |
| task_01_best | frozen/full | 3.302232 | 0.2380 | 1.2720 |
| task_01_best | fast_adapt/full | 3.302103 | 0.2356 | 1.2719 |
| task_01_best | online_write/full | 3.302103 | 0.2356 | 1.3355 |
| task_02_best | frozen/full | 2.581951 | 0.3665 | 0.8893 |
| task_02_best | fast_adapt/full | 2.582091 | 0.3665 | 0.8893 |
| task_02_best | online_write/full | 2.582090 | 0.3665 | 0.9459 |
| task_03_best | frozen/full | 3.115949 | 0.2149 | 1.1813 |
| task_03_best | fast_adapt/full | 3.115972 | 0.2144 | 1.1813 |
| task_03_best | online_write/full | 3.115967 | 0.2144 | 1.2415 |
| task_04_best | frozen/full | 3.049414 | 0.2060 | 1.1452 |
| task_04_best | fast_adapt/full | 3.049265 | 0.2050 | 1.1451 |
| task_04_best | online_write/full | 3.049272 | 0.2050 | 1.2470 |
| task_05_best | frozen/full | 2.985291 | 0.3867 | 1.1705 |
| task_05_best | fast_adapt/full | 2.985353 | 0.3867 | 1.1705 |
| task_05_best | online_write/full | 2.985352 | 0.3867 | 1.2732 |
| task_06_best | frozen/full | 2.783086 | 0.3375 | 0.9708 |
| task_06_best | fast_adapt/full | 2.783143 | 0.3370 | 0.9708 |
| task_06_best | online_write/full | 2.783157 | 0.3370 | 1.0390 |
| task_07_best | frozen/full | 3.142655 | 0.2397 | 1.1596 |
| task_07_best | fast_adapt/full | 3.142529 | 0.2397 | 1.1595 |
| task_07_best | online_write/full | 3.142527 | 0.2397 | 1.2384 |
| task_08_best | frozen/full | 3.098248 | 0.2530 | 1.1312 |
| task_08_best | fast_adapt/full | 3.098215 | 0.2530 | 1.1311 |
| task_08_best | online_write/full | 3.098224 | 0.2530 | 1.1905 |
| task_09_best | frozen/full | 3.134137 | 0.2310 | 1.1201 |
| task_09_best | fast_adapt/full | 3.134199 | 0.2305 | 1.1200 |
| task_09_best | online_write/full | 3.134233 | 0.2305 | 1.1894 |

## Continual retention and anchors

CLNLL averages only anchor laws whose first occurrence is no later than the checkpoint. Forgetting is current NLL minus the best NLL since that law was first seen.

| checkpoint | variant | CLNLL | avg forgetting | seen laws |
|---:|---|---:|---:|---:|
| task_00_best | frozen/full | 3.024473 | 0.000000 | 1 |
| task_01_best | frozen/full | 3.278532 | 0.175848 | 2 |
| task_02_best | frozen/full | 3.120746 | 0.193198 | 3 |
| task_03_best | frozen/full | 3.131732 | 0.204184 | 3 |
| task_04_best | frozen/full | 3.078447 | 0.090557 | 4 |
| task_05_best | frozen/full | 3.166170 | 0.179931 | 5 |
| task_06_best | frozen/full | 3.184315 | 0.198076 | 5 |
| task_07_best | frozen/full | 3.147172 | 0.144414 | 6 |
| task_08_best | frozen/full | 3.132567 | 0.137558 | 6 |
| task_09_best | frozen/full | 3.125622 | 0.136473 | 7 |

## Stage-level plasticity

`adaptation_gain_nll = pre_nll - post_nll`; positive means the current task improved after training.

| task | shift type | variant | pre NLL | post NLL | adaptation gain |
|---:|---|---|---:|---:|---:|
| task_00 | initial | frozen/full | NA | 3.054703 | NA |
| task_01 | novel | frozen/full | 3.878900 | 3.302232 | 0.576668 |
| task_02 | novel | frozen/full | 3.087300 | 2.581951 | 0.505349 |
| task_03 | exact_recurrence | frozen/full | 3.410060 | 3.115949 | 0.294111 |
| task_04 | near_recurrence | frozen/full | 3.367471 | 3.049414 | 0.318057 |
| task_05 | specialization | frozen/full | 3.187393 | 2.985291 | 0.202102 |
| task_06 | transient_anomaly | frozen/full | 3.223537 | 2.783086 | 0.440450 |
| task_07 | merge | frozen/full | 3.264909 | 3.142655 | 0.122254 |
| task_08 | long_gap_recurrence | frozen/full | 3.137307 | 3.098248 | 0.039059 |
| task_09 | mixture | frozen/full | 3.373573 | 3.134137 | 0.239436 |

## Transfer and adaptation contract

FWT compares the same task test set from the fixed C_init and the pre-task checkpoint. Only genuinely unseen persistent-law tasks enter the average; recurrence tasks remain diagnostics.

- Average FWT: `0.051564` (available).

| protocol | task | K min | K max | adaptation AUC | status |
|---|---:|---:|---:|---:|---|
| fast_adapt | 1 | 0 | 32 | -0.001218 | available |
| fast_adapt | 2 | 0 | 32 | -0.003757 | available |
| fast_adapt | 3 | 0 | 32 | -0.012534 | available |
| fast_adapt | 4 | 0 | 32 | -0.005530 | available |
| fast_adapt | 5 | 0 | 32 | -0.003248 | available |
| fast_adapt | 6 | 0 | 32 | -0.006065 | available |
| fast_adapt | 7 | 0 | 32 | -0.005583 | available |
| fast_adapt | 8 | 0 | 32 | -0.004604 | available |
| fast_adapt | 9 | 0 | 32 | -0.002969 | available |
| online_write | 1 | 0 | 32 | -0.001211 | available |
| online_write | 2 | 0 | 32 | -0.003757 | available |
| online_write | 3 | 0 | 32 | -0.012536 | available |
| online_write | 4 | 0 | 32 | -0.005557 | available |
| online_write | 5 | 0 | 32 | -0.003249 | available |
| online_write | 6 | 0 | 32 | -0.006063 | available |
| online_write | 7 | 0 | 32 | -0.005584 | available |
| online_write | 8 | 0 | 32 | -0.004602 | available |
| online_write | 9 | 0 | 32 | -0.002967 | available |

| returned law | first task | return task | shift | RRR | status |
|---|---:|---:|---|---:|---|
| A_1 | 0 | 3 | exact_recurrence | NA | not_available_missing_first_gain |
| A_1 | 0 | 8 | long_gap_recurrence | NA | not_available_missing_first_gain |

## HM-specific state

Topology action counts come from committed transaction events; no leaf-count difference is inferred.

| task | nodes | leaves | episodic rows | episodic bytes | semantic bytes | split | merge | prune | NISE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 12 | 189643 | 1692 | 0 | 0 | 0 | 0.171528 |
| 1 | 1 | 1 | 36 | 375058 | 1692 | 2 | 2 | 0 | 0.449382 |
| 2 | 3 | 2 | 72 | 726310 | 2780 | 5 | 2 | 2 | 0.555815 |
| 3 | 3 | 2 | 96 | 908662 | 2780 | 8 | 2 | 4 | 0.450355 |
| 4 | 5 | 3 | 96 | 908530 | 3868 | 10 | 3 | 5 | 0.354625 |
| 5 | 3 | 2 | 108 | 1025982 | 2780 | 10 | 3 | 6 | 0.413353 |
| 6 | 3 | 2 | 132 | 1209486 | 2780 | 10 | 3 | 6 | 0.492167 |
| 7 | 3 | 2 | 132 | 1229710 | 2780 | 10 | 3 | 6 | 0.398074 |
| 8 | 3 | 2 | 132 | 1249230 | 2780 | 10 | 3 | 6 | 0.366925 |
| 9 | 3 | 2 | 144 | 1365654 | 2780 | 10 | 3 | 6 | 0.374863 |

## Schedule-driven diagnostics

For NLL differences, positive values mean the right-hand condition has lower NLL; definitions come from task shift_type and paired controls in the protocol.

| metric | formula | value | expected |
|---|---|---:|---|
| A_1_retention_before_task3 | `L_2,A_1 - L_0,A_1` | 0.351569 | near_zero_or_negative |
| A_1_exact_recurrence_recovery_task3 | `L_2,A_1 - L_3,A_1` | 0.295577 | positive |
| B_1_near_recurrence_impact_task5 | `L_5,B_1 - L_4,B_1` | 0.170023 | near_zero_or_negative |
| B_prime_1_near_recurrence_gain_task5 | `L_4,B_prime_1 - L_5,B_prime_1` | -0.158096 | positive |
| A_1_specialization_impact_task6 | `L_6,A_1 - L_5,A_1` | 0.277261 | near_zero_or_negative |
| A_2_specialization_gain_task6 | `L_5,A_2 - L_6,A_2` | -0.267433 | positive |
| controls_task_06_no_transient_transient_control_adaptation_task6 | `L_{5,6}^control - L_{6,6}^control` | 0.464307 | positive |
| controls_task_06_no_transient_transient_excess_adaptation_task6 | `(L_{5,6} - L_{6,6}) - (L_{5,6}^control - L_{6,6}^control)` | -0.023856 | near_zero_or_negative |
| A_1_long_gap_reference_task8 | `L_7,A_1 - L_0,A_1` | 0.074914 | near_zero_or_negative |
| A_1_long_gap_recurrence_recovery_task8 | `L_7,A_1 - L_8,A_1` | 0.049270 | positive |
| E_1_B_1_mixture_adaptation_task9 | `P_9^pre - P_9^post` | 0.239436 | positive |
| A_1_rrr_task3 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |
| A_1_rrr_task8 | `(L_first_pre - L_return_pre) / (L_first_pre - L_first_post)` | NA | not_available_missing_first_gain |

## Hawkes law recovery

NISE compares causal total and representative event-type intensity curves against `ground_truth/regimes.npz`.

| checkpoint | variant | regime | NISE | sequences |
|---:|---|---|---:|---:|
| task_00_best | frozen/full | A_1 | 0.171528 | 64 |
| task_01_best | frozen/full | A_1 | 0.588593 | 64 |
| task_01_best | frozen/full | B_1 | 0.310171 | 64 |
| task_02_best | frozen/full | A_1 | 0.946428 | 64 |
| task_02_best | frozen/full | B_1 | 0.643728 | 64 |
| task_02_best | frozen/full | C_1 | 0.077290 | 64 |
| task_03_best | frozen/full | A_1 | 0.264843 | 64 |
| task_03_best | frozen/full | B_1 | 0.672795 | 64 |
| task_03_best | frozen/full | C_1 | 0.413425 | 64 |
| task_04_best | frozen/full | A_1 | 0.322454 | 64 |
| task_04_best | frozen/full | B_1 | 0.342113 | 64 |
| task_04_best | frozen/full | B_prime_1 | 0.362386 | 64 |
| task_04_best | frozen/full | C_1 | 0.391547 | 64 |
| task_05_best | frozen/full | A_1 | 0.021679 | 64 |
| task_05_best | frozen/full | A_2 | 0.015069 | 64 |
| task_05_best | frozen/full | B_1 | 0.598966 | 64 |
| task_05_best | frozen/full | B_prime_1 | 0.631552 | 64 |
| task_05_best | frozen/full | C_1 | 0.799498 | 64 |
| task_06_best | frozen/full | A_1 | 0.508065 | 64 |
| task_06_best | frozen/full | A_2 | 0.527862 | 64 |
| task_06_best | frozen/full | B_1 | 0.669206 | 64 |
| task_06_best | frozen/full | B_prime_1 | 0.655303 | 64 |
| task_06_best | frozen/full | C_1 | 0.100399 | 64 |
| task_07_best | frozen/full | A_1 | 0.308627 | 64 |
| task_07_best | frozen/full | A_2 | 0.293184 | 64 |
| task_07_best | frozen/full | A_merge | 0.302931 | 64 |
| task_07_best | frozen/full | B_1 | 0.558570 | 64 |
| task_07_best | frozen/full | B_prime_1 | 0.571566 | 64 |
| task_07_best | frozen/full | C_1 | 0.353568 | 64 |
| task_08_best | frozen/full | A_1 | 0.225033 | 64 |
| task_08_best | frozen/full | A_2 | 0.216564 | 64 |
| task_08_best | frozen/full | A_merge | 0.220200 | 64 |
| task_08_best | frozen/full | B_1 | 0.582436 | 64 |
| task_08_best | frozen/full | B_prime_1 | 0.596799 | 64 |
| task_08_best | frozen/full | C_1 | 0.360520 | 64 |
| task_09_best | frozen/full | A_1 | 0.362428 | 64 |
| task_09_best | frozen/full | A_2 | 0.373765 | 64 |
| task_09_best | frozen/full | A_merge | 0.357903 | 64 |
| task_09_best | frozen/full | B_1 | 0.398966 | 64 |
| task_09_best | frozen/full | B_prime_1 | 0.416712 | 64 |
| task_09_best | frozen/full | C_1 | 0.437562 | 64 |
| task_09_best | frozen/full | E_1 | 0.276702 | 64 |

## Unseen/OOD novelty control

Transient/unseen anchors are reported separately and never enter CLNLL, average forgetting, or average seen-task NLL.

| checkpoint | variant | regime | NLL/event | accuracy | time MAE |
|---:|---|---|---:|---:|---:|
| task_00_best | frozen/full | B_1 | 3.845968 | 0.0739 | 1.4640 |
| task_00_best | frozen/full | C_1 | 3.258135 | 0.1387 | 1.0764 |
| task_00_best | frozen/full | B_prime_1 | 3.799152 | 0.0784 | 1.4622 |
| task_00_best | frozen/full | A_2 | 3.154206 | 0.3050 | 1.2131 |
| task_00_best | frozen/full | X_transient | 3.174471 | 0.2532 | 1.1304 |
| task_00_best | frozen/full | A_merge | 3.012941 | 0.3265 | 1.1459 |
| task_00_best | frozen/full | E_1 | 3.419324 | 0.1503 | 1.1057 |
| task_01_best | frozen/full | C_1 | 3.098224 | 0.2013 | 0.9985 |
| task_01_best | frozen/full | B_prime_1 | 3.163078 | 0.2425 | 1.2054 |
| task_01_best | frozen/full | A_2 | 3.469072 | 0.1131 | 1.2229 |
| task_01_best | frozen/full | X_transient | 3.388749 | 0.1278 | 1.1205 |
| task_01_best | frozen/full | A_merge | 3.362968 | 0.1479 | 1.1621 |
| task_01_best | frozen/full | E_1 | 3.306632 | 0.1302 | 1.0436 |
| task_02_best | frozen/full | B_prime_1 | 3.368191 | 0.1410 | 1.1893 |
| task_02_best | frozen/full | A_2 | 3.482879 | 0.0868 | 1.2011 |
| task_02_best | frozen/full | X_transient | 3.408642 | 0.0733 | 1.0919 |
| task_02_best | frozen/full | A_merge | 3.369241 | 0.0893 | 1.1251 |
| task_02_best | frozen/full | E_1 | 3.379427 | 0.0871 | 0.9916 |
| task_03_best | frozen/full | B_prime_1 | 3.457758 | 0.0742 | 1.2263 |
| task_03_best | frozen/full | A_2 | 3.188611 | 0.2140 | 1.1849 |
| task_03_best | frozen/full | X_transient | 3.153553 | 0.1732 | 1.0869 |
| task_03_best | frozen/full | A_merge | 3.066175 | 0.2321 | 1.1182 |
| task_03_best | frozen/full | E_1 | 3.251869 | 0.2168 | 1.0180 |
| task_04_best | frozen/full | A_2 | 3.261655 | 0.1979 | 1.2095 |
| task_04_best | frozen/full | X_transient | 3.219343 | 0.1670 | 1.1159 |
| task_04_best | frozen/full | A_merge | 3.129941 | 0.2174 | 1.1388 |
| task_04_best | frozen/full | E_1 | 3.198547 | 0.1970 | 1.0351 |
| task_05_best | frozen/full | X_transient | 3.085007 | 0.3017 | 1.0853 |
| task_05_best | frozen/full | A_merge | 2.941170 | 0.3583 | 1.1094 |
| task_05_best | frozen/full | E_1 | 3.343246 | 0.1016 | 1.0154 |
| task_06_best | frozen/full | X_transient | 3.229560 | 0.1616 | 1.1032 |
| task_06_best | frozen/full | A_merge | 3.211498 | 0.1636 | 1.1354 |
| task_06_best | frozen/full | E_1 | 3.319924 | 0.1077 | 1.0175 |
| task_07_best | frozen/full | X_transient | 3.132653 | 0.2322 | 1.0991 |
| task_07_best | frozen/full | E_1 | 3.228184 | 0.1804 | 1.0317 |
| task_08_best | frozen/full | X_transient | 3.115924 | 0.2336 | 1.0944 |
| task_08_best | frozen/full | E_1 | 3.240880 | 0.1760 | 1.0247 |
| task_09_best | frozen/full | X_transient | 3.175214 | 0.1951 | 1.0712 |

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
