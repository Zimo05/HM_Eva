# Continual Hawkes benchmark

Protocol: `CL-core-v2`
Event dimension: `8`; exponential decay bases: `[0.5, 1.5]`

Model-facing task and control CSVs contain only `event_times,event_types`.
`benchmark_manifest.json` is the protocol source of truth; stream and ground-truth manifests are oracle-only.
Anchors are independently sampled and are evaluated read-only after each checkpoint.
Each task also has independent `adapt_support.csv` and `adapt_query.csv` files; the manifest records the fixed adaptation exposure values `K={0,1,2,4,8,16,32}`.
Every K curve starts from a fresh clone of the same pre-task checkpoint and scores the same fixed query set.

## Stage schedule

| task_id | stage | regime mixture | shift | recurrence_of |
|---:|---|---|---|---|
| 0 | A_1_initial | A_1:1 | initial |  |
| 1 | B_1_novel | B_1:1 | novel |  |
| 2 | C_1_novel | C_1:1 | novel |  |
| 3 | A_1_exact_recurrence | A_1:1 | exact_recurrence | A_1 |
| 4 | B_prime_1_near_recurrence | B_prime_1:1 | near_recurrence | B_1 |
| 5 | A_2_specialization | A_2:1 | specialization | A_1 |
| 6 | C_1_with_transient_X | C_1:0.9, X_transient:0.1 | transient_anomaly | C_1 |
| 7 | A_merge | A_merge:1 | merge | A_1|A_2 |
| 8 | A_1_long_gap_recurrence | A_1:1 | long_gap_recurrence | A_1 |
| 9 | E_1_B_1_mixture | E_1:0.7, B_1:0.3 | mixture |  |

Task 6 has a matched `controls/task_06_no_transient/` C_1-only stream.
Persistent-law averages exclude `X_transient`; it is reported as diagnostic OOD evidence.
The old recurrence-only suite belongs under `legacy/recurrence_v1/` when retained locally.
