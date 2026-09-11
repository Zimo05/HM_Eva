# Continual Hawkes benchmark

`hm_continual_v2` is the canonical `CL-core-v2` paper benchmark. Generate it
from the repository root with:

```bash
python Datasets/CL/generate_continual_hawkes.py \
  --benchmark unified \
  --output Datasets/CL/hm_continual_v2 \
  --seed 7
```

The ten-stage protocol is:

```text
A_1 -> B_1 -> C_1 -> A_1 -> B_prime_1 -> A_2
    -> (C_1 + X_transient) -> A_merge -> A_1 -> (E_1 + B_1)
```

`benchmark_manifest.json` is the protocol source of truth. It records task
semantics, regime weights, first-seen tasks, persistent versus diagnostic
regimes, frozen anchors, matched controls, and the adaptation support/query
files with their fixed `K={0,1,2,4,8,16,32}` values. Task CSVs contain only
`event_times,event_types`; every adaptation curve starts from the same fresh
pre-task checkpoint clone and scores the same query file.

HM continual training writes `checkpoint/task_XX_last.pt` for the final
state and `checkpoint/task_XX_best.pt` for the task-local validation choice.
The evaluator consumes the `*_best.pt` files, while the next task resumes
from the preceding task's best checkpoint. `train.csv` is used for updates,
`val.csv` only for checkpoint selection, and `test.csv` only for reporting.

The old recurrence-only suite is retained under
`legacy/recurrence_v1/` when present and remains available through
`--benchmark recurrence`. The existing drift, hierarchy, and transient
generators remain auxiliary stress tests.
