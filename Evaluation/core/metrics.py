from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Mapping, Sequence


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def prediction_metrics(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("no event predictions")
    true = [int(row["true_type"]) for row in rows]
    pred = [int(row["predicted_type"]) for row in rows]
    labels = sorted(set(true) | set(pred))
    f1s = []
    support = Counter(true)
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, pred))
        fp = sum(a != label and b == label for a, b in zip(true, pred))
        fn = sum(a == label and b != label for a, b in zip(true, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    errors = [float(r["predicted_delta_time"]) - float(r["true_delta_time"]) for r in rows]
    return {
        "nll_per_event": mean(float(r["event_nll"]) for r in rows),
        "accuracy": mean(a == b for a, b in zip(true, pred)),
        "macro_f1": mean(f1s),
        "time_mae": mean(abs(x) for x in errors),
        "time_rmse": math.sqrt(mean(x * x for x in errors)),
        "num_events": len(rows),
        "per_type_support": dict(sorted(support.items())),
        "majority_accuracy": max(support.values()) / len(true),
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def paired_permutation_test(left: Sequence[float], right: Sequence[float], seed: int, samples: int = 10000) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have equal non-zero length")
    diffs = [a - b for a, b in zip(left, right)]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        trial = abs(mean(d if rng.random() < 0.5 else -d for d in diffs))
        extreme += trial >= observed
    return (extreme + 1) / (samples + 1)


def adaptation_auc(points: Mapping[int, float]) -> float:
    ordered = sorted((int(k), float(v)) for k, v in points.items())
    if len(ordered) < 2 or ordered[-1][0] <= 0:
        raise ValueError("adaptation AUC needs at least two K values and Kmax > 0")
    area = sum((b_k - a_k) * (a_v + b_v) / 2 for (a_k, a_v), (b_k, b_v) in zip(ordered, ordered[1:]))
    return area / ordered[-1][0]


def continual_metrics(matrix: Mapping[int, Mapping[str, float]], first_seen: Mapping[str, int]) -> list[dict]:
    output = []
    for task in sorted(matrix):
        seen = [law for law, start in first_seen.items() if start <= task and law in matrix[task]]
        forgetting = []
        bwt = []
        for law in seen:
            start = first_seen[law]
            history = [matrix[t][law] for t in sorted(matrix) if start <= t <= task and law in matrix[t]]
            forgetting.append(matrix[task][law] - min(history))
            if start < task and law in matrix.get(start, {}):
                bwt.append(matrix[start][law] - matrix[task][law])
        output.append({"task": task, "clnll": mean(matrix[task][law] for law in seen), "average_forgetting": mean(forgetting), "average_bwt": mean(bwt), "seen_laws": len(seen)})
    return output
