from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

from .cl_metrics import compute_retention_metrics


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
    event_nlls = [
        float(row["event_nll"])
        for row in rows
        if row.get("event_nll") not in (None, "")
    ]
    return {
        "nll_per_event": mean(event_nlls) if event_nlls else None,
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
    """Return normalized adaptation AUC over the actual K span.

    ``K`` is the number of support events, so every curve must include its
    ``K=0`` pre-adaptation baseline.  Keeping this compatibility wrapper here
    lets older callers use the same contract as the canonical CL engine.
    """

    ordered = sorted((int(k), float(v)) for k, v in points.items())
    if len(ordered) < 2 or ordered[0][0] != 0:
        raise ValueError("adaptation AUC requires at least two K values including K=0")
    span = ordered[-1][0] - ordered[0][0]
    if span <= 0:
        raise ValueError("adaptation AUC requires a positive K span")
    area = sum(
        (b_k - a_k) * (a_v + b_v) / 2
        for (a_k, a_v), (b_k, b_v) in zip(ordered, ordered[1:])
    )
    return area / span


def continual_metrics(matrix: Mapping[int, Mapping[str, float]], first_seen: Mapping[str, int]) -> list[dict]:
    """Compatibility view over the canonical frozen-anchor contract."""

    _, summaries = compute_retention_metrics(
        matrix,
        first_seen=first_seen,
        persistent_regimes=first_seen,
    )
    return [
        {
            "task": row["checkpoint_task"],
            "clnll": row["clnll"],
            "average_forgetting": row["average_forgetting"],
            "average_bwt": row["average_bwt"],
            "seen_laws": row["seen_law_count"],
        }
        for row in summaries
    ]
