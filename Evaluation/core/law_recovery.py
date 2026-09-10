from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Train.Inference import MemoryTreeInference


def _assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        left, right = linear_sum_assignment(cost)
        return list(zip(left.tolist(), right.tolist()))
    except ImportError:
        remaining = set(range(cost.shape[1]))
        pairs = []
        for left in range(cost.shape[0]):
            if not remaining:
                break
            right = min(remaining, key=lambda index: cost[left, index])
            remaining.remove(right)
            pairs.append((left, right))
        return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Match learned HM leaves to DWS Hawkes laws")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    inference = MemoryTreeInference.from_checkpoint(args.checkpoint, device=args.device)
    tree, hawkes = inference.tree, inference.hawkes
    learned = []
    for leaf_id in tree.leaf_ids:
        theta = tree.semantic_theta(leaf_id).detach()
        dimension = hawkes.num_types
        mu = F.softplus(theta[:dimension]).cpu().numpy()
        weights = F.softplus(theta[dimension:]).reshape(
            dimension, dimension, hawkes.num_basis
        ).cpu().numpy()
        learned.append((leaf_id, mu, weights.sum(axis=-1)))
    truth_payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    truth = [(law_id, np.asarray(value["mu"]), np.asarray(value["A"])) for law_id, value in sorted(truth_payload.items(), key=lambda item: int(item[0]))]
    cost = np.empty((len(learned), len(truth)), dtype=np.float64)
    for i, (_, mu, amplitude) in enumerate(learned):
        for j, (_, true_mu, true_amplitude) in enumerate(truth):
            numerator = np.square(mu - true_mu).sum() + np.square(amplitude - true_amplitude).sum()
            denominator = np.square(true_mu).sum() + np.square(true_amplitude).sum()
            cost[i, j] = numerator / max(denominator, 1e-12)
    pairs = _assignment(cost)
    rows = [{"leaf_id": learned[i][0], "law_id": truth[j][0], "normalized_parameter_error": float(cost[i, j])} for i, j in pairs]
    result = {
        "learned_leaf_count": len(learned),
        "ground_truth_law_count": len(truth),
        "matched_count": len(rows),
        "mean_normalized_parameter_error": float(np.mean([row["normalized_parameter_error"] for row in rows])) if rows else None,
        "matching": rows,
        "note": "Hungarian matching on normalized squared baseline-plus-kernel-amplitude error; predictive intensity NISE remains reported by the continual evaluator.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
