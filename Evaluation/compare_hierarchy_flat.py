"""Compare full HMT with the root-only flat-memory continual baseline.

Run with ``--recommend-capacity`` after Stage A to estimate the flat bank
capacity from HMT's measured persistent storage. After Stage B, run without
that flag to write a paper-ready mean/std table over the requested seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parent / "results" / "continual" / "HM"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _semantic_structural_bytes(row: dict[str, Any]) -> float | None:
    """Read semantic bytes across the legacy and prototype-aware schemas."""

    semantic = _number(row, "semantic_bytes")
    if semantic is None:
        return None
    if _number(row, "semantic_tree_tensor_bytes") is None:
        # The initial prototype-aware schema reported prototype bytes beside,
        # rather than inside, semantic_bytes.
        semantic += _number(row, "router_prototype_bytes") or 0.0
    return semantic


def _ordered_checkpoint_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    rows = [
        row for row in rows
        if _number(row, "checkpoint_task") is not None
    ]
    if not rows:
        raise ValueError(f"no checkpoint rows in {path}")
    return sorted(rows, key=lambda row: int(float(row["checkpoint_task"])))


def _result_directory(
    root: Path,
    condition: str,
    seed: int,
    run_id: str | None,
) -> Path:
    path = root / condition / f"seed_{seed}"
    return path / run_id if run_id else path


def _final_metrics(result_dir: Path) -> dict[str, float | None]:
    summary_rows = _ordered_checkpoint_rows(
        result_dir / "continual_summary.csv"
    )
    tree_rows = _ordered_checkpoint_rows(result_dir / "checkpoint_tree.csv")
    summary = summary_rows[-1]
    tree = tree_rows[-1]
    episodic = _number(tree, "episodic_bytes")
    semantic = _semantic_structural_bytes(tree)
    router_prototype = _number(tree, "router_prototype_bytes")
    persistent_total = _number(tree, "total_memory_bytes")
    nodes = _number(tree, "node_count")
    leaves = _number(tree, "leaf_count")
    memory_capacity = None
    config_path = result_dir / "cl_config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        if isinstance(model, dict):
            value = model.get("memory_capacity_per_node")
            if value is not None:
                memory_capacity = float(value)
    return {
        "clnll": _number(summary, "clnll"),
        "forgetting": _number(summary, "average_forgetting"),
        "bwt": _number(summary, "average_bwt"),
        "episodic_bytes": episodic,
        "semantic_bytes": semantic,
        "semantic_tree_tensor_bytes": _number(tree, "semantic_tree_tensor_bytes"),
        "router_prototype_bytes": router_prototype,
        "persistent_bytes": (
            persistent_total
            if persistent_total is not None
            else (
                episodic + semantic
                if episodic is not None and semantic is not None
                else None
            )
        ),
        "node_count": nodes,
        "leaf_count": leaves,
        "memory_capacity_per_node": memory_capacity,
    }


def _row_bytes_per_episode(checkpoint_rows: list[dict[str, str]]) -> float:
    points = [
        (_number(row, "episodic_rows"), _number(row, "episodic_bytes"))
        for row in checkpoint_rows
    ]
    points = [
        (rows, size) for rows, size in points
        if rows is not None and size is not None
    ]
    unique_rows = {rows for rows, _ in points}
    if len(unique_rows) < 2:
        raise ValueError(
            "HMT checkpoint_tree.csv needs at least two distinct episodic "
            "row counts to estimate bytes per row"
        )
    mean_rows = sum(rows for rows, _ in points) / len(points)
    mean_bytes = sum(size for _, size in points) / len(points)
    variance = sum((rows - mean_rows) ** 2 for rows, _ in points)
    slope = sum(
        (rows - mean_rows) * (size - mean_bytes)
        for rows, size in points
    ) / variance
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"invalid estimated episodic bytes per row: {slope}")
    return slope


def _recommended_capacity(
    root: Path,
    condition: str,
    seed: int,
    run_id: str | None,
    budget_statistic: str,
) -> dict[str, Any]:
    result_dir = _result_directory(root, condition, seed, run_id)
    rows = _ordered_checkpoint_rows(result_dir / "checkpoint_tree.csv")
    first, last = rows[0], rows[-1]
    if _number(first, "node_count") != 1:
        raise ValueError(
            f"Stage A for seed {seed} must begin with a root-only "
            "checkpoint to estimate the flat semantic storage."
        )
    row_bytes = _row_bytes_per_episode(rows)
    first_semantic = _semantic_structural_bytes(first)
    if first_semantic is None:
        raise ValueError(
            f"missing initial root semantic bytes in {result_dir / 'checkpoint_tree.csv'}"
        )
    budgets = []
    for row in rows:
        total = _number(row, "total_memory_bytes")
        episodic = _number(row, "episodic_bytes")
        semantic = _semantic_structural_bytes(row)
        if total is not None:
            budgets.append(total)
        elif episodic is not None and semantic is not None:
            budgets.append(episodic + semantic)
    if not budgets:
        raise ValueError(f"missing persistent-byte metrics in {result_dir}")
    target = budgets[-1] if budget_statistic == "final" else sum(budgets) / len(budgets)
    episodic_budget = target - first_semantic
    if episodic_budget <= 0:
        raise ValueError(
            f"target persistent budget {target:.0f} is below the root semantic "
            f"storage {first_semantic:.0f} for seed {seed}"
        )
    capacity = max(1, int(round(episodic_budget / row_bytes)))
    estimated_total = first_semantic + capacity * row_bytes
    error_pct = abs(estimated_total - target) / target * 100.0
    return {
        "seed": seed,
        "budget_statistic": budget_statistic,
        "target_persistent_bytes": target,
        "root_semantic_bytes": first_semantic,
        "estimated_episodic_bytes_per_row": row_bytes,
        "recommended_memory_capacity_per_node": capacity,
        "estimated_flat_persistent_bytes": estimated_total,
        "estimated_budget_error_pct": error_pct,
        "hmt_final_task": int(float(last["checkpoint_task"])),
    }


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None, None
    return (
        sum(finite) / len(finite),
        statistics.stdev(finite) if len(finite) > 1 else 0.0,
    )


def _aggregate_rows(
    strategy: str,
    seed_rows: list[tuple[int, dict[str, float | None]]],
    paired_errors: list[float | None] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": strategy,
        "n_seeds": len(seed_rows),
        "seeds": ",".join(str(seed) for seed, _ in seed_rows),
    }
    for metric in (
        "clnll", "forgetting", "bwt", "episodic_bytes", "semantic_bytes",
        "semantic_tree_tensor_bytes", "router_prototype_bytes", "persistent_bytes",
        "node_count", "leaf_count",
        "memory_capacity_per_node",
    ):
        mean, std = _mean_std([values[metric] for _, values in seed_rows])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    match_mean, match_std = _mean_std(paired_errors or [])
    row["paired_storage_error_pct_mean"] = match_mean
    row["paired_storage_error_pct_std"] = match_std
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storage-match and compare HMT with flat_memory."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--hierarchy-condition", default="full")
    parser.add_argument("--flat-condition", default="flat_memory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--recommend-capacity",
        action="store_true",
        help="Stage A only: estimate per-seed flat bank capacities from HMT storage.",
    )
    parser.add_argument(
        "--budget-statistic",
        choices=("final", "mean"),
        default="final",
        help="Use final or mean HMT persistent bytes as the Stage B target.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else None

    if args.recommend_capacity:
        rows = [
            _recommended_capacity(
                root,
                args.hierarchy_condition,
                seed,
                args.run_id,
                args.budget_statistic,
            )
            for seed in args.seeds
        ]
        output = output or (root / "flat_memory_capacity_recommendations.csv")
        _write_csv(output, rows)
        print(f"Wrote per-seed storage-match capacities: {output}")
        for row in rows:
            print(
                f"seed {row['seed']}: memory_capacity_per_node="
                f"{row['recommended_memory_capacity_per_node']} "
                f"(estimated budget error {row['estimated_budget_error_pct']:.2f}%)"
            )
        return

    hierarchy_rows = []
    flat_rows = []
    paired_errors = []
    for seed in args.seeds:
        hierarchy = _final_metrics(
            _result_directory(root, args.hierarchy_condition, seed, args.run_id)
        )
        flat = _final_metrics(
            _result_directory(root, args.flat_condition, seed, args.run_id)
        )
        if flat["node_count"] != 1:
            raise ValueError(
                f"{args.flat_condition} seed {seed} is not root-only "
                f"(node_count={flat['node_count']})"
            )
        hmt_bytes = hierarchy["persistent_bytes"]
        flat_bytes = flat["persistent_bytes"]
        paired_errors.append(
            abs(flat_bytes - hmt_bytes) / hmt_bytes * 100.0
            if hmt_bytes and flat_bytes is not None
            else None
        )
        hierarchy_rows.append((seed, hierarchy))
        flat_rows.append((seed, flat))

    rows = [
        _aggregate_rows(args.hierarchy_condition, hierarchy_rows),
        _aggregate_rows(args.flat_condition, flat_rows, paired_errors),
    ]
    output = output or (root / "hierarchy_flat_comparison.csv")
    _write_csv(output, rows)
    print(f"Wrote hierarchy/flat comparison table: {output}")
    over_budget = [
        (seed, error)
        for seed, error in zip(args.seeds, paired_errors)
        if error is not None and error > 5.0
    ]
    if over_budget:
        details = ", ".join(
            f"seed {seed}: {error:.2f}%" for seed, error in over_budget
        )
        print(f"Storage mismatch exceeds 5%: {details}")


if __name__ == "__main__":
    main()
