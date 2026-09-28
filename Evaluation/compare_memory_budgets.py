"""Aggregate fixed-memory-budget CL runs and draw the memory Pareto figure.

Each ``--budget-run LABEL=RUN_ID`` points to one fixed byte budget run under
``RESULTS_ROOT/CONDITION/seed_SEED/RUN_ID``. The x axis always uses measured
serialized persistent-memory bytes from the final checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
from typing import Any


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "continual" / "HM"
DEFAULT_CONDITIONS = ("full", "flat_memory", "no_sleep", "no_merge_prune")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _final_row(path: Path) -> dict[str, str]:
    rows = [
        row for row in _read_rows(path)
        if _number(row, "checkpoint_task") is not None
    ]
    if not rows:
        raise ValueError(f"no checkpoint rows in {path}")
    return max(rows, key=lambda row: int(float(row["checkpoint_task"])))


def _metrics(result_dir: Path) -> dict[str, float]:
    summary = _final_row(result_dir / "continual_summary.csv")
    tree = _final_row(result_dir / "checkpoint_tree.csv")
    total = _number(tree, "total_memory_bytes")
    if total is None:
        episodic = _number(tree, "episodic_bytes")
        semantic = _number(tree, "semantic_bytes")
        if episodic is None or semantic is None:
            raise ValueError(f"missing persistent-memory bytes in {result_dir}")
        total = episodic + semantic
    result = {
        "persistent_memory_bytes": total,
        "persistent_memory_mb": total / 1_000_000.0,
    }
    for key in ("clnll", "average_forgetting", "average_bwt"):
        value = _number(summary, key)
        if value is None:
            raise ValueError(f"missing {key} in {result_dir / 'continual_summary.csv'}")
        result[key] = value
    return result


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _parse_budget_run(value: str) -> tuple[str, str]:
    label, separator, run_id = value.partition("=")
    if not separator or not label.strip() or not run_id.strip():
        raise argparse.ArgumentTypeError(
            "budget runs must use LABEL=RUN_ID, for example 0.5B=budget_0_5B"
        )
    return label.strip(), run_id.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate >=3 seeded memory-budget runs and draw CL-NLL/forgetting Pareto curves."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument(
        "--budget-run",
        action="append",
        type=_parse_budget_run,
        required=True,
        metavar="LABEL=RUN_ID",
        help="Repeat once per fixed budget; run IDs are shared across conditions/seeds.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        parser.error("the final budget-sweep figure requires at least three distinct seeds")
    root = args.results_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "memory_budget_sweep"
    )

    per_seed: list[dict[str, Any]] = []
    for budget_label, run_id in args.budget_run:
        for condition in args.conditions:
            for seed in args.seeds:
                result_dir = root / condition / f"seed_{seed}" / run_id
                per_seed.append({
                    "budget_label": budget_label,
                    "run_id": run_id,
                    "condition": condition,
                    "seed": seed,
                    **_metrics(result_dir),
                })

    summary_rows: list[dict[str, Any]] = []
    for budget_label, run_id in args.budget_run:
        for condition in args.conditions:
            group = [
                row for row in per_seed
                if row["budget_label"] == budget_label
                and row["condition"] == condition
            ]
            summary: dict[str, Any] = {
                "budget_label": budget_label,
                "run_id": run_id,
                "condition": condition,
                "n_seeds": len(group),
            }
            for metric in (
                "persistent_memory_bytes",
                "persistent_memory_mb",
                "clnll",
                "average_forgetting",
                "average_bwt",
            ):
                mean, std = _mean_std([float(row[metric]) for row in group])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            summary_rows.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("budget_sweep_per_seed.csv", per_seed),
        ("budget_sweep_summary.csv", summary_rows),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {
        condition: plt.get_cmap("tab10")(index % 10)
        for index, condition in enumerate(args.conditions)
    }
    for condition in args.conditions:
        points = [row for row in summary_rows if row["condition"] == condition]
        points.sort(key=lambda row: row["persistent_memory_mb_mean"])
        for axis, metric, ylabel in (
            (axes[0], "clnll", "CL-NLL ↓"),
            (axes[1], "average_forgetting", "Average forgetting ↓"),
        ):
            x = [row["persistent_memory_mb_mean"] for row in points]
            y = [row[f"{metric}_mean"] for row in points]
            axis.errorbar(
                x,
                y,
                xerr=[row["persistent_memory_mb_std"] for row in points],
                yerr=[row[f"{metric}_std"] for row in points],
                marker="o",
                capsize=3,
                color=colors[condition],
                label=condition,
            )
            for row in points:
                axis.annotate(
                    row["budget_label"],
                    (row["persistent_memory_mb_mean"], row[f"{metric}_mean"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            axis.set_xlabel("Persistent memory (MB)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    figure.savefig(output_dir / "memory_budget_pareto.png", dpi=220)
    plt.close(figure)
    print(f"Wrote per-seed data, summary table, and figure to {output_dir}")


if __name__ == "__main__":
    main()
