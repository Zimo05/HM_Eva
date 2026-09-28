"""Aggregate continual-learning summary metrics across seeds.

The default layout is::

    Evaluation/results/continual/<model>/sequential/seed_*/continual_summary.csv

An optional model-selection layer is also accepted::

    Evaluation/results/continual/<selection>/<model>/sequential/seed_*/continual_summary.csv

For each seed, the script computes the mean over that seed's checkpoint rows.
The evaluator-generated aggregate row (the row with an empty
``checkpoint_task``) is deliberately ignored, so a seed is never counted
twice.  The output contains one row per seed; no cross-seed mean is computed.

Example::

    python Evaluation/continual_analysis.py \
      --results-root Evaluation/results \
      --strategies sequential replay joint
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("clnll", "average_forgetting", "average_bwt")
SEED_FIELDS = (
    "selection",
    "model",
    "strategy",
    "seed",
    "clnll",
    "average_forgetting",
    "average_bwt",
    "num_checkpoints",
    "source",
)
def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean_metric(rows: list[dict[str, str]], metric: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(metric))) is not None]
    return mean(values) if values else None


def _seed_metrics(path: Path) -> tuple[dict[str, float | int | str | None], Path]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty continual summary: {path}")

    checkpoint_rows = [
        row for row in rows
        if str(row.get("checkpoint_task", "")).strip() != ""
    ]
    if not checkpoint_rows:
        raise ValueError(f"no usable checkpoint rows: {path}")

    # Average each metric independently so an unavailable BWT value is not
    # treated as zero.  The aggregate row, when present, is not included.
    values = {
        metric: _mean_metric(checkpoint_rows, metric)
        for metric in METRICS
    }
    source = "checkpoint_rows"
    seed_name = path.parent.name
    seed = seed_name.removeprefix("seed_")
    try:
        seed_value: int | str = int(seed)
    except ValueError:
        seed_value = seed
    result: dict[str, float | int | str | None] = {
        "seed": seed_value,
        **values,
        "num_checkpoints": sum(
            1 for row in rows
            if str(row.get("checkpoint_task", "")).strip() != ""
        ),
        "source": source,
    }
    return result, path


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def analyze(
    results_root: Path,
    strategy: str,
    selection: str | None = None,
) -> list[dict[str, Any]]:
    continual_root = results_root.expanduser().resolve() / "continual"
    summary_paths = []
    patterns = (
        (f"*/{strategy}/seed_*/continual_summary.csv", ""),
        (f"*/*/{strategy}/seed_*/continual_summary.csv", None),
    )
    for pattern, fixed_selection in patterns:
        for path in continual_root.glob(pattern):
            relative_parts = path.parent.relative_to(continual_root).parts
            # .../<model>/<strategy>/<seed> or
            # .../<selection>/<model>/<strategy>/<seed>
            current_selection = (
                fixed_selection
                if fixed_selection is not None
                else "/".join(relative_parts[:-3])
            )
            if selection is not None and current_selection != selection:
                continue
            summary_paths.append((path, current_selection))
    summary_paths = list({(path, current_selection) for path, current_selection in summary_paths})
    summary_paths.sort(key=lambda item: str(item[0]))
    if not summary_paths:
        raise FileNotFoundError(
            "no continual summaries found below "
            f"{continual_root} for selection={selection!r}, "
            f"strategy={strategy!r}"
        )

    per_seed: list[dict[str, Any]] = []
    for path, current_selection in summary_paths:
        relative_parts = path.parent.relative_to(continual_root).parts
        model = relative_parts[-3]
        values, _ = _seed_metrics(path)
        per_seed.append({
            "selection": current_selection or "default",
            "model": model,
            "strategy": strategy,
            **values,
        })

    per_seed.sort(key=lambda row: (str(row["model"]), str(row["seed"])))
    return per_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="root containing continual/<model>/<strategy>/seed_*",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=("sequential", "replay", "joint"),
        help="continual strategies to analyze (default: sequential replay joint)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="backward-compatible shorthand for analyzing one strategy",
    )
    parser.add_argument(
        "--selection",
        default=None,
        help="optional layer after continual, for example 'base'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the two CSV outputs (default: results root)",
    )
    args = parser.parse_args()
    strategies = [args.strategy] if args.strategy else list(args.strategies)
    per_seed: list[dict[str, Any]] = []
    for strategy in strategies:
        per_seed.extend(analyze(args.results_root, strategy, args.selection))
    per_seed.sort(
        key=lambda row: (
            str(row["selection"]),
            str(row["model"]),
            str(row["strategy"]),
            str(row["seed"]),
        )
    )
    output_dir = (args.output_dir or args.results_root).expanduser().resolve()
    per_seed_path = output_dir / "continual_analysis_per_seed.csv"
    _write_csv(per_seed_path, SEED_FIELDS, per_seed)

    print(f"per-seed: {per_seed_path}")
    for row in per_seed:
        print(
            f"{row['selection']}/{row['model']}/{row['strategy']}/seed_{row['seed']}: "
            f"clnll_mean={row['clnll']} "
            f"average_forgetting_mean={row['average_forgetting']} "
            f"average_bwt_mean={row['average_bwt']}"
        )


if __name__ == "__main__":
    main()
