#!/usr/bin/env python3
"""Aggregate continual intensity leaf summaries at the seed directory level."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


LEAF_FIELDS = (
    "checkpoint_task",
    "grid_samples",
    "grid_support",
    "history_rule",
    "model",
    "nise_mean",
    "nise_median",
    "regime_id",
    "sequence_count",
)
OUTPUT_FIELDS = (*LEAF_FIELDS, "evaluation_scope", "first_seen_task")


def _checkpoint_task(path: Path) -> int:
    name = path.parent.parent.name
    prefix = "checkpoint_task_"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected checkpoint directory: {name}")
    return int(name.removeprefix(prefix))


def _read_protocol(seed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    protocol_path = seed_dir / "benchmark_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    anchors = protocol.get("anchors")
    first_seen = protocol.get("first_seen")
    if not isinstance(anchors, list) or not isinstance(first_seen, dict):
        raise ValueError(f"invalid continual protocol: {protocol_path}")
    return anchors, {str(key): int(value) for key, value in first_seen.items()}


def aggregate_model(seed_dir: Path, model: str) -> dict[str, Any]:
    anchors, first_seen = _read_protocol(seed_dir)
    anchor_specs = {
        str(anchor["regime_id"]): str(anchor["evaluation_scope"])
        for anchor in anchors
    }
    anchor_order = {regime_id: index for index, regime_id in enumerate(anchor_specs)}
    leaf_paths = list(
        (seed_dir / "intensity_curves").glob(
            "checkpoint_task_*/*/intensity_summary.csv"
        )
    )
    if not leaf_paths:
        raise FileNotFoundError(f"no intensity leaf summaries under {seed_dir}")
    leaf_paths.sort(
        key=lambda path: (
            _checkpoint_task(path),
            anchor_order.get(path.parent.name, len(anchor_order)),
            path.parent.name,
        )
    )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for leaf_path in leaf_paths:
        with leaf_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != LEAF_FIELDS:
                raise ValueError(
                    f"unexpected columns in {leaf_path}: {reader.fieldnames}"
                )
            leaf_rows = list(reader)
        if len(leaf_rows) != 1:
            raise ValueError(f"expected one row in {leaf_path}, found {len(leaf_rows)}")

        row = leaf_rows[0]
        checkpoint_task = _checkpoint_task(leaf_path)
        regime_id = leaf_path.parent.name
        if int(row["checkpoint_task"]) != checkpoint_task:
            raise ValueError(f"checkpoint mismatch in {leaf_path}")
        if row["regime_id"] != regime_id:
            raise ValueError(f"regime mismatch in {leaf_path}")
        if row["model"] != model:
            raise ValueError(f"model mismatch in {leaf_path}: {row['model']} != {model}")
        if regime_id not in anchor_specs or regime_id not in first_seen:
            raise ValueError(f"regime {regime_id} is absent from {seed_dir}/benchmark_protocol.json")

        key = (checkpoint_task, regime_id)
        if key in seen:
            raise ValueError(f"duplicate intensity summary for {key} under {seed_dir}")
        seen.add(key)
        rows.append(
            {
                **row,
                "evaluation_scope": anchor_specs[regime_id],
                "first_seen_task": first_seen[regime_id],
            }
        )

    output_path = seed_dir / "intensity_summary.csv"
    temp_path = seed_dir / f".intensity_summary.csv.{os.getpid()}.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_path)

    metric_rows: list[dict[str, Any]] = []
    metric_fields: tuple[str, ...] | None = None
    for summary_path in leaf_paths:
        metrics_path = summary_path.with_name("intensity_metrics.csv")
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            current_fields = tuple(reader.fieldnames or ())
            if metric_fields is None:
                metric_fields = current_fields
            elif current_fields != metric_fields:
                raise ValueError(
                    f"inconsistent metric columns in {metrics_path}: {reader.fieldnames}"
                )
            for row in reader:
                checkpoint_task = _checkpoint_task(metrics_path)
                regime_id = metrics_path.parent.name
                if int(row["checkpoint_task"]) != checkpoint_task:
                    raise ValueError(f"checkpoint mismatch in {metrics_path}")
                if row["regime_id"] != regime_id:
                    raise ValueError(f"regime mismatch in {metrics_path}")
                if row["model"] != model:
                    raise ValueError(
                        f"model mismatch in {metrics_path}: {row['model']} != {model}"
                    )
                metric_rows.append(
                    {
                        **row,
                        "evaluation_scope": anchor_specs[regime_id],
                        "first_seen_task": first_seen[regime_id],
                    }
                )
    if not metric_fields or not metric_rows:
        raise ValueError(f"no intensity metric rows under {seed_dir}")
    metrics_output_path = seed_dir / "intensity_metrics.csv"
    metrics_temp_path = seed_dir / f".intensity_metrics.csv.{os.getpid()}.tmp"
    with metrics_temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*metric_fields, "evaluation_scope", "first_seen_task"),
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    metrics_temp_path.replace(metrics_output_path)

    tasks = sorted({checkpoint_task for checkpoint_task, _ in seen})
    expected = {(task, regime_id) for task in tasks for regime_id in anchor_specs}
    missing = sorted(expected - seen)
    return {
        "model": model,
        "summary_output": str(output_path),
        "metrics_output": str(metrics_output_path),
        "rows": len(rows),
        "metric_rows": len(metric_rows),
        "tasks": tasks,
        "missing_within_available_task_range": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("Evaluation/results/continual"),
    )
    parser.add_argument("--strategy", default="sequential")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("models", nargs="+", help="Model directory names")
    args = parser.parse_args()

    reports = []
    for model in args.models:
        seed_dir = args.results_root / model / args.strategy / f"seed_{args.seed}"
        reports.append(aggregate_model(seed_dir, model))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
