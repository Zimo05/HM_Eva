"""Evaluate saved HM checkpoints on current and past Taobao time windows.

This is a checkpoint-evaluation utility, not a trainer. It expects the output
of ``prepare_taobao_chronological_cl.py`` and evaluates each checkpoint on all
past/current windows plus one next-window test-only adaptation probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HAWKES_MEMORY_ROOT = ROOT / "Models" / "HawkesMemory"
MEMORY_ROOT = ROOT / "Models" / "HawkesMemory" / "Memory"
for _path in (ROOT, HAWKES_MEMORY_ROOT, MEMORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from Evaluate import _load_variant_inference
from EvaluateCL import _discover_checkpoints


def _read_sequences(path: Path, max_sequences: int | None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if not {"event_times", "event_types"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain event_times and event_types")
    sequences = []
    for index, row in frame.iterrows():
        times = json.loads(row["event_times"])
        types = json.loads(row["event_types"])
        if not times or len(times) != len(types):
            raise ValueError(f"{path}:{index + 2} has mismatched or empty events")
        times = [float(value) for value in times]
        types = [int(value) for value in types]
        if any(not math.isfinite(value) for value in times):
            raise ValueError(f"{path}:{index + 2} has non-finite event times")
        if times[0] < 0 or any(right < left for left, right in zip(times, times[1:])):
            raise ValueError(f"{path}:{index + 2} is not in chronological order")
        sequences.append({
            "times": times,
            "types": types,
            "source_index": int(index),
        })
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    if not sequences:
        raise ValueError(f"{path} contains no sequences")
    return sequences


def _checkpoint_event_dim(checkpoint: Path) -> int:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    value = config.get("num_event_types") if isinstance(config, dict) else None
    if value is None:
        raise ValueError(f"{checkpoint} does not declare model_config.num_event_types")
    return int(value)


def _evaluate(
    inference,
    static_cache,
    sequences: list[dict[str, Any]],
    event_dim: int,
    window_id: int,
) -> dict[str, Any]:
    nll_total = 0.0
    scored_events = 0
    total_events = 0
    for sequence in sequences:
        if any(event < 0 or event >= event_dim for event in sequence["types"]):
            raise ValueError(
                f"window {window_id} contains an event type outside [0, {event_dim - 1}]"
            )
        result = inference.run_sequence(
            sequence,
            frontier_static_cache=static_cache,
        )
        for event in result.get("events", ()):
            total_events += 1
            if int(event.get("event_index", 0)) < 1:
                continue
            value = float(event["nll"])
            if not math.isfinite(value):
                raise ValueError(
                    f"checkpoint produced a non-finite NLL on window {window_id}"
                )
            nll_total += value
            scored_events += 1
    if scored_events == 0:
        raise ValueError(f"window {window_id} has no scoreable events")
    return {
        "nll": nll_total / scored_events,
        "nll_sum": nll_total,
        "scored_events": scored_events,
        "events": total_events,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_windows(
    checkpoint_task: int,
    available_window_ids: set[int],
) -> list[int]:
    """Return seen test windows plus one next-window adaptation probe."""

    windows = sorted(
        window_id
        for window_id in available_window_ids
        if window_id <= checkpoint_task
    )
    next_window = checkpoint_task + 1
    if next_window in available_window_ids:
        windows.append(next_window)
    return windows


def _summaries(
    cells: dict[tuple[int, int], dict[str, Any]],
    checkpoint_ids: list[int],
    windows: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    continual: list[dict[str, Any]] = []
    adaptation: list[dict[str, Any]] = []
    for task in checkpoint_ids:
        seen = [
            cells[(task, window)] for window in sorted(windows)
            if window <= task and (task, window) in cells
        ]
        total_events = sum(int(row["scored_events"]) for row in seen)
        cl_nll = (
            sum(float(row["nll_sum"]) for row in seen) / total_events
            if total_events else None
        )
        bwt_values = []
        forgetting_values = []
        for window in sorted(windows):
            if window >= task or (task, window) not in cells:
                continue
            baseline = cells.get((window, window))
            if baseline is not None:
                # For loss, positive BWT means the later checkpoint improved.
                bwt_values.append(
                    float(baseline["nll"]) - float(cells[(task, window)]["nll"])
                )
            historical = [
                float(cells[(checkpoint, window)]["nll"])
                for checkpoint in checkpoint_ids
                if window <= checkpoint < task and (checkpoint, window) in cells
            ]
            if historical:
                # Loss analogue of average forgetting: current loss minus the
                # best loss observed after the window was first learned.
                forgetting_values.append(
                    float(cells[(task, window)]["nll"]) - min(historical)
                )
        continual.append({
            "checkpoint_task": task,
            "window_label": windows[task]["label"],
            "cl_nll": cl_nll,
            "evaluated_past_windows": len(seen),
            "average_bwt": (
                sum(bwt_values) / len(bwt_values) if bwt_values else None
            ),
            "average_forgetting": (
                sum(forgetting_values) / len(forgetting_values)
                if forgetting_values else None
            ),
        })
        previous_task = task - 1
        if (
            previous_task in checkpoint_ids
            and (previous_task, task) in cells
            and (task, task) in cells
        ):
            prior_loss = float(cells[(previous_task, task)]["nll"])
            current_loss = float(cells[(task, task)]["nll"])
            adaptation.append({
                "task_id": task,
                "window_label": windows[task]["label"],
                "nll_pre_checkpoint": prior_loss,
                "nll_post_checkpoint": current_loss,
                "adaptation_gain_nll": prior_loss - current_loss,
            })
    return continual, adaptation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HM CL checkpoints on a globally chronological Taobao stream"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    manifest_path = data_dir / "chronological_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing chronological manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("global_clock") != "UTC absolute event timestamp":
        raise ValueError("input manifest does not certify a global absolute time clock")
    windows = {
        int(item["window_id"]): {
            "label": (
                item["start_date"]
                if item["start_date"] == item["end_date"]
                else f"{item['start_date']}..{item['end_date']}"
            ),
            "item": item,
        }
        for item in manifest.get("windows", ())
    }
    if not windows:
        raise ValueError("chronological manifest contains no time windows")
    checkpoint_paths = _discover_checkpoints(args.checkpoint_dir.expanduser())
    checkpoint_ids = sorted(set(checkpoint_paths).intersection(windows))
    if not checkpoint_ids:
        raise FileNotFoundError(
            "no task_XX checkpoint matches a chronological window id under "
            f"{args.checkpoint_dir}"
        )
    expected_types = int(manifest["event_type_count"])
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    matrix_rows: list[dict[str, Any]] = []
    for checkpoint_task in checkpoint_ids:
        checkpoint = checkpoint_paths[checkpoint_task]
        event_dim = _checkpoint_event_dim(checkpoint)
        if event_dim != expected_types:
            raise ValueError(
                f"event vocabulary mismatch: manifest={expected_types}, "
                f"checkpoint {checkpoint}={event_dim}"
            )
        _canonical, _protocol, _view, _settings, inference, static_cache = (
            _load_variant_inference(checkpoint, "frozen/full", args.device)
        )
        for window_id in _evaluation_windows(checkpoint_task, set(windows)):
            test_path = data_dir / f"window_{window_id:03d}" / "test.csv"
            sequences = _read_sequences(test_path, args.max_sequences)
            result = _evaluate(
                inference, static_cache, sequences, event_dim, window_id
            )
            label = windows[window_id]["label"]
            cell = {
                "checkpoint_task": checkpoint_task,
                "evaluation_window": window_id,
                "window_label": label,
                "evaluation_role": (
                    "pre_update_adaptation_probe"
                    if window_id > checkpoint_task
                    else "post_update_test"
                ),
                "checkpoint": str(checkpoint.resolve()),
                **result,
            }
            cells[(checkpoint_task, window_id)] = result
            matrix_rows.append(cell)

    continual_rows, adaptation_rows = _summaries(
        cells, checkpoint_ids, windows
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "taobao_chronological_nll_matrix.csv", matrix_rows)
    _write_csv(output_dir / "taobao_chronological_cl_summary.csv", continual_rows)
    _write_csv(output_dir / "taobao_chronological_adaptation.csv", adaptation_rows)
    summary = {
        "benchmark_id": manifest.get("benchmark_id"),
        "manifest": str(manifest_path),
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "checkpoint_tasks": checkpoint_ids,
        "window_count": len(windows),
        "training_invoked": False,
        "ground_truth_regimes_used": False,
        "evaluation": (
            "frozen/full NLL on each past/current chronological test window, "
            "plus one next-window pre-update adaptation probe"
        ),
        "metrics": {
            "cl_nll": "scored-event-weighted mean NLL over windows r <= checkpoint t",
            "average_bwt": "mean(NLL(C_r,D_r) - NLL(C_t,D_r)); positive means later checkpoint improved",
            "average_forgetting": "mean(NLL(C_t,D_r) - best prior post-learning NLL on D_r)",
            "adaptation_gain_nll": "NLL(C_(t-1),D_t) - NLL(C_t,D_t); positive means adaptation improved",
        },
        "adaptation_probe": (
            "The upper-diagonal C_t on D_(t+1) matrix row is a pre-update "
            "probe only and is excluded from CL-NLL, BWT, and Forgetting."
        ),
        "note": "No synthetic regime labels, ground-truth laws, or synthetic RRR are used.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Chronological Taobao evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
