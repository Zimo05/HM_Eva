"""Post-process CL checkpoints to measure semantic/episodic/working memory.

This entry point never invokes the continual training runner. Checkpoint
boundaries and the fixed recurrent anchor are resolved from the benchmark
manifest, then the same anchor sequences are replayed against each checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HAWKES_MEMORY_ROOT = ROOT / "Models" / "HawkesMemory"
MEMORY_ROOT = ROOT / "Models" / "HawkesMemory" / "Memory"
for _path in (ROOT, HAWKES_MEMORY_ROOT, MEMORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from Evaluation.core.cl_protocol import CLProtocol
from Evaluate import _load_variant_inference
from EvaluateCL import _discover_checkpoints, _load_cl_dataset


def _contains_regime(value: str | None, regime_id: str) -> bool:
    return regime_id in {
        token.strip() for token in str(value or "").split("|") if token.strip()
    }


def _select_recurrent_regime(protocol: CLProtocol) -> str:
    """Select the manifest regime with the complete A-style recurrence arc."""

    candidates = []
    for regime_id in protocol.persistent_regimes:
        recurrence_types = {
            task.shift_type
            for task in protocol.tasks.values()
            if _contains_regime(task.recurrence_of, regime_id)
        }
        score = sum(
            label in recurrence_types
            for label in ("exact_recurrence", "specialization", "long_gap_recurrence")
        )
        if score:
            candidates.append((score, protocol.first_seen.get(regime_id, 10**9), regime_id))
    if not candidates:
        raise ValueError(
            "benchmark manifest has no persistent regime with recurrence metadata"
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def _run_condition(
    checkpoint: Path,
    sequences: list[dict[str, Any]],
    variant: str,
    device: str | None,
) -> dict[str, Any]:
    _canonical, _protocol, _memory_view, settings, inference, static_cache = (
        _load_variant_inference(checkpoint, variant, device)
    )
    nll_values: list[float] = []
    semantic_vectors: list[torch.Tensor] = []
    episodic_norms: list[float] = []
    working_norms: list[float] = []
    event_count = 0
    for source in sequences:
        result = inference.run_sequence(
            source,
            frontier_static_cache=static_cache,
            capture_memory_components=True,
        )
        for event in result["events"]:
            event_count += 1
            if int(event["event_index"]) >= 1:
                nll_values.append(float(event["nll"]))
            semantic_vectors.append(event["semantic_theta_used"].detach().float())
            episodic_norms.append(
                float(event["episodic_delta_used"].detach().norm().item())
            )
            working_norms.append(
                float(event["working_delta_used"].detach().norm().item())
            )
    if not nll_values:
        raise ValueError(f"{variant} produced no scored events for {checkpoint}")
    semantic_mean = torch.stack(semantic_vectors).mean(dim=0)
    return {
        "nll": sum(nll_values) / len(nll_values),
        "scored_events": len(nll_values),
        "events": event_count,
        "semantic_mean": semantic_mean,
        "episodic_used_norm": sum(episodic_norms) / max(len(episodic_norms), 1),
        "working_norm": sum(working_norms) / max(len(working_norms), 1),
        "episodic_enabled": bool(settings["episodic"]),
        "working_enabled": bool(settings["working"]),
    }


def _stage_for_task(
    protocol: CLProtocol,
    task_id: int,
    regime_id: str,
) -> tuple[str, str, str]:
    task = protocol.task(task_id)
    if task_id == protocol.first_seen.get(regime_id):
        return "first_A", "first A", str(task.stage_label or "first occurrence")
    labels = {
        "exact_recurrence": "exact recurrence",
        "specialization": "specialization",
        "long_gap_recurrence": "long-gap recurrence",
    }
    if _contains_regime(task.recurrence_of, regime_id) and task.shift_type in labels:
        return task.shift_type, labels[task.shift_type], str(task.stage_label or task.shift_type)
    return task.shift_type, "", str(task.stage_label or task.shift_type)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output: Path, boundaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    x = [int(row["checkpoint_task"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, [row["nll_semantic"] for row in rows], marker="o", label="semantic only")
    axes[0].plot(x, [row["nll_semantic_episodic"] for row in rows], marker="o", label="semantic + episodic")
    axes[0].plot(x, [row["nll_full"] for row in rows], marker="o", label="full (working enabled)")
    axes[0].set_ylabel("NLL on fixed recurrent anchor")
    axes[0].set_xlabel("checkpoint task")
    axes[0].set_title("Memory contribution")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [row["semantic_shift_norm"] for row in rows], marker="o", label="semantic shift")
    axes[1].plot(x, [row["episodic_used_norm"] for row in rows], marker="o", label="episodic used")
    axes[1].plot(x, [row["working_norm"] for row in rows], marker="o", label="working memory")
    axes[1].set_ylabel("Mean parameter norm")
    axes[1].set_xlabel("checkpoint task")
    axes[1].set_title("Parameter migration")
    axes[1].legend(frameon=False)
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(int(boundary["task_id"]), color="0.55", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process existing HM CL checkpoints for memory lifecycle diagnostics"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-start", type=int, default=None)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    protocol = CLProtocol.load(args.data_root.expanduser())
    checkpoint_paths = _discover_checkpoints(args.checkpoint_dir.expanduser())
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no task_XX_best.pt checkpoints found under {args.checkpoint_dir}"
        )
    regime_id = _select_recurrent_regime(protocol)
    anchor = next(
        (item for item in protocol.anchors if item.regime_id == regime_id), None
    )
    if anchor is None:
        raise ValueError(f"manifest has no fixed anchor for recurrent regime {regime_id}")
    sequences = _load_cl_dataset(anchor.path, protocol.event_dim, args.max_sequences)
    lower = min(checkpoint_paths) if args.task_start is None else int(args.task_start)
    upper = max(checkpoint_paths) if args.task_end is None else int(args.task_end)
    selected_tasks = [
        task for task in sorted(checkpoint_paths)
        if lower <= task <= upper and task in protocol.tasks
    ]
    if not selected_tasks:
        raise ValueError("selected checkpoint task range is empty")
    first_checkpoint_task = protocol.first_seen.get(regime_id)
    if first_checkpoint_task not in checkpoint_paths:
        first_checkpoint_task = min(selected_tasks)

    rows: list[dict[str, Any]] = []
    semantic_only_cache: dict[int, dict[str, Any]] = {}
    reference_task = first_checkpoint_task
    if reference_task in checkpoint_paths:
        semantic_only_cache[reference_task] = _run_condition(
            checkpoint_paths[reference_task],
            sequences,
            "frozen/semantic_only",
            args.device,
        )
    semantic_reference = (
        semantic_only_cache[reference_task]["semantic_mean"].clone()
        if reference_task in semantic_only_cache else None
    )
    boundaries: dict[tuple[int, str], dict[str, Any]] = {}
    for task_id in selected_tasks:
        shift_type, boundary_name, stage_label = _stage_for_task(
            protocol, task_id, regime_id
        )
        if task_id not in semantic_only_cache:
            semantic_only_cache[task_id] = _run_condition(
                checkpoint_paths[task_id],
                sequences,
                "frozen/semantic_only",
                args.device,
            )
        variants = {
            "semantic": semantic_only_cache[task_id],
            "semantic_episodic": _run_condition(
                checkpoint_paths[task_id], sequences, "frozen/episodic_only", args.device
            ),
            "full": _run_condition(
                checkpoint_paths[task_id], sequences, "fast_adapt/full", args.device
            ),
        }
        semantic_vector = variants["semantic"]["semantic_mean"]
        if semantic_reference is None:
            semantic_reference = semantic_vector.clone()
        semantic_shift = float((semantic_vector - semantic_reference).norm().item())
        nll_semantic = float(variants["semantic"]["nll"])
        nll_sem_epi = float(variants["semantic_episodic"]["nll"])
        nll_full = float(variants["full"]["nll"])
        row = {
            "checkpoint_task": task_id,
            "stage_label": stage_label,
            "shift_type": shift_type,
            "boundary": boundary_name,
            "regime_id": regime_id,
            "checkpoint": str(checkpoint_paths[task_id].resolve()),
            "scored_events": variants["full"]["scored_events"],
            "nll_semantic": nll_semantic,
            "nll_semantic_episodic": nll_sem_epi,
            "nll_full": nll_full,
            "episodic_gain": nll_semantic - nll_sem_epi,
            "working_gain": nll_sem_epi - nll_full,
            # Reset test: frozen/full has episodic retrieval but no working
            # adaptation; frozen/semantic_only removes both memory additions.
            "nll_frozen_full": nll_sem_epi,
            "nll_frozen_semantic_only": nll_semantic,
            "nll_fast_adapt_full": nll_full,
            "semantic_shift_norm": semantic_shift,
            "episodic_used_norm": variants["semantic_episodic"]["episodic_used_norm"],
            "working_norm": variants["full"]["working_norm"],
        }
        rows.append(row)
        if boundary_name:
            boundaries[(task_id, boundary_name)] = {
                "task_id": task_id,
                "boundary": boundary_name,
                "stage_label": stage_label,
            }

    output_dir = args.output_dir.expanduser()
    _write_csv(output_dir / "memory_lifecycle_A.csv", rows)
    boundary_rows = [boundaries[key] for key in sorted(boundaries)]
    _write_csv(output_dir / "memory_lifecycle_boundaries.csv", boundary_rows)
    _plot(rows, output_dir / "memory_lifecycle_A.png", boundary_rows)
    summary = {
        "benchmark_id": protocol.benchmark_id,
        "manifest": str(protocol.manifest_path),
        "recurrent_regime": regime_id,
        "anchor": str(anchor.path),
        "checkpoint_count": len(rows),
        "training_invoked": False,
        "reset_test": ["fast_adapt/full", "frozen/full", "frozen/semantic_only"],
        "boundaries": boundary_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Memory lifecycle diagnostics written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
