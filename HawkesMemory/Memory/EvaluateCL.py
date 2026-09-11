"""Protocol-separated continual-learning evaluation for Hawkes Memory Tree checkpoints.

The ordinary :mod:`Evaluate` entry point evaluates one flat CSV with one
train/validation/test split.  CL has a different contract: checkpoint ``C_t``
is evaluated on its current task, the next task before learning, and the
independent frozen anchor banks.  The primary benchmark runs frozen,
fast-adapt, and online-write inference as separate state transitions.
Mechanism ablations remain separate memory-view runs.

Run from the repository root with ``PYTHONPATH`` containing both the project
root and ``Memory``::

    PYTHONPATH="$PWD:$PWD/Memory" python -u -m EvaluateCL \
      --data-root "$PWD/Datasets/CL/hm_continual_v2" \
      --checkpoint-dir "/path/to/continual/checkpoints" \
      --output-dir "$PWD/Memory/Eval/CL" \
      --device cuda
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from Evaluate import (
        SUPPORTED_ROUTER_KINDS,
        _curve_summary,
        _jsonable,
        adaptation_curve_rows,
        dataset_fingerprint,
        run_variant,
        run_variant_compact,
        write_csv,
    )
except ModuleNotFoundError:
    from Memory.Evaluate import (
        SUPPORTED_ROUTER_KINDS,
        _curve_summary,
        _jsonable,
        adaptation_curve_rows,
        dataset_fingerprint,
        run_variant,
        run_variant_compact,
        write_csv,
    )

try:
    from Evaluation.core.cl_metrics import (
        AdaptationRecord,
        CLMetricEngine,
        FrozenAnchorRecord,
        HMStateRecord,
        TaskBoundaryRecord,
        build_frozen_anchor_matrix,
        compute_retention_metrics,
        compute_task_boundary_metrics,
    )
    from Evaluation.core.cl_protocol import CLProtocol
except ModuleNotFoundError:
    from core.cl_metrics import (
        AdaptationRecord,
        CLMetricEngine,
        FrozenAnchorRecord,
        HMStateRecord,
        TaskBoundaryRecord,
        build_frozen_anchor_matrix,
        compute_retention_metrics,
        compute_task_boundary_metrics,
    )
    from core.cl_protocol import CLProtocol

from Train.Inference import (
    EvaluationProtocol,
    InferenceConfig,
    MemoryTreeInference,
    inference_config_for_protocol,
)


BEST_CHECKPOINT_RE = re.compile(r"^task_(\d+)_best\.pt$")
LEGACY_CHECKPOINT_RE = re.compile(r"^task_(\d+)\.pt$")
# Compatibility name for callers that imported the old exact-name matcher.
CHECKPOINT_RE = LEGACY_CHECKPOINT_RE
SCALAR_METRICS = (
    "events",
    "sequences",
    "nll_per_event",
    "accuracy",
    "local_time_mae",
)
CL_PROTOCOL_VARIANTS = (
    "frozen/full",
    "fast_adapt/full",
    "online_write/full",
)
LEGACY_CL_VARIANT_MAP = {
    # The historical name was misleading: its implementation enabled
    # sequence-local Working Memory adaptation.  Preserve that behavior under
    # the explicit FAST_ADAPT protocol when old scripts are resumed.
    "full_frozen": "fast_adapt/full",
    "full_online": "online_write/full",
    "no_working": "frozen/full",
}


def _canonical_variant(variant: str) -> str:
    canonical = LEGACY_CL_VARIANT_MAP.get(variant, variant)
    if canonical not in CL_PROTOCOL_VARIANTS:
        raise ValueError(
            f"unsupported CL protocol variant {variant!r}; expected one of "
            f"{CL_PROTOCOL_VARIANTS}"
        )
    return canonical


@dataclass(frozen=True)
class EvaluationSet:
    """One model-facing CL evaluation CSV."""

    name: str
    kind: str
    path: Path
    task_id: int | None = None
    regime_id: str | None = None
    stage_label: str | None = None
    evaluation_scope: str | None = None


@dataclass(frozen=True)
class GroundTruthLaw:
    """Positive Hawkes parameters used only by the CL evaluation layer."""

    regime_id: str
    mu: np.ndarray
    W: np.ndarray
    betas: np.ndarray
    kind: str
    parent_regime: str | None = None


def _parse_sequence_list(value: Any, cast: type) -> list[Any]:
    """Parse the Python/JSON list representation used by CL CSV files."""

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse sequence value {value!r}") from error
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"sequence value must be a list, got {type(parsed).__name__}"
        )
    return [cast(item) for item in parsed]


def _load_cl_dataset(
    path: Path,
    expected_types: int,
    max_sequences: int | None = None,
) -> list[dict[str, Any]]:
    """Load IDs directly; do not infer a new sparse mapping for each split."""

    frame = pd.read_csv(path)
    required = {"event_times", "event_types"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    dataset: list[dict[str, Any]] = []
    for source_index, row in frame.iterrows():
        try:
            times = _parse_sequence_list(row["event_times"], float)
            types = _parse_sequence_list(row["event_types"], int)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid sequence at {path}:{source_index + 2}"
            ) from error
        if not times or len(times) != len(types):
            raise ValueError(
                f"{path}:{source_index + 2} has mismatched or empty events"
            )
        if any(not math.isfinite(value) for value in times):
            raise ValueError(f"{path}:{source_index + 2} contains non-finite times")
        if times[0] < 0.0 or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(
                f"{path}:{source_index + 2} event_times must be non-decreasing"
            )
        if any(event_type < 0 or event_type >= expected_types for event_type in types):
            raise ValueError(
                f"{path}:{source_index + 2} contains an event type outside "
                f"[0, {expected_types - 1}]"
            )
        dataset.append({
            "times": torch.tensor(times, dtype=torch.float32),
            "types": torch.tensor(types, dtype=torch.long),
            "source_index": int(source_index),
        })

    if max_sequences is not None:
        dataset = dataset[:max_sequences]
    if not dataset:
        raise ValueError(f"{path} contains no valid sequences")
    return dataset


def _normalise_data_root(path: Path) -> Path:
    """Accept the canonical generated root and the older nested layout."""

    path = path.expanduser()
    if (path / "task_00").is_dir():
        return path
    nested = path / "Data"
    if (nested / "task_00").is_dir():
        return nested
    return path


def _discover_task_sets(
    data_root: Path,
    protocol: CLProtocol,
) -> dict[int, EvaluationSet]:
    result: dict[int, EvaluationSet] = {}
    for task_id in protocol.task_ids:
        test_path = protocol.split_path(task_id, "test")
        result[task_id] = EvaluationSet(
            name=f"task_{task_id:02d}_test",
            kind="task_test",
            path=test_path,
            task_id=task_id,
        )
    return result


def _discover_checkpoints(checkpoint_dir: Path) -> dict[int, Path]:
    best: dict[int, Path] = {}
    legacy: dict[int, Path] = {}
    for path in sorted(checkpoint_dir.glob("task_*.pt")):
        if not path.is_file():
            continue
        best_match = BEST_CHECKPOINT_RE.fullmatch(path.name)
        if best_match is not None:
            best[int(best_match.group(1))] = path
            continue
        legacy_match = LEGACY_CHECKPOINT_RE.fullmatch(path.name)
        if legacy_match is not None:
            legacy[int(legacy_match.group(1))] = path
    return {
        task_id: best.get(task_id, legacy_path)
        for task_id, legacy_path in legacy.items()
    } | best


def _read_stage_metadata(
    protocol: CLProtocol,
) -> dict[int, dict[str, Any]]:
    """Read labels from the oracle manifest; never use its parameters."""

    metadata: dict[int, dict[str, Any]] = {}
    for task_id in protocol.task_ids:
        task = protocol.task(task_id)
        weights = dict(task.regime_weights)
        metadata[task_id] = {
            "stage_label": task.stage_label,
            "regime_id": next(iter(weights)) if len(weights) == 1 else None,
            "shift_type": task.shift_type,
            "recurrence_of": task.recurrence_of,
            "paired_control": task.paired_control,
            "regime_weights": json.dumps(weights, separators=(",", ":")),
        }
    return metadata


def _discover_anchors(
    data_root: Path,
    protocol: CLProtocol,
) -> list[EvaluationSet]:
    anchors: list[EvaluationSet] = []
    for item in protocol.anchors:
        regime_id = item.regime_id
        path = item.path
        anchors.append(EvaluationSet(
            name=f"anchor_{_safe_name(regime_id)}",
            kind="anchor",
            path=path,
            regime_id=regime_id,
            stage_label="frozen_anchor",
            evaluation_scope=item.evaluation_scope,
        ))
    return anchors


def _paired_control_set(
    protocol: CLProtocol,
    task_id: int,
    *,
    pre_update: bool,
) -> EvaluationSet | None:
    """Build the manifest-declared matched-control evaluation set."""

    task = protocol.task(task_id)
    if task.paired_control is None:
        return None
    path = protocol.control_split_path(task.paired_control, "test")
    suffix = "_pre" if pre_update else ""
    return EvaluationSet(
        name=f"task_{task_id:02d}_control{suffix}",
        kind="matched_control_pre" if pre_update else "matched_control",
        path=path,
        task_id=task_id,
        stage_label=f"{task.stage_label or task.shift_type}_control",
        evaluation_scope="matched_control",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return value


def _mean(values: Iterable[Any]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return sum(clean) / len(clean) if clean else None


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, dict):
        raise TypeError(f"checkpoint must contain a dictionary: {path}")
    router_kind = metadata.get("model_config", {}).get("router_kind")
    if router_kind not in SUPPORTED_ROUTER_KINDS:
        raise ValueError(
            f"incompatible CL checkpoint {path}: router_kind={router_kind!r}; "
            f"expected one of {sorted(SUPPORTED_ROUTER_KINDS)}"
        )
    config = metadata.get("model_config", {})
    if "num_event_types" not in config:
        raise KeyError(f"checkpoint has no model_config.num_event_types: {path}")
    return metadata


def _tree_health(checkpoint: Path) -> dict[str, Any]:
    """Collect topology facts without evaluating any data."""

    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device="cpu",
        inference_config=inference_config_for_protocol(
            EvaluationProtocol.FROZEN,
        ),
    )
    tree = inference.tree
    leaf_depths = [int(tree.nodes[node_id].depth) for node_id in tree.leaf_ids]
    memory_rows = sum(len(bank) for bank in tree.episodic_memory.banks.values())
    def serialized_size(value: Any) -> int | None:
        try:
            buffer = io.BytesIO()
            torch.save(value, buffer)
            return int(buffer.tell())
        except Exception:
            return None

    episodic_state = getattr(tree.episodic_memory, "state_dict", lambda: {})()
    semantic_state = tree.semantic_theta_table()
    return {
        "node_count": len(tree.all_node_ids),
        "leaf_count": len(tree.leaf_ids),
        "leaf_ids": list(tree.leaf_ids),
        "max_depth": max(leaf_depths, default=0),
        "mean_leaf_depth": _mean(leaf_depths),
        "memory_rows": memory_rows,
        "episodic_rows": memory_rows,
        "episodic_bytes": serialized_size(episodic_state),
        "semantic_bytes": serialized_size(semantic_state),
    }


def _batch_cache_dir(
    output_dir: Path,
    checkpoint_task: int,
    variant: str,
) -> Path:
    return (
        output_dir
        / "cache"
        / f"checkpoint_task_{checkpoint_task:02d}"
        / "checkpoint_batch"
        / _safe_name(variant)
    )


def _load_or_run_batch(
    *,
    checkpoint: Path,
    checkpoint_task: int,
    evaluation_sets: Sequence[EvaluationSet],
    variant: str,
    evaluation_cache: Mapping[Path, Sequence[Mapping[str, Any]]],
    data_sha_cache: Mapping[Path, str],
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], float, bool]:
    """Evaluate each set from a fresh checkpoint for one protocol.

    ``ONLINE_WRITE`` mutates the in-memory episodic bank.  A single inference
    object must therefore never be shared by two evaluation sets (or by two
    protocol rows).  Read-only protocols use the compact path, while the
    online protocol uses the ordinary causal path so its writes and usage
    updates remain active across the sequences within one evaluation set.
    """

    variant = _canonical_variant(variant)
    capture_event_rows = bool(
        args.save_event_predictions or not variant.startswith("frozen/")
    )

    cache = _batch_cache_dir(args.output_dir, checkpoint_task, variant)
    metrics_path = cache / "metrics.json"
    events_path = cache / "event_rows.json"
    meta_path = cache / "meta.json"
    expected_meta = {
        "cache_format": "protocol_eval_per_set_v2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "variant": variant,
        "sequence_batch_size": int(args.eval_batch_size),
        "save_event_predictions": bool(args.save_event_predictions),
        "capture_event_rows": capture_event_rows,
        "evaluation_sets": [
            {
                "name": evaluation_set.name,
                "kind": evaluation_set.kind,
                "path": str(evaluation_set.path.resolve()),
                "data_sha256": data_sha_cache[evaluation_set.path],
                "task_id": evaluation_set.task_id,
                "regime_id": evaluation_set.regime_id,
                "stage_label": evaluation_set.stage_label,
                "evaluation_scope": evaluation_set.evaluation_scope,
                "sequence_count": len(
                    evaluation_cache[evaluation_set.path]
                ),
            }
            for evaluation_set in evaluation_sets
        ],
    }
    cache_valid = False
    if (
        args.resume
        and metrics_path.is_file()
        and meta_path.is_file()
        and (
            not capture_event_rows
            or events_path.is_file()
        )
    ):
        try:
            cache_valid = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                == expected_meta
            )
        except (OSError, json.JSONDecodeError):
            cache_valid = False
    if cache_valid:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        events = (
            json.loads(events_path.read_text(encoding="utf-8"))
            if capture_event_rows else []
        )
        return metrics, events, 0.0, True

    progress_dir = None
    if args.resume:
        progress_dir = cache
        cache.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(expected_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    def aggregate_event_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        events = len(rows)
        if not events:
            return {
                "events": 0,
                "sequences": len({row.get("source_index") for row in rows}),
                "nll_per_event": None,
                "accuracy": None,
                "local_time_mae": None,
            }
        return {
            "events": events,
            "sequences": len({row.get("source_index") for row in rows}),
            "nll_per_event": _mean(float(row["nll"]) for row in rows),
            "accuracy": _mean(
                float(
                    int(row.get("predicted_type_at_event_time", -1))
                    == int(row.get("true_type", -2))
                )
                for row in rows
            ),
            "local_time_mae": _mean(
                abs(float(row.get("predicted_time", 0.0))
                    - float(row.get("true_time", 0.0)))
                for row in rows
            ),
        }

    metrics: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    elapsed = 0.0
    if not evaluation_sets:
        raise ValueError(
            f"no evaluation sets available for checkpoint task_{checkpoint_task:02d}"
        )
    for evaluation_set in evaluation_sets:
        set_sequences = [
            {
                **dict(sequence),
                # These fields are sequence metadata.  They are used for
                # aggregation only and never enter model computation.
                "eval_set_id": evaluation_set.name,
                "eval_kind": evaluation_set.kind,
                "eval_task": evaluation_set.task_id,
                "regime_id": evaluation_set.regime_id,
                "stage_label": evaluation_set.stage_label,
                "_sequence_position": sequence_position,
            }
            for sequence_position, sequence in enumerate(
                evaluation_cache[evaluation_set.path]
            )
        ]
        if not set_sequences:
            metrics[evaluation_set.name] = {
                "events": 0,
                "sequences": 0,
                "nll_per_event": None,
                "accuracy": None,
                "local_time_mae": None,
            }
            continue
        set_progress_dir = None
        if progress_dir is not None:
            set_progress_dir = progress_dir / _safe_name(evaluation_set.name)
        if variant.startswith("online_write/"):
            checkpoint_before = _sha256(checkpoint)
            rows, _inference, set_elapsed = run_variant(
                checkpoint,
                set_sequences,
                variant,
                args.device,
                progress_dir=set_progress_dir,
                prototype_duplicate_threshold=args.prototype_duplicate_threshold,
                prototype_mode_threshold=args.prototype_mode_threshold,
                prototype_context_alias_capacity=args.prototype_context_alias_capacity,
                verbose=args.verbose,
            )
            if _sha256(checkpoint) != checkpoint_before:
                raise RuntimeError(
                    f"ONLINE_WRITE mutated checkpoint file {checkpoint}; "
                    "writes must remain in the disposable inference object"
                )
            set_metrics = aggregate_event_rows(rows)
            if capture_event_rows:
                event_rows.extend(rows)
        else:
            set_metrics_by_group, set_events, _inference, set_elapsed = (
                run_variant_compact(
                    checkpoint,
                    set_sequences,
                    variant,
                    args.device,
                    sequence_batch_size=args.eval_batch_size,
                    progress_dir=set_progress_dir,
                    prototype_duplicate_threshold=args.prototype_duplicate_threshold,
                    prototype_mode_threshold=args.prototype_mode_threshold,
                    prototype_context_alias_capacity=args.prototype_context_alias_capacity,
                    capture_event_predictions=capture_event_rows,
                    verbose=args.verbose,
                )
            )
            set_metrics = set_metrics_by_group.get(evaluation_set.name, {})
            if capture_event_rows:
                event_rows.extend(set_events)
        metrics[evaluation_set.name] = set_metrics
        elapsed += float(set_elapsed)
    if args.resume:
        metrics_path.write_text(
            json.dumps(_jsonable(metrics), ensure_ascii=False),
            encoding="utf-8",
        )
        if capture_event_rows:
            events_path.write_text(
                json.dumps(_jsonable(event_rows), ensure_ascii=False),
                encoding="utf-8",
            )
    return metrics, event_rows, elapsed, False


def _metric_row(
    *,
    checkpoint_task: int,
    checkpoint: Path,
    evaluation_set: EvaluationSet,
    variant: str,
    metrics: Mapping[str, Any],
    tree: Mapping[str, Any],
    elapsed: float,
    from_cache: bool,
    data_sha256: str,
) -> dict[str, Any]:
    variant = _canonical_variant(variant)
    row: dict[str, Any] = {
        "checkpoint_task": checkpoint_task,
        "checkpoint": str(checkpoint.resolve()),
        "eval_name": evaluation_set.name,
        "eval_kind": evaluation_set.kind,
        "eval_task": evaluation_set.task_id,
        "regime_id": evaluation_set.regime_id,
        "stage_label": evaluation_set.stage_label,
        "evaluation_scope": evaluation_set.evaluation_scope,
        "data_path": str(evaluation_set.path.resolve()),
        "data_sha256": data_sha256,
        "variant": variant,
        "protocol": variant.split("/", 1)[0],
        "memory_view": variant.split("/", 1)[1],
        "elapsed_seconds": elapsed,
        "from_cache": from_cache,
        "leaf_count": tree.get("leaf_count"),
        "node_count": tree.get("node_count"),
        "max_depth": tree.get("max_depth"),
        "mean_leaf_depth": tree.get("mean_leaf_depth"),
        "memory_rows": tree.get("memory_rows"),
    }
    for name in SCALAR_METRICS:
        if name in metrics:
            row[name] = _finite_scalar(metrics[name])
    events = metrics.get("events")
    row["events_per_second"] = (
        float(events) / elapsed if events and elapsed > 0.0 else None
    )
    return row


def _decorate_event_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fields = {
        "checkpoint_task": metric_row["checkpoint_task"],
        "checkpoint": metric_row["checkpoint"],
        "eval_name": metric_row["eval_name"],
        "eval_kind": metric_row["eval_kind"],
        "eval_task": metric_row["eval_task"],
        "regime_id": metric_row["regime_id"],
        "stage_label": metric_row["stage_label"],
        "evaluation_scope": metric_row.get("evaluation_scope"),
    }
    return [{**fields, **dict(row)} for row in rows]


class _EventPredictionWriter:
    """Stream event predictions so a full CL matrix stays memory-bounded."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", newline="", encoding="utf-8-sig")
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] = []

    def write(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric_row: Mapping[str, Any],
    ) -> None:
        decorated = _decorate_event_rows(rows, metric_row)
        if not decorated:
            return
        if self.writer is None:
            self.fieldnames = sorted({
                key for row in decorated for key in row
            })
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            self.writer.writeheader()
        for row in decorated:
            values = {}
            for key in self.fieldnames:
                value = row.get(key)
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(
                        _jsonable(value), ensure_ascii=False
                    )
                values[key] = value
            self.writer.writerow(values)

    def close(self) -> None:
        self.handle.close()


def _continual_summary(
    metric_rows: Sequence[Mapping[str, Any]],
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    tree_by_checkpoint: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize current-task quality and checkpoint topology.

    Retention is deliberately not computed from task-test averages here.  The
    CL retention metrics come from ``_law_metrics`` and the frozen anchor
    matrix, which prevents future/unseen laws from contaminating the result.
    """

    output: list[dict[str, Any]] = []
    task_rows = [row for row in metric_rows if row["eval_kind"] == "task_test"]
    for variant in variants:
        for checkpoint_task in checkpoint_tasks:
            current = next(
                (
                    row for row in task_rows
                    if row["variant"] == variant
                    and row["checkpoint_task"] == checkpoint_task
                    and row["eval_task"] == checkpoint_task
                ),
                None,
            )
            tree = tree_by_checkpoint[checkpoint_task]
            output.append({
                "variant": variant,
                "checkpoint_task": checkpoint_task,
                "current_nll_per_event": current.get("nll_per_event") if current else None,
                "current_accuracy": current.get("accuracy") if current else None,
                "current_local_time_mae": current.get("local_time_mae") if current else None,
                "leaf_count": tree.get("leaf_count"),
                "node_count": tree.get("node_count"),
                "memory_rows": tree.get("memory_rows"),
            })
    return output


def _law_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    first_seen: Mapping[str, int],
    protocol: CLProtocol | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute CLNLL, forgetting, and BWT through the canonical engine."""

    anchors = [row for row in metric_rows if row["eval_kind"] == "anchor"]
    law_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        records = [
            FrozenAnchorRecord(
                checkpoint_task=int(row["checkpoint_task"]),
                regime_id=str(row["regime_id"]),
                nll_per_event=row.get("nll_per_event"),
                num_events=int(row.get("events") or 0),
                evaluation_scope=str(row.get("evaluation_scope", "persistent")),
            )
            for row in anchors
            if row.get("variant") == variant and row.get("regime_id") is not None
        ]
        if protocol is None:
            from Evaluation.core.cl_metrics import (
                build_frozen_anchor_matrix,
                compute_retention_metrics,
            )
            matrix = build_frozen_anchor_matrix(
                records,
                first_seen=first_seen,
                persistent_regimes=first_seen,
            )
            raw_law_rows, raw_summary_rows = compute_retention_metrics(matrix)
        else:
            retention = CLMetricEngine(protocol).retention(records)
            raw_law_rows = retention["law_rows"]
            raw_summary_rows = retention["summary_rows"]
        law_rows.extend({"variant": variant, **row} for row in raw_law_rows)
        summary_rows.extend({"variant": variant, **row} for row in raw_summary_rows)
        # Keep a stable row for a selected checkpoint whose anchors were not
        # completed, while leaving all metric values explicitly unavailable.
        existing = {int(row["checkpoint_task"]) for row in raw_summary_rows}
        for checkpoint_task in checkpoint_tasks:
            if checkpoint_task in existing:
                continue
            summary_rows.append({
                "variant": variant,
                "checkpoint_task": int(checkpoint_task),
                "seen_law_count": 0,
                "clnll": None,
                "average_forgetting": None,
                "average_bwt": None,
                "bwt_law_count": 0,
            })
    return (
        law_rows,
        sorted(summary_rows, key=lambda row: (row["variant"], row["checkpoint_task"])),
    )


def _stage_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
    protocol: CLProtocol | None = None,
    scratch_nll_by_task: Mapping[int, float | None] | None = None,
) -> list[dict[str, Any]]:
    """Compute task-boundary fields through the canonical metric engine."""

    output: list[dict[str, Any]] = []
    task_ids = sorted({
        int(row["eval_task"])
        for row in metric_rows
        if row.get("eval_task") is not None
        and row["eval_kind"] in {"task_test", "task_test_pre"}
    })
    for variant in variants:
        records: list[TaskBoundaryRecord] = []
        source_rows: dict[
            int, tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]
        ] = {}
        for task_id in task_ids:
            pre = next(
                (
                    row for row in metric_rows
                    if row["variant"] == variant
                    and row["eval_kind"] == "task_test_pre"
                    and int(row["eval_task"]) == task_id
                ),
                None,
            )
            post = next(
                (
                    row for row in metric_rows
                    if row["variant"] == variant
                    and row["eval_kind"] == "task_test"
                    and int(row["eval_task"]) == task_id
                    and int(row["checkpoint_task"]) == task_id
                ),
                None,
            )
            if pre is None and post is None:
                continue
            spec = protocol.task(task_id) if protocol is not None else None
            records.append(TaskBoundaryRecord(
                task_id=task_id,
                pre_nll=pre.get("nll_per_event") if pre else None,
                post_nll=post.get("nll_per_event") if post else None,
                scratch_nll=(
                    scratch_nll_by_task.get(task_id)
                    if scratch_nll_by_task is not None else None
                ),
                shift_type=spec.shift_type if spec else None,
                recurrence_of=spec.recurrence_of if spec else None,
            ))
            source_rows[task_id] = (pre, post)
        computed = (
            CLMetricEngine(protocol).task_boundaries(records)
            if protocol is not None
            else compute_task_boundary_metrics(records)
        )
        for row in computed:
            pre, post = source_rows[row["task_id"]]
            source = post or pre or {}
            output.append({
                "task_id": row["task_id"],
                "variant": variant,
                "stage_label": source.get("stage_label"),
                "regime_id": source.get("regime_id"),
                "shift_type": row["shift_type"],
                "recurrence_of": row["recurrence_of"],
                "pre_checkpoint_task": pre.get("checkpoint_task") if pre else None,
                "post_checkpoint_task": post.get("checkpoint_task") if post else None,
                "pre_nll_per_event": row["pre_nll"],
                "post_nll_per_event": row["post_nll"],
                "scratch_nll_per_event": row["scratch_nll"],
                "adaptation_gain_nll": row["adaptation_gain_nll"],
                "fwt_nll": row["fwt_nll"],
                "fwt_eligible": row["fwt_eligible"],
                "fwt_status": row["fwt_status"],
                "new_persistent_regimes": row["new_persistent_regimes"],
            })
    return output


def _anchor_nll_matrix(
    metric_rows: Sequence[Mapping[str, Any]],
    persistent_regimes: Iterable[str] | None = None,
    protocol: CLProtocol | None = None,
) -> list[dict[str, Any]]:
    """Make the paper-style checkpoint × persistent-regime NLL matrix.

    Transient anchors can be useful diagnostics, but including them in the
    CL matrix would let an intentionally unseen anomaly affect the persistent
    law averages reported alongside it.
    """

    anchors = [
        row
        for row in metric_rows
        if row["eval_kind"] == "anchor"
        and row.get("regime_id") is not None
    ]
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in anchors}):
        records = [
            FrozenAnchorRecord(
                checkpoint_task=int(row["checkpoint_task"]),
                regime_id=str(row["regime_id"]),
                nll_per_event=row.get("nll_per_event"),
                num_events=int(row.get("events") or 0),
                evaluation_scope=str(row.get("evaluation_scope", "persistent")),
            )
            for row in anchors
            if str(row["variant"]) == variant
        ]
        if protocol is not None:
            matrix = CLMetricEngine(protocol).frozen_anchor_matrix(records)
        else:
            matrix = build_frozen_anchor_matrix(
                records,
                persistent_regimes=persistent_regimes,
            )
        output.extend({"variant": variant, **row} for row in matrix.to_rows())
    return output


def _frozen_anchor_records(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str = "frozen/full",
) -> list[FrozenAnchorRecord]:
    """Reduce evaluator rows to the canonical persistent-anchor records."""

    variant = _canonical_variant(variant)
    return [
        FrozenAnchorRecord(
            checkpoint_task=int(row["checkpoint_task"]),
            regime_id=str(row["regime_id"]),
            nll_per_event=row.get("nll_per_event"),
            num_events=int(row.get("events") or 0),
            evaluation_scope=str(row.get("evaluation_scope", "persistent")),
        )
        for row in metric_rows
        if row.get("eval_kind") == "anchor"
        and row.get("variant") == variant
        and row.get("regime_id") is not None
    ]


def _task_boundary_records(
    metric_rows: Sequence[Mapping[str, Any]],
    protocol: CLProtocol,
    *,
    variant: str = "frozen/full",
    scratch_nll_by_task: Mapping[int, float | None] | None = None,
) -> list[TaskBoundaryRecord]:
    """Reduce task-test pre/post rows to one record per protocol task."""

    variant = _canonical_variant(variant)
    output: list[TaskBoundaryRecord] = []
    for task_id in protocol.task_ids:
        pre = next(
            (
                row for row in metric_rows
                if row.get("variant") == variant
                and row.get("eval_kind") == "task_test_pre"
                and row.get("eval_task") is not None
                and int(row["eval_task"]) == task_id
            ),
            None,
        )
        post = next(
            (
                row for row in metric_rows
                if row.get("variant") == variant
                and row.get("eval_kind") == "task_test"
                and row.get("eval_task") is not None
                and int(row["eval_task"]) == task_id
                and int(row["checkpoint_task"]) == task_id
            ),
            None,
        )
        if pre is None and post is None:
            continue
        spec = protocol.task(task_id)
        output.append(TaskBoundaryRecord(
            task_id=task_id,
            pre_nll=pre.get("nll_per_event") if pre else None,
            post_nll=post.get("nll_per_event") if post else None,
            scratch_nll=(
                scratch_nll_by_task.get(task_id)
                if scratch_nll_by_task is not None else None
            ),
            shift_type=spec.shift_type,
            recurrence_of=spec.recurrence_of,
        ))
    return output


def _read_topology_events(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Read committed topology transactions emitted by the HM trainer."""

    candidates = (
        checkpoint_dir.parent / "topology_events.jsonl",
        checkpoint_dir / "topology_events.jsonl",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Older runs may have used this path for plain-text
                # diagnostics.  Those lines are not topology transactions.
                continue
            if not isinstance(value, dict):
                continue
            action = str(value.get("action", ""))
            if action not in {"split", "merge", "topology_prune"}:
                continue
            try:
                task_id = int(value["task_id"])
                epoch = int(value["global_epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                **value,
                "task_id": task_id,
                "global_epoch": epoch,
                "action": action,
                "committed": bool(value.get("committed", True)),
            })
    return rows


def _hm_state_records(
    checkpoint_tasks: Sequence[int],
    tree_by_checkpoint: Mapping[int, Mapping[str, Any]],
    topology_events: Sequence[Mapping[str, Any]],
) -> list[HMStateRecord]:
    """Attach cumulative committed topology counts to checkpoint state."""

    rows: list[HMStateRecord] = []
    for task_id in checkpoint_tasks:
        events = [
            row for row in topology_events
            if bool(row.get("committed", True))
            and int(row.get("task_id", task_id)) <= int(task_id)
        ]
        tree = tree_by_checkpoint.get(task_id, {})
        rows.append(HMStateRecord(
            task_id=int(task_id),
            node_count=tree.get("node_count"),
            leaf_count=tree.get("leaf_count"),
            episodic_rows=tree.get("episodic_rows", tree.get("memory_rows")),
            episodic_bytes=tree.get("episodic_bytes"),
            semantic_bytes=tree.get("semantic_bytes"),
            split_count=sum(row.get("action") == "split" for row in events),
            merge_count=sum(row.get("action") == "merge" for row in events),
            prune_count=sum(
                row.get("action") == "topology_prune" for row in events
            ),
            nise=None,
        ))
    return rows


def _concat_support_prefix(
    support_sequences: Sequence[Mapping[str, Any]],
    K: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Flatten the first K support events into one causal prefix."""

    if K == 0:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    times_parts: list[torch.Tensor] = []
    type_parts: list[torch.Tensor] = []
    remaining = int(K)
    last_time = 0.0
    for sequence in support_sequences:
        times = torch.as_tensor(sequence["times"], dtype=torch.float32).reshape(-1)
        types = torch.as_tensor(sequence["types"], dtype=torch.long).reshape(-1)
        take = min(remaining, int(times.numel()))
        if take <= 0:
            break
        local_times = times[:take]
        shift = 0.0 if not times_parts else last_time + 1.0 - float(local_times[0])
        local_times = local_times + shift
        times_parts.append(local_times)
        type_parts.append(types[:take])
        last_time = float(local_times[-1])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        return None
    return torch.cat(times_parts), torch.cat(type_parts)


def _adaptation_records(
    protocol: CLProtocol,
    checkpoint_paths: Mapping[int, Path],
    selected_tasks: Sequence[int],
    expected_types: int,
    args: argparse.Namespace,
    *,
    variant: str = "fast_adapt/full",
) -> list[AdaptationRecord]:
    """Run independent K-indexed support/query evaluations from fresh clones."""

    if getattr(args, "no_adaptation_evaluation", False):
        return []
    variant = _canonical_variant(variant)
    records: list[AdaptationRecord] = []
    for task_id in selected_tasks:
        spec = protocol.adaptation(task_id)
        if spec is None:
            continue
        previous_tasks = [
            candidate for candidate in protocol.task_ids
            if candidate < task_id and candidate in checkpoint_paths
        ]
        if not previous_tasks:
            continue
        checkpoint = checkpoint_paths[max(previous_tasks)]
        support = _load_cl_dataset(
            spec["support"], expected_types, args.max_sequences
        )
        query = _load_cl_dataset(
            spec["query"], expected_types, args.max_sequences
        )
        for K in spec["K"]:
            prefix = _concat_support_prefix(support, int(K))
            if prefix is None:
                records.append(AdaptationRecord(
                    task_id=int(task_id),
                    K=int(K),
                    pre_nll=None,
                    adapted_nll=None,
                    protocol=variant.split("/", 1)[0],
                ))
                continue
            support_times, support_types = prefix
            combined: list[dict[str, Any]] = []
            for query_index, sequence in enumerate(query):
                query_times = torch.as_tensor(
                    sequence["times"], dtype=torch.float32
                ).reshape(-1)
                query_types = torch.as_tensor(
                    sequence["types"], dtype=torch.long
                ).reshape(-1)
                if support_times.numel():
                    shift = float(support_times[-1]) + 1.0 - float(query_times[0])
                else:
                    shift = -float(query_times[0])
                combined.append({
                    "times": torch.cat((support_times, query_times + shift)),
                    "types": torch.cat((support_types, query_types)),
                    "source_index": (
                        int(task_id) * 1_000_000
                        + int(K) * 10_000
                        + query_index
                    ),
                })
            event_rows, _inference, _elapsed = run_variant(
                checkpoint,
                combined,
                variant,
                args.device,
                verbose=args.verbose,
            )
            query_rows = [
                row for row in event_rows
                if int(row.get("event_index", -1)) >= int(K)
            ]
            adapted_nll = _mean(row.get("nll") for row in query_rows)
            records.append(AdaptationRecord(
                task_id=int(task_id),
                K=int(K),
                pre_nll=None,
                adapted_nll=adapted_nll,
                protocol=variant.split("/", 1)[0],
            ))
    return records


def _fwt_scratch_nlls(
    scratch_checkpoint: Path | None,
    protocol: CLProtocol,
    selected_tasks: Sequence[int],
    evaluation_cache: Mapping[Path, Sequence[Mapping[str, Any]]],
    expected_types: int,
    args: argparse.Namespace,
) -> dict[int, float | None]:
    """Evaluate every task from the same saved C_init for protocol-scoped FWT."""

    if scratch_checkpoint is None:
        return {}
    scratch_checkpoint = Path(scratch_checkpoint).expanduser().resolve()
    if not scratch_checkpoint.is_file():
        raise FileNotFoundError(
            f"FWT scratch checkpoint does not exist: {scratch_checkpoint}"
        )
    output: dict[int, float | None] = {}
    for task_id in selected_tasks:
        path = protocol.split_path(task_id, "test")
        sequences = evaluation_cache.get(path)
        if not sequences:
            continue
        event_rows, _inference, _elapsed = run_variant(
            scratch_checkpoint,
            [
                {
                    **dict(sequence),
                    "eval_set_id": f"task_{task_id:02d}_scratch",
                    "eval_kind": "task_test",
                    "eval_task": task_id,
                    "regime_id": None,
                    "stage_label": "fwt_scratch",
                }
                for sequence in sequences
            ],
            "frozen/full",
            args.device,
            verbose=args.verbose,
        )
        output[int(task_id)] = _mean(row.get("nll") for row in event_rows)
    return output


def _special_case_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    variant: str = "frozen/full",
    protocol: CLProtocol | None = None,
) -> list[dict[str, Any]]:
    """Compute schedule-specific diagnostics from protocol task metadata."""

    if protocol is None:
        return []
    variant = _canonical_variant(variant)

    lookup = {
        (int(row["checkpoint_task"]), str(row.get("regime_id"))): row.get("nll_per_event")
        for row in metric_rows
        if row["eval_kind"] == "anchor" and row["variant"] == variant
    }
    task_lookup = {
        (
            int(row["checkpoint_task"]),
            int(row["eval_task"]),
            str(row["eval_kind"]),
        ): row.get("nll_per_event")
        for row in metric_rows
        if row.get("eval_task") is not None and row["variant"] == variant
    }

    tasks = [protocol.task(task_id) for task_id in protocol.task_ids]
    task_ids = list(protocol.task_ids)
    tasks_by_id = {task.task_id: task for task in tasks}
    first_seen = dict(protocol.first_seen)
    output: list[dict[str, Any]] = []

    def add_metric(
        name: str,
        formula: str,
        left_key: tuple[int, str],
        right_key: tuple[int, str],
        expected: str,
    ) -> None:
        left = lookup.get(left_key)
        right = lookup.get(right_key)
        if left is None or right is None:
            return
        output.append({
            "metric": name,
            "variant": variant,
            "formula": formula,
            "value": float(left) - float(right),
            "left_value": left,
            "right_value": right,
            "expected": expected,
        })

    def add_task_difference(
        name: str,
        formula: str,
        left_key: tuple[int, int, str],
        right_key: tuple[int, int, str],
        expected: str,
    ) -> None:
        left = task_lookup.get(left_key)
        right = task_lookup.get(right_key)
        if left is None or right is None:
            return
        output.append({
            "metric": name,
            "variant": variant,
            "formula": formula,
            "value": float(left) - float(right),
            "left_value": left,
            "right_value": right,
            "expected": expected,
        })

    def previous_task(task_id: int) -> int | None:
        prior = [candidate for candidate in task_ids if candidate < task_id]
        return max(prior) if prior else None

    def next_task(task_id: int) -> int | None:
        later = [candidate for candidate in task_ids if candidate > task_id]
        return min(later) if later else None

    for task in tasks:
        task_id = task.task_id
        shift_type = task.shift_type
        parent_id = task.recurrence_of
        if shift_type in {"exact_recurrence", "long_gap_recurrence"} and parent_id:
            prior_task = previous_task(task_id)
            seen_task = first_seen.get(parent_id)
            if prior_task is not None and seen_task is not None:
                prefix = _safe_name(parent_id)
                if shift_type == "exact_recurrence":
                    add_metric(
                        f"{prefix}_retention_before_task{task_id}",
                        f"L_{prior_task},{parent_id} - L_{seen_task},{parent_id}",
                        (prior_task, parent_id),
                        (seen_task, parent_id),
                        "near_zero_or_negative",
                    )
                else:
                    add_metric(
                        f"{prefix}_long_gap_reference_task{task_id}",
                        f"L_{prior_task},{parent_id} - L_{seen_task},{parent_id}",
                        (prior_task, parent_id),
                        (seen_task, parent_id),
                        "near_zero_or_negative",
                    )
                add_metric(
                    f"{prefix}_{shift_type}_recovery_task{task_id}",
                    f"L_{prior_task},{parent_id} - L_{task_id},{parent_id}",
                    (prior_task, parent_id),
                    (task_id, parent_id),
                    "positive",
                )

        if shift_type in {"near_recurrence", "specialization", "new_specialization"}:
            later_task = next_task(task_id)
            if later_task is not None:
                regime_ids = list(task.regime_weights)
                if parent_id and parent_id in first_seen:
                    add_metric(
                        f"{_safe_name(parent_id)}_{shift_type}_impact_task{later_task}",
                        f"L_{later_task},{parent_id} - L_{task_id},{parent_id}",
                        (later_task, parent_id),
                        (task_id, parent_id),
                        "near_zero_or_negative",
                    )
                for regime_id in regime_ids:
                    if regime_id == parent_id:
                        continue
                    add_metric(
                        f"{_safe_name(regime_id)}_{shift_type}_gain_task{later_task}",
                        f"L_{task_id},{regime_id} - L_{later_task},{regime_id}",
                        (task_id, regime_id),
                        (later_task, regime_id),
                        "positive",
                    )

        if shift_type in {"transient", "transient_anomaly"}:
            prior_task = previous_task(task_id)
            if prior_task is None or task.paired_control is None:
                continue
            control_prefix = _safe_name(task.paired_control)
            add_task_difference(
                f"{control_prefix}_transient_control_adaptation_task{task_id}",
                f"L_{{{prior_task},{task_id}}}^control - "
                f"L_{{{task_id},{task_id}}}^control",
                (prior_task, task_id, "matched_control_pre"),
                (task_id, task_id, "matched_control"),
                "positive",
            )
            stream_pre = task_lookup.get(
                (prior_task, task_id, "task_test_pre")
            )
            stream_post = task_lookup.get(
                (task_id, task_id, "task_test")
            )
            control_pre = task_lookup.get(
                (prior_task, task_id, "matched_control_pre")
            )
            control_post = task_lookup.get(
                (task_id, task_id, "matched_control")
            )
            if all(value is not None for value in (
                stream_pre, stream_post, control_pre, control_post
            )):
                stream_gain = float(stream_pre) - float(stream_post)
                control_gain = float(control_pre) - float(control_post)
                output.append({
                    "metric": f"{control_prefix}_transient_excess_adaptation_task{task_id}",
                    "variant": variant,
                    "formula": (
                        f"(L_{{{prior_task},{task_id}}} - L_{{{task_id},{task_id}}})"
                        f" - (L_{{{prior_task},{task_id}}}^control - "
                        f"L_{{{task_id},{task_id}}}^control)"
                    ),
                    "value": stream_gain - control_gain,
                    "left_value": stream_gain,
                    "right_value": control_gain,
                    "expected": "near_zero_or_negative",
                })
    mixture_tasks = [
        task.task_id
        for task in tasks
        if task.shift_type == "mixture"
    ]
    for task_id in mixture_tasks:
        prior_task = previous_task(task_id)
        if prior_task is None:
            continue
        mixture_pre = task_lookup.get((prior_task, task_id, "task_test_pre"))
        mixture_post = task_lookup.get((task_id, task_id, "task_test"))
        if mixture_pre is None or mixture_post is None:
            continue
        weights = tasks_by_id[task_id].regime_weights
        label = "_".join(str(regime_id) for regime_id in weights)
        output.append({
            "metric": f"{_safe_name(label)}_mixture_adaptation_task{task_id}",
            "variant": variant,
            "formula": f"P_{task_id}^pre - P_{task_id}^post",
            "value": float(mixture_pre) - float(mixture_post),
            "left_value": mixture_pre,
            "right_value": mixture_post,
            "expected": "positive",
        })
    return output


def _load_ground_truth(
    data_root: Path,
    expected_types: int,
    expected_basis: int,
) -> tuple[dict[str, GroundTruthLaw], dict[str, Any]]:
    """Load oracle Hawkes laws for diagnostics, never for model setup."""

    ground_truth_dir = data_root / "ground_truth"
    metadata_path = ground_truth_dir / "regimes.json"
    arrays_path = ground_truth_dir / "regimes.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return {}, {
            "available": False,
            "reason": "ground_truth/regimes.json or regimes.npz is missing",
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    betas = np.asarray(metadata.get("betas", ()), dtype=np.float64).reshape(-1)
    if betas.size != expected_basis or np.any(~np.isfinite(betas)) or np.any(betas <= 0):
        raise ValueError(
            f"ground-truth betas must contain {expected_basis} positive finite values; "
            f"got {betas.tolist()}"
        )
    regimes = metadata.get("regimes")
    if not isinstance(regimes, dict):
        raise ValueError(f"invalid regimes metadata in {metadata_path}")

    laws: dict[str, GroundTruthLaw] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for raw_regime_id, raw_info in regimes.items():
            regime_id = str(raw_regime_id)
            info = raw_info if isinstance(raw_info, dict) else {}
            array_key = str(info.get("array_key", regime_id))
            mu_key = f"{array_key}__mu"
            W_key = f"{array_key}__W"
            if mu_key not in arrays or W_key not in arrays:
                raise KeyError(
                    f"ground-truth arrays for {regime_id!r} are missing "
                    f"({mu_key}, {W_key})"
                )
            mu = np.asarray(arrays[mu_key], dtype=np.float64).copy()
            W = np.asarray(arrays[W_key], dtype=np.float64).copy()
            expected_W_shape = (expected_types, expected_types, expected_basis)
            if mu.shape != (expected_types,) or W.shape != expected_W_shape:
                raise ValueError(
                    f"ground-truth shape mismatch for {regime_id!r}: "
                    f"mu={mu.shape}, W={W.shape}; expected "
                    f"{(expected_types,)}, {expected_W_shape}"
                )
            if (
                np.any(~np.isfinite(mu))
                or np.any(~np.isfinite(W))
                or np.any(mu < 0.0)
                or np.any(W < 0.0)
            ):
                raise ValueError(f"ground-truth law {regime_id!r} is not finite/non-negative")
            laws[regime_id] = GroundTruthLaw(
                regime_id=regime_id,
                mu=mu,
                W=W,
                betas=betas.copy(),
                kind=str(info.get("kind", "unknown")),
                parent_regime=(
                    str(info["parent_regime"])
                    if info.get("parent_regime") else None
                ),
            )
    return laws, {
        "available": True,
        "metadata_path": str(metadata_path.resolve()),
        "arrays_path": str(arrays_path.resolve()),
        "betas": betas.tolist(),
        "regime_count": len(laws),
    }


def _hawkes_intensity_at_time(
    event_times: np.ndarray,
    event_types: np.ndarray,
    time: float,
    mu: np.ndarray,
    W: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Evaluate the strict-causal exponential Hawkes intensity at one time."""

    intensity = np.asarray(mu, dtype=np.float64).copy()
    mask = event_times < float(time)
    if not np.any(mask):
        return np.maximum(intensity, 1e-12)
    past_times = event_times[mask]
    past_types = event_types[mask].astype(np.int64, copy=False)
    kernels = np.exp(
        -(float(time) - past_times)[:, None] * betas[None, :]
    )
    intensity += (
        W[:, past_types, :] * kernels[None, :, :]
    ).sum(axis=(1, 2))
    return np.maximum(intensity, 1e-12)


def _hawkes_intensity_curve(
    event_times: np.ndarray,
    event_types: np.ndarray,
    grid: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    return np.stack([
        _hawkes_intensity_at_time(
            event_times, event_types, float(time), mu, W, betas
        )
        for time in grid
    ], axis=0)


def _decode_model_law(
    raw_theta: torch.Tensor,
    expected_types: int,
    expected_basis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the model's unconstrained semantic/effective theta to mu/W."""

    theta = raw_theta.detach().reshape(-1)
    expected_size = expected_types + expected_types * expected_types * expected_basis
    if theta.numel() != expected_size:
        raise ValueError(
            f"model theta has {theta.numel()} values; expected {expected_size}"
        )
    positive = F.softplus(theta)
    mu = positive[:expected_types].cpu().numpy().astype(np.float64, copy=True)
    W = positive[expected_types:].reshape(
        expected_types, expected_types, expected_basis
    ).cpu().numpy().astype(np.float64, copy=True)
    return mu, W


def _representative_event_types(
    event_types: np.ndarray,
    expected_types: int,
    count: int = 3,
) -> list[int]:
    frequencies = Counter(int(value) for value in event_types.tolist())
    representatives = [item for item, _ in frequencies.most_common(max(count, 1))]
    return [item for item in representatives if 0 <= item < expected_types]


def _nise(
    predicted: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, np.ndarray]:
    # NumPy 2.x removed ``trapz``; keep a lazy fallback for older versions.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    squared_error = (predicted - target) ** 2
    numerator = float(integrate(squared_error.sum(axis=1), grid))
    denominator = float(integrate((target ** 2).sum(axis=1), grid)) + 1e-12
    nise = numerator / denominator
    per_type = np.asarray([
        float(integrate(squared_error[:, index], grid))
        / (float(integrate(target[:, index] ** 2, grid)) + 1e-12)
        for index in range(target.shape[1])
    ], dtype=np.float64)
    return float(nise), per_type


def _batched_trapezoid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Integrate ``values[B, G, ...]`` over one possibly different grid/row."""

    if values.ndim < 2 or grid.ndim != 2 or values.size(0) != grid.size(0):
        raise ValueError("batched trapezoid inputs must be [B, G, ...] and [B, G]")
    if values.size(1) < 2 or grid.size(1) != values.size(1):
        raise ValueError("batched trapezoid requires matching grids of length >= 2")
    widths = (grid[:, 1:] - grid[:, :-1]).clamp_min(0.0)
    return (
        0.5
        * (values[:, 1:] + values[:, :-1])
        * widths.reshape(widths.size(0), widths.size(1), *([1] * (values.ndim - 2)))
    ).sum(dim=1)


def _hawkes_intensity_curves_batched(
    event_times: torch.Tensor,
    event_types: torch.Tensor,
    valid: torch.Tensor,
    grid: torch.Tensor,
    mu: torch.Tensor,
    W: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """Evaluate strict-causal exponential Hawkes curves as ``[B, G, D]``.

    ``mu/W`` may be constant per sequence (``[B, D]``/``[B, D, D, M]``) or
    selected at every grid point (``[B, G, D]``/``[B, G, D, D, M]``).  The
    latter is used for the model's causal parameter snapshots.
    """
    if (
        event_times.ndim != 2
        or event_types.shape != event_times.shape
        or valid.shape != event_times.shape
        or valid.dtype != torch.bool
        or grid.ndim != 2
        or grid.size(0) != event_times.size(0)
    ):
        raise ValueError("event batch tensors must align as [B, L] and [B, G]")
    if betas.ndim != 1:
        raise ValueError("Hawkes betas must be one-dimensional")
    batch_size, _, event_type_count = (
        event_times.size(0),
        event_times.size(1),
        int(mu.size(-1)),
    )
    if event_type_count <= 0:
        raise ValueError("Hawkes intensity requires at least one event type")
    if W.size(-2) != event_type_count or W.size(-3) != event_type_count:
        raise ValueError("Hawkes branching matrix has incompatible type dimensions")
    if W.size(-1) != betas.numel():
        raise ValueError("Hawkes branching matrix and betas have incompatible bases")

    deltas = grid[:, :, None] - event_times[:, None, :]
    causal = valid[:, None, :] & deltas.gt(0.0)
    kernels = torch.exp(
        -deltas.clamp_min(0.0).unsqueeze(-1) * betas.reshape(1, 1, 1, -1)
    ) * causal.unsqueeze(-1).to(event_times.dtype)
    safe_types = event_types.clamp(0, event_type_count - 1)
    source_one_hot = F.one_hot(
        safe_types,
        num_classes=event_type_count,
    ).to(event_times.dtype)
    source_kernel = (
        kernels.unsqueeze(-2)
        * source_one_hot[:, None, :, :, None]
    )

    if mu.ndim == 2:
        mu_grid = mu[:, None, :].expand(batch_size, grid.size(1), -1)
    elif mu.ndim == 3 and mu.shape[:2] == grid.shape:
        mu_grid = mu
    else:
        raise ValueError("mu must have shape [B, D] or [B, G, D]")
    if W.ndim == 4:
        W_grid = W[:, None, :, :, :].expand(
            batch_size,
            grid.size(1),
            -1,
            -1,
            -1,
        )
    elif W.ndim == 5 and W.shape[:2] == grid.shape:
        W_grid = W
    else:
        raise ValueError("W must have shape [B, D, D, M] or [B, G, D, D, M]")
    excitation = torch.einsum(
        "bgdsm,bglsm->bgd",
        W_grid,
        source_kernel,
    )
    return (mu_grid + excitation).clamp_min(1e-12)


def _batched_nise(
    predicted: torch.Tensor,
    target: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total and per-type NISE for a padded GPU batch."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target curves must align as [B, G, D]")
    squared_error = (predicted - target).square()
    numerator_by_type = _batched_trapezoid(squared_error, grid)
    denominator_by_type = _batched_trapezoid(target.square(), grid) + 1e-12
    nise_by_type = numerator_by_type / denominator_by_type
    nise = numerator_by_type.sum(dim=-1) / (
        denominator_by_type.sum(dim=-1) + 1e-12
    )
    return nise, nise_by_type


def _batched_law_evaluation(
    sequences: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Sequence[Mapping[str, Any] | Any]],
    law: GroundTruthLaw,
    *,
    model_betas: torch.Tensor,
    expected_types: int,
    expected_basis: int,
    device: torch.device,
    intensity_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build target/predicted curves and NISE in one device batch."""

    if not sequences or len(sequences) != len(snapshots):
        raise ValueError("law batch sequences and snapshots must be non-empty/aligned")
    lengths = [int(sequence["times"].numel()) for sequence in sequences]
    if any(length <= 0 for length in lengths):
        raise ValueError("law evaluation batches cannot contain empty sequences")
    if any(len(rows) != length for rows, length in zip(snapshots, lengths)):
        raise ValueError("every law sequence needs one causal snapshot per event")
    batch_size = len(sequences)
    max_length = max(lengths)
    parameter_dim = expected_types + expected_types * expected_types * expected_basis
    event_times = torch.zeros(
        batch_size, max_length, device=device, dtype=torch.float64
    )
    event_types = torch.zeros(
        batch_size, max_length, device=device, dtype=torch.long
    )
    valid = torch.zeros(
        batch_size, max_length, device=device, dtype=torch.bool
    )
    theta = torch.zeros(
        batch_size, max_length, parameter_dim, device=device, dtype=torch.float64
    )
    horizons = []
    for row_index, (sequence, sequence_snapshots, length) in enumerate(
        zip(sequences, snapshots, lengths)
    ):
        times = torch.as_tensor(
            sequence["times"], device=device, dtype=torch.float64
        ).reshape(-1)
        types = torch.as_tensor(
            sequence["types"], device=device, dtype=torch.long
        ).reshape(-1)
        if times.numel() != length or types.numel() != length:
            raise ValueError("law sequence event tensors are misaligned")
        event_times[row_index, :length] = times
        event_types[row_index, :length] = types
        valid[row_index, :length] = True
        theta[row_index, :length] = torch.stack([
            torch.as_tensor(snapshot, device=device, dtype=torch.float64).reshape(-1)
            for snapshot in sequence_snapshots
        ])
        horizons.append(max(float(times[-1].detach().cpu()), 1e-6))

    unit_grid = torch.linspace(
        0.0,
        1.0,
        max(intensity_samples, 2),
        device=device,
        dtype=torch.float64,
    )
    horizon_tensor = torch.as_tensor(
        horizons, device=device, dtype=torch.float64
    )
    grid = horizon_tensor[:, None] * unit_grid[None, :]
    target_mu = torch.as_tensor(law.mu, device=device, dtype=torch.float64)
    target_W = torch.as_tensor(law.W, device=device, dtype=torch.float64)
    betas = torch.as_tensor(model_betas, device=device, dtype=torch.float64)
    target_curve = _hawkes_intensity_curves_batched(
        event_times,
        event_types,
        valid,
        grid,
        target_mu[None, :].expand(batch_size, -1),
        target_W[None, :].expand(batch_size, -1, -1, -1),
        torch.as_tensor(law.betas, device=device, dtype=torch.float64),
    )

    expected_size = parameter_dim
    if theta.size(-1) != expected_size:
        raise ValueError(
            f"model theta has {theta.size(-1)} values; expected {expected_size}"
        )
    positive = F.softplus(theta)
    model_mu_by_event = positive[..., :expected_types]
    model_W_by_event = positive[..., expected_types:].reshape(
        batch_size,
        max_length,
        expected_types,
        expected_types,
        expected_basis,
    )
    snapshot_indices = (
        (valid[:, None, :] & event_times[:, None, :].lt(grid[:, :, None]))
        .sum(dim=-1)
        .clamp_min(0)
    )
    max_snapshot_indices = torch.as_tensor(
        lengths, device=device, dtype=torch.long
    ).sub(1).clamp_min(0)[:, None]
    snapshot_indices = torch.minimum(snapshot_indices, max_snapshot_indices)
    mu_grid = model_mu_by_event.gather(
        1,
        snapshot_indices[:, :, None].expand(-1, -1, expected_types),
    )
    W_grid = model_W_by_event.gather(
        1,
        snapshot_indices[:, :, None, None, None].expand(
            -1,
            -1,
            expected_types,
            expected_types,
            expected_basis,
        ),
    )
    predicted_curve = _hawkes_intensity_curves_batched(
        event_times,
        event_types,
        valid,
        grid,
        mu_grid,
        W_grid,
        betas,
    )
    nise, nise_by_type = _batched_nise(predicted_curve, target_curve, grid)
    return grid, target_curve, predicted_curve, nise, nise_by_type


def _plot_intensity_curve(
    path: Path,
    *,
    grid: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    event_times: np.ndarray,
    event_types: np.ndarray,
    regime_id: str,
    checkpoint_task: int,
    anchor_index: int,
    nise: float,
    expected_types: int,
) -> str | None:
    """Write a compact total-plus-representative-types Hawkes plot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    representatives = _representative_event_types(
        event_types, expected_types, count=3
    )
    figure, axes = plt.subplots(
        1 + len(representatives),
        1,
        figsize=(11, 2.7 * (1 + len(representatives))),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    total_target = target.sum(axis=1)
    total_predicted = predicted.sum(axis=1)
    axes[0].plot(grid, total_target, ":", linewidth=1.8, label="ground truth")
    axes[0].plot(grid, total_predicted, "-", linewidth=1.4, label="predicted")
    axes[0].set_ylabel("total intensity")
    axes[0].set_title(
        f"checkpoint task_{checkpoint_task:02d} | {regime_id} | "
        f"anchor {anchor_index:03d} | NISE={nise:.5f}"
    )
    axes[0].legend(loc="upper right")
    for axis, event_type in zip(axes[1:], representatives):
        axis.plot(
            grid, target[:, event_type], ":", linewidth=1.8,
            label="ground truth",
        )
        axis.plot(
            grid, predicted[:, event_type], "-", linewidth=1.4,
            label="predicted",
        )
        axis.set_ylabel(f"type {event_type}")
        axis.legend(loc="upper right")
    for axis in axes:
        # Ticks are drawn in axis coordinates so they remain visible without
        # distorting the intensity y-scale.
        axis.vlines(
            event_times,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="0.65",
            linewidth=0.35,
            alpha=0.45,
        )
        axis.grid(alpha=0.18)
    axes[-1].set_xlabel("time")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path.resolve())


def _law_inference(
    checkpoint: Path,
    args: argparse.Namespace,
) -> MemoryTreeInference:
    """Build the frozen, read-only inference used for Hawkes NISE."""

    inference = MemoryTreeInference.from_checkpoint(
        checkpoint,
        device=args.device,
        inference_config=inference_config_for_protocol(
            EvaluationProtocol.FROZEN,
            probe_write_counterfactuals=False,
            write_probe_seed=42,
            prototype_duplicate_threshold=args.prototype_duplicate_threshold,
            prototype_mode_threshold=args.prototype_mode_threshold,
            prototype_context_alias_capacity=args.prototype_context_alias_capacity,
        ),
    )
    return inference


def _hawkes_law_evaluation(
    *,
    checkpoint_paths: Mapping[int, Path],
    checkpoint_tasks: Sequence[int],
    anchors: Sequence[EvaluationSet],
    evaluation_cache: Mapping[Path, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[str, GroundTruthLaw],
    regime_first_seen: Mapping[str, int],
    args: argparse.Namespace,
    expected_types: int,
    transient_regimes: Iterable[str] = (),
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate causal intensity curves and return NISE only."""

    if not anchors or not ground_truth:
        return [], []

    variant = "frozen/full"
    expected_basis = len(next(iter(ground_truth.values())).betas)
    diagnostic_regimes = {str(regime_id) for regime_id in transient_regimes}
    intensity_rows: list[dict[str, Any]] = []
    for checkpoint_task in checkpoint_tasks:
        checkpoint = checkpoint_paths[checkpoint_task]
        inference = _law_inference(checkpoint, args)
        model_betas = inference.hawkes.decays.detach()
        if model_betas.numel() != expected_basis:
            raise ValueError(
                f"checkpoint task_{checkpoint_task:02d} has {model_betas.numel()} "
                f"decay bases, expected {expected_basis}"
            )
        # The routing table is checkpoint-static.  Reuse it for every anchor
        # batch; the causal Working Memory state is still reset per sequence
        # inside ``run_sequence`` below.
        static_cache = inference.tree.frontier_routing.build_static_cache(
            detach=True
        )
        for anchor in anchors:
            regime_id = str(anchor.regime_id)
            law = ground_truth.get(regime_id)
            if law is None:
                continue
            sequences = list(evaluation_cache.get(anchor.path, ()))
            for batch_start in range(0, len(sequences), args.eval_batch_size):
                batch = sequences[
                    batch_start:batch_start + args.eval_batch_size
                ]
                prepared, static_cache = inference.prepare_sequence_batch(
                    batch,
                    frontier_static_cache=static_cache,
                )
                if not all(item.get("z") is not None for item in prepared):
                    raise RuntimeError(
                        "Hawkes law evaluation requires a padded CausalPrefixEncoder"
                    )
                batch_results = inference.run_sequence_batch_compact(
                    prepared,
                    frontier_static_cache=static_cache,
                    capture_prediction_theta=True,
                )
                snapshots_by_sequence: list[list[torch.Tensor]] = []
                for result in batch_results:
                    events = result.get("events", ())
                    snapshots = [
                        event.get("prediction_theta") for event in events
                    ]
                    if not events or any(snapshot is None for snapshot in snapshots):
                        raise RuntimeError(
                            "inference did not expose causal prediction_theta; "
                            "please use the matching Memory/Train/Inference.py"
                        )
                    snapshots_by_sequence.append(snapshots)

                grid, target_curve, predicted_curve, nise, nise_by_type = (
                    _batched_law_evaluation(
                        batch,
                        snapshots_by_sequence,
                        law,
                        model_betas=model_betas,
                        expected_types=expected_types,
                        expected_basis=expected_basis,
                        device=inference.device,
                        intensity_samples=args.intensity_samples,
                    )
                )
                first_seen_task = regime_first_seen.get(regime_id)
                scope = (
                    "ood_unseen"
                    if (
                        law.kind == "transient"
                        or regime_id in diagnostic_regimes
                        or first_seen_task is None
                        or first_seen_task > checkpoint_task
                    )
                    else "seen_law"
                )
                for offset, sequence in enumerate(batch):
                    anchor_index = batch_start + offset
                    event_times = sequence["times"].detach().cpu().numpy().astype(
                        np.float64, copy=False
                    )
                    event_types = sequence["types"].detach().cpu().numpy().astype(
                        np.int64, copy=False
                    )
                    grid_row = grid[offset].detach().cpu().numpy()
                    target_row = target_curve[offset].detach().cpu().numpy()
                    predicted_row = predicted_curve[offset].detach().cpu().numpy()
                    nise_value = float(nise[offset].detach().cpu())
                    plot_path = None
                    if anchor_index < args.intensity_plot_anchors:
                        plot_path = _plot_intensity_curve(
                            args.output_dir
                            / "intensity_curves"
                            / f"checkpoint_task_{checkpoint_task:02d}"
                            / f"{_safe_name(regime_id)}_{anchor_index:03d}.png",
                            grid=grid_row,
                            target=target_row,
                            predicted=predicted_row,
                            event_times=event_times,
                            event_types=event_types,
                            regime_id=regime_id,
                            checkpoint_task=checkpoint_task,
                            anchor_index=anchor_index,
                            nise=nise_value,
                            expected_types=expected_types,
                        )
                    intensity_row = {
                        "checkpoint_task": checkpoint_task,
                        "checkpoint": str(checkpoint.resolve()),
                        "variant": variant,
                        "regime_id": regime_id,
                        "anchor_index": anchor_index,
                        "events": int(len(event_times)),
                        "evaluation_scope": scope,
                        "first_seen_task": first_seen_task,
                        "nise": nise_value,
                        "plot_path": plot_path,
                    }
                    for event_type, value in enumerate(
                        nise_by_type[offset].detach().cpu().tolist()
                    ):
                        intensity_row[f"nise_type_{event_type}"] = float(value)
                    intensity_rows.append(intensity_row)

    def grouped_summary(
        rows: Sequence[Mapping[str, Any]],
        value_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[
                (
                    int(row["checkpoint_task"]),
                    str(row["variant"]),
                    str(row["regime_id"]),
                    str(row["evaluation_scope"]),
                )
            ].append(row)
        output: list[dict[str, Any]] = []
        for key in sorted(groups):
            checkpoint_task, row_variant, regime_id, scope = key
            current = groups[key]
            item: dict[str, Any] = {
                "checkpoint_task": checkpoint_task,
                "variant": row_variant,
                "regime_id": regime_id,
                "evaluation_scope": scope,
                "first_seen_task": regime_first_seen.get(regime_id),
                "sequence_count": len(current),
            }
            for value_name in value_names:
                values = [row.get(value_name) for row in current]
                item[f"{value_name}_mean"] = _mean(values)
                clean = [
                    float(value) for value in values
                    if value is not None and math.isfinite(float(value))
                ]
                item[f"{value_name}_median"] = (
                    float(np.median(clean)) if clean else None
                )
            output.append(item)
        return output

    intensity_summary = grouped_summary(intensity_rows, ("nise",))
    return intensity_rows, intensity_summary


def _ood_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, GroundTruthLaw],
    regime_first_seen: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Keep transient/unseen anchors out of CL averages and report them here."""

    output = []
    for row in metric_rows:
        if row.get("eval_kind") != "anchor":
            continue
        regime_id = str(row.get("regime_id"))
        law = ground_truth.get(regime_id)
        if law is None:
            continue
        is_persistent_anchor = (
            row.get("evaluation_scope") == "persistent"
            or (
                row.get("evaluation_scope") is None
                and law.kind != "transient"
            )
        )
        first_seen_task = regime_first_seen.get(regime_id)
        if is_persistent_anchor and (
            first_seen_task is not None
            and first_seen_task <= int(row.get("checkpoint_task", -1))
        ):
            continue
        output.append({
            "checkpoint_task": row.get("checkpoint_task"),
            "variant": row.get("variant"),
            "regime_id": regime_id,
            "evaluation_scope": "ood_unseen",
            "first_seen_task": first_seen_task,
            "nll_per_event": row.get("nll_per_event"),
            "accuracy": row.get("accuracy"),
            "local_time_mae": row.get("local_time_mae"),
        })
    return output


def _plot_summary_figures(
    output_dir: Path,
    *,
    continual_rows: Sequence[Mapping[str, Any]],
    anchor_matrix_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    special_rows: Sequence[Mapping[str, Any]],
    intensity_summary_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Plot the compact CL figures most useful for diagnosis and a paper."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(
            f"[CL Plot] summary plots skipped: matplotlib unavailable ({error})",
            flush=True,
        )
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def finite(value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def save(figure: Any, filename: str) -> None:
        path = plot_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(str(path.resolve()))

    def plot_by_variant(
        axis: Any,
        rows: Sequence[Mapping[str, Any]],
        metric: str,
    ) -> bool:
        variants = sorted({str(row.get("variant")) for row in rows})
        plotted = False
        task_ticks: set[int] = set()
        for variant in variants:
            points = []
            for row in rows:
                if str(row.get("variant")) != variant:
                    continue
                task = finite(row.get("checkpoint_task"))
                value = finite(row.get(metric))
                if task is not None and value is not None:
                    points.append((int(task), value))
            points.sort()
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                label=variant,
            )
            task_ticks.update(point[0] for point in points)
            plotted = True
        axis.set_xlabel("checkpoint task")
        if task_ticks:
            axis.set_xticks(sorted(task_ticks))
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)
        return plotted

    # Current-task prediction quality separates immediate fit from retention.
    quality_metrics = (
        ("current_nll_per_event", "Current-task NLL/event", "lower is better"),
        ("current_accuracy", "Current-task accuracy", "higher is better"),
        ("current_local_time_mae", "Current-task time MAE", "lower is better"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    quality_plotted = False
    for axis, (metric, title, direction) in zip(axes[0], quality_metrics):
        quality_plotted |= plot_by_variant(axis, continual_rows, metric)
        axis.set_title(f"{title}\n({direction})")
    if quality_plotted:
        save(figure, "current_task_quality.png")
    else:
        plt.close(figure)

    # Stability metrics on laws seen by each checkpoint.
    continual_metrics = (
        ("clnll", "Seen-law CLNLL", "lower is better"),
        ("average_forgetting", "Average forgetting", "near zero is best"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), squeeze=False)
    continual_plotted = False
    for axis, (metric, title, direction) in zip(axes[0], continual_metrics):
        continual_plotted |= plot_by_variant(axis, continual_rows, metric)
        axis.set_title(f"{title}\n({direction})")
        if metric == "average_forgetting":
            axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    if continual_plotted:
        save(figure, "continual_learning.png")
    else:
        plt.close(figure)

    # Frozen-anchor checkpoint x law matrix. Prefer the paper's main variant.
    matrix_variants = sorted({str(row.get("variant")) for row in anchor_matrix_rows})
    matrix_variant = (
        "frozen/full" if "frozen/full" in matrix_variants
        else (matrix_variants[0] if matrix_variants else None)
    )
    matrix_rows = sorted(
        (
            row for row in anchor_matrix_rows
            if str(row.get("variant")) == matrix_variant
        ),
        key=lambda row: int(row["checkpoint_task"]),
    )
    metadata_columns = {"checkpoint_task", "variant"}
    regime_columns = sorted({
        key
        for row in matrix_rows
        for key in row
        if key not in metadata_columns
        and finite(row.get(key)) is not None
    })
    if matrix_rows and regime_columns:
        matrix = np.full(
            (len(matrix_rows), len(regime_columns)), np.nan, dtype=np.float64
        )
        for row_index, row in enumerate(matrix_rows):
            for column_index, regime_id in enumerate(regime_columns):
                value = finite(row.get(regime_id))
                if value is not None:
                    matrix[row_index, column_index] = value
        figure, axis = plt.subplots(
            figsize=(max(8.0, 1.05 * len(regime_columns)),
                     max(4.0, 0.58 * len(matrix_rows) + 1.8))
        )
        masked = np.ma.masked_invalid(matrix)
        colour_map = plt.get_cmap("viridis").copy()
        colour_map.set_bad("white")
        image = axis.imshow(masked, aspect="auto", cmap=colour_map)
        axis.set_xticks(range(len(regime_columns)), labels=regime_columns)
        axis.set_yticks(
            range(len(matrix_rows)),
            labels=[f"C{int(row['checkpoint_task'])}" for row in matrix_rows],
        )
        axis.set_xlabel("frozen anchor law")
        axis.set_ylabel("checkpoint")
        axis.set_title(f"Anchor NLL/event matrix — {matrix_variant} (lower is better)")
        figure.colorbar(image, ax=axis, label="NLL/event")
        if matrix.size <= 140:
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if not math.isfinite(float(value)):
                        continue
                    red, green, blue, _ = colour_map(image.norm(value))
                    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if luminance > 0.55 else "white",
                    )
        save(figure, "anchor_nll_heatmap.png")

    # Plasticity of each stage before versus after learning the current task.
    stage_variants = sorted({str(row.get("variant")) for row in stage_rows})
    stage_variant = (
        "frozen/full" if "frozen/full" in stage_variants
        else (stage_variants[0] if stage_variants else None)
    )
    stage_points = sorted(
        (
            (int(row["task_id"]), value)
            for row in stage_rows
            if str(row.get("variant")) == stage_variant
            and (value := finite(row.get("adaptation_gain_nll"))) is not None
        ),
        key=lambda item: item[0],
    )
    if stage_points:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        values = [point[1] for point in stage_points]
        axis.bar(
            [point[0] for point in stage_points],
            values,
            color=["#2a9d8f" if value >= 0.0 else "#e76f51" for value in values],
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        stage_by_task = {
            int(row["task_id"]): row for row in stage_rows
            if str(row.get("variant")) == stage_variant
        }
        axis.set_xticks(
            [point[0] for point in stage_points],
            labels=[
                f"{task_id}\n{stage_by_task.get(task_id, {}).get('shift_type') or 'stage'}"
                for task_id, _ in stage_points
            ],
        )
        axis.set_xlabel("task / shift type")
        axis.set_ylabel("pre NLL - post NLL")
        axis.set_title(f"Stage adaptation gain by protocol shift — {stage_variant} (positive is better)")
        axis.grid(axis="y", alpha=0.25)
        save(figure, "stage_adaptation_gain.png")

    # Underlying Hawkes-law recovery, averaged over seen anchor laws.
    law_variants = sorted({
        str(row.get("variant")) for row in intensity_summary_rows
    })
    law_variant = (
        "frozen/full" if "frozen/full" in law_variants
        else (law_variants[0] if law_variants else None)
    )
    seen_intensity = [
        row for row in intensity_summary_rows
        if str(row.get("variant")) == law_variant
        and row.get("evaluation_scope") == "seen_law"
    ]
    law_tasks = sorted({int(row["checkpoint_task"]) for row in seen_intensity})
    if law_tasks:
        points = []
        for task_id in law_tasks:
            value = _mean(
                row.get("nise_mean")
                for row in seen_intensity
                if int(row["checkpoint_task"]) == task_id
            )
            value = finite(value)
            if value is not None:
                points.append((task_id, value))
        if points:
            figure, axis = plt.subplots(figsize=(8.5, 4.2))
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                color="#264653",
            )
            axis.set_xlabel("checkpoint task")
            axis.set_ylabel("NISE")
            axis.set_title(f"Hawkes intensity NISE — {law_variant}\n(lower is better)")
            axis.grid(alpha=0.25)
            save(figure, "hawkes_law_recovery.png")

    # Structural growth is separated from memory-row growth because the scales differ.
    topology_points = sorted(
        checkpoint_rows, key=lambda row: int(row["checkpoint_task"])
    )
    if topology_points:
        tasks = [int(row["checkpoint_task"]) for row in topology_points]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), squeeze=False)
        structural_plotted = False
        for metric, label in (("node_count", "nodes"), ("leaf_count", "leaves")):
            values = [finite(row.get(metric)) for row in topology_points]
            valid = [(task, value) for task, value in zip(tasks, values) if value is not None]
            if valid:
                axes[0, 0].plot(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )
                structural_plotted = True
        memory_values = [finite(row.get("memory_rows")) for row in topology_points]
        valid_memory = [
            (task, value) for task, value in zip(tasks, memory_values)
            if value is not None
        ]
        if valid_memory:
            axes[0, 1].plot(
                [item[0] for item in valid_memory],
                [item[1] for item in valid_memory],
                marker="o",
                linewidth=1.8,
                color="#e76f51",
            )
        axes[0, 0].set_title("Tree size")
        axes[0, 0].set_ylabel("count")
        axes[0, 1].set_title("Persistent episodic memory")
        axes[0, 1].set_ylabel("memory rows")
        for axis in axes[0]:
            axis.set_xlabel("checkpoint task")
            axis.grid(alpha=0.25)
        if structural_plotted:
            axes[0, 0].legend()
        if structural_plotted or valid_memory:
            save(figure, "topology_and_memory_growth.png")
        else:
            plt.close(figure)

    # Dataset-specific recurrence/specialization diagnostics become available gradually.
    special_points = [
        (str(row.get("metric")), value)
        for row in special_rows
        if (value := finite(row.get("value"))) is not None
    ]
    if special_points:
        figure, axis = plt.subplots(
            figsize=(max(9.0, 0.85 * len(special_points)), 5.0)
        )
        values = [point[1] for point in special_points]
        axis.bar(
            range(len(special_points)),
            values,
            color=["#2a9d8f" if value >= 0.0 else "#e76f51" for value in values],
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_xticks(
            range(len(special_points)),
            labels=[point[0] for point in special_points],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("NLL difference / gain")
        axis.set_title("Protocol shift diagnostics")
        axis.grid(axis="y", alpha=0.25)
        save(figure, "special_case_metrics.png")

    print(f"[CL Plot] wrote {len(written)} figures to {plot_dir}", flush=True)
    return written


def _write_report(
    path: Path,
    *,
    data_root: Path,
    checkpoint_dir: Path,
    checkpoint_tasks: Sequence[int],
    variants: Sequence[str],
    metric_rows: Sequence[Mapping[str, Any]],
    continual_rows: Sequence[Mapping[str, Any]],
    law_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    special_rows: Sequence[Mapping[str, Any]],
    intensity_summary_rows: Sequence[Mapping[str, Any]],
    ood_rows: Sequence[Mapping[str, Any]],
    summary_plot_paths: Sequence[str],
    tree_by_checkpoint: Mapping[int, Mapping[str, Any]],
    skipped_tasks: Sequence[int],
    anchors_enabled: bool,
    metric_report: Mapping[str, Any] | None = None,
) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float) and not math.isfinite(value):
            return "NA"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        "# Hawkes Memory Tree CL Evaluation",
        "",
        f"- Data root: `{data_root.resolve()}`",
        f"- Benchmark manifest: `{(data_root / 'benchmark_manifest.json').resolve()}`",
        "- Persistent-law averages follow `persistent_regimes`; diagnostic transient anchors are reported in the OOD section.",
        f"- Checkpoints: `{checkpoint_dir.resolve()}`",
        f"- Checkpoint tasks: `{list(checkpoint_tasks)}`",
        f"- Variants: `{list(variants)}`",
        "- Task-test protocol: checkpoint `task_k_best` is evaluated on `D_k^test`; "
        "`D_{k+1}^test` is also evaluated before learning when available.",
        f"- Frozen anchors: `{'enabled' if anchors_enabled else 'disabled'}`.",
        "",
        "## Checkpoint topology",
        "",
        "| checkpoint | nodes | leaves | max depth | memory rows |",
        "|---:|---:|---:|---:|---:|",
    ]
    for task_id in checkpoint_tasks:
        tree = tree_by_checkpoint[task_id]
        lines.append(
            f"| task_{task_id:02d}_best | {tree.get('node_count')} | "
            f"{tree.get('leaf_count')} | {tree.get('max_depth')} | "
            f"{tree.get('memory_rows')} |"
        )

    lines.extend([
        "",
        "## Current-task test quality",
        "",
        "| checkpoint | variant | NLL/event | accuracy | time MAE |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in metric_rows:
        if row["eval_kind"] != "task_test" or row["eval_task"] != row["checkpoint_task"]:
            continue
        lines.append(
            f"| task_{int(row['checkpoint_task']):02d}_best | {row['variant']} | "
            f"{fmt(row.get('nll_per_event'), 6)} | {fmt(row.get('accuracy'))} | "
            f"{fmt(row.get('local_time_mae'))} |"
        )

    lines.extend([
        "",
        "## Continual retention and anchors",
        "",
        "CLNLL averages only anchor laws whose first occurrence is no later than the checkpoint. "
        "Forgetting is current NLL minus the best NLL since that law was first seen.",
        "",
        "| checkpoint | variant | CLNLL | avg forgetting | seen laws |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in continual_rows:
        lines.append(
            f"| task_{int(row['checkpoint_task']):02d}_best | {row['variant']} | "
            f"{fmt(row.get('clnll'), 6)} | "
            f"{fmt(row.get('average_forgetting'), 6)} | "
            f"{fmt(row.get('seen_law_count'), 0)} |"
        )

    lines.extend([
        "",
        "## Stage-level plasticity",
        "",
        "`adaptation_gain_nll = pre_nll - post_nll`; positive means the current task improved after training.",
        "",
        "| task | shift type | variant | pre NLL | post NLL | adaptation gain |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for row in stage_rows:
        lines.append(
            f"| task_{int(row['task_id']):02d} | {row.get('shift_type') or 'NA'} | {row['variant']} | "
            f"{fmt(row.get('pre_nll_per_event'), 6)} | "
            f"{fmt(row.get('post_nll_per_event'), 6)} | "
            f"{fmt(row.get('adaptation_gain_nll'), 6)} |"
        )

    if metric_report is not None:
        fwt = metric_report.get("fwt", {})
        adaptation = metric_report.get("adaptation", {})
        rrr = metric_report.get("rrr", {})
        lines.extend([
            "",
            "## Transfer and adaptation contract",
            "",
            "FWT compares the same task test set from the fixed C_init and the pre-task checkpoint. "
            "Only genuinely unseen persistent-law tasks enter the average; recurrence tasks remain diagnostics.",
            "",
            f"- Average FWT: `{fmt(fwt.get('average_fwt'), 6)}` "
            f"({fwt.get('status', 'not_available')}).",
            "",
            "| protocol | task | K min | K max | adaptation AUC | status |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for row in adaptation.get("summary", ()):
            lines.append(
                f"| {row.get('protocol') or 'default'} | {row.get('task_id')} | "
                f"{fmt(row.get('K_min'), 0)} | {fmt(row.get('K_max'), 0)} | "
                f"{fmt(row.get('adaptation_auc'), 6)} | {row.get('status')} |"
            )
        if rrr.get("rows"):
            lines.extend([
                "",
                "| returned law | first task | return task | shift | RRR | status |",
                "|---|---:|---:|---|---:|---|",
            ])
            for row in rrr["rows"]:
                lines.append(
                    f"| {row.get('regime_id')} | {row.get('first_task')} | "
                    f"{row.get('return_task')} | {row.get('shift_type')} | "
                    f"{fmt(row.get('rrr'), 6)} | {row.get('status')} |"
                )

        hm_state = metric_report.get("hm_state", ())
        if hm_state:
            lines.extend([
                "",
                "## HM-specific state",
                "",
                "Topology action counts come from committed transaction events; no leaf-count difference is inferred.",
                "",
                "| task | nodes | leaves | episodic rows | episodic bytes | semantic bytes | split | merge | prune | NISE |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in hm_state:
                lines.append(
                    f"| {row.get('task_id')} | {row.get('node_count')} | "
                    f"{row.get('leaf_count')} | {row.get('episodic_rows')} | "
                    f"{row.get('episodic_bytes')} | {row.get('semantic_bytes')} | "
                    f"{row.get('split_count')} | {row.get('merge_count')} | "
                    f"{row.get('prune_count')} | {fmt(row.get('nise'), 6)} |"
                )

    lines.extend([
        "",
        "## Schedule-driven diagnostics",
        "",
        "For NLL differences, positive values mean the right-hand condition has lower NLL; definitions come from task shift_type and paired controls in the protocol.",
        "",
        "| metric | formula | value | expected |",
        "|---|---|---:|---|",
    ])
    for row in special_rows:
        lines.append(
            f"| {row['metric']} | `{row['formula']}` | "
            f"{fmt(row.get('value'), 6)} | {row.get('expected')} |"
        )

    seen_intensity = [
        row for row in intensity_summary_rows
        if row.get("evaluation_scope") == "seen_law"
    ]
    if seen_intensity:
        lines.extend([
            "",
            "## Hawkes law recovery",
            "",
            "NISE compares causal total and representative event-type intensity curves "
            "against `ground_truth/regimes.npz`.",
            "",
            "| checkpoint | variant | regime | NISE | sequences |",
            "|---:|---|---|---:|---:|",
        ])
        for row in seen_intensity:
            lines.append(
                f"| task_{int(row['checkpoint_task']):02d}_best | {row['variant']} | "
                f"{row['regime_id']} | {fmt(row.get('nise_mean'), 6)} | "
                f"{fmt(row.get('sequence_count'), 0)} |"
            )

    if ood_rows:
        lines.extend([
            "",
            "## Unseen/OOD novelty control",
            "",
            "Transient/unseen anchors are reported separately and never enter CLNLL, "
            "average forgetting, or average seen-task NLL.",
            "",
            "| checkpoint | variant | regime | NLL/event | accuracy | time MAE |",
            "|---:|---|---|---:|---:|---:|",
        ])
        for row in ood_rows:
            lines.append(
                f"| task_{int(row['checkpoint_task']):02d}_best | {row['variant']} | "
                f"{row['regime_id']} | {fmt(row.get('nll_per_event'), 6)} | "
                f"{fmt(row.get('accuracy'))} | "
                f"{fmt(row.get('local_time_mae'))} |"
            )

    if law_rows:
        lines.extend([
            "",
            f"The full per-law anchor table contains `{len(law_rows)}` rows; "
            "see `law_metrics.csv` for start/current/best NLL, forgetting, and BWT.",
        ])

    if skipped_tasks:
        lines.extend([
            "",
            "## Skipped task IDs",
            "",
            f"No matching task test/checkpoint pair was available for: `{list(skipped_tasks)}`.",
        ])
    if summary_plot_paths:
        lines.extend(["", "## Summary plots", ""])
        for plot_value in summary_plot_paths:
            plot_path = Path(plot_value)
            try:
                relative_path = plot_path.relative_to(path.parent)
            except ValueError:
                relative_path = plot_path
            title = plot_path.stem.replace("_", " ").title()
            lines.extend([
                f"### {title}",
                "",
                f"![{title}]({relative_path.as_posix()})",
                "",
            ])
    lines.extend([
        "",
        "## Output files",
        "",
        "- `task_metrics.csv`: checkpoint × task-test × variant metrics.",
        "- `control_metrics.csv`: checkpoint × manifest-declared matched-control metrics.",
        "- `anchor_metrics.csv`: checkpoint × frozen-anchor × variant metrics.",
        "- `continual_summary.csv`: current quality, CLNLL, forgetting, and checkpoint topology.",
        "- `law_metrics.csv`: per-law CLNLL support, forgetting, and BWT terms.",
        "- `stage_metrics.csv`: pre/post task-test adaptation gains.",
        "- `fwt_metrics.csv`: protocol-scoped forward transfer with fixed scratch baseline when supplied.",
        "- `adaptation_points.csv` / `adaptation_summary.csv`: fixed-query K-indexed adaptation curves and normalized AUC.",
        "- `rrr_metrics.csv`: protocol-driven exact/long-gap recurrence retention ratios.",
        "- `hm_state.csv`: HM-only memory, topology transaction counts, and NISE.",
        "- `cl_metrics.json`: canonical CL metric contract shared with baseline runners.",
        "- `anchor_nll_matrix.csv`: paper-style wide checkpoint × regime NLL matrix.",
        "- `special_case_metrics.csv`: schedule-driven recurrence, near-recurrence, specialization, mixture, and transient diagnostics.",
        "- `intensity_metrics.csv` / `intensity_summary.csv`: causal intensity-curve NISE and checkpoint summaries.",
        "- `ood_metrics.csv`: transient/unseen-anchor novelty control, excluded from CL averages.",
        "- `intensity_curves/`: optional total-plus-representative-type GT/prediction plots.",
        "- `plots/`: current quality, CLNLL/forgetting, anchor heatmap, adaptation, NISE, and topology figures.",
        "- `checkpoint_tree.csv`: leaf/node counts and checkpoint memory sizes.",
        "- `summary.json`: machine-readable copy of the complete evaluation manifest.",
        "- `event_predictions.csv`: written only when `--save-event-predictions` is supplied.",
        "- `protocol_comparison.csv`: one comparison table across the selected protocols.",
        "- `frozen/`: strict frozen anchor matrix, CLNLL, forgetting, BWT, and law metrics.",
        "- `fast_adapt/`: official fixed-query adaptation curve plus event-exposure diagnostics; it is excluded from CL aggregates.",
        "- `online_write/`: independent fixed-query write curve plus event-exposure diagnostics; each eval set starts from a fresh checkpoint load.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol_outputs(
    output_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    continual_rows: Sequence[Mapping[str, Any]],
    law_rows: Sequence[Mapping[str, Any]],
    anchor_matrix_rows: Sequence[Mapping[str, Any]],
    adaptation_points: Sequence[Mapping[str, Any]] = (),
    adaptation_summary: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Materialize the protocol × memory-view output contract."""

    task_fields = (
        "checkpoint_task", "checkpoint", "eval_name", "eval_kind", "eval_task",
        "regime_id", "stage_label", "variant", "protocol", "memory_view",
        "nll_per_event", "accuracy", "local_time_mae", "events",
    )
    anchor_fields = (
        "checkpoint_task", "checkpoint", "eval_name", "regime_id", "variant",
        "protocol", "memory_view", "nll_per_event", "accuracy", "local_time_mae",
    )
    curve_fields = (
        "checkpoint_task", "eval_task", "eval_set_id", "regime", "source_index",
        "protocol", "memory_view", "K", "exposure_events", "nll", "accuracy",
        "time_MAE", "retrieval_hit", "working_norm", "write_count",
    )
    curve_summary_fields = (
        "protocol", "K", "exposure_events", "events", "nll", "accuracy",
        "time_MAE", "retrieval_hit", "working_norm", "write_count",
    )
    adaptation_fields = (
        "protocol", "task_id", "K", "pre_nll", "adapted_nll", "gain_nll",
    )
    adaptation_summary_fields = (
        "protocol", "task_id", "K_min", "K_max", "K_count",
        "adaptation_auc", "status",
    )
    anchor_matrix_fields = (
        "checkpoint_task", "variant",
        *sorted({
            key
            for row in anchor_matrix_rows
            for key in row
            if key not in {"checkpoint_task", "variant"}
        }),
    )
    comparison_rows = [
        {
            "checkpoint_task": row.get("checkpoint_task"),
            "eval_task": row.get("eval_task"),
            "eval_name": row.get("eval_name"),
            "protocol": row.get("protocol", str(row.get("variant", "")).split("/", 1)[0]),
            "memory_view": row.get("memory_view", "full"),
            "variant": row.get("variant"),
            "nll_per_event": row.get("nll_per_event"),
            "accuracy": row.get("accuracy"),
            "local_time_mae": row.get("local_time_mae"),
            "events": row.get("events"),
        }
        for row in metric_rows
    ]
    write_csv(
        output_dir / "protocol_comparison.csv",
        comparison_rows,
        fieldnames=(
            "checkpoint_task", "eval_task", "eval_name", "protocol", "memory_view",
            "variant", "nll_per_event", "accuracy", "local_time_mae", "events",
        ),
    )

    for protocol_name in ("frozen", "fast_adapt", "online_write"):
        variant = f"{protocol_name}/full"
        protocol_dir = output_dir / protocol_name
        protocol_dir.mkdir(parents=True, exist_ok=True)
        selected_metrics = [
            row for row in metric_rows if row.get("variant") == variant
        ]
        write_csv(
            protocol_dir / "task_metrics.csv",
            [
                row for row in selected_metrics
                if row.get("eval_kind") in {"task_test", "task_test_pre"}
            ],
            fieldnames=task_fields,
        )
        write_csv(
            protocol_dir / "anchor_metrics.csv",
            [row for row in selected_metrics if row.get("eval_kind") == "anchor"],
            fieldnames=anchor_fields,
        )
        if protocol_name == "frozen":
            write_csv(
                protocol_dir / "anchor_nll_matrix.csv",
                anchor_matrix_rows,
                fieldnames=anchor_matrix_fields,
            )
            write_csv(
                protocol_dir / "continual_summary.csv",
                continual_rows,
                fieldnames=(
                    "checkpoint_task", "variant", "current_nll_per_event",
                    "current_accuracy", "clnll", "average_forgetting", "average_bwt",
                ),
            )
            write_csv(
                protocol_dir / "law_metrics.csv",
                law_rows,
                fieldnames=(
                    "checkpoint_task", "variant", "regime_id", "forgetting_nll", "bwt_nll",
                ),
            )
            continue

        selected_events = [
            row for row in event_rows if row.get("variant") == variant
        ]
        curve = adaptation_curve_rows(
            selected_events, protocol=protocol_name, memory_view="full"
        )
        # Keep the event-index exposure diagnostic separate from the official
        # task-level K curve.  The latter uses a fixed query set and a fresh
        # pre-task clone for every K.
        write_csv(protocol_dir / "exposure_curve.csv", curve, fieldnames=curve_fields)
        summary_name = (
            "reaccess_summary.csv"
            if protocol_name == "fast_adapt" else "online_summary.csv"
        )
        write_csv(
            protocol_dir / summary_name,
            _curve_summary(curve),
            fieldnames=curve_summary_fields,
        )
        write_csv(
            protocol_dir / "adaptation_curve.csv",
            [row for row in adaptation_points
             if row.get("protocol") == protocol_name],
            fieldnames=adaptation_fields,
        )
        write_csv(
            protocol_dir / "adaptation_summary.csv",
            [row for row in adaptation_summary
             if row.get("protocol") == protocol_name],
            fieldnames=adaptation_summary_fields,
        )
        if protocol_name == "online_write":
            write_csv(protocol_dir / "write_metrics.csv", curve, fieldnames=curve_fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hawkes Memory Tree CL checkpoints on task tests and frozen anchors"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("frozen", "fast_adapt", "online_write", "all"),
        default="all",
        help="protocol selection; each selected protocol reloads every checkpoint",
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=CL_PROTOCOL_VARIANTS + tuple(LEGACY_CL_VARIANT_MAP),
        default=None,
        help="canonical protocol/full keys; legacy names are accepted as aliases",
    )
    parser.add_argument("--task-start", type=int, default=None)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument(
        "--current-only", action="store_true",
        help="evaluate each checkpoint only on its own task test set",
    )
    parser.add_argument(
        "--no-anchors", action="store_true",
        help="skip the independent frozen anchor banks",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help=(
            "number of variable-length sequences used for each padded encoder "
            "batch; reduce it when GPU memory is tight"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="reuse completed matrix cells")
    parser.add_argument(
        "--save-event-predictions", action="store_true",
        help="write the large combined event_predictions.csv artifact",
    )
    parser.add_argument(
        "--intensity-samples",
        type=int,
        default=256,
        help="number of time-grid samples used by the NISE integral",
    )
    parser.add_argument(
        "--intensity-plot-anchors",
        type=int,
        default=2,
        help="number of anchor sequences per regime/checkpoint to plot",
    )
    parser.add_argument(
        "--no-hawkes-law-evaluation",
        action="store_true",
        help="skip ground-truth intensity NISE and intensity plots",
    )
    parser.add_argument(
        "--no-summary-plots",
        action="store_true",
        help="skip automatic plots derived from the aggregate CL metrics",
    )
    parser.add_argument(
        "--no-adaptation-evaluation",
        action="store_true",
        help="skip independent support/query adaptation curves",
    )
    parser.add_argument(
        "--fwt-scratch-checkpoint",
        type=Path,
        default=None,
        help=(
            "fixed C_init checkpoint for protocol-scoped FWT; when omitted, "
            "FWT is reported as unavailable rather than inferred from C_0"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print per-sequence progress from the underlying evaluator",
    )
    parser.add_argument("--prototype-duplicate-threshold", type=float, default=None)
    parser.add_argument("--prototype-mode-threshold", type=float, default=None)
    parser.add_argument("--prototype-context-alias-capacity", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.intensity_samples < 2:
        raise ValueError("--intensity-samples must be at least 2")
    if args.intensity_plot_anchors < 0:
        raise ValueError("--intensity-plot-anchors cannot be negative")
    if (
        args.task_start is not None
        and args.task_end is not None
        and args.task_start > args.task_end
    ):
        raise ValueError("--task-start cannot be greater than --task-end")

    args.data_root = args.data_root.expanduser()
    args.checkpoint_dir = args.checkpoint_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    data_root = _normalise_data_root(args.data_root)
    protocol = CLProtocol.load(data_root)
    task_sets = _discover_task_sets(data_root, protocol)
    checkpoint_paths = _discover_checkpoints(args.checkpoint_dir)
    if not task_sets:
        raise FileNotFoundError(f"no task_XX/test.csv files found below {data_root}")
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no task_XX_best.pt (or legacy task_XX.pt) checkpoints found "
            f"below {args.checkpoint_dir}"
        )

    available_ids = sorted(set(task_sets).intersection(checkpoint_paths))
    range_start, range_end = protocol.resolve_range(args.task_start, args.task_end)
    selected_ids = [
        task_id for task_id in available_ids
        if range_start <= task_id <= range_end
    ]
    if not selected_ids:
        raise ValueError(
            "no task has both a test.csv and checkpoint after applying the task range; "
            f"data={sorted(task_sets)}, checkpoints={sorted(checkpoint_paths)}"
        )
    skipped_ids = sorted(set(task_sets).symmetric_difference(checkpoint_paths))

    stage_metadata = _read_stage_metadata(protocol)
    for task_id, evaluation_set in list(task_sets.items()):
        metadata = stage_metadata.get(task_id, {})
        task_sets[task_id] = EvaluationSet(
            name=evaluation_set.name,
            kind=evaluation_set.kind,
            path=evaluation_set.path,
            task_id=task_id,
            regime_id=metadata.get("regime_id"),
            stage_label=metadata.get("stage_label"),
            evaluation_scope="persistent",
        )
    # Persistent-law metrics are deliberately scoped by the protocol.  The
    # manifest may list transient diagnostic laws in first_seen, but they must
    # not enter CL-NLL, average forgetting, or BWT averages.
    regime_first_seen = {
        regime_id: protocol.first_seen[regime_id]
        for regime_id in protocol.persistent_regimes
        if regime_id in protocol.first_seen
    }
    all_regime_first_seen = dict(protocol.first_seen)

    if args.variants is not None:
        variants = list(dict.fromkeys(
            _canonical_variant(variant) for variant in args.variants
        ))
    else:
        variants_by_protocol = {
            "frozen": ["frozen/full"],
            "fast_adapt": ["fast_adapt/full"],
            "online_write": ["online_write/full"],
            "all": list(CL_PROTOCOL_VARIANTS),
        }
        variants = variants_by_protocol[args.protocol]
    anchors = (
        []
        if args.no_anchors
        else _discover_anchors(data_root, protocol)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[CL Eval] data={data_root} checkpoints={args.checkpoint_dir} "
        f"tasks={selected_ids} variants={variants} anchors={len(anchors)}",
        flush=True,
    )

    checkpoint_meta: dict[int, dict[str, Any]] = {}
    checkpoint_sha: dict[int, str] = {}
    benchmark_manifest_sha = _sha256(protocol.manifest_path)
    expected_types: int | None = None
    expected_basis: int | None = None
    tree_by_checkpoint: dict[int, dict[str, Any]] = {}
    for task_id in selected_ids:
        checkpoint = checkpoint_paths[task_id]
        metadata = _checkpoint_meta(checkpoint)
        protocol_metadata = metadata.get("cl_protocol", {})
        saved_manifest_sha = protocol_metadata.get("benchmark_sha256")
        if (
            saved_manifest_sha is not None
            and saved_manifest_sha != benchmark_manifest_sha
        ):
            raise ValueError(
                f"checkpoint task_{task_id:02d} was created from a different "
                "benchmark_manifest.json"
            )
        saved_task_id = protocol_metadata.get("task_id")
        if saved_task_id is not None and int(saved_task_id) != int(task_id):
            raise ValueError(
                f"checkpoint task_{task_id:02d} declares cl task "
                f"{saved_task_id}"
            )
        checkpoint_meta[task_id] = metadata
        checkpoint_sha[task_id] = _sha256(checkpoint)
        current_types = int(metadata["model_config"]["num_event_types"])
        if expected_types is None:
            expected_types = current_types
        elif current_types != expected_types:
            raise ValueError(
                f"checkpoint type mismatch at task_{task_id:02d}: "
                f"{current_types} != {expected_types}"
            )
        current_basis = int(metadata["model_config"]["num_basis"])
        if expected_basis is None:
            expected_basis = current_basis
        elif current_basis != expected_basis:
            raise ValueError(
                f"checkpoint basis mismatch at task_{task_id:02d}: "
                f"{current_basis} != {expected_basis}"
            )
        print(
            f"[CL Eval] loading topology for checkpoint task_{task_id:02d}",
            flush=True,
        )
        tree_by_checkpoint[task_id] = _tree_health(checkpoint)
    assert expected_types is not None
    assert expected_basis is not None
    if expected_types != protocol.event_dim:
        raise ValueError(
            f"checkpoint event dimension {expected_types} disagrees with "
            f"protocol event_dim {protocol.event_dim}"
        )
    if expected_basis != len(protocol.betas):
        raise ValueError(
            f"checkpoint basis count {expected_basis} disagrees with "
            f"protocol betas {len(protocol.betas)}"
        )
    ground_truth: dict[str, GroundTruthLaw] = {}
    ground_truth_meta: dict[str, Any] = {"available": False, "disabled": True}
    if (
        not args.no_hawkes_law_evaluation
        and anchors
        and "frozen/full" in variants
    ):
        ground_truth, ground_truth_meta = _load_ground_truth(
            data_root, expected_types, expected_basis
        )
        if ground_truth:
            print(
                f"[CL Eval] Hawkes law layer: regimes={len(ground_truth)} "
                "variant=frozen/full "
                f"grid={args.intensity_samples}",
                flush=True,
            )
        else:
            print(
                "[CL Eval] Hawkes law layer skipped: "
                f"{ground_truth_meta.get('reason', 'no ground truth')}",
                flush=True,
            )

    evaluation_cache: dict[Path, list[dict[str, Any]]] = {}
    data_sha_cache: dict[Path, str] = {}
    metric_rows: list[dict[str, Any]] = []
    task_matrix_rows: list[dict[str, Any]] = []
    anchor_matrix_rows: list[dict[str, Any]] = []
    control_matrix_rows: list[dict[str, Any]] = []
    protocol_event_rows: list[dict[str, Any]] = []
    event_writer = (
        _EventPredictionWriter(args.output_dir / "event_predictions.csv")
        if args.save_event_predictions
        else None
    )
    protocol_task_ids = list(protocol.task_ids)
    for checkpoint_task in selected_ids:
        checkpoint = checkpoint_paths[checkpoint_task]
        evaluation_sets: list[EvaluationSet] = []
        if args.current_only:
            evaluation_sets.append(task_sets[checkpoint_task])
        else:
            evaluation_sets.append(task_sets[checkpoint_task])
            next_task = next(
                (task_id for task_id in protocol_task_ids if task_id > checkpoint_task),
                None,
            )
            if next_task is not None and next_task in task_sets:
                next_set = task_sets[next_task]
                evaluation_sets.append(EvaluationSet(
                    name=f"{next_set.name}_pre",
                    kind="task_test_pre",
                    path=next_set.path,
                    task_id=next_set.task_id,
                    regime_id=next_set.regime_id,
                    stage_label=next_set.stage_label,
                    evaluation_scope=next_set.evaluation_scope,
                ))
                next_control = _paired_control_set(
                    protocol, next_task, pre_update=True
                )
                if next_control is not None:
                    evaluation_sets.append(next_control)
        current_control = _paired_control_set(
            protocol, checkpoint_task, pre_update=False
        )
        if current_control is not None:
            evaluation_sets.append(current_control)
        evaluation_sets.extend(anchors)
        for evaluation_set in evaluation_sets:
            if evaluation_set.path not in evaluation_cache:
                evaluation_cache[evaluation_set.path] = _load_cl_dataset(
                    evaluation_set.path,
                    expected_types,
                    args.max_sequences,
                )
                data_sha_cache[evaluation_set.path] = dataset_fingerprint(
                    evaluation_set.path
                )
        for variant in variants:
            print(
                f"[CL Eval] checkpoint=task_{checkpoint_task:02d} "
                f"sets={[item.name for item in evaluation_sets]} "
                f"variant={variant} "
                f"sequences={sum(len(evaluation_cache[item.path]) for item in evaluation_sets)} "
                f"batch_size={args.eval_batch_size}",
                flush=True,
            )
            metrics_by_set, event_rows, elapsed, from_cache = (
                _load_or_run_batch(
                    checkpoint=checkpoint,
                    checkpoint_task=checkpoint_task,
                    evaluation_sets=evaluation_sets,
                    variant=variant,
                    evaluation_cache=evaluation_cache,
                    data_sha_cache=data_sha_cache,
                    checkpoint_sha256=checkpoint_sha[checkpoint_task],
                    args=args,
                )
            )
            for evaluation_set in evaluation_sets:
                metrics = metrics_by_set.get(evaluation_set.name, {})
                current_metric_row = _metric_row(
                    checkpoint_task=checkpoint_task,
                    checkpoint=checkpoint,
                    evaluation_set=evaluation_set,
                    variant=variant,
                    metrics=metrics,
                    tree=tree_by_checkpoint[checkpoint_task],
                    elapsed=elapsed,
                    from_cache=from_cache,
                    data_sha256=data_sha_cache[evaluation_set.path],
                )
                metric_rows.append(current_metric_row)
                if evaluation_set.kind in {"task_test", "task_test_pre"}:
                    task_matrix_rows.append(current_metric_row)
                elif evaluation_set.kind in {"matched_control", "matched_control_pre"}:
                    control_matrix_rows.append(current_metric_row)
                else:
                    anchor_matrix_rows.append(current_metric_row)
                selected_event_rows = [
                    row for row in event_rows
                    if row.get("eval_set_id") == evaluation_set.name
                ]
                protocol_event_rows.extend(
                    _decorate_event_rows(selected_event_rows, current_metric_row)
                )
                if event_writer is not None:
                    event_writer.write(selected_event_rows, current_metric_row)

    # Reduce all raw observations to the canonical CL contract.  Retention,
    # forgetting, and BWT are defined on the frozen checkpoint matrix.  The
    # adaptation curves use independent support/query files and a fresh
    # pre-task clone for every K; they never enter the frozen CL aggregates.
    frozen_metric_rows = [
        row for row in metric_rows if row.get("variant") == "frozen/full"
    ]
    scratch_nll_by_task = _fwt_scratch_nlls(
        getattr(args, "fwt_scratch_checkpoint", None),
        protocol,
        selected_ids,
        evaluation_cache,
        expected_types,
        args,
    )
    frozen_records = _frozen_anchor_records(frozen_metric_rows)
    boundary_records = _task_boundary_records(
        frozen_metric_rows,
        protocol,
        scratch_nll_by_task=scratch_nll_by_task,
    )
    adaptation_records: list[AdaptationRecord] = []
    for adaptation_variant in ("fast_adapt/full", "online_write/full"):
        if adaptation_variant not in variants:
            continue
        adaptation_records.extend(_adaptation_records(
            protocol,
            checkpoint_paths,
            selected_ids,
            expected_types,
            args,
            variant=adaptation_variant,
        ))
    topology_events = _read_topology_events(args.checkpoint_dir)
    hm_state_records = _hm_state_records(
        selected_ids,
        tree_by_checkpoint,
        topology_events,
    )
    metric_engine = CLMetricEngine(protocol)
    metric_report = metric_engine.evaluate(
        frozen_anchor_records=frozen_records,
        task_boundary_records=boundary_records,
        adaptation_records=adaptation_records,
        hm_state_records=hm_state_records,
    )

    frozen_variants = ["frozen/full"] if frozen_metric_rows else []
    continual_rows = _continual_summary(
        frozen_metric_rows, selected_ids, frozen_variants, tree_by_checkpoint
    )
    law_rows = [
        {"variant": "frozen/full", **row}
        for row in metric_report["law_metrics"]
    ]
    law_summary_rows = [
        {"variant": "frozen/full", **row}
        for row in metric_report["continual_summary"]
    ]
    existing_summary_tasks = {
        int(row["checkpoint_task"]) for row in law_summary_rows
    }
    for checkpoint_task in selected_ids:
        if checkpoint_task in existing_summary_tasks:
            continue
        law_summary_rows.append({
            "variant": "frozen/full",
            "checkpoint_task": int(checkpoint_task),
            "seen_law_count": 0,
            "clnll": None,
            "average_forgetting": None,
            "average_bwt": None,
            "bwt_law_count": 0,
        })
    law_summary_rows.sort(key=lambda row: int(row["checkpoint_task"]))
    law_summary_by_key = {
        (row["variant"], row["checkpoint_task"]): row
        for row in law_summary_rows
    }
    for row in continual_rows:
        row.update(law_summary_by_key.get(
            (row["variant"], row["checkpoint_task"]),
            {},
        ))
    stage_rows = _stage_metrics(
        frozen_metric_rows,
        frozen_variants,
        protocol=protocol,
        scratch_nll_by_task=scratch_nll_by_task,
    )
    anchor_matrix_rows_wide = [
        {"variant": "frozen/full", **row}
        for row in metric_report["frozen_anchor_matrix"]
    ]
    special_rows = _special_case_metrics(
        frozen_metric_rows, variant="frozen/full", protocol=protocol
    )
    for row in metric_report["rrr"]["rows"]:
        special_rows.append({
            "metric": f"{_safe_name(row['regime_id'])}_rrr_task{row['return_task']}",
            "variant": "frozen/full",
            "formula": (
                "(L_first_pre - L_return_pre) / "
                "(L_first_pre - L_first_post)"
            ),
            "value": row.get("rrr"),
            "left_value": row.get("first_pre_nll"),
            "right_value": row.get("return_pre_nll"),
            "expected": row.get("status", "diagnostic"),
        })
    hm_state_rows = metric_report["hm_state"]
    hm_state_by_task = {
        int(row["task_id"]): row for row in hm_state_rows
    }
    checkpoint_rows = []
    for task_id in selected_ids:
        hm_state = hm_state_by_task.get(task_id, {})
        checkpoint_rows.append({
            "checkpoint_task": task_id,
            "checkpoint": str(checkpoint_paths[task_id].resolve()),
            "checkpoint_sha256": checkpoint_sha[task_id],
            **tree_by_checkpoint[task_id],
            "episodic_rows": hm_state.get("episodic_rows"),
            "episodic_bytes": hm_state.get("episodic_bytes"),
            "semantic_bytes": hm_state.get("semantic_bytes"),
            "split_count": hm_state.get("split_count"),
            "merge_count": hm_state.get("merge_count"),
            "prune_count": hm_state.get("prune_count"),
            "nise": hm_state.get("nise"),
            "history_epochs": len(checkpoint_meta[task_id].get("history", [])),
        })

    write_csv(args.output_dir / "all_metrics.csv", metric_rows)
    write_csv(args.output_dir / "task_metrics.csv", task_matrix_rows)
    write_csv(args.output_dir / "control_metrics.csv", control_matrix_rows)
    write_csv(args.output_dir / "anchor_metrics.csv", anchor_matrix_rows)
    write_csv(args.output_dir / "continual_summary.csv", continual_rows)
    write_csv(args.output_dir / "law_metrics.csv", law_rows)
    write_csv(args.output_dir / "stage_metrics.csv", stage_rows)
    write_csv(args.output_dir / "anchor_nll_matrix.csv", anchor_matrix_rows_wide)
    write_csv(args.output_dir / "special_case_metrics.csv", special_rows)
    write_csv(args.output_dir / "checkpoint_tree.csv", checkpoint_rows)
    if event_writer is not None:
        event_writer.close()

    if "frozen/full" in variants:
        intensity_rows, intensity_summary_rows = _hawkes_law_evaluation(
            checkpoint_paths=checkpoint_paths,
            checkpoint_tasks=selected_ids,
            anchors=anchors,
            evaluation_cache=evaluation_cache,
            ground_truth=ground_truth,
            regime_first_seen=all_regime_first_seen,
            args=args,
            expected_types=expected_types,
            transient_regimes=protocol.transient_regimes,
        )
    else:
        intensity_rows, intensity_summary_rows = [], []
    ood_rows = _ood_metrics(
        frozen_metric_rows, ground_truth, regime_first_seen
    )
    nise_by_task: dict[int, float | None] = {}
    for task_id in selected_ids:
        values = []
        for row in intensity_summary_rows:
            if (
                row.get("evaluation_scope") != "seen_law"
                or row.get("variant") != "frozen/full"
            ):
                continue
            try:
                row_task = int(row.get("checkpoint_task"))
            except (TypeError, ValueError):
                continue
            if row_task == int(task_id):
                values.append(row.get("nise_mean"))
        nise_by_task[int(task_id)] = _mean(values)
    for row in hm_state_rows:
        row["nise"] = nise_by_task.get(int(row["task_id"]))
    for row in checkpoint_rows:
        row["nise"] = nise_by_task.get(int(row["checkpoint_task"]))
    write_csv(args.output_dir / "intensity_metrics.csv", intensity_rows)
    write_csv(args.output_dir / "intensity_summary.csv", intensity_summary_rows)
    write_csv(args.output_dir / "ood_metrics.csv", ood_rows)
    write_csv(
        args.output_dir / "fwt_metrics.csv",
        metric_report["fwt"]["rows"],
        fieldnames=(
            "task_id", "pre_nll", "post_nll", "scratch_nll", "shift_type",
            "recurrence_of", "new_persistent_regimes", "adaptation_gain_nll",
            "fwt_nll", "fwt_eligible", "fwt_status",
        ),
    )
    write_csv(
        args.output_dir / "adaptation_points.csv",
        metric_report["adaptation"]["points"],
        fieldnames=(
            "protocol", "task_id", "K", "pre_nll", "adapted_nll", "gain_nll",
        ),
    )
    write_csv(
        args.output_dir / "adaptation_summary.csv",
        metric_report["adaptation"]["summary"],
        fieldnames=(
            "protocol", "task_id", "K_min", "K_max", "K_count",
            "adaptation_auc", "status",
        ),
    )
    write_csv(
        args.output_dir / "rrr_metrics.csv",
        metric_report["rrr"]["rows"],
        fieldnames=(
            "regime_id", "first_task", "return_task", "shift_type",
            "first_pre_nll", "first_post_nll", "return_pre_nll",
            "first_gain_nll", "rrr", "status",
        ),
    )
    write_csv(args.output_dir / "hm_state.csv", hm_state_rows)
    (args.output_dir / "cl_metrics.json").write_text(
        json.dumps(_jsonable(metric_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_protocol_outputs(
        args.output_dir,
        metric_rows,
        protocol_event_rows,
        continual_rows,
        law_rows,
        anchor_matrix_rows_wide,
        metric_report["adaptation"]["points"],
        metric_report["adaptation"]["summary"],
    )

    summary_plot_paths = (
        []
        if args.no_summary_plots
        else _plot_summary_figures(
            args.output_dir,
            continual_rows=continual_rows,
            anchor_matrix_rows=anchor_matrix_rows_wide,
            stage_rows=stage_rows,
            special_rows=special_rows,
            intensity_summary_rows=intensity_summary_rows,
            checkpoint_rows=checkpoint_rows,
        )
    )

    summary = {
        "data_root": str(data_root.resolve()),
        "benchmark_manifest": str(
            protocol.manifest_path.resolve()
        ),
        "benchmark": protocol.benchmark_id,
        "benchmark_version": protocol.version,
        "persistent_regimes": sorted(protocol.persistent_regimes),
        "transient_regimes": sorted(protocol.transient_regimes),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "protocol": "frozen",
        "protocols": variants,
        "primary_protocol": "frozen",
        "memory_view": "full",
        "variants": variants,
        "task_ids": selected_ids,
        "available_data_task_ids": sorted(task_sets),
        "available_checkpoint_task_ids": sorted(checkpoint_paths),
        "skipped_task_ids": skipped_ids,
        "current_only": bool(args.current_only),
        "anchors_enabled": not args.no_anchors,
        "anchor_files": [str(item.path.resolve()) for item in anchors],
        "controls": list(protocol.controls),
        "tasks": [
            {
                "task_id": task_id,
                "test_path": str(task_sets[task_id].path.resolve()),
                "stage_label": task_sets[task_id].stage_label,
                "regime_id": task_sets[task_id].regime_id,
                "metadata": stage_metadata.get(task_id, {}),
            }
            for task_id in selected_ids
        ],
        "tree": tree_by_checkpoint,
        "regime_first_seen": regime_first_seen,
        "all_regime_first_seen": all_regime_first_seen,
        "ground_truth": ground_truth_meta,
        "hawkes_law_evaluation": {
            "enabled": (
                not args.no_hawkes_law_evaluation
                and bool(anchors)
                and "frozen/full" in variants
            ),
            "variant": "frozen/full",
            "intensity_samples": args.intensity_samples,
            "intensity_plot_anchors": args.intensity_plot_anchors,
        },
        "metrics": metric_rows,
        "control_metrics": control_matrix_rows,
        "continual_summary": continual_rows,
        "law_metrics": law_rows,
        "stage_metrics": stage_rows,
        "anchor_nll_matrix": anchor_matrix_rows_wide,
        "cl_metrics": metric_report,
        "fwt": metric_report["fwt"],
        "adaptation": metric_report["adaptation"],
        "rrr": metric_report["rrr"],
        "hm_state": hm_state_rows,
        "topology_events": [dict(row) for row in topology_events],
        "fwt_scratch_checkpoint": (
            None
            if getattr(args, "fwt_scratch_checkpoint", None) is None
            else str(Path(args.fwt_scratch_checkpoint).expanduser().resolve())
        ),
        "special_case_metrics": special_rows,
        "intensity_metrics": intensity_rows,
        "intensity_summary": intensity_summary_rows,
        "ood_metrics": ood_rows,
        "summary_plots": summary_plot_paths,
        "note": (
            "CL evaluation uses model-facing task CSVs and independent frozen anchors. "
            "Oracle manifests provide task/law labels; ground-truth laws are used only "
            "for the separate Hawkes intensity NISE layer."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        args.output_dir / "report.md",
        data_root=data_root,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_tasks=selected_ids,
        variants=variants,
        metric_rows=metric_rows,
        continual_rows=continual_rows,
        law_rows=law_rows,
        stage_rows=stage_rows,
        special_rows=special_rows,
        intensity_summary_rows=intensity_summary_rows,
        ood_rows=ood_rows,
        summary_plot_paths=summary_plot_paths,
        tree_by_checkpoint=tree_by_checkpoint,
        skipped_tasks=skipped_ids,
        anchors_enabled=not args.no_anchors,
        metric_report=metric_report,
    )
    print(f"[Done] CL report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
