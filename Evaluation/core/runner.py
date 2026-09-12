from __future__ import annotations

import json
import csv
import gzip
import io
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from .adapters import copy_checkpoint_contract, copy_prediction_contract, evaluate_hm, normalize_native_metrics, resolved_device, run_command, stationary_command
from .cl_protocol import CLProtocol
from .data import (
    continual_root,
    continual_protocol,
    dataset_inputs,
    prepare_continual_baseline_dataset,
    prepare_continual_hm_dataset,
    prepare_hm_dataset,
)
from .io import sha256, write_csv, write_json
from .manifest import begin, build_manifest, finish
from .cl_metrics import (
    AdaptationRecord,
    CLMetricEngine,
    FrozenAnchorRecord,
    TaskBoundaryRecord,
)
from .paths import DATASETS_ROOT, EVALUATION_ROOT, MODELS_ROOT, PROJECT_ROOT, result_dir
from .report import write_report
from .resources import resource_record
from .specs import JobSpec


def run_stationary_job(*, dataset: str, model: str, args, condition: str = "full", script: str = "") -> Path:
    spec = JobSpec(dataset=dataset, model=model, condition=condition, script=script)
    target = result_dir(spec, args)
    prepared = target / "prepared" if dataset == "dws" or model == "HM" else None
    command, cwd, env = stationary_command(spec, args, target, prepared=prepared)
    inputs = dataset_inputs(dataset, getattr(args, "variant", None))
    if args.checkpoint is not None:
        inputs.append(args.checkpoint)
    manifest = build_manifest(spec, args, inputs, command)
    if not begin(target, manifest, args.resume):
        print(f"Complete result already exists: {target}")
        return target
    if args.dry_run:
        finish(target, "dry_run")
        print(" ".join(command))
        return target
    started = time.perf_counter()
    try:
        if prepared is not None:
            prepare_hm_dataset(
                spec.dataset,
                args.seed,
                prepared,
                getattr(args, "variant", None),
            )
        run_command(command, cwd, env, target / "logs" / "train.log")
        copy_checkpoint_contract(target)
        if model == "HM":
            evaluate_hm(spec, args, target, env)
        copy_prediction_contract(target)
        metrics = normalize_native_metrics(spec, target)
        write_json(target / "metrics.json", metrics)
        source_sequence_metrics = target / "native" / "sequence_metrics.csv"
        if source_sequence_metrics.exists():
            shutil.copy2(source_sequence_metrics, target / "sequence_metrics.csv")
        else:
            write_csv(target / "sequence_metrics.csv", [{"split": "test", **metrics}])
        resources = resource_record(target, started, command)
        write_json(target / "resources.json", resources)
        write_report(target / "report.md", spec, metrics, command)
        (target / "plots").mkdir(exist_ok=True)
        (target / "tree_events.jsonl").touch(exist_ok=True)
        finish(target, "complete")
    except Exception as error:
        finish(target, "failed", f"{error}\n{traceback.format_exc()}")
        raise
    print(f"Result directory: {target}")
    return target


def _continual_hm_command(
    args,
    target: Path,
    data_root: Path,
    task: int,
    previous: Path | None,
    strategy: str,
    protocol: CLProtocol,
) -> tuple[list[str], Path, dict[str, str]]:
    import os
    import sys
    memory = MODELS_ROOT / "HawkesMemory" / "Memory"
    last_checkpoint = target / "checkpoint" / f"task_{task:02d}_last.pt"
    best_checkpoint = target / "checkpoint" / f"task_{task:02d}_best.pt"
    prepared = target / "prepared" / f"hm_task_{task:02d}"
    data_path, split_manifest = prepare_continual_hm_dataset(
        data_root,
        prepared,
        task,
        protocol,
    )
    cold_start_epochs = 0 if previous is not None else (1 if args.smoke else 5)
    initial_checkpoint = target / "checkpoint" / f"initial_seed{args.seed}.pt"
    topology_log_path = target / "topology_diagnostics.log"
    topology_events_path = target / "topology_events.jsonl"
    cl_config_path = target / "cl_config.json"
    benchmark_manifest_path = target / "benchmark_protocol.json"
    command = [
        args.python_executable or sys.executable,
        "-m",
        "Train.Train",
        "--data-path",
        str(data_path),
        "--split-manifest",
        str(split_manifest),
        "--split",
        "train",
        "--checkpoint",
        str(last_checkpoint),
        "--best-checkpoint",
        str(best_checkpoint),
        "--tree-init-depth",
        "0",
        "--seed",
        str(args.seed),
        "--device",
        resolved_device(args.device),
        "--epochs",
        str(args.epochs or (1 if args.smoke else 30)),
        "--cold-start-epochs",
        str(cold_start_epochs),
        "--unified-topology-log-path",
        str(topology_log_path),
        "--topology-events-path",
        str(topology_events_path),
        "--cl-config",
        str(cl_config_path),
        "--benchmark-manifest",
        str(benchmark_manifest_path),
        "--cl-task-id",
        str(task),
    ]
    if previous is None:
        command += ["--initial-checkpoint-output", str(initial_checkpoint)]
    if previous is not None:
        command += [
            "--resume",
            str(previous),
            "--cl-previous-checkpoint",
            str(previous),
        ]
    if strategy in {
        "no_working",
        "no_episodic",
        "fixed_topology",
        "no_sleep",
        "heuristic_controller",
        "no_merge_prune",
    }:
        command += ["--evaluation-ablation", strategy]
    if args.smoke:
        command += ["--max-sequences", "2", "--max-events-per-sequence", "12", "--no-training-plots"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), str(MODELS_ROOT / "HawkesMemory"), str(memory), env.get("PYTHONPATH", "")))
    return command, memory, env


def _hm_state_bytes(checkpoint: Path) -> dict[str, int]:
    result = {"checkpoint_bytes": checkpoint.stat().st_size}
    try:
        import torch
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        buffer = io.BytesIO()
        torch.save(payload.get("tree_state_dict", {}), buffer)
        result["tree_and_episodic_bytes"] = buffer.tell()
    except Exception as error:
        result["tree_and_episodic_bytes"] = checkpoint.stat().st_size
        result["measurement_warning"] = str(error)
    return result


def _continual_cl_config(args, protocol: CLProtocol, strategy: str) -> dict[str, Any]:
    """Return the learner protocol shared by every HM continual stage."""

    return {
        "format_version": 1,
        "benchmark_id": protocol.benchmark_id,
        "benchmark_version": protocol.version,
        "epochs_per_task": int(args.epochs or (1 if args.smoke else 30)),
        "cold_start_epochs": int(1 if args.smoke else 5),
        "training": {
            "seed": int(args.seed),
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
            "optimizer_impl": "auto",
            "sleep_every": 1,
            "evaluation_ablation": strategy,
            "controller_target_version": 5,
            "controller_train_heads": ["adapt", "retrieve", "write"],
            "controller_write_ranking": False,
        },
        "optimizer": {"learning_rate": 1e-3, "weight_decay": 1e-5},
        "model": {
            "z_dim": 50,
            "node_dim": 64,
            "memory_key_dim": 64,
            "num_basis": 2,
            "decays": [0.5, 1.5],
            "semantic_blend": 0.1,
        },
        "frontier": {
            "frontier_budget": 4,
            "frontier_min_experts": 2,
            "routing_temperature": 1.5,
            "exploration_epsilon": 0.0,
            "confidence_weight": 0.25,
            "expansion_compute_cost": 0.05,
            "posterior_temperature": 1.0,
            "credible_mass": 0.90,
            "owner_confidence_threshold": 0.80,
            "max_writes_per_sequence": 8,
        },
        "wake": {
            "prototype_duplicate_threshold": 0.98,
            "prototype_duplicate_quantile": 0.85,
            "prototype_mode_threshold": 0.90,
            "prototype_mode_capacity": 12,
            "prototype_context_alias_capacity": 3,
            "count_similarity_low": 0.35,
            "count_similarity_high": 0.65,
            "count_exponent": 2.0,
            "count_saturation": 3.0,
            "count_topk": None,
            "lambda_route_mi": 0.2,
            "lambda_route_posterior": 0.0,
            "lambda_route_distill": 1.0,
            "lambda_route_mix": 0.0,
            "route_energy_temperature": 1.0,
            "route_encoder_warmup_epochs": 2,
            "route_encoder_grad_scale": 0.1,
            "route_encoder_reliability_decay": 0.9,
            "route_teacher_temperature": 1.0,
            "lambda_route_probe": 0.1,
            "route_probe_leaves": 2,
            "route_probe_leaf_smoothing": 0.05,
            "route_probe_residual_temperature": 1.0,
            "route_probe_gain_temperature": 0.1,
            "route_probe_complexity_weight": 0.01,
            "route_probe_residual_rank": 2,
            "route_probe_residual_grad_clip": 0.0,
            "lambda_route_balance": 0.05,
            "route_balance_batch_size": 64,
            "wake_wavefront_batch_size": 64,
            "retrieval_microbatch": 1024,
            "route_balance_max_steps": 8,
            "route_balance_target_kl": 0.1,
        },
        "sleep": {
            "light_replay_budget": 32,
            "split_min_structural_strength": 0.0,
            "split_min_effective_sample_size": 0.0,
            "split_route_loss_weight": 1.0,
            "split_anchor_weight": 1e-2,
            "deep_availability_tau": 3.0,
            "deep_probe_interval": 5,
            "deep_computation_cost": 0.05,
            "deep_prior_probability": 0.15,
            "deep_prior_weight": 0.01,
            "deep_evidence_budget": 32,
            "topology_inertia_strength": 0.01,
            "topology_inertia_tau": 3.0,
        },
        "structure": {
            "prune_warmup_epochs": 10,
            "merge_kwargs": {
                "min_replay": 8,
                "stale_weight": 0.2,
                "dynamics_weight": 0.1,
                "loss_weight": 0.1,
                "gate_temperature": 1.0,
                "budget_ratio": 0.95,
                "dual_lr": 1e-6,
                "dual_initial": 0.0,
            },
        },
        "permissions": {
            "gradient_update": "train.csv",
            "checkpoint_selection": "val.csv",
            "final_evaluation": "test.csv",
            "selection_forbidden": ["anchors/*", "ground_truth/*", "test.csv"],
        },
    }


def _file_record(path: Path, *, role: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "role": role}


def _continual_stage_manifest(
    protocol: CLProtocol,
    config_path: Path,
    *,
    stages: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "benchmark_id": protocol.benchmark_id,
        "benchmark_manifest": _file_record(
            protocol.manifest_path, role="protocol"
        ),
        "cl_config": _file_record(config_path, role="learner_configuration"),
        "permissions": {
            "gradient_update": "train.csv",
            "checkpoint_selection": "val.csv",
            "final_evaluation": "test.csv",
            "selection_forbidden": ["anchors/*", "ground_truth/*", "test.csv"],
        },
        "stages": {} if stages is None else stages,
    }


def _validate_continual_stage_manifest(
    stage_manifest: dict[str, Any],
    protocol: CLProtocol,
    config_path: Path,
) -> None:
    """Reject reruns whose recorded benchmark or learner protocol changed."""

    if stage_manifest.get("benchmark_id") != protocol.benchmark_id:
        raise RuntimeError(
            "continual stage manifest benchmark_id does not match the active "
            "benchmark protocol"
        )
    expected_benchmark = _file_record(
        protocol.manifest_path, role="protocol"
    )
    recorded_benchmark = stage_manifest.get("benchmark_manifest")
    if not isinstance(recorded_benchmark, dict) or any(
        recorded_benchmark.get(key) != expected_benchmark[key]
        for key in ("path", "sha256")
    ):
        raise RuntimeError(
            "continual stage manifest points at a different "
            "benchmark_manifest.json"
        )
    expected_config = _file_record(
        config_path, role="learner_configuration"
    )
    recorded_config = stage_manifest.get("cl_config")
    if not isinstance(recorded_config, dict) or any(
        recorded_config.get(key) != expected_config[key]
        for key in ("path", "sha256")
    ):
        raise RuntimeError(
            "continual stage manifest points at a different cl_config.json"
        )


def _baseline_command(model: str, args, prepared: Path, output: Path,
                      initial: Path | None, evaluate_only: bool,
                      dataset_label: str | None = None):
    import os
    import sys
    python = args.python_executable or sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(__import__("pathlib").Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")))
    if model == "RMTPP":
        epochs = args.epochs or (1 if args.smoke else 80)
        batch = args.batch_size or (2 if args.smoke else 16)
    elif model in {"S2P2", "AttNHP"}:
        epochs = args.epochs or (1 if args.smoke else 80)
        batch = args.batch_size or (2 if args.smoke else 64)
    else:
        epochs = args.epochs or (1 if args.smoke else (1 if model == "TPP_LLM" else 20))
        batch = args.batch_size or (2 if args.smoke else 32)
    device = resolved_device(args.device)
    if model == "RMTPP":
        command = [
            python,
            str(MODELS_ROOT / "RMTPP" / "run_experiment.py"),
            "--dataset", "taobao",
            "--prepared-data-dir", str(prepared),
            "--output-dir", str(output),
            "--archive", str(output) + ".tar.gz",
            "--overwrite",
            "--seed", str(args.seed),
            "--gpu", ("-1" if device == "cpu" else device.split(":")[-1]),
            "--epochs", str(epochs),
            "--batch-size", str(batch),
            "--learning-rate", "5e-4",
            "--hidden-size", "64",
            "--mc-samples", "32",
            "--early-stop-patience", "20",
            "--lr-patience", "6",
            "--lr-factor", "0.3",
        ]
        if dataset_label:
            command += ["--dataset-label", dataset_label]
        cwd = MODELS_ROOT / "RMTPP"
        if initial:
            command += ["--initial-checkpoint", str(initial)]
        if evaluate_only:
            command += ["--evaluate-only"]
    elif model == "THP":
        command = [python, str(MODELS_ROOT / "THP" / "run_experiment.py"),
                   "--dataset", "taobao", "--prepared-data-dir", str(prepared),
                   "--output-dir", str(output), "--archive", str(output) + ".tar.gz",
                   "--overwrite", "--seed", str(args.seed), "--device", device,
                   "--epochs", str(epochs), "--batch-size", str(batch)]
        if dataset_label:
            command += ["--dataset-label", dataset_label]
        cwd = MODELS_ROOT / "THP"
        if initial:
            command += ["--initial-checkpoint", str(initial)]
        if evaluate_only:
            command += ["--evaluate-only"]
    elif model in {"S2P2", "AttNHP"}:
        command = [
            python,
            str(MODELS_ROOT / "EasyTPP" / "run_experiment.py"),
            "--model", model,
            "--dataset", "taobao",
            "--prepared-data-dir", str(prepared),
            "--output-dir", str(output),
            "--archive", str(output) + ".tar.gz",
            "--overwrite",
            "--seed", str(args.seed),
            "--device", device,
            "--epochs", str(epochs),
            "--batch-size", str(batch),
        ]
        if dataset_label:
            command += ["--dataset-label", dataset_label]
        if args.smoke:
            command += ["--max-sequences", "4", "--max-events-per-sequence", "16"]
        cwd = MODELS_ROOT / "EasyTPP"
        if initial:
            command += ["--initial-checkpoint", str(initial)]
        if evaluate_only:
            command += ["--evaluate-only"]
    elif model == "TPP_LLM":
        command = [python, str(MODELS_ROOT / "TPP-LLM" / "scripts" / "train_tpp_llm.py"), "--evaluation_dataset", "taobao", "--prepared_data", "--data_path", str(prepared), "--evaluation_output", str(output), "--anonymous_labels", "--device", device, "--seed", str(args.seed), "--peft_type", "lora", "--lora_rank", "16", "--num_train_epochs", str(0 if evaluate_only else epochs), "--train_batch_size", str(batch), "--eval_batch_size", str(batch)]
        cwd = MODELS_ROOT / "TPP-LLM"
        if initial:
            command += ["--initial_checkpoint", str(initial)]
    else:
        raise KeyError(model)
    return command, cwd, env


def _native_metric(model: str, output: Path) -> dict:
    if model == "TPP_LLM":
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    candidates = sorted(output.rglob("*test*.csv"))
    if not candidates:
        raise FileNotFoundError(f"test metrics absent below {output}")
    with candidates[-1].open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    values = {key.strip().lower(): value for key, value in row.items()}
    ll = float(values["log-likelihood"])
    raw_events = values.get("num_events", values.get("events", 0))
    try:
        num_events = int(float(raw_events))
    except (TypeError, ValueError):
        num_events = 0
    return {
        "nll_per_event": -ll,
        "accuracy": float(values["accuracy"]),
        "time_rmse": float(values["rmse"]),
        "num_events": num_events,
    }


def _serialized_replay_row(row: dict[str, str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("event_times", "event_types"),
        lineterminator="\r\n",
    )
    writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _select_task_balanced_replay_rows(
    candidates: dict[int, list[tuple[int, dict[str, str], int]]],
    budget: int,
    header_bytes: int,
    selection_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Select a deterministic task-balanced byte reservoir.

    Replay is a model-facing training resource, so its selection cannot use
    ground-truth law labels.  Tasks compete by selected serialized bytes and
    retain source order within each task.  The final output is sorted by task
    and source identity so repeated runs are byte-identical.
    """

    positions = {task: 0 for task in candidates}
    task_bytes = {task: 0 for task in candidates}
    remaining = max(0, int(budget) - int(header_bytes))
    selected: list[tuple[int, int, dict[str, str]]] = []

    while remaining > 0:
        fitting_by_task: dict[int, tuple[int, dict[str, str], int]] = {}
        for task in sorted(candidates):
            entries = candidates[task]
            position = positions[task]
            # If the next source row is too large, skip it permanently;
            # remaining bytes only decrease, so it can never fit later.
            while position < len(entries) and entries[position][2] > remaining:
                position += 1
            positions[task] = position
            if position < len(entries):
                fitting_by_task[task] = entries[position]
        if not fitting_by_task:
            break
        task = min(fitting_by_task, key=lambda value: (task_bytes[value], value))
        source_index, row, row_bytes = fitting_by_task[task]
        selected.append((task, source_index, row))
        positions[task] += 1
        task_bytes[task] += row_bytes
        remaining -= row_bytes

    selected.sort(key=lambda item: (item[0], item[1]))
    if selection_metadata is not None:
        selection_metadata.update({
            "header_bytes": int(header_bytes),
            "selected_rows": len(selected),
            "task_bytes": {str(task): int(value) for task, value in sorted(task_bytes.items())},
        })
    return [row for _, _, row in selected]


def _make_replay_buffer(
    data_root: Path,
    through_task: int,
    budget: int,
    output: Path,
    protocol: CLProtocol | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    """Build a byte-bounded, task-balanced replay CSV.

    The small legacy helper remains usable for isolated ``task_XX/train.csv``
    fixtures.  Production runs pass the loaded protocol.  No ground-truth
    manifest is consulted because replay selection is part of the learner's
    training path.
    """

    legacy_layout = protocol is None and not (
        data_root / "benchmark_manifest.json"
    ).is_file()
    if protocol is None and not legacy_layout:
        protocol = continual_protocol(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates: dict[int, list[tuple[int, dict[str, str], int]]] = {}
    header_bytes = len("event_times,event_types\r\n".encode("utf-8"))
    task_ids = (
        list(range(through_task + 1))
        if legacy_layout
        else protocol.task_ids_between(protocol.task_ids[0], through_task)
    )
    for task in task_ids:
        source = (
            data_root / f"task_{task:02d}" / "train.csv"
            if legacy_layout
            else protocol.split_path(task, "train")
        )
        source_index = 0
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row = {
                    "event_times": row.get("event_times", ""),
                    "event_types": row.get("event_types", ""),
                }
                candidates.setdefault(task, []).append(
                    (source_index, row, len(_serialized_replay_row(row)))
                )
                source_index += 1
    rows = _select_task_balanced_replay_rows(
        candidates,
        int(budget),
        header_bytes,
        selection_metadata,
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
        writer.writeheader()
        writer.writerows(rows)
    actual = output.stat().st_size
    if actual > budget:
        raise RuntimeError(f"replay serialization {actual} exceeds HM budget {budget}")
    return output, actual


def _write_adaptation_prefix(
    source: Path,
    output: Path,
    exposure_events: int,
) -> bool:
    """Write exactly ``exposure_events`` observed support events.

    The benchmark exposure axis counts observed support events.  A one-event
    prefix is therefore intentionally valid even though it contains no
    supervised transition; the caller records that point as a no-update
    baseline instead of silently changing its meaning to ``K + 1``.
    """

    if exposure_events <= 0:
        raise ValueError("adaptation prefix exposure_events must be positive")
    target_events = int(exposure_events)
    parts: list[tuple[list[float], list[int]]] = []
    remaining = target_events
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    last_time = 0.0
    for row in source_rows:
        times = [float(value) for value in json.loads(row["event_times"])]
        types = [int(value) for value in json.loads(row["event_types"])]
        if len(times) != len(types) or not times:
            raise ValueError(f"invalid adaptation support row in {source}")
        take = min(remaining, len(times))
        local_times = times[:take]
        shift = 0.0 if not parts else last_time + 1.0 - local_times[0]
        shifted = [value + shift for value in local_times]
        parts.append((shifted, types[:take]))
        last_time = shifted[-1]
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        return False
    times = [value for part, _ in parts for value in part]
    types = [value for _, part in parts for value in part]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
        writer.writeheader()
        writer.writerow({
            "event_times": json.dumps(times, separators=(",", ":")),
            "event_types": json.dumps(types, separators=(",", ":")),
        })
    return True


def _run_baseline_adaptation(
    model: str,
    strategy: str,
    args,
    target: Path,
    data_root: Path,
    protocol: CLProtocol,
    task: int,
    pre_checkpoint: Path,
) -> tuple[list[AdaptationRecord], list[dict[str, Any]]]:
    """Run baseline K-shot adaptation from a fresh copy of the pre-task model."""

    spec = protocol.adaptation(task)
    if spec is None:
        return [], []
    if not pre_checkpoint.is_file():
        raise FileNotFoundError(f"baseline adaptation checkpoint required: {pre_checkpoint}")

    adapted_nll: dict[int, float | None] = {}
    status_rows: list[dict[str, Any]] = []
    adaptation_root = target / "native" / f"task_{task:02d}" / "adaptation"
    prepared_root = target / "prepared" / f"task_{task:02d}_adaptation"
    query_path = Path(spec["query"])
    for K in sorted(int(raw_k) for raw_k in spec["K"]):
        output = adaptation_root / f"K_{K:02d}"
        if K == 0:
            # The K=0 point is evaluated on the fixed query from the unchanged
            # pre-task checkpoint.  The task train split only supplies the
            # native loader with a valid, unused training dataset.
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                prepared_root / f"K_{K:02d}",
                [task],
                eval_csv=query_path,
                protocol=protocol,
            )
            evaluate_only = True
            status = "baseline"
        else:
            support_path = prepared_root / f"support_K_{K:02d}.csv"
            if not _write_adaptation_prefix(Path(spec["support"]), support_path, K):
                adapted_nll[K] = None
                status_rows.append({
                    "protocol": "baseline",
                    "task_id": int(task),
                    "K": K,
                    "status": "unavailable_missing_support",
                    "observed_events": K,
                    "train_events": 0,
                })
                continue
            if K == 1:
                # There is no supervised transition in a one-event prefix.
                # Reusing the K=0 query evaluation is the exact zero-gradient
                # result and avoids asking a native trainer to fabricate a
                # target event.
                adapted_nll[K] = adapted_nll.get(0)
                output.mkdir(parents=True, exist_ok=True)
                write_json(output / "metrics.json", {
                    "nll_per_event": adapted_nll[K],
                    "status": "no_update_insufficient_transition",
                    "observed_events": K,
                })
                status_rows.append({
                    "protocol": "baseline",
                    "task_id": int(task),
                    "K": K,
                    "status": "no_update_insufficient_transition",
                    "observed_events": K,
                    "train_events": 0,
                })
                continue
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                prepared_root / f"K_{K:02d}",
                [],
                eval_csv=query_path,
                protocol=protocol,
                train_csvs=[support_path],
                current_task=task,
                validation_csv=support_path,
                train_min_events=1,
                validation_min_events=1,
            )
            evaluate_only = False
            status = "trained"
        status_rows.append({
            "protocol": "baseline",
            "task_id": int(task),
            "K": K,
            "status": status,
            "observed_events": K,
            "train_events": K,
        })
        command, cwd, env = _baseline_command(
            model,
            args,
            prepared,
            output,
            pre_checkpoint,
            evaluate_only,
            dataset_label=protocol.benchmark_id,
        )
        run_command(
            command,
            cwd,
            env,
            target / "logs" / f"task_{task:02d}_adapt_K_{K:02d}.log",
        )
        adapted_nll[K] = _native_metric(model, output).get("nll_per_event")

    pre_nll = adapted_nll.get(0)
    records = [
        AdaptationRecord(
            task_id=int(task),
            K=K,
            pre_nll=pre_nll,
            adapted_nll=adapted_nll.get(K),
            protocol="baseline",
        )
        for K in sorted(int(value) for value in spec["K"])
    ]
    return records, status_rows


def _baseline_cl_summary(
    rows: list[dict],
    protocol: CLProtocol,
    adaptation_records: list[AdaptationRecord] | None = None,
) -> dict:
    def optional_float(row: dict | None, key: str) -> float | None:
        if not row or row.get(key) is None:
            return None
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    task_rows = {
        (int(row["task"]), str(row["evaluation"])): row
        for row in rows
        if row.get("evaluation") != "frozen_anchor"
    }
    boundaries = [
        TaskBoundaryRecord(
            task_id=task_id,
            pre_nll=optional_float(task_rows.get((task_id, "pre_update")), "nll_per_event"),
            post_nll=optional_float(task_rows.get((task_id, "post_update")), "nll_per_event"),
            scratch_nll=optional_float(task_rows.get((task_id, "scratch_pre")), "nll_per_event"),
            shift_type=protocol.task(task_id).shift_type,
            recurrence_of=protocol.task(task_id).recurrence_of,
        )
        for task_id in protocol.task_ids
    ]
    anchors = [
        FrozenAnchorRecord(
            checkpoint_task=int(row["task"]),
            regime_id=str(row["anchor"]),
            nll_per_event=optional_float(row, "nll_per_event"),
            num_events=int(row.get("num_events") or 0),
            evaluation_scope=str(row.get("evaluation_scope", "persistent")),
        )
        for row in rows
        if row.get("evaluation") == "frozen_anchor"
    ]
    report = CLMetricEngine(protocol).evaluate(
        frozen_anchor_records=anchors,
        task_boundary_records=boundaries,
        adaptation_records=adaptation_records or (),
    )
    boundary_rows = report["task_boundaries"]
    transfer = [
        {
            "task": row["task_id"],
            "shift_type": row["shift_type"],
            "recurrence_of": row["recurrence_of"],
            "adaptation_gain_nll": row["adaptation_gain_nll"],
            "forward_transfer_nll": row["fwt_nll"],
            "fwt_eligible": row["fwt_eligible"],
            "fwt_status": row["fwt_status"],
        }
        for row in boundary_rows
    ]
    post_rows = [
        row for row in rows
        if row.get("evaluation") == "post_update"
    ]
    final_post = max(post_rows, key=lambda row: int(row["task"])) if post_rows else {}
    rrr_rows = report["rrr"]["rows"]
    first_rrr = rrr_rows[0] if rrr_rows else {}
    return {
        "nll_per_event": final_post.get("nll_per_event"),
        "accuracy": final_post.get("accuracy"),
        "time_rmse": final_post.get("time_rmse"),
        "continual_summary": report["continual_summary"],
        "transfer": transfer,
        # Task-index AUC was never an adaptation curve.  The canonical field
        # is populated only when independent K-indexed records are supplied.
        "adaptation_auc": report["adaptation"]["average_adaptation_auc"],
        "adaptation": report["adaptation"],
        "fwt": report["fwt"],
        "rrr": report["rrr"],
        "recurrence_retention_ratio": first_rrr.get("rrr"),
        "recurrence_retention_regime": first_rrr.get("regime_id"),
        # Preserve the old result key for consumers that only read the scalar.
        "recurrence_retention_ratio_A": first_rrr.get("rrr"),
        "cl_metrics": report,
        "tsr": None,
        "tsr_note": "TSR is topology-specific and is not defined for flat baselines.",
    }


def _run_baseline_continual(model: str, strategy: str, args, target: Path,
                            data_root: Path, protocol: CLProtocol):
    native = target / "native"
    previous = args.checkpoint
    task_ids = protocol.task_ids_between(args.task_start, args.task_end)
    first_protocol_task = protocol.task_ids[0]
    start_index = protocol.task_ids.index(task_ids[0])
    if previous is None and start_index > 0:
        previous_task = protocol.task_ids[start_index - 1]
        previous = target / "checkpoint" / f"task_{previous_task:02d}.pt"
    if previous is not None and not previous.is_file():
        raise FileNotFoundError(f"resume checkpoint required for task-start: {previous}")
    stage_rows = []
    adaptation_records: list[AdaptationRecord] = []
    adaptation_status_rows: list[dict[str, Any]] = []
    replay_rows = []
    prediction_target = target / "predictions.jsonl.gz"
    with gzip.open(prediction_target, "wt", encoding="utf-8") as combined_predictions:
        for task in task_ids:
            pre_checkpoint = previous
            task_root = native / f"task_{task:02d}"
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                target / "prepared" / f"task_{task:02d}_pre",
                [task],
                protocol=protocol,
            )
            pre_output = task_root / "pre_update"
            command, cwd, env = _baseline_command(
                model,
                args,
                prepared,
                pre_output,
                previous,
                True,
                dataset_label=protocol.benchmark_id,
            )
            run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_pre.log")
            stage_rows.append({"task": task, "evaluation": "pre_update", **_native_metric(model, pre_output)})
            if task != first_protocol_task:
                scratch_output = task_root / "scratch_pre"
                command, cwd, env = _baseline_command(
                    model,
                    args,
                    prepared,
                    scratch_output,
                    None,
                    True,
                    dataset_label=protocol.benchmark_id,
                )
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_scratch.log")
                stage_rows.append({"task": task, "evaluation": "scratch_pre", **_native_metric(model, scratch_output)})
            if pre_checkpoint is not None and protocol.adaptation(task) is not None:
                records, status_rows = _run_baseline_adaptation(
                    model,
                    strategy,
                    args,
                    target,
                    data_root,
                    protocol,
                    task,
                    pre_checkpoint,
                )
                adaptation_records.extend(records)
                adaptation_status_rows.extend(status_rows)
            train_tasks = list(protocol.task_ids_between(first_protocol_task, task)) if strategy == "joint" else [task]
            validation_csvs = (
                # The protocol gives every task the same fixed validation
                # sequence count, so this union preserves task-level coverage
                # without consulting oracle law labels.
                [protocol.val_path(validation_task) for validation_task in train_tasks]
                if strategy == "joint"
                else None
            )
            replay_csvs = []
            task_index = protocol.task_ids.index(task)
            if strategy == "replay" and task_index > 0:
                manifest_path = args.hm_resource_root / "resource_manifest.json"
                resources = json.loads(manifest_path.read_text(encoding="utf-8"))
                previous_task = protocol.task_ids[task_index - 1]
                budget = int(resources["stages"][str(previous_task)]["tree_and_episodic_bytes"])
                replay_selection: dict[str, Any] = {}
                replay_path, actual = _make_replay_buffer(
                    data_root,
                    previous_task,
                    budget,
                    target / "prepared" / f"replay_{task:02d}.csv",
                    protocol=protocol,
                    selection_metadata=replay_selection,
                )
                replay_csvs = [replay_path]
                replay_rows.append({
                    "task": task,
                    "budget_bytes": budget,
                    "actual_bytes": actual,
                    "selected_rows": replay_selection.get("selected_rows", 0),
                    "task_bytes": json.dumps(
                        replay_selection.get("task_bytes", {}),
                        sort_keys=True,
                    ),
                })
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                target / "prepared" / f"task_{task:02d}_train",
                train_tasks,
                replay_csvs=replay_csvs,
                protocol=protocol,
                validation_csvs=validation_csvs,
            )
            train_output = task_root / "train"
            command, cwd, env = _baseline_command(
                model,
                args,
                prepared,
                train_output,
                previous if strategy != "joint" else None,
                False,
                dataset_label=protocol.benchmark_id,
            )
            run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_train.log")
            produced = (
                train_output / "best.pt"
                if model == "TPP_LLM"
                else train_output / "checkpoint" / "best.pt"
            )
            checkpoint = target / "checkpoint" / f"task_{task:02d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, checkpoint)
            previous = checkpoint
            stage_rows.append({"task": task, "evaluation": "post_update", **_native_metric(model, train_output)})
            source_predictions = train_output / "predictions.jsonl.gz"
            if source_predictions.exists():
                with gzip.open(source_predictions, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        row["task"] = task
                        row["evaluation"] = "post_update"
                        combined_predictions.write(json.dumps(row) + "\n")
            for anchor_spec in protocol.anchors:
                anchor = anchor_spec.path
                anchor_id = anchor_spec.regime_id
                prepared_anchor = prepare_continual_baseline_dataset(
                    model,
                    data_root,
                    target / "prepared" / f"task_{task:02d}_anchor_{anchor_id}",
                    [task],
                    eval_csv=anchor,
                    protocol=protocol,
                )
                anchor_output = task_root / "anchors" / anchor_id
                command, cwd, env = _baseline_command(
                    model,
                    args,
                    prepared_anchor,
                    anchor_output,
                    checkpoint,
                    True,
                    dataset_label=protocol.benchmark_id,
                )
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_anchor_{anchor_id}.log")
                stage_rows.append({
                    "task": task,
                    "evaluation": "frozen_anchor",
                    "anchor": anchor_id,
                    "evaluation_scope": anchor_spec.evaluation_scope,
                    **_native_metric(model, anchor_output),
                })
    with (target / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in stage_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(stage_rows)
    if replay_rows:
        with (target / "replay_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "task", "budget_bytes", "actual_bytes", "selected_rows",
                    "task_bytes",
                ),
            )
            writer.writeheader()
            writer.writerows(replay_rows)
    if adaptation_status_rows:
        write_csv(
            target / "adaptation_status.csv",
            adaptation_status_rows,
            fieldnames=(
                "protocol", "task_id", "K", "status",
                "observed_events", "train_events",
            ),
        )
    summary = _baseline_cl_summary(stage_rows, protocol, adaptation_records)
    cl_report = summary["cl_metrics"]
    write_csv(target / "frozen_anchor_matrix.csv", cl_report["frozen_anchor_matrix"])
    write_csv(target / "law_metrics.csv", cl_report["law_metrics"])
    write_csv(target / "continual_summary.csv", cl_report["continual_summary"])
    write_csv(target / "task_boundaries.csv", cl_report["task_boundaries"])
    write_csv(target / "fwt_metrics.csv", cl_report["fwt"]["rows"])
    write_csv(
        target / "adaptation_points.csv",
        cl_report["adaptation"]["points"],
        fieldnames=(
            "protocol", "task_id", "K", "pre_nll", "adapted_nll", "gain_nll",
        ),
    )
    write_csv(
        target / "adaptation_summary.csv",
        cl_report["adaptation"]["summary"],
        fieldnames=(
            "protocol", "task_id", "K_min", "K_max", "K_count",
            "adaptation_auc", "status",
        ),
    )
    write_csv(target / "rrr_metrics.csv", cl_report["rrr"]["rows"])
    write_json(target / "cl_metrics.json", cl_report)
    return {
        **summary,
        "benchmark": protocol.benchmark_id,
        "benchmark_version": protocol.version,
        "persistent_regimes": sorted(protocol.persistent_regimes),
        "transient_regimes": sorted(protocol.transient_regimes),
        "stages": stage_rows,
        "replay": replay_rows,
    }


def run_continual_job(*, model: str, strategy: str, args, script: str = "") -> Path:
    spec = JobSpec(dataset="continual", model=model, condition=strategy, kind="continual", script=script)
    data_root = continual_root(args.data_root)
    protocol = continual_protocol(data_root)
    task_start, task_end = protocol.resolve_range(args.task_start, args.task_end)
    task_ids = protocol.task_ids_between(task_start, task_end)
    # Keep the resolved range in the result manifest and in all downstream
    # calls.  The CLI only accepts an optional end; the protocol supplies it.
    args.task_start = task_start
    args.task_end = task_end
    target = result_dir(spec, args)
    csv_inputs = sorted(
        path for path in data_root.rglob("*.csv")
        if not path.name.startswith("._")
    )
    inputs = [protocol.manifest_path] + csv_inputs + [
        data_root / "ground_truth" / "regimes.json",
        data_root / "ground_truth" / "regimes.npz",
        data_root / "ground_truth_manifest.csv",
    ]
    if strategy == "replay" and getattr(args, "hm_resource_root", None):
        inputs.append(args.hm_resource_root / "resource_manifest.json")
    if args.checkpoint is not None:
        inputs.append(args.checkpoint)
    manifest = build_manifest(spec, args, inputs, None)
    if not begin(target, manifest, args.resume):
        print(f"Complete result already exists: {target}")
        return target
    if args.dry_run:
        finish(target, "dry_run")
        print(f"{model} {strategy}: tasks {list(task_ids)}")
        return target
    started = time.perf_counter()
    commands: list[list[str]] = []
    protocol_copy = target / "benchmark_protocol.json"
    protocol_hash_path = target / "benchmark_protocol.sha256"
    protocol_copy_sha256 = ""
    try:
        if protocol_copy.exists() and not protocol_copy.is_file():
            raise RuntimeError(
                f"benchmark protocol output is not a regular file: {protocol_copy}"
            )
        if protocol_copy.is_file():
            if sha256(protocol_copy) != sha256(protocol.manifest_path):
                raise RuntimeError(
                    "existing benchmark_protocol.json differs from the active "
                    "benchmark_manifest.json"
                )
        else:
            shutil.copy2(protocol.manifest_path, protocol_copy)
        protocol_copy_sha256 = sha256(protocol_copy)
        protocol_hash_path.write_text(
            f"{protocol_copy_sha256}  {protocol_copy.name}\n",
            encoding="utf-8",
        )
        (target / "topology_events.jsonl").touch(exist_ok=True)
        if model == "HM":
            previous = args.checkpoint
            start_index = protocol.task_ids.index(task_start)
            if previous is None and start_index > 0:
                previous_task = protocol.task_ids[start_index - 1]
                previous = target / "checkpoint" / f"task_{previous_task:02d}_best.pt"
            if previous is not None and not previous.exists():
                raise FileNotFoundError(f"resume checkpoint required for task-start: {previous}")
            config_path = target / "cl_config.json"
            if config_path.exists():
                cl_config = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(cl_config, dict):
                    raise ValueError(f"invalid continual learner config: {config_path}")
                if cl_config.get("benchmark_id") != protocol.benchmark_id:
                    raise RuntimeError(
                        "existing cl_config.json belongs to a different "
                        "benchmark protocol"
                    )
            else:
                cl_config = _continual_cl_config(args, protocol, strategy)
                write_json(config_path, cl_config)
            stage_manifest_path = target / "stage_manifest.json"
            if stage_manifest_path.exists():
                stage_manifest = json.loads(
                    stage_manifest_path.read_text(encoding="utf-8")
                )
                if not isinstance(stage_manifest, dict):
                    raise ValueError(
                        f"invalid continual stage manifest: {stage_manifest_path}"
                    )
                stage_manifest.setdefault("stages", {})
            else:
                stage_manifest = _continual_stage_manifest(
                    protocol, config_path
                )
                write_json(stage_manifest_path, stage_manifest)
            _validate_continual_stage_manifest(
                stage_manifest, protocol, config_path
            )
            resource_manifest = {"format_version": 1, "measurement": "torch serialized tree_state_dict (semantic tree plus episodic buffers)", "stages": {}}
            existing_resource = target / "resource_manifest.json"
            if existing_resource.exists():
                resource_manifest = json.loads(existing_resource.read_text(encoding="utf-8"))
            for task in task_ids:
                is_initial_task = previous is None
                command, cwd, env = _continual_hm_command(
                    args, target, data_root, task, previous, strategy,
                    protocol=protocol,
                )
                commands.append(command)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}.log")
                if is_initial_task:
                    initial_checkpoint = target / "checkpoint" / f"initial_seed{args.seed}.pt"
                    if not initial_checkpoint.is_file():
                        raise FileNotFoundError(
                            "task 0 did not produce the pre-update FWT checkpoint: "
                            f"{initial_checkpoint}"
                        )
                last_checkpoint = target / "checkpoint" / f"task_{task:02d}_last.pt"
                best_checkpoint = target / "checkpoint" / f"task_{task:02d}_best.pt"
                if not last_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"task {task} did not produce its last checkpoint: "
                        f"{last_checkpoint}"
                    )
                if not best_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"task {task} did not produce its best checkpoint: "
                        f"{best_checkpoint}"
                    )
                stage_manifest["stages"][str(task)] = {
                    "task_id": task,
                    "stage_label": protocol.task(task).stage_label,
                    "shift_type": protocol.task(task).shift_type,
                    "train": _file_record(
                        protocol.train_path(task), role="gradient_update"
                    ),
                    "validation": _file_record(
                        protocol.val_path(task), role="checkpoint_selection"
                    ),
                    "test": _file_record(
                        protocol.split_path(task, "test"),
                        role="final_evaluation_only",
                    ),
                    "previous_checkpoint": (
                        None
                        if previous is None
                        else _file_record(previous, role="resume_input")
                    ),
                    "last_checkpoint": _file_record(
                        last_checkpoint, role="last_state"
                    ),
                    "best_checkpoint": _file_record(
                        best_checkpoint, role="selection_state"
                    ),
                }
                write_json(stage_manifest_path, stage_manifest)
                previous = best_checkpoint
                resource_manifest["stages"][str(task)] = {
                    **_hm_state_bytes(best_checkpoint),
                    "checkpoint": str(best_checkpoint.resolve()),
                    "checkpoint_role": "best",
                }
                write_json(existing_resource, resource_manifest)
            eval_command = [
                args.python_executable or __import__("sys").executable,
                "-m", "EvaluateCL",
                "--data-root", str(data_root),
                "--checkpoint-dir", str(target / "checkpoint"),
                "--output-dir", str(target / "native"),
                "--task-start", str(args.task_start),
                "--task-end", str(args.task_end),
                "--device", resolved_device(args.device),
                "--resume",
                "--eval-batch-size", str(args.eval_batch_size),
            ]
            # Keep the default command free of event-row output.  The legacy
            # boolean remains an explicit alias; the scope option is passed
            # through for supplementary/final and debug runs.
            if getattr(args, "save_event_predictions", False):
                eval_command.append("--save-event-predictions")
            else:
                event_scope = getattr(args, "event_prediction_scope", "none")
                if event_scope != "none":
                    eval_command += ["--event-prediction-scope", event_scope]
            if strategy == "no_working":
                eval_command += ["--variants", "frozen/full"]
            elif strategy == "no_episodic":
                eval_command += [
                    "--variants", "frozen/full", "frozen/no_episodic"
                ]
            initial_checkpoint = target / "checkpoint" / f"initial_seed{args.seed}.pt"
            if initial_checkpoint.is_file():
                eval_command += [
                    "--fwt-scratch-checkpoint",
                    str(initial_checkpoint),
                ]
            if args.smoke:
                eval_command += ["--max-sequences", "1", "--no-summary-plots", "--no-hawkes-law-evaluation"]
            commands.append(eval_command)
            run_command(eval_command, cwd, env, target / "logs" / "evaluate.log")
            summary_path = target / "native" / "summary.json"
            metrics = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            native = target / "native"
            # Keep the runner-level CL artifact contract identical to the
            # baseline path. EvaluateCL keeps its richer native tree under
            # ``native/`` and calls the frozen checkpoint × regime matrix
            # ``anchor_nll_matrix.csv``; the outer contract exposes that same
            # matrix as ``frozen_anchor_matrix.csv``.
            hm_artifacts = {
                "task_metrics.csv": "task_metrics.csv",
                "anchor_metrics.csv": "anchor_metrics.csv",
                "continual_summary.csv": "continual_summary.csv",
                "anchor_nll_matrix.csv": "frozen_anchor_matrix.csv",
                "law_metrics.csv": "law_metrics.csv",
                "fwt_metrics.csv": "fwt_metrics.csv",
                "rrr_metrics.csv": "rrr_metrics.csv",
                "checkpoint_tree.csv": "checkpoint_tree.csv",
            }
            for source_name, target_name in hm_artifacts.items():
                source = native / source_name
                if source.is_file():
                    shutil.copy2(source, target / target_name)
            task_metrics = native / "task_metrics.csv"
            if task_metrics.is_file():
                shutil.copy2(task_metrics, target / "sequence_metrics.csv")
            predictions = target / "native" / "event_predictions.csv"
            if predictions.exists() and getattr(args, "event_prediction_scope", "none") != "none":
                shutil.copy2(predictions, target / "event_predictions.csv")
                with predictions.open("r", newline="", encoding="utf-8-sig") as source, gzip.open(target / "predictions.jsonl.gz", "wt", encoding="utf-8") as destination:
                    for row in csv.DictReader(source):
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                with gzip.open(target / "predictions.jsonl.gz", "wt", encoding="utf-8"):
                    pass
        else:
            metrics = _run_baseline_continual(
                model, strategy, args, target, data_root, protocol
            )
        metrics["benchmark_protocol"] = str(protocol_copy.resolve())
        metrics["benchmark_protocol_sha256"] = protocol_copy_sha256
        write_json(target / "metrics.json", metrics)
        resource_command = [token for cmd in commands for token in cmd]
        if not resource_command:
            resource_command = [model, strategy, args.device]
        write_json(target / "resources.json", resource_record(target, started, resource_command))
        write_report(target / "report.md", spec, {"tasks": len(task_ids)}, commands[-1] if commands else [model, strategy])
        (target / "plots").mkdir(exist_ok=True)
        (target / "tree_events.jsonl").touch(exist_ok=True)
        finish(target, "complete")
    except Exception as error:
        finish(target, "failed", f"{error}\n{traceback.format_exc()}")
        raise
    print(f"Result directory: {target}")
    return target


def run_diagnostic_job(*, kind: str, args, script: str = "") -> Path:
    if kind in {"law_recovery", "frontier", "residual_rank"}:
        condition = kind if kind != "residual_rank" else f"residual_rank_{args.rank}"
        target = run_stationary_job(dataset="dws", model="HM", args=args, condition=condition, script=script)
        if kind == "frontier" and not args.dry_run:
            checkpoint = target / "checkpoint" / "best.pt"
            if not checkpoint.exists():
                checkpoint = target / "checkpoint" / "model.pt"
            memory = MODELS_ROOT / "HawkesMemory" / "Memory"
            _, _, base_env = stationary_command(
                JobSpec(dataset="dws", model="HM", condition=condition),
                args, target,
            )
            all_leaf = target / "native_all_leaf"
            command = [args.python_executable or __import__("sys").executable, "-m", "Evaluate", "--checkpoint", str(checkpoint), "--data-path", str(target / "prepared" / "canonical.csv"), "--split-manifest", str(target / "prepared" / "split_manifest.json"), "--output-dir", str(all_leaf), "--protocol", "frozen", "--variants", "frozen/full", "--seed", str(args.seed), "--device", resolved_device(args.device), "--save-event-predictions"]
            if args.smoke:
                command += ["--max-test-sequences", "2", "--bootstrap-samples", "50", "--no-plots"]
            all_leaf_env = dict(base_env)
            all_leaf_env["HM_EVAL_ALL_LEAF"] = "1"
            run_command(command, memory, all_leaf_env, target / "logs" / "all_leaf.log")
            active_summary = json.loads((target / "native" / "summary.json").read_text(encoding="utf-8"))
            exhaustive_summary = json.loads((all_leaf / "summary.json").read_text(encoding="utf-8"))
            active = active_summary["variants"].get("frozen/full", active_summary["variants"].get("full_frozen", {}))
            exhaustive = exhaustive_summary["variants"].get("frozen/full", exhaustive_summary["variants"].get("full_frozen", {}))
            comparison = {
                **{key: active.get(key) for key in ("nll_per_event", "accuracy", "macro_f1", "time_mae", "time_rmse", "num_events")},
                "checkpoint": str(checkpoint), "retrained": False,
                "active_frontier": active, "all_leaf": exhaustive,
            }
            write_json(target / "frontier_comparison.json", comparison)
            write_json(target / "metrics.json", comparison)
            write_report(target / "report.md", JobSpec(dataset="dws", model="HM", condition=condition), {"retrained": False}, command)
        if kind == "law_recovery" and not args.dry_run:
            checkpoint = target / "checkpoint" / "best.pt"
            if not checkpoint.exists():
                checkpoint = target / "checkpoint" / "model.pt"
            _, _, env = stationary_command(
                JobSpec(dataset="dws", model="HM", condition=condition),
                args, target,
            )
            output = target / "law_recovery.json"
            command = [args.python_executable or __import__("sys").executable, str(EVALUATION_ROOT / "core" / "law_recovery.py"), "--checkpoint", str(checkpoint), "--ground-truth", str(DATASETS_ROOT / "Data" / f"tree_{args.variant}" / f"parameters_{args.variant}.json"), "--output", str(output), "--device", resolved_device(args.device)]
            run_command(command, MODELS_ROOT / "HawkesMemory" / "Memory", env, target / "logs" / "law_recovery.log")
            recovery = json.loads(output.read_text(encoding="utf-8"))
            summary = json.loads((target / "native" / "summary.json").read_text(encoding="utf-8"))
            frozen_metrics = summary.get("variants", {}).get("frozen/full", summary.get("variants", {}).get("full_frozen", {}))
            recovery.update({key: frozen_metrics.get(key) for key in ("nll_per_event", "accuracy", "macro_f1", "time_mae", "time_rmse", "num_events")})
            recovery["routing_recovery"] = summary.get("routing", summary.get("tree", {}))
            write_json(target / "metrics.json", recovery)
            write_report(target / "report.md", JobSpec(dataset="dws", model="HM", condition=condition), recovery, command)
        return target
    return run_continual_job(model="HM", strategy=kind, args=args, script=script)
