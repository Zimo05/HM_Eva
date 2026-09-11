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
    prepare_hm_dataset,
)
from .io import sha256, write_csv, write_json
from .manifest import begin, build_manifest, finish
from .cl_metrics import (
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
    command, cwd, env = stationary_command(spec, args, target)
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
        if model == "HM":
            prepare_hm_dataset(
                spec.dataset,
                args.seed,
                target / "prepared",
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
    train_path = protocol.train_path(task)
    validation_path = protocol.val_path(task)
    if train_path.name != "train.csv":
        raise ValueError(
            f"continual protocol train split must be train.csv: {train_path}"
        )
    if validation_path.name not in {"val.csv", "validation.csv"}:
        raise ValueError(
            "continual protocol validation split must be val.csv or "
            f"validation.csv: {validation_path}"
        )
    cold_start_epochs = 0 if previous is not None else (1 if args.smoke else 5)
    topology_events_path = target / "topology_events.jsonl"
    command = [
        args.python_executable or sys.executable,
        "-m",
        "Train.Train",
        "--data-path",
        str(train_path),
        "--validation-data-path",
        str(validation_path),
        "--checkpoint",
        str(last_checkpoint),
        "--best-checkpoint",
        str(best_checkpoint),
        "--cl-config",
        str(target / "cl_config.json"),
        "--benchmark-manifest",
        str(protocol.manifest_path),
        "--cl-task-id",
        str(task),
        "--tree-init-depth",
        "0",
        "--seed",
        str(args.seed),
        "--device",
        resolved_device(args.device),
        "--epochs",
        str(args.epochs or (1 if args.smoke else 20)),
        "--cold-start-epochs",
        str(cold_start_epochs),
        "--unified-topology-log-path",
        str(topology_events_path),
    ]
    if previous is None:
        command += [
            "--initial-checkpoint-output",
            str(target / "checkpoint" / f"initial_seed{args.seed}.pt"),
        ]
    if strategy in {"no_working", "no_episodic", "fixed_topology", "no_sleep", "heuristic_controller", "no_merge_prune"}:
        command += ["--evaluation-ablation", strategy]
    if previous is not None:
        command += [
            "--resume",
            str(previous),
            "--cl-previous-checkpoint",
            str(previous),
        ]
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
        "epochs_per_task": int(args.epochs or (1 if args.smoke else 20)),
        "cold_start_epochs": int(1 if args.smoke else 5),
        "training": {
            "seed": int(args.seed),
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
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
                      initial: Path | None, evaluate_only: bool):
    import os
    import sys
    python = args.python_executable or sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(__import__("pathlib").Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")))
    epochs = args.epochs or (1 if args.smoke else (1 if model == "TPP_LLM" else 20))
    batch = args.batch_size or (2 if args.smoke else 32)
    device = resolved_device(args.device)
    if model == "RMTPP":
        command = [python, str(MODELS_ROOT / "RMTPP" / "run_experiment.py"), "--dataset", "taobao", "--prepared-data-dir", str(prepared), "--output-dir", str(output), "--archive", str(output) + ".tar.gz", "--overwrite", "--seed", str(args.seed), "--gpu", ("-1" if device == "cpu" else device.split(":")[-1]), "--epochs", str(epochs), "--batch-size", str(batch)]
        cwd = MODELS_ROOT / "RMTPP"
        if initial:
            command += ["--initial-checkpoint", str(initial)]
        if evaluate_only:
            command += ["--evaluate-only"]
    elif model == "THP":
        command = [python, str(MODELS_ROOT / "THP" / "run_experiment.py"), "--dataset", "taobao", "--prepared-data-dir", str(prepared), "--output-dir", str(output), "--archive", str(output) + ".tar.gz", "--overwrite", "--seed", str(args.seed), "--device", device, "--epochs", str(epochs), "--batch-size", str(batch)]
        cwd = MODELS_ROOT / "THP"
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


def _make_replay_buffer(
    data_root: Path,
    through_task: int,
    budget: int,
    output: Path,
    protocol: CLProtocol | None = None,
) -> tuple[Path, int]:
    # Keep the small legacy helper usable for callers that construct an
    # isolated ``task_XX/train.csv`` fixture.  The production continual runner
    # always passes the loaded manifest protocol.
    legacy_layout = protocol is None and not (
        data_root / "benchmark_manifest.json"
    ).is_file()
    if protocol is None and not legacy_layout:
        protocol = continual_protocol(data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    used = len("event_times,event_types\r\n".encode("utf-8"))
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
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                encoded = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
                if used + len(encoded) > budget:
                    continue
                rows.append(row)
                used += len(encoded)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
        writer.writeheader()
        writer.writerows(rows)
    actual = output.stat().st_size
    if actual > budget:
        raise RuntimeError(f"replay serialization {actual} exceeds HM budget {budget}")
    return output, actual


def _baseline_cl_summary(rows: list[dict], protocol: CLProtocol) -> dict:
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
    replay_rows = []
    prediction_target = target / "predictions.jsonl.gz"
    with gzip.open(prediction_target, "wt", encoding="utf-8") as combined_predictions:
        for task in task_ids:
            task_root = native / f"task_{task:02d}"
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                target / "prepared" / f"task_{task:02d}_pre",
                [task],
                protocol=protocol,
            )
            pre_output = task_root / "pre_update"
            command, cwd, env = _baseline_command(model, args, prepared, pre_output, previous, True)
            run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_pre.log")
            stage_rows.append({"task": task, "evaluation": "pre_update", **_native_metric(model, pre_output)})
            if task != first_protocol_task:
                scratch_output = task_root / "scratch_pre"
                command, cwd, env = _baseline_command(model, args, prepared, scratch_output, None, True)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_scratch.log")
                stage_rows.append({"task": task, "evaluation": "scratch_pre", **_native_metric(model, scratch_output)})
            train_tasks = list(protocol.task_ids_between(first_protocol_task, task)) if strategy == "joint" else [task]
            replay_csvs = []
            task_index = protocol.task_ids.index(task)
            if strategy == "replay" and task_index > 0:
                manifest_path = args.hm_resource_root / "resource_manifest.json"
                resources = json.loads(manifest_path.read_text(encoding="utf-8"))
                previous_task = protocol.task_ids[task_index - 1]
                budget = int(resources["stages"][str(previous_task)]["tree_and_episodic_bytes"])
                replay_path, actual = _make_replay_buffer(
                    data_root,
                    previous_task,
                    budget,
                    target / "prepared" / f"replay_{task:02d}.csv",
                    protocol=protocol,
                )
                replay_csvs = [replay_path]
                replay_rows.append({"task": task, "budget_bytes": budget, "actual_bytes": actual})
            prepared = prepare_continual_baseline_dataset(
                model,
                data_root,
                target / "prepared" / f"task_{task:02d}_train",
                train_tasks,
                replay_csvs=replay_csvs,
                protocol=protocol,
            )
            train_output = task_root / "train"
            command, cwd, env = _baseline_command(model, args, prepared, train_output, previous if strategy != "joint" else None, False)
            run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_train.log")
            produced = train_output / ("best.pt" if model == "TPP_LLM" else "checkpoint" / "best.pt")
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
                command, cwd, env = _baseline_command(model, args, prepared_anchor, anchor_output, checkpoint, True)
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
            writer = csv.DictWriter(handle, fieldnames=("task", "budget_bytes", "actual_bytes")); writer.writeheader(); writer.writerows(replay_rows)
    summary = _baseline_cl_summary(stage_rows, protocol)
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
                command, cwd, env = _continual_hm_command(
                    args, target, data_root, task, previous, strategy,
                    protocol=protocol,
                )
                commands.append(command)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}.log")
                if previous is None:
                    initial_checkpoint = target / "checkpoint" / f"initial_seed{args.seed}.pt"
                    if not initial_checkpoint.is_file():
                        raise FileNotFoundError(
                            f"fresh CL stage did not produce its initial scratch checkpoint: "
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
            eval_command = [args.python_executable or __import__("sys").executable, "-m", "EvaluateCL", "--data-root", str(data_root), "--checkpoint-dir", str(target / "checkpoint"), "--output-dir", str(target / "native"), "--task-start", str(args.task_start), "--task-end", str(args.task_end), "--device", resolved_device(args.device), "--resume", "--save-event-predictions"]
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
            for name in ("task_metrics.csv", "anchor_metrics.csv", "continual_summary.csv", "checkpoint_tree.csv"):
                source = target / "native" / name
                if source.exists() and name == "task_metrics.csv":
                    shutil.copy2(source, target / "sequence_metrics.csv")
            predictions = target / "native" / "event_predictions.csv"
            if predictions.exists():
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
