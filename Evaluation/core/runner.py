from __future__ import annotations

import json
import csv
import gzip
import io
import shutil
import time
import traceback
from pathlib import Path

from .adapters import copy_checkpoint_contract, copy_prediction_contract, evaluate_hm, normalize_native_metrics, resolved_device, run_command, stationary_command
from .data import continual_root, dataset_inputs, prepare_continual_baseline_dataset, prepare_hm_dataset
from .io import write_csv, write_json
from .manifest import begin, build_manifest, finish
from .metrics import adaptation_auc, continual_metrics
from .paths import DATASETS_ROOT, EVALUATION_ROOT, MODELS_ROOT, result_dir
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


def _continual_hm_command(args, target: Path, data_root: Path, task: int, previous: Path | None, strategy: str) -> tuple[list[str], Path, dict[str, str]]:
    import os
    import sys
    memory = MODELS_ROOT / "HawkesMemory" / "Memory"
    checkpoint = target / "checkpoint" / f"task_{task:02d}.pt"
    command = [args.python_executable or sys.executable, "-m", "Train.Train", "--data-path", str(data_root / f"task_{task:02d}" / "train.csv"), "--checkpoint", str(checkpoint), "--best-checkpoint", str(checkpoint), "--tree-init-depth", "0", "--seed", str(args.seed), "--device", resolved_device(args.device), "--epochs", str(args.epochs or (1 if args.smoke else 20)), "--cold-start-epochs", str(1 if args.smoke else 5)]
    if strategy in {"no_working", "no_episodic", "fixed_topology", "no_sleep", "heuristic_controller", "no_merge_prune"}:
        command += ["--evaluation-ablation", strategy]
    if previous is not None:
        command += ["--resume", str(previous), "--cold-start-epochs", "0"]
    if args.smoke:
        command += ["--max-sequences", "2", "--max-events-per-sequence", "12", "--no-training-plots"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(MODELS_ROOT / "HawkesMemory"), str(memory), env.get("PYTHONPATH", "")))
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
    return {"nll_per_event": -ll, "accuracy": float(values["accuracy"]), "time_rmse": float(values["rmse"])}


def _make_replay_buffer(data_root: Path, through_task: int, budget: int,
                        output: Path) -> tuple[Path, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    used = len("event_times,event_types\r\n".encode("utf-8"))
    for task in range(through_task + 1):
        source = data_root / f"task_{task:02d}" / "train.csv"
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


def _baseline_cl_summary(rows: list[dict]) -> dict:
    first_seen = {
        "A_1": 0, "B_1": 1, "C_1": 2, "B_prime_1": 4,
        "A_2": 5, "A_merge": 7, "E_1": 9,
    }
    matrix: dict[int, dict[str, float]] = {}
    for row in rows:
        if row["evaluation"] == "frozen_anchor" and row.get("anchor") in first_seen:
            matrix.setdefault(int(row["task"]), {})[row["anchor"]] = float(row["nll_per_event"])
    cl_rows = continual_metrics(matrix, first_seen) if matrix else []
    task_rows = {(int(row["task"]), row["evaluation"]): row for row in rows if row["evaluation"] != "frozen_anchor"}
    transfer = []
    for task in sorted({key[0] for key in task_rows}):
        pre = task_rows.get((task, "pre_update"))
        post = task_rows.get((task, "post_update"))
        scratch = task_rows.get((task, "scratch_pre"))
        transfer.append({
            "task": task,
            "adaptation_gain_nll": (float(pre["nll_per_event"]) - float(post["nll_per_event"])) if pre and post else None,
            "forward_transfer_nll": (float(scratch["nll_per_event"]) - float(pre["nll_per_event"])) if scratch and pre else None,
        })
    gains = {row["task"]: row["adaptation_gain_nll"] for row in transfer if row["adaptation_gain_nll"] is not None}
    first_pre = task_rows.get((0, "pre_update"))
    first_post = task_rows.get((0, "post_update"))
    return_pre = task_rows.get((8, "pre_update"))
    rrr = None
    if first_pre and first_post and return_pre:
        denominator = float(first_pre["nll_per_event"]) - float(first_post["nll_per_event"])
        if abs(denominator) > 1e-8:
            rrr = (float(first_pre["nll_per_event"]) - float(return_pre["nll_per_event"])) / denominator
    final_post = task_rows.get((max((key[0] for key in task_rows), default=0), "post_update"), {})
    return {
        "nll_per_event": final_post.get("nll_per_event"),
        "accuracy": final_post.get("accuracy"),
        "time_rmse": final_post.get("time_rmse"),
        "continual_summary": cl_rows,
        "transfer": transfer,
        "adaptation_auc": adaptation_auc(gains) if len(gains) >= 2 else None,
        "recurrence_retention_ratio_A": rrr,
        "tsr": None,
        "tsr_note": "TSR is topology-specific and is not defined for flat baselines.",
    }


def _run_baseline_continual(model: str, strategy: str, args, target: Path,
                            data_root: Path):
    native = target / "native"
    previous = args.checkpoint
    if previous is None and args.task_start:
        previous = target / "checkpoint" / f"task_{args.task_start - 1:02d}.pt"
    if previous is not None and not previous.is_file():
        raise FileNotFoundError(f"resume checkpoint required for task-start: {previous}")
    stage_rows = []
    replay_rows = []
    prediction_target = target / "predictions.jsonl.gz"
    with gzip.open(prediction_target, "wt", encoding="utf-8") as combined_predictions:
        for task in range(args.task_start, args.task_end + 1):
            task_root = native / f"task_{task:02d}"
            prepared = prepare_continual_baseline_dataset(model, data_root, target / "prepared" / f"task_{task:02d}_pre", [task])
            pre_output = task_root / "pre_update"
            command, cwd, env = _baseline_command(model, args, prepared, pre_output, previous, True)
            run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_pre.log")
            stage_rows.append({"task": task, "evaluation": "pre_update", **_native_metric(model, pre_output)})
            if task > 0:
                scratch_output = task_root / "scratch_pre"
                command, cwd, env = _baseline_command(model, args, prepared, scratch_output, None, True)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_scratch.log")
                stage_rows.append({"task": task, "evaluation": "scratch_pre", **_native_metric(model, scratch_output)})
            train_tasks = list(range(0, task + 1)) if strategy == "joint" else [task]
            replay_csvs = []
            if strategy == "replay" and task:
                manifest_path = args.hm_resource_root / "resource_manifest.json"
                resources = json.loads(manifest_path.read_text(encoding="utf-8"))
                budget = int(resources["stages"][str(task - 1)]["tree_and_episodic_bytes"])
                replay_path, actual = _make_replay_buffer(data_root, task - 1, budget, target / "prepared" / f"replay_{task:02d}.csv")
                replay_csvs = [replay_path]
                replay_rows.append({"task": task, "budget_bytes": budget, "actual_bytes": actual})
            prepared = prepare_continual_baseline_dataset(model, data_root, target / "prepared" / f"task_{task:02d}_train", train_tasks, replay_csvs=replay_csvs)
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
            for anchor in sorted((data_root / "anchors").glob("*.csv")):
                prepared_anchor = prepare_continual_baseline_dataset(model, data_root, target / "prepared" / f"task_{task:02d}_anchor_{anchor.stem}", [task], eval_csv=anchor)
                anchor_output = task_root / "anchors" / anchor.stem
                command, cwd, env = _baseline_command(model, args, prepared_anchor, anchor_output, checkpoint, True)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}_anchor_{anchor.stem}.log")
                stage_rows.append({"task": task, "evaluation": "frozen_anchor", "anchor": anchor.stem, **_native_metric(model, anchor_output)})
    with (target / "sequence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in stage_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(stage_rows)
    if replay_rows:
        with (target / "replay_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("task", "budget_bytes", "actual_bytes")); writer.writeheader(); writer.writerows(replay_rows)
    return {**_baseline_cl_summary(stage_rows), "stages": stage_rows, "replay": replay_rows}


def run_continual_job(*, model: str, strategy: str, args, script: str = "") -> Path:
    spec = JobSpec(dataset="continual", model=model, condition=strategy, kind="continual", script=script)
    target = result_dir(spec, args)
    data_root = continual_root(args.data_root)
    inputs = sorted(data_root.rglob("*.csv")) + [
        data_root / "ground_truth" / "regimes.json"
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
        print(f"{model} {strategy}: tasks {args.task_start}..{args.task_end}")
        return target
    started = time.perf_counter()
    commands: list[list[str]] = []
    try:
        if model == "HM":
            previous = args.checkpoint
            if previous is None and args.task_start:
                previous = target / "checkpoint" / f"task_{args.task_start - 1:02d}.pt"
            if previous is not None and not previous.exists():
                raise FileNotFoundError(f"resume checkpoint required for task-start: {previous}")
            resource_manifest = {"format_version": 1, "measurement": "torch serialized tree_state_dict (semantic tree plus episodic buffers)", "stages": {}}
            existing_resource = target / "resource_manifest.json"
            if existing_resource.exists():
                resource_manifest = json.loads(existing_resource.read_text(encoding="utf-8"))
            for task in range(args.task_start, args.task_end + 1):
                command, cwd, env = _continual_hm_command(args, target, data_root, task, previous, strategy)
                commands.append(command)
                run_command(command, cwd, env, target / "logs" / f"task_{task:02d}.log")
                previous = target / "checkpoint" / f"task_{task:02d}.pt"
                resource_manifest["stages"][str(task)] = _hm_state_bytes(previous)
                write_json(existing_resource, resource_manifest)
            eval_command = [args.python_executable or __import__("sys").executable, "-m", "EvaluateCL", "--data-root", str(data_root), "--checkpoint-dir", str(target / "checkpoint"), "--output-dir", str(target / "native"), "--task-start", str(args.task_start), "--task-end", str(args.task_end), "--device", resolved_device(args.device), "--resume", "--save-event-predictions"]
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
            metrics = _run_baseline_continual(model, strategy, args, target, data_root)
        write_json(target / "metrics.json", metrics)
        resource_command = [token for cmd in commands for token in cmd]
        if not resource_command:
            resource_command = [model, strategy, args.device]
        write_json(target / "resources.json", resource_record(target, started, resource_command))
        write_report(target / "report.md", spec, {"tasks": args.task_end - args.task_start + 1}, commands[-1] if commands else [model, strategy])
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
            command = [args.python_executable or __import__("sys").executable, "-m", "Evaluate", "--checkpoint", str(checkpoint), "--data-path", str(target / "prepared" / "canonical.csv"), "--split-manifest", str(target / "prepared" / "split_manifest.json"), "--output-dir", str(all_leaf), "--protocol", "frozen", "--variants", "full_frozen", "--seed", str(args.seed), "--device", resolved_device(args.device), "--save-event-predictions"]
            if args.smoke:
                command += ["--max-test-sequences", "2", "--bootstrap-samples", "50", "--no-plots"]
            all_leaf_env = dict(base_env)
            all_leaf_env["HM_EVAL_ALL_LEAF"] = "1"
            run_command(command, memory, all_leaf_env, target / "logs" / "all_leaf.log")
            active = json.loads((target / "native" / "summary.json").read_text(encoding="utf-8"))["variants"]["full_frozen"]
            exhaustive = json.loads((all_leaf / "summary.json").read_text(encoding="utf-8"))["variants"]["full_frozen"]
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
            recovery.update({key: summary.get("variants", {}).get("full_frozen", {}).get(key) for key in ("nll_per_event", "accuracy", "macro_f1", "time_mae", "time_rmse", "num_events")})
            recovery["routing_recovery"] = summary.get("routing", summary.get("tree", {}))
            write_json(target / "metrics.json", recovery)
            write_report(target / "report.md", JobSpec(dataset="dws", model="HM", condition=condition), recovery, command)
        return target
    return run_continual_job(model="HM", strategy=kind, args=args, script=script)
