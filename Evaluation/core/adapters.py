from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import write_json
from .paths import MODELS_ROOT, PROJECT_ROOT


def python_for(args) -> str:
    return args.python_executable or sys.executable


def device_index(device: str) -> str:
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu":
        return "-1"
    return device.split(":", 1)[1] if ":" in device else "0"


def resolved_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def stationary_command(
    spec,
    args,
    result_dir: Path,
    prepared: Path | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    native = result_dir / "native"
    python = python_for(args)
    epochs = args.epochs or (1 if args.smoke else None)
    batch = args.batch_size or (4 if args.smoke else None)
    device = resolved_device(args.device)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), env.get("PYTHONPATH", "")))
    if spec.model in {"RMTPP", "FullyNN"}:
        script = MODELS_ROOT / spec.model / "run_experiment.py"
        command = [python, str(script), "--dataset", spec.dataset, "--seed", str(args.seed), "--output-dir", str(native), "--archive", str(result_dir / "native.tar.gz"), "--overwrite", "--gpu", device_index(args.device)]
        if spec.dataset == "dws":
            command += ["--variant", args.variant]
        if prepared is not None:
            command += ["--prepared-data-dir", str(prepared)]
        if epochs:
            command += ["--epochs", str(epochs)]
        if batch:
            command += ["--batch-size", str(batch)]
        return command, MODELS_ROOT / spec.model, env
    if spec.model == "THP":
        script = MODELS_ROOT / "THP" / "run_experiment.py"
        dataset = f"dws_{args.variant}" if spec.dataset == "dws" else spec.dataset
        command = [python, str(script), "--dataset", dataset, "--seed", str(args.seed), "--device", device, "--output-dir", str(native), "--archive", str(result_dir / "native.tar.gz"), "--overwrite"]
        if prepared is not None:
            command += ["--prepared-data-dir", str(prepared)]
        if epochs:
            command += ["--epochs", str(epochs)]
        if batch:
            command += ["--batch-size", str(batch)]
        return command, MODELS_ROOT / "THP", env
    if spec.model == "TPP_LLM":
        script = MODELS_ROOT / "TPP-LLM" / "scripts" / "train_tpp_llm.py"
        data_dir = prepared or (result_dir / "prepared" / "tpp_llm")
        command = [python, str(script), "--evaluation_dataset", spec.dataset, "--evaluation_output", str(native), "--data_path", str(data_dir), "--device", device, "--seed", str(args.seed), "--peft_type", "lora", "--lora_rank", "16", "--anonymous_labels"]
        if prepared is not None:
            command += ["--prepared_data"]
        if spec.dataset == "dws":
            command += ["--evaluation_variant", args.variant]
        if epochs:
            command += ["--num_train_epochs", str(epochs)]
        if batch:
            command += ["--train_batch_size", str(batch), "--eval_batch_size", str(batch)]
        return command, MODELS_ROOT / "TPP-LLM", env
    if spec.model == "HM":
        prepared = prepared or (result_dir / "prepared")
        # Dataset preparation belongs to the runner, after the result manifest
        # has been accepted.  Keeping command construction side-effect free is
        # important for --dry-run and for clean failure reporting.
        data_path = prepared / "canonical.csv"
        split_manifest = prepared / "split_manifest.json"
        memory = MODELS_ROOT / "HawkesMemory" / "Memory"
        checkpoint = result_dir / "checkpoint" / "model.pt"
        best = result_dir / "checkpoint" / "best.pt"
        command = [python, "-m", "Train.Train", "--data-path", str(data_path), "--split-manifest", str(split_manifest), "--split", "train", "--tree-init-depth", "0", "--checkpoint", str(checkpoint), "--best-checkpoint", str(best), "--seed", str(args.seed), "--device", device]
        if spec.condition in {
            "no_working",
            "no_episodic",
            "fixed_topology",
            "no_sleep",
            "heuristic_controller",
            "no_merge_prune",
        }:
            command += ["--evaluation-ablation", spec.condition]
        if getattr(args, "rank", None) is not None:
            rank = "8" if args.rank == "D" else args.rank
            command += ["--residual-init-rank", str(rank)]
        command += ["--epochs", str(epochs or 20), "--cold-start-epochs", str(1 if args.smoke else 5)]
        if args.smoke:
            command += ["--max-sequences", "4", "--max-events-per-sequence", "16", "--no-training-plots"]
        env["PYTHONPATH"] = os.pathsep.join((str(MODELS_ROOT / "HawkesMemory"), str(memory), env.get("PYTHONPATH", "")))
        return command, memory, env
    raise KeyError(f"unsupported model: {spec.model}")


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")


def _find_test_csv(native: Path) -> Path:
    candidates = sorted(native.rglob("*test*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no test metric CSV below {native}")
    return candidates[-1]


def normalize_native_metrics(spec, result_dir: Path) -> dict[str, Any]:
    native = result_dir / "native"
    if spec.model == "HM":
        summary = native / "summary.json"
        if not summary.exists():
            return {"state": "trained", "evaluation_pending": True}
        payload = json.loads(summary.read_text(encoding="utf-8"))
        return payload.get("variants", {}).get(
            "frozen/full",
            payload.get("variants", {}).get("full_frozen", payload),
        )
    if spec.model == "TPP_LLM":
        path = native / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))
    path = _find_test_csv(native)
    with path.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    lowered = {key.lower().strip(): value for key, value in row.items()}
    def value(*names: str) -> float | None:
        for name in names:
            if name in lowered and lowered[name] not in {"", "nan", None}:
                return float(lowered[name])
        return None
    loglike = value("log-likelihood", "likelihood", "log_likelihood")
    metrics = {"nll_per_event": -loglike if loglike is not None else None, "accuracy": value("accuracy"), "time_rmse": value("rmse"), "source": str(path)}
    predictions = next(iter(sorted(native.rglob("predictions.jsonl.gz"))), None)
    if predictions is not None:
        rows = []
        with gzip.open(predictions, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows:
            labels = sorted({int(row["true_type"]) for row in rows} | {int(row["predicted_type"]) for row in rows})
            f1s = []
            for label in labels:
                tp = sum(int(row["true_type"]) == label == int(row["predicted_type"]) for row in rows)
                fp = sum(int(row["true_type"]) != label and int(row["predicted_type"]) == label for row in rows)
                fn = sum(int(row["true_type"]) == label and int(row["predicted_type"]) != label for row in rows)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            errors = [float(row["predicted_delta_time"]) - float(row["true_delta_time"]) for row in rows]
            metrics.update({
                "macro_f1": sum(f1s) / len(f1s),
                "time_mae": sum(abs(error) for error in errors) / len(errors),
                "num_events": len(rows),
            })
    return metrics


def evaluate_hm(spec, args, result_dir: Path, env: dict[str, str]) -> None:
    prepared = result_dir / "prepared"
    data_path = prepared / "canonical.csv"
    split_manifest = prepared / "split_manifest.json"
    checkpoint = result_dir / "checkpoint" / "best.pt"
    if not checkpoint.exists():
        checkpoint = result_dir / "checkpoint" / "model.pt"
    memory = MODELS_ROOT / "HawkesMemory" / "Memory"
    variants = {
        "no_working": "frozen/full",
        "no_episodic": "frozen/no_episodic",
    }
    command = [python_for(args), "-m", "Evaluate", "--checkpoint", str(checkpoint), "--data-path", str(data_path), "--split-manifest", str(split_manifest), "--output-dir", str(result_dir / "native"), "--protocol", "both", "--seed", str(args.seed), "--device", resolved_device(args.device), "--resume", "--save-event-predictions"]
    if spec.condition in variants:
        command += ["--variants", variants[spec.condition]]
    if args.smoke:
        command += ["--max-test-sequences", "2", "--bootstrap-samples", "50", "--no-plots"]
    run_command(command, memory, env, result_dir / "logs" / "evaluate.log")


def copy_checkpoint_contract(result_dir: Path) -> None:
    checkpoint_dir = result_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_dir.iterdir()):
        return
    candidates = sorted((result_dir / "native").rglob("*best.pt"))
    if candidates:
        shutil.copy2(candidates[-1], checkpoint_dir / "best.pt")


def copy_prediction_contract(result_dir: Path) -> None:
    target = result_dir / "predictions.jsonl.gz"
    if target.exists():
        return
    candidates = sorted((result_dir / "native").rglob("predictions.jsonl.gz"))
    if candidates:
        shutil.copy2(candidates[-1], target)
        return
    hm_csv = result_dir / "native" / "event_predictions.csv"
    if hm_csv.exists():
        previous_time: dict[tuple[str, int], float] = {}
        with hm_csv.open("r", newline="", encoding="utf-8") as source, gzip.open(target, "wt", encoding="utf-8") as output:
            for row in csv.DictReader(source):
                key = (row["variant"], int(row["source_index"]))
                true_time = float(row["true_time"])
                true_delta = true_time - previous_time.get(key, 0.0)
                previous_time[key] = true_time
                output.write(json.dumps({
                    "sequence_id": int(row["source_index"]),
                    "event_index": int(row["event_index"]),
                    "true_type": int(row["true_type"]),
                    "predicted_type": int(row["predicted_type_at_event_time"]),
                    "type_probabilities": json.loads(row["type_probabilities"]),
                    "true_delta_time": true_delta,
                    "predicted_delta_time": float(row["predicted_delta"]),
                    "event_nll": float(row["nll"]),
                    "variant": row["variant"],
                }) + "\n")
