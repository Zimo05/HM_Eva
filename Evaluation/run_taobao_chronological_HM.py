"""Train HM over real Taobao windows in absolute chronological order.

This runner is deliberately independent of the synthetic ``CLProtocol``
machinery. Each UTC window gets one ordinary ``Train.py`` invocation; later
windows resume the immediately preceding window's best-validation checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = ROOT / "Models" / "HawkesMemory" / "Memory"
TRAIN_ENTRY = MEMORY_ROOT / "Train" / "Train.py"
DEFAULT_TAOBAO_CHRONOLOGICAL_CONFIG = {
    "format_version": 1,
    "benchmark_id": "taobao_real_chronological",
    "epochs_per_task": 60,
    "cold_start_epochs": 15,
    "training": {
        "learning_rate": 0.0007,
        "weight_decay": 0.00001,
        "grad_clip": 3.5,
        "optimizer_impl": "auto",
        "sleep_every": 1,
        "evaluation_ablation": "full",
    },
    "model": {
        "z_dim": 50,
        "node_dim": 128,
        "memory_key_dim": 64,
        "memory_capacity_per_node": 128,
        "num_basis": 2,
        "decays": [0.00005, 0.00015],
        # Tiny decay rates leave little branching-stability margin after
        # cold start; start from the fitted Hawkes parameters exactly.
        "semantic_blend": 0.0,
    },
}


def _load_chronological_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_TAOBAO_CHRONOLOGICAL_CONFIG)
    if path is not None:
        text = path.expanduser().read_text(encoding="utf-8").replace("\u00a0", " ")
        supplied = json.loads(text)
        if not isinstance(supplied, dict):
            raise ValueError("--cl-config must contain a JSON object")
        if supplied.get("format_version") != 1:
            raise ValueError("Taobao chronology cl_config.json requires format_version 1")
        if supplied.get("benchmark_id") != "taobao_real_chronological":
            raise ValueError(
                "cl_config.json benchmark_id must be 'taobao_real_chronological'"
            )
        allowed_root = {
            "format_version", "benchmark_id", "epochs_per_task",
            "cold_start_epochs", "training", "model",
        }
        unknown_root = set(supplied).difference(allowed_root)
        if unknown_root:
            raise ValueError(f"unsupported cl_config.json fields: {sorted(unknown_root)}")
        for key in ("epochs_per_task", "cold_start_epochs"):
            if key in supplied:
                config[key] = supplied[key]
        for section in ("training", "model"):
            values = supplied.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f"cl_config.json {section} must be an object")
            unknown = set(values).difference(config[section])
            if unknown:
                raise ValueError(
                    f"unsupported cl_config.json {section} fields: {sorted(unknown)}"
                )
            config[section].update(values)

    return _validate_chronological_config(config)


def _validate_chronological_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["epochs_per_task"] <= 0 or config["cold_start_epochs"] < 0:
        raise ValueError("epochs_per_task must be positive and cold_start_epochs non-negative")
    training = config["training"]
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if training["grad_clip"] <= 0 or training["sleep_every"] <= 0:
        raise ValueError("grad_clip and sleep_every must be positive")
    if training["optimizer_impl"] not in {"auto", "standard", "foreach", "fused"}:
        raise ValueError("optimizer_impl must be auto, standard, foreach, or fused")
    if training["evaluation_ablation"] not in {
        "full", "no_working", "no_episodic", "fixed_topology", "no_sleep",
        "heuristic_controller", "no_merge_prune", "flat_memory",
    }:
        raise ValueError("unsupported evaluation_ablation")
    model = config["model"]
    for key in ("z_dim", "node_dim", "memory_key_dim", "memory_capacity_per_node", "num_basis"):
        if model[key] <= 0:
            raise ValueError(f"model.{key} must be positive")
    decays = model["decays"]
    if not isinstance(decays, list) or len(decays) != model["num_basis"]:
        raise ValueError("model.decays must have num_basis positive entries")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in decays):
        raise ValueError("model.decays must contain only positive finite values")
    if not 0.0 <= model["semantic_blend"] <= 1.0:
        raise ValueError("model.semantic_blend must lie in [0, 1]")
    model["decays"] = [float(value) for value in decays]
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_training_command(
    *,
    python_executable: str,
    data_path: Path,
    split_manifest: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    task_id: int,
    seed: int,
    device: str,
    epochs: int,
    cold_start_epochs: int,
    memory_capacity_per_node: int,
    persistent_memory_budget_bytes: int | None,
    previous_checkpoint: Path | None,
    decays: tuple[float, ...] = (0.00005, 0.00015),
    learning_rate: float = 0.0007,
    weight_decay: float = 0.00001,
    grad_clip: float = 3.5,
    optimizer_impl: str = "auto",
    sleep_every: int = 1,
    evaluation_ablation: str = "full",
    z_dim: int = 50,
    node_dim: int = 128,
    memory_key_dim: int = 64,
    semantic_blend: float = 0.0,
) -> list[str]:
    last = checkpoint_dir / f"task_{task_id:02d}_last.pt"
    best = checkpoint_dir / f"task_{task_id:02d}_best.pt"
    command = [
        python_executable,
        str(TRAIN_ENTRY),
        "--data-path", str(data_path),
        "--split-manifest", str(split_manifest),
        "--split", "train",
        "--checkpoint", str(last),
        "--best-checkpoint", str(best),
        "--tree-init-depth", "0",
        "--seed", str(seed),
        "--device", device,
        "--epochs", str(epochs),
        "--cold-start-epochs", str(cold_start_epochs if previous_checkpoint is None else 0),
        "--learning-rate", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--grad-clip", str(grad_clip),
        "--optimizer-impl", optimizer_impl,
        "--sleep-every", str(sleep_every),
        "--evaluation-ablation", evaluation_ablation,
        "--z-dim", str(z_dim),
        "--node-dim", str(node_dim),
        "--memory-key-dim", str(memory_key_dim),
        "--memory-capacity-per-node", str(memory_capacity_per_node),
        "--num-basis", str(len(decays)),
        "--decays", *(str(value) for value in decays),
        "--semantic-blend", str(semantic_blend),
        "--no-training-plots",
        "--validation-history-path",
        str(output_dir / "validation" / f"task_{task_id:02d}.json"),
        "--unified-topology-log-path", str(output_dir / "topology_diagnostics.log"),
        "--topology-events-path", str(output_dir / "topology_events.jsonl"),
    ]
    if persistent_memory_budget_bytes is not None:
        command.extend([
            "--persistent-memory-budget-bytes",
            str(persistent_memory_budget_bytes),
        ])
    if previous_checkpoint is not None:
        command.extend(["--resume", str(previous_checkpoint)])
    return command


def _write_run_manifest(
    output_dir: Path,
    data_manifest_path: Path,
    manifest: dict[str, Any],
    *,
    task_ids: list[int],
    seed: int,
    config: dict[str, Any],
    config_path: Path | None,
    persistent_memory_budget_bytes: int | None,
) -> None:
    training = config["training"]
    model = config["model"]
    run = {
        "workflow": "standalone_taobao_chronological_HM",
        "benchmark_id": config["benchmark_id"],
        "data_manifest": str(data_manifest_path.resolve()),
        "data_manifest_sha256": _sha256(data_manifest_path),
        "global_clock": manifest["global_clock"],
        "task_ids": task_ids,
        "checkpoint_pattern": "checkpoint/task_{task_id:02d}_best.pt",
        "cl_config": (
            {
                "path": str(config_path.expanduser().resolve()),
                "sha256": _sha256(config_path.expanduser().resolve()),
            }
            if config_path is not None else None
        ),
        "model": model,
        "training": {
            "seed": seed,
            "epochs_per_window": config["epochs_per_task"],
            "cold_start_epochs": config["cold_start_epochs"],
            **training,
            "persistent_memory_budget_bytes": persistent_memory_budget_bytes,
        },
        "sequence": (
            "train window t, select by its chronological validation split, "
            "then resume task_t_best.pt on window t+1"
        ),
        "windows": [
            {
                "task_id": int(item["window_id"]),
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "event_count": int(item["event_count"]),
            }
            for item in manifest["windows"]
            if int(item["window_id"]) in task_ids
        ],
    }
    (output_dir / "chronological_training_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HM sequentially on prepared UTC Taobao windows"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cl-config",
        type=Path,
        default=None,
        help="Chronological run settings JSON; mapped to native Train.py options.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cold-start-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--optimizer-impl",
        choices=("auto", "standard", "foreach", "fused"),
        default=None,
    )
    parser.add_argument("--sleep-every", type=int, default=None)
    parser.add_argument(
        "--evaluation-ablation",
        choices=(
            "full", "no_working", "no_episodic", "fixed_topology", "no_sleep",
            "heuristic_controller", "no_merge_prune", "flat_memory",
        ),
        default=None,
    )
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--node-dim", type=int, default=None)
    parser.add_argument("--memory-key-dim", type=int, default=None)
    parser.add_argument("--memory-capacity-per-node", type=int, default=None)
    parser.add_argument("--num-basis", type=int, default=None)
    parser.add_argument("--decays", type=float, nargs="+", default=None)
    parser.add_argument("--semantic-blend", type=float, default=None)
    parser.add_argument("--persistent-memory-budget-bytes", type=int, default=None)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_chronological_config(args.cl_config)
    for argument, key in (
        ("epochs", "epochs_per_task"),
        ("cold_start_epochs", "cold_start_epochs"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    for argument, key in (
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("grad_clip", "grad_clip"),
        ("optimizer_impl", "optimizer_impl"),
        ("sleep_every", "sleep_every"),
        ("evaluation_ablation", "evaluation_ablation"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config["training"][key] = value
    for argument, key in (
        ("z_dim", "z_dim"),
        ("node_dim", "node_dim"),
        ("memory_key_dim", "memory_key_dim"),
        ("memory_capacity_per_node", "memory_capacity_per_node"),
        ("num_basis", "num_basis"),
        ("semantic_blend", "semantic_blend"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config["model"][key] = value
    if args.decays is not None:
        config["model"]["decays"] = args.decays
        if args.num_basis is None:
            config["model"]["num_basis"] = len(args.decays)
    config = _validate_chronological_config(config)

    args.epochs = config["epochs_per_task"]
    args.cold_start_epochs = config["cold_start_epochs"]
    args.learning_rate = config["training"]["learning_rate"]
    args.weight_decay = config["training"]["weight_decay"]
    args.grad_clip = config["training"]["grad_clip"]
    args.optimizer_impl = config["training"]["optimizer_impl"]
    args.sleep_every = config["training"]["sleep_every"]
    args.evaluation_ablation = config["training"]["evaluation_ablation"]
    args.z_dim = config["model"]["z_dim"]
    args.node_dim = config["model"]["node_dim"]
    args.memory_key_dim = config["model"]["memory_key_dim"]
    args.memory_capacity_per_node = config["model"]["memory_capacity_per_node"]
    args.num_basis = config["model"]["num_basis"]
    args.decays = tuple(config["model"]["decays"])
    args.semantic_blend = config["model"]["semantic_blend"]

    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.epochs <= 0 or args.cold_start_epochs < 0:
        raise ValueError("--epochs must be positive and cold-start epochs non-negative")
    if args.memory_capacity_per_node <= 0:
        raise ValueError("--memory-capacity-per-node must be positive")
    if (
        args.persistent_memory_budget_bytes is not None
        and args.persistent_memory_budget_bytes <= 0
    ):
        raise ValueError("--persistent-memory-budget-bytes must be positive")

    data_dir = args.data_dir.expanduser().resolve()
    data_manifest_path = data_dir / "chronological_manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"missing chronological manifest: {data_manifest_path}")
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_id") != config["benchmark_id"]:
        raise ValueError(
            "cl_config.json benchmark_id does not match chronological data: "
            f"{config['benchmark_id']!r} vs {manifest.get('benchmark_id')!r}"
        )
    if manifest.get("global_clock") != "UTC absolute event timestamp":
        raise ValueError("training requires verified absolute UTC timestamps")
    windows = sorted(
        manifest.get("windows", ()), key=lambda item: int(item["window_id"])
    )
    if not windows:
        raise ValueError("chronological manifest has no windows")
    all_task_ids = [int(item["window_id"]) for item in windows]
    if all_task_ids != list(range(len(all_task_ids))):
        raise ValueError("chronological window IDs must be contiguous from zero")
    task_end = all_task_ids[-1] if args.task_end is None else int(args.task_end)
    if args.task_start < 0 or task_end < args.task_start:
        raise ValueError("invalid task range")
    task_ids = [
        task_id for task_id in all_task_ids
        if args.task_start <= task_id <= task_end
    ]
    if not task_ids:
        raise ValueError("requested task range has no chronological windows")

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_dir,
        data_manifest_path,
        manifest,
        task_ids=task_ids,
        seed=args.seed,
        config=config,
        config_path=args.cl_config,
        persistent_memory_budget_bytes=args.persistent_memory_budget_bytes,
    )

    previous_checkpoint: Path | None = None
    first_task_index = all_task_ids.index(task_ids[0])
    if first_task_index > 0:
        predecessor_id = all_task_ids[first_task_index - 1]
        previous_checkpoint = checkpoint_dir / f"task_{predecessor_id:02d}_best.pt"
        if not previous_checkpoint.is_file():
            raise FileNotFoundError(
                f"starting at window {task_ids[0]} requires predecessor "
                f"checkpoint {previous_checkpoint}"
            )

    environment = os.environ.copy()
    python_paths = [str(ROOT), str(ROOT / "Models" / "HawkesMemory"), str(MEMORY_ROOT)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    window_by_id = {int(item["window_id"]): item for item in windows}
    for task_id in task_ids:
        best = checkpoint_dir / f"task_{task_id:02d}_best.pt"
        if args.resume and best.is_file():
            previous_checkpoint = best
            print(f"Skipping completed window {task_id}: {best}", flush=True)
            continue
        if task_id > 0 and previous_checkpoint is None:
            predecessor = checkpoint_dir / f"task_{task_id - 1:02d}_best.pt"
            if not predecessor.is_file():
                raise FileNotFoundError(
                    f"window {task_id} must resume from {predecessor}; "
                    "run earlier windows first"
                )
            previous_checkpoint = predecessor

        window = window_by_id[task_id]
        task_dir = data_dir / f"window_{task_id:03d}"
        data_path = task_dir / str(window.get("training_data", "combined.csv"))
        split_manifest = task_dir / str(
            window.get("training_split_manifest", "split_manifest.json")
        )
        if not data_path.is_file() or not split_manifest.is_file():
            raise FileNotFoundError(
                f"window {task_id} has no combined training data/split manifest"
            )
        command = _build_training_command(
            python_executable=args.python_executable,
            data_path=data_path,
            split_manifest=split_manifest,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            task_id=task_id,
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            cold_start_epochs=args.cold_start_epochs,
            memory_capacity_per_node=args.memory_capacity_per_node,
            persistent_memory_budget_bytes=args.persistent_memory_budget_bytes,
            previous_checkpoint=previous_checkpoint,
            decays=args.decays,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            optimizer_impl=args.optimizer_impl,
            sleep_every=args.sleep_every,
            evaluation_ablation=args.evaluation_ablation,
            z_dim=args.z_dim,
            node_dim=args.node_dim,
            memory_key_dim=args.memory_key_dim,
            semantic_blend=args.semantic_blend,
        )
        print(
            f"Training chronological window {task_id} "
            f"({window['start_date']}..{window['end_date']})",
            flush=True,
        )
        if args.dry_run:
            print(shlex.join(command), flush=True)
        else:
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"task_{task_id:02d}.log"
            with log_path.open("w", encoding="utf-8") as log_handle:
                subprocess.run(
                    command,
                    cwd=MEMORY_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not best.is_file():
                raise FileNotFoundError(
                    f"window {task_id} training completed without {best}; "
                    f"see {log_path}"
                )
        previous_checkpoint = best

    print(f"Chronological HM checkpoints are in {checkpoint_dir}")


if __name__ == "__main__":
    main()
