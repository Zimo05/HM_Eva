from __future__ import annotations

import csv
import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .hm_bootstrap import (
    STATIONARY_HM_DATASETS,
    STATIONARY_HM_WAKE_WAVEFRONT_BATCH_SIZE,
    expected_stationary_hm_upstream,
)
from .io import sha256, write_json
from .metrics import prediction_metrics
from .paths import DATASETS_ROOT, MODELS_ROOT, PROJECT_ROOT


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


HM_UPSTREAM_MANIFEST_ENV = "HM_UPSTREAM_MANIFEST"
HM_UPSTREAM_MANIFEST_NAME = "hm_upstream_manifest.json"


def hm_upstream_manifest_path() -> Path:
    """Return the manifest describing the upstream H-tree used by HM eval."""

    override = os.environ.get(HM_UPSTREAM_MANIFEST_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (DATASETS_ROOT / "DWS" / HM_UPSTREAM_MANIFEST_NAME).resolve()


def resolve_hm_upstream_h_tree(
    variant: str | None,
    *,
    manifest_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate the variant-specific upstream H-tree artifact.

    The manifest is deliberately the source of truth for stationary HM
    evaluation.  This keeps artifact selection reproducible and avoids
    silently falling back to a root-only tree when a variant is misconfigured.
    """

    if variant is None:
        raise ValueError("DWS HM evaluation requires a variant for H-tree resolution")

    manifest = (manifest_path or hm_upstream_manifest_path()).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"HM upstream manifest not found: {manifest}; "
            f"set {HM_UPSTREAM_MANIFEST_ENV} to an explicit manifest if needed"
        )

    with manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"HM upstream manifest must be a JSON object: {manifest}")

    variants = payload.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError(f"HM upstream manifest has no 'variants' mapping: {manifest}")

    variant_key = str(variant)
    entry = variants.get(variant_key)
    if entry is None:
        entry = variants.get(f"variant_{variant_key}")
    if entry is None:
        available = ", ".join(sorted(str(key) for key in variants))
        raise KeyError(
            f"HM upstream manifest has no entry for DWS variant {variant_key!r} "
            f"(available: {available})"
        )

    if isinstance(entry, str):
        artifact_name = entry
        entry_metadata: dict[str, Any] = {}
    elif isinstance(entry, Mapping):
        entry_metadata = dict(entry)
        artifact_name = (
            entry.get("h_tree")
            or entry.get("h_tree_path")
            or entry.get("path")
        )
    else:
        raise ValueError(
            f"HM upstream manifest entry for variant {variant_key!r} must be a path or object"
        )

    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError(
            f"HM upstream manifest entry for variant {variant_key!r} has no H-tree path"
        )

    artifact = Path(artifact_name).expanduser()
    if not artifact.is_absolute():
        artifact = manifest.parent / artifact
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"HM upstream H-tree for DWS variant {variant_key!r} not found: {artifact}"
        )

    expected_hash = entry_metadata.get("sha256") or entry_metadata.get("artifact_sha256")
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(
                f"invalid sha256 for DWS variant {variant_key!r} in {manifest}"
            )
        actual_hash = sha256(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                f"H-tree sha256 mismatch for DWS variant {variant_key!r}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    summary_name = (
        entry_metadata.get("sequence_summary")
        or entry_metadata.get("sequence_summary_path")
        or entry_metadata.get("summary_csv")
    )
    sequence_summary: Path | None = None
    if summary_name is not None:
        if not isinstance(summary_name, str) or not summary_name:
            raise ValueError(
                f"invalid sequence_summary for DWS variant {variant_key!r}"
            )
        sequence_summary = Path(summary_name).expanduser()
        if not sequence_summary.is_absolute():
            sequence_summary = manifest.parent / sequence_summary
        sequence_summary = sequence_summary.resolve()
        if not sequence_summary.is_file():
            raise FileNotFoundError(
                f"H-tree sequence summary for DWS variant {variant_key!r} "
                f"not found: {sequence_summary}"
            )
        expected_summary_hash = entry_metadata.get(
            "sequence_summary_sha256",
            entry_metadata.get("summary_sha256"),
        )
        if expected_summary_hash is not None:
            if (
                not isinstance(expected_summary_hash, str)
                or len(expected_summary_hash) != 64
            ):
                raise ValueError(
                    f"invalid sequence_summary sha256 for DWS variant "
                    f"{variant_key!r} in {manifest}"
                )
            actual_summary_hash = sha256(sequence_summary)
            if actual_summary_hash != expected_summary_hash:
                raise ValueError(
                    f"sequence_summary sha256 mismatch for DWS variant "
                    f"{variant_key!r}: expected {expected_summary_hash}, "
                    f"got {actual_summary_hash}"
                )

    node_dim_value = entry_metadata.get(
        "node_dim",
        entry_metadata.get(
            "d_model",
            payload.get("node_dim", payload.get("d_model", 128)),
        ),
    )
    try:
        node_dim = int(node_dim_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid node_dim for DWS variant {variant_key!r}: {node_dim_value!r}"
        ) from exc
    if node_dim <= 0:
        raise ValueError(
            f"node_dim for DWS variant {variant_key!r} must be positive: {node_dim}"
        )

    metadata = {
        "manifest_path": str(manifest),
        "variant": variant_key,
        "node_dim": node_dim,
        "entry": entry_metadata,
    }
    if sequence_summary is not None:
        metadata["sequence_summary_path"] = str(sequence_summary)
    return artifact, metadata


def hm_upstream_input_paths(variant: str | None) -> list[Path]:
    """Return manifest and artifact paths for result-manifest provenance."""

    artifact, metadata = resolve_hm_upstream_h_tree(variant)
    inputs = [Path(metadata["manifest_path"]), artifact]
    sequence_summary = metadata.get("sequence_summary_path")
    if sequence_summary is not None:
        inputs.append(Path(sequence_summary))
    return inputs


def stationary_command(
    spec,
    args,
    result_dir: Path,
    prepared: Path | None = None,
    hm_upstream: Any | None = None,
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
    if spec.model in {"S2P2", "AttNHP"}:
        script = MODELS_ROOT / "EasyTPP" / "run_experiment.py"
        dataset = f"dws_{args.variant}" if spec.dataset == "dws" else spec.dataset
        command = [
            python,
            str(script),
            "--model", spec.model,
            "--dataset", dataset,
            "--seed", str(args.seed),
            "--device", device,
            "--output-dir", str(native),
            "--archive", str(result_dir / "native.tar.gz"),
            "--overwrite",
        ]
        if spec.dataset == "dws":
            command += ["--variant", args.variant]
        if prepared is not None:
            command += ["--prepared-data-dir", str(prepared)]
        if epochs:
            command += ["--epochs", str(epochs)]
        if batch:
            command += ["--batch-size", str(batch)]
        if getattr(args, "smoke", False):
            command += ["--max-sequences", "4", "--max-events-per-sequence", "16"]
        return command, MODELS_ROOT / "EasyTPP", env
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
        eval_batch = int(getattr(args, "eval_batch_size", 64))
        if eval_batch <= 0:
            raise ValueError("eval_batch_size must be positive")
        # The upstream algorithm is shared, but the Memory wake transaction
        # starts from conservative dataset-specific values because the
        # multi-type Hawkes/residual tensors are substantially larger.
        wake_wavefront_batch_size = STATIONARY_HM_WAKE_WAVEFRONT_BATCH_SIZE.get(
            spec.dataset,
            int(getattr(args, "batch_size", None) or eval_batch),
        )
        upstream_h_tree: Path | None = None
        train_sequence_summary: Path | None = None
        upstream_node_dim: int | None = None
        if hm_upstream is not None:
            def upstream_value(name: str, default: Any = None) -> Any:
                if isinstance(hm_upstream, Mapping):
                    return hm_upstream.get(name, default)
                return getattr(hm_upstream, name, default)

            upstream_h_tree_value = upstream_value("h_tree")
            train_sequence_summary_value = upstream_value("sequence_summary")
            if upstream_h_tree_value is None or train_sequence_summary_value is None:
                raise ValueError(
                    "HM upstream descriptor must provide h_tree and sequence_summary"
                )
            upstream_h_tree = Path(upstream_h_tree_value).expanduser().resolve()
            train_sequence_summary = (
                Path(train_sequence_summary_value).expanduser().resolve()
            )
            upstream_node_dim = int(upstream_value("node_dim", 128))
        elif spec.dataset in STATIONARY_HM_DATASETS:
            # The runner replaces this deterministic descriptor with the
            # completed train-only bootstrap.  Keeping the fallback here
            # makes command construction useful for dry-runs and unit tests
            # without ever falling back to a root-only stationary HM model.
            descriptor = expected_stationary_hm_upstream(
                prepared or result_dir / "prepared",
                spec.dataset,
            )
            upstream_h_tree = descriptor.h_tree
            train_sequence_summary = descriptor.sequence_summary
            upstream_node_dim = descriptor.node_dim
        elif spec.dataset == "dws":
            upstream_h_tree, upstream_metadata = resolve_hm_upstream_h_tree(
                str(args.variant)
            )
            upstream_node_dim = int(upstream_metadata["node_dim"])
            summary_path = upstream_metadata.get("sequence_summary_path")
            if summary_path is None:
                raise ValueError(
                    "DWS HM upstream manifest must provide sequence_summary"
                )
            train_sequence_summary = (
                (prepared or result_dir / "prepared")
                / "sequence_summary_train.csv"
            )
        else:
            raise ValueError(
                "HM requires a train-only upstream H-tree descriptor for "
                f"dataset {spec.dataset!r}"
            )
        # Dataset preparation belongs to the runner, before the final result
        # manifest is assembled so canonical/upstream hashes can be recorded.
        # Keeping command construction itself side-effect free is important for
        # --dry-run and for clean failure reporting.
        data_path = prepared / "canonical.csv"
        split_manifest = prepared / "split_manifest.json"
        memory = MODELS_ROOT / "HawkesMemory" / "Memory"
        checkpoint = result_dir / "checkpoint" / "model.pt"
        best = result_dir / "checkpoint" / "best.pt"
        if spec.dataset == "stackoverflow":
            # StackOverflow has more event types and a slower long-tail
            # convergence profile.  Keep this override local to StackOverflow
            # so the established DWS/Retweet/Taobao contract is unchanged.
            hm_default_epochs = 120
            frontier_budget = "5"
            frontier_routing_temperature = "0.9"
            residual_init_scale = "0.06"
        else:
            hm_default_epochs = 60
            frontier_budget = "7"
            frontier_routing_temperature = "1.10"
            residual_init_scale = "0.08"
        if spec.dataset in {"taobao", "stackoverflow"}:
            prototype_duplicate_threshold = "0.97"
            prototype_mode_threshold = "0.92"
            prototype_mode_capacity = "16"
            prototype_duplicate_quantile = "0.88"
            prototype_mode_quantile = "0.92"
        else:
            prototype_duplicate_threshold = None
            prototype_mode_threshold = None
            prototype_mode_capacity = None
            prototype_duplicate_quantile = None
            prototype_mode_quantile = None
        command = [
            python,
            str(memory / "Train" / "Train.py"),
            "--data-path",
            str(data_path),
            "--split-manifest",
            str(split_manifest),
            "--split",
            "train",
            "--tree-init-depth",
            "0",
            "--num-basis",
            "2",
            "--decays",
            "0.5",
            "1.5",
        ]
        if upstream_h_tree is not None:
            command += [
                "--h-tree",
                str(upstream_h_tree),
                # Keep stationary HM architecture identical to the upstream
                # run_HM.sh contract.  In particular, do not let
                # TrainingCLI's generic node_dim=64 default silently create
                # a different checkpoint from the 128-dimensional H-tree.
                "--z-dim",
                "50",
                "--node-dim",
                str(upstream_node_dim),
                "--memory-key-dim",
                "64",
                # Restore the complete standalone frontier configuration,
                # not just its maximum width.  These values determine both
                # the local router and posterior/owner selection semantics.
                "--frontier-min-experts",
                "2",
                "--frontier-budget",
                frontier_budget,
                "--frontier-routing-temperature",
                frontier_routing_temperature,
                "--frontier-exploration",
                "0",
                "--frontier-confidence-weight",
                "0.60",
                "--frontier-compute-cost",
                "0.005",
                "--frontier-posterior-temperature",
                "0.85",
                "--frontier-credible-mass",
                "0.30",
                "--frontier-owner-confidence",
                "0.50",
                "--max-writes-per-sequence",
                "8",
                "--semantic-blend",
                "0",
                "--leaf-symmetry-scale",
                "0",
                # Restore the upstream route-gradient and controller-teacher
                # contract instead of inheriting TrainingCLI defaults.
                "--route-mix-weight",
                "0",
                "--route-posterior-weight",
                "0",
                "--route-distill-weight",
                "0.25",
                "--route-mi-weight",
                "0.15",
                "--route-balance-weight",
                "0.10",
                "--route-energy-temperature",
                "1.0",
                "--route-encoder-warmup-epochs",
                "0",
                "--route-encoder-grad-scale",
                "0.08",
                "--route-encoder-reliability-decay",
                "0.80",
                "--route-teacher-temperature",
                "0.85",
                "--route-balance-batch-size",
                "32",
                "--wake-wavefront-batch-size",
                str(wake_wavefront_batch_size),
                # Preserve the upstream structural transaction thresholds.
                "--prune-warmup-epochs",
                "12",
                "--merge-min-replay",
                "12",
                "--light-replay-budget",
                "128",
                "--deep-min-interval",
                "3",
                "--deep-computation-cost",
                "0.05",
                "--deep-prior-probability",
                "0.10",
                "--deep-prior-weight",
                "0.01",
                "--deep-evidence-budget",
                "32",
                "--topology-inertia-strength",
                "0.03",
                "--topology-inertia-tau",
                "3.0",
            ]
            if spec.dataset == "retweet":
                # Retweet is the isolated rollout target for the shared-bank
                # snapshot protocol.  Other stationary datasets, DWS, and CL
                # keep the established ordered transaction semantics.
                command += ["--wake-transaction-mode", "snapshot"]
            if not getattr(args, "smoke", False):
                # Alignment and residual signatures require the complete
                # training population so every H-tree leaf has non-zero
                # mass.  Strict inductive H-trees also cannot be paired with
                # a truncated manifest train split, so smoke keeps the full
                # split and only reduces the training epochs below.
                command += [
                    "--sequence-summary",
                    str(train_sequence_summary),
                    "--residual-init-scale",
                    residual_init_scale,
                    "--residual-init-rank",
                    "4",
                    "--residual-init-grad-clip",
                    "0",
                    "--alignment-epochs",
                    "5",
                    "--alignment-batch-size",
                    "16",
                    "--alignment-lr",
                    "0.001",
                    "--alignment-weight-decay",
                    "0.00001",
                    "--alignment-temperature",
                    "1.0",
                    "--alignment-grad-clip",
                    "5.0",
                ]
        if prototype_duplicate_threshold is not None:
            # Taobao/StackOverflow prototype policy.  DWS and the other
            # datasets keep the checkpoint/TrainingCLI defaults unchanged.
            command += [
                "--prototype-duplicate-threshold",
                prototype_duplicate_threshold,
                "--prototype-mode-threshold",
                prototype_mode_threshold,
                "--prototype-mode-capacity",
                prototype_mode_capacity,
                "--prototype-duplicate-quantile",
                prototype_duplicate_quantile,
                "--prototype-mode-quantile",
                prototype_mode_quantile,
            ]
        command += [
            "--checkpoint",
            str(checkpoint),
            "--best-checkpoint",
            str(best),
            "--seed",
            str(args.seed),
            "--device",
            device,
        ]
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
        if args.smoke:
            cold_start_epochs = 1
        elif spec.dataset in {"taobao", "stackoverflow"}:
            cold_start_epochs = 40
        elif spec.dataset in STATIONARY_HM_DATASETS:
            cold_start_epochs = 20
        else:
            cold_start_epochs = 5
        command += [
            "--epochs", str(epochs or hm_default_epochs),
            "--cold-start-epochs", str(cold_start_epochs),
            "--validation-batch-size", str(eval_batch),
        ]
        if args.smoke:
            if upstream_h_tree is None:
                command += ["--max-sequences", "4", "--max-events-per-sequence", "16"]
            command += ["--no-training-plots"]
        if spec.dataset in STATIONARY_HM_DATASETS:
            # This policy is intentionally limited to stationary discovery
            # HM runs. DWS keeps its external Hawkes contract, while
            # continual HM uses _continual_hm_command and no projection flag.
            command += [
                "--stability-constrained-cold-start",
                "--cold-start-rho-base",
                "0.88",
                "--residual-rho-safe",
                "0.95",
            ]
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


def _as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_probabilities(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    return [float(item) for item in value]


def _prediction_rows(result_dir: Path, native: Path) -> list[dict[str, Any]]:
    candidates = []
    root_predictions = result_dir / "predictions.jsonl.gz"
    if root_predictions.is_file():
        candidates.append(root_predictions)
    candidates.extend(sorted(native.rglob("predictions.jsonl.gz")))
    for path in candidates:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    # HM's native evaluator predates the shared JSONL contract.  Keep this
    # fallback for direct normalization of an existing result directory.
    hm_csv = native / "event_predictions.csv"
    if not hm_csv.is_file():
        return []
    previous_time: dict[tuple[str, int], float] = {}
    rows: list[dict[str, Any]] = []
    with hm_csv.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (row.get("variant", ""), int(row["source_index"]))
            true_time = float(row["true_time"])
            true_delta = true_time - previous_time.get(key, 0.0)
            previous_time[key] = true_time
            rows.append({
                "sequence_id": int(row["source_index"]),
                "event_index": int(row["event_index"]),
                "true_type": int(row["true_type"]),
                "predicted_type": int(
                    row.get("predicted_type", row.get("predicted_type_at_event_time", -1))
                ),
                "predicted_type_at_event_time": int(
                    row.get("predicted_type_at_event_time", row.get("predicted_type", -1))
                ),
                "type_probabilities": _as_probabilities(
                    row.get("type_probabilities")
                ),
                "forecast_type_probabilities": _as_probabilities(
                    row.get("forecast_type_probabilities")
                    or row.get("prefix_type_probabilities")
                ),
                "true_delta_time": true_delta,
                "predicted_delta_time": float(row["predicted_delta"]),
                "event_nll": float(row["nll"]),
            })
    return rows


def _canonical_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        forecast_probabilities = _as_probabilities(
            row.get("forecast_type_probabilities")
            or row.get("prefix_type_probabilities")
        )
        probabilities = forecast_probabilities or _as_probabilities(
            row.get("type_probabilities")
        )
        predicted_type = row.get("forecast_predicted_type")
        if predicted_type in (None, ""):
            predicted_type = row.get("predicted_type")
        if predicted_type in (None, "") and probabilities:
            predicted_type = max(range(len(probabilities)), key=probabilities.__getitem__)
        if predicted_type in (None, ""):
            predicted_type = row.get("predicted_type_at_event_time")
        row["predicted_type"] = int(predicted_type) if predicted_type not in (None, "") else None
        if probabilities is not None:
            row["type_probabilities"] = probabilities
        if row.get("event_nll") in (None, "") and row.get("nll") not in (None, ""):
            row["event_nll"] = row["nll"]
        canonical.append(row)
    return canonical


def _extract_num_types(payload: Any) -> int | None:
    if isinstance(payload, Mapping):
        for key in ("dim_process", "num_types", "num_event_types", "expected_types"):
            value = _as_float(payload.get(key))
            if value is not None and value > 0 and value.is_integer():
                return int(value)
        for value in payload.values():
            found = _extract_num_types(value)
            if found is not None:
                return found
    return None


def _resolve_num_types(result_dir: Path, native: Path, rows: list[Mapping[str, Any]]) -> int | None:
    json_candidates = [
        native / "adapter_manifest.json",
        native / "log" / "run_config.json",
        native / "log" / "summary.json",
        native / "summary.json",
        result_dir / "manifest.json",
    ]
    for path in json_candidates:
        if not path.is_file():
            continue
        try:
            found = _extract_num_types(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            found = None
        if found is not None:
            return found

    for path in (result_dir / "prepared" / "train.pkl", native / "train.pkl"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                found = _extract_num_types(pickle.load(handle))
        except (OSError, EOFError, pickle.PickleError):
            found = None
        if found is not None:
            return found

    probability_lengths = [
        len(probabilities)
        for row in rows
        for probabilities in [_as_probabilities(row.get("forecast_type_probabilities") or row.get("type_probabilities"))]
        if probabilities
    ]
    if probability_lengths:
        return max(probability_lengths)
    labels = []
    for row in rows:
        for key in ("true_type", "predicted_type", "predicted_type_at_event_time"):
            value = row.get(key)
            if value not in (None, ""):
                labels.append(int(value))
    return max(labels) + 1 if labels else None


def _merge_prediction_metrics(
    metrics: dict[str, Any],
    result_dir: Path,
    native: Path,
) -> dict[str, Any]:
    raw_rows = _prediction_rows(result_dir, native)
    rows = _canonical_prediction_rows(raw_rows)
    num_types = _resolve_num_types(result_dir, native, rows)
    fallback_nll = _as_float(metrics.get("nll_per_event"))
    if rows:
        native_events = metrics.get("events")
        canonical = prediction_metrics(
            rows,
            num_types=num_types,
            fallback_nll=fallback_nll,
        )
        metrics.update(canonical)
        if native_events is not None:
            metrics["native_events"] = native_events
        metrics["events"] = canonical["num_events"]
        if "local_time_mae" in metrics:
            metrics["native_local_time_mae"] = metrics["local_time_mae"]
            metrics["local_time_mae"] = canonical["time_mae"]
        if "local_time_rmse" in metrics:
            metrics["native_local_time_rmse"] = metrics["local_time_rmse"]
            metrics["local_time_rmse"] = canonical["time_rmse"]
    elif "num_events" not in metrics and "events" in metrics:
        metrics["num_events"] = metrics["events"]
    if num_types is not None:
        metrics["num_types"] = num_types
    metrics["prediction_population"] = "event_index >= 1 (causal next-event)"
    metrics["time_prediction_protocol"] = "causal next-event one-step estimator"
    return metrics


def normalize_native_metrics(spec, result_dir: Path) -> dict[str, Any]:
    native = result_dir / "native"
    if spec.model == "HM":
        summary = native / "summary.json"
        if not summary.exists():
            return {"state": "trained", "evaluation_pending": True}
        payload = json.loads(summary.read_text(encoding="utf-8"))
        selected = payload.get("variants", {}).get(
            "frozen/full",
            payload.get("variants", {}).get("full_frozen", payload),
        )
        metrics = dict(selected) if isinstance(selected, Mapping) else {}
        metrics["source"] = str(summary)
        return _merge_prediction_metrics(metrics, result_dir, native)
    if spec.model == "TPP_LLM":
        path = native / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        return _merge_prediction_metrics(dict(metrics), result_dir, native)
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
    metrics = {
        "nll_per_event": -loglike if loglike is not None else None,
        "accuracy": value("accuracy"),
        "time_mae": value("mae", "time_mae"),
        "time_rmse": value("rmse", "time_rmse"),
        "num_events": value("num_events", "numevents", "rmse numevents"),
        "source": str(path),
    }
    if metrics["num_events"] is not None:
        metrics["num_events"] = int(metrics["num_events"])
    return _merge_prediction_metrics(metrics, result_dir, native)


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
    eval_batch = int(getattr(args, "eval_batch_size", 64))
    if eval_batch <= 0:
        raise ValueError("eval_batch_size must be positive")
    if spec.dataset in {"taobao", "stackoverflow"}:
        eval_prototype_duplicate_threshold = "0.97"
        eval_prototype_mode_threshold = "0.92"
    else:
        eval_prototype_duplicate_threshold = None
        eval_prototype_mode_threshold = None
    command = [
        python_for(args), "-m", "Evaluate",
        "--checkpoint", str(checkpoint),
        "--data-path", str(data_path),
        "--split-manifest", str(split_manifest),
        "--output-dir", str(result_dir / "native"),
        "--protocol", "both",
        "--seed", str(args.seed),
        "--device", resolved_device(args.device),
        "--eval-batch-size", str(eval_batch),
        "--resume", "--save-event-predictions",
    ]
    if eval_prototype_duplicate_threshold is not None:
        # Keep evaluation-time retrieval thresholds aligned with the
        # Taobao/StackOverflow training checkpoint; DWS retains its original
        # evaluation defaults.
        command += [
            "--prototype-duplicate-threshold",
            eval_prototype_duplicate_threshold,
            "--prototype-mode-threshold",
            eval_prototype_mode_threshold,
        ]
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
                probabilities = row.get("type_probabilities")
                forecast_probabilities = (
                    row.get("forecast_type_probabilities")
                    or row.get("prefix_type_probabilities")
                )
                if forecast_probabilities:
                    probabilities = forecast_probabilities
                try:
                    parsed_probabilities = json.loads(probabilities) if probabilities else None
                except json.JSONDecodeError:
                    parsed_probabilities = None
                predicted_type = row.get("predicted_type")
                if predicted_type in (None, "") and forecast_probabilities:
                    try:
                        values = [float(value) for value in json.loads(forecast_probabilities)]
                        predicted_type = max(range(len(values)), key=values.__getitem__)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        predicted_type = None
                if predicted_type in (None, ""):
                    predicted_type = row.get("predicted_type_at_event_time")
                output.write(json.dumps({
                    "sequence_id": int(row["source_index"]),
                    "event_index": int(row["event_index"]),
                    "true_type": int(row["true_type"]),
                    "predicted_type": int(predicted_type),
                    "predicted_type_at_event_time": (
                        int(row["predicted_type_at_event_time"])
                        if row.get("predicted_type_at_event_time") not in (None, "")
                        else int(predicted_type)
                    ),
                    "type_probabilities": parsed_probabilities,
                    "forecast_type_probabilities": parsed_probabilities,
                    "true_delta_time": true_delta,
                    "predicted_delta_time": float(row["predicted_delta"]),
                    "event_nll": float(row["nll"]),
                    "variant": row["variant"],
                }) + "\n")
