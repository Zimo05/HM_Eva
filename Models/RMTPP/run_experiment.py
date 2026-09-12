#!/usr/bin/env python3
"""Run a complete EasyTPP RMTPP experiment on supported benchmarks.

The driver follows the FullyNN baseline runner's experiment layout: adapt the
data, train RMTPP, select the best validation checkpoint, evaluate test data,
plot metrics, and create a compact tar.gz bundle.  Unlike the older
validation-only plots, every RMTPP curve contains both train and validation.
"""

import argparse
import csv
import gzip
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch


RMTPP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RMTPP_ROOT.parents[1]
EASYTPP_ROOT = PROJECT_ROOT / "Models" / "EasyTPP"
LOCAL_CACHE_ROOT = Path(tempfile.gettempdir()) / "rmtpp_baseline_cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT / "xdg"))
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_ROOT / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(LOCAL_CACHE_ROOT / "huggingface" / "datasets")
)
sys.path.insert(0, str(EASYTPP_ROOT))

from data_configuration import DataConfiguration  # noqa: E402
from easy_tpp.config_factory import Config  # noqa: E402
from easy_tpp.runner import Runner  # noqa: E402
from easy_tpp.utils import RunnerPhase, logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train RMTPP, evaluate the best validation checkpoint, plot "
            "train/validation likelihood, RMSE and accuracy, then create tar.gz."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "dws", "amazon", "covid_policy_tracker", "taxi", "taobao",
            "retweet", "stackoverflow",
        ),
        default="dws",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help=(
            "Optional display/run label for prepared data; defaults to the "
            "canonical dataset name."
        ),
    )
    parser.add_argument("--variant", choices=("8", "13", "15", "17", "20"), default="13")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--mc-samples", type=int, default=32,
        help="Monte Carlo samples per interval for training log-likelihood.",
    )
    parser.add_argument("--thinning-num-sample", type=int, default=1)
    parser.add_argument("--thinning-num-exp", type=int, default=500)
    parser.add_argument("--dtime-max", type=float, default=120.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--selection-metric", choices=("loglike", "acc", "rmse"),
        default="loglike",
    )
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA index; use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: Models/RMTPP/Result/<group>/<dataset>_training_results",
    )
    parser.add_argument(
        "--archive", default=None,
        help="Output tar.gz. Default: <output-dir>.tar.gz",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing result directory/archive.",
    )
    parser.add_argument("--prepared-data-dir", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    positive = (
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.mc_samples,
        args.thinning_num_sample,
        args.thinning_num_exp,
        args.early_stop_patience,
        args.lr_patience,
    )
    if any(value < 1 for value in positive):
        raise ValueError(
            "epoch, batch/model sizes, sampling sizes, and patience must be positive"
        )
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr-factor must be between 0 and 1")
    if args.learning_rate <= 0 or args.dtime_max <= 0 or args.min_delta < 0:
        raise ValueError(
            "learning-rate/dtime-max must be positive and min-delta non-negative"
        )
    if args.gpu < -1:
        raise ValueError("--gpu must be -1 or a non-negative CUDA index")


def _normalise_dataset_label(label):
    """Turn a benchmark label into a safe, stable native run name."""

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return value.strip("._-") or "dataset"


def prepare_paths(args):
    dataset_label = getattr(args, "dataset_label", None)
    if dataset_label:
        run_name = _normalise_dataset_label(dataset_label)
        result_group = run_name
    elif args.dataset == "dws":
        run_name, result_group = "dws_{}".format(args.variant), "DWS"
    elif args.dataset == "amazon":
        run_name, result_group = "amazon", "Amazon"
    elif args.dataset == "covid_policy_tracker":
        run_name, result_group = "covid_policy_tracker", "Covid-Policy-Tracker"
    elif args.dataset == "taxi":
        run_name, result_group = "taxi", "Taxi"
    elif args.dataset == "taobao":
        run_name, result_group = "taobao", "Taobao"
    elif args.dataset == "retweet":
        run_name, result_group = "retweet", "Retweet"
    else:
        run_name, result_group = "stackoverflow", "StackOverflow"

    default_dir = (
        RMTPP_ROOT / "Result" / result_group / (run_name + "_training_results")
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else default_dir
    )
    archive = (
        Path(args.archive).expanduser().resolve()
        if args.archive else Path(str(output) + ".tar.gz")
    )
    if archive.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("--archive must end with .tar.gz")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                "{} already contains results; pass --overwrite or choose "
                "--output-dir".format(output)
            )
        shutil.rmtree(output)
    if archive.exists():
        if not args.overwrite:
            raise FileExistsError(
                "{} already exists; pass --overwrite or choose --archive".format(
                    archive
                )
            )
        archive.unlink()

    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(
        prefix=run_name + "_", dir=str(LOCAL_CACHE_ROOT)
    ))
    paths = {
        "run_name": run_name,
        "output": output,
        "archive": archive,
        "log": output / "log",
        "csv": output / "csv",
        "plot": output / "plot",
        "scratch": scratch,
        "checkpoints": scratch / "checkpoints",
        "config": scratch / "config",
        "runtime": scratch / "runtime",
    }
    for key in (
        "log", "csv", "plot", "checkpoints", "config", "runtime"
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    return paths


def read_records(path):
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError("{} is empty".format(path))
    return records


def read_num_event_types(train_json):
    return int(read_records(train_json)[0]["dim_process"])


def compute_training_stats(train_json):
    records = read_records(train_json)
    intervals = np.asarray([
        float(delta)
        for record in records
        for delta in record["time_since_last_event"][1:]
    ], dtype=np.float64)
    if intervals.size == 0 or np.any(intervals < 0):
        raise ValueError("Training data must contain non-negative intervals")
    return {
        "num_sequences": len(records),
        "num_intervals": int(intervals.size),
        "mean": float(np.mean(intervals)),
        "std": float(np.std(intervals)),
        "median": float(np.median(intervals)),
        "p90": float(np.quantile(intervals, 0.90)),
        "p99": float(np.quantile(intervals, 0.99)),
        "max": float(np.max(intervals)),
    }


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_config(args, paths, adapted_dir, num_event_types):
    config = {
        "pipeline_config_id": "runner_config",
        "data": {
            paths["run_name"]: {
                "data_format": "json",
                "train_dir": str(adapted_dir / "train.json"),
                "valid_dir": str(adapted_dir / "dev.json"),
                "test_dir": str(adapted_dir / "test.json"),
                "data_specs": {
                    "num_event_types": num_event_types,
                    "pad_token_id": num_event_types,
                    "padding_side": "right",
                    "truncation_side": "right",
                },
            }
        },
        "RMTPP_train": {
            "base_config": {
                "stage": "train",
                "backend": "torch",
                "dataset_id": paths["run_name"],
                "runner_id": "std_tpp",
                "model_id": "RMTPP",
                "base_dir": str(paths["runtime"]),
            },
            "trainer_config": {
                "batch_size": args.batch_size,
                "max_epoch": args.epochs,
                "shuffle": True,
                "optimizer": "adam",
                "learning_rate": args.learning_rate,
                "valid_freq": 1,
                "use_tfb": False,
                "metrics": ["acc", "rmse"],
                "seed": args.seed,
                "gpu": args.gpu,
            },
            "model_config": {
                "hidden_size": args.hidden_size,
                "loss_integral_num_sample_per_step": args.mc_samples,
                "thinning": {
                    "num_seq": 10,
                    "num_sample": args.thinning_num_sample,
                    "num_exp": args.thinning_num_exp,
                    "look_ahead_time": 10,
                    "patience_counter": 5,
                    "over_sample_rate": 5,
                    "num_samples_boundary": 5,
                    "dtime_max": args.dtime_max,
                    "num_step_gen": 1,
                },
            },
        },
    }
    config_path = paths["config"] / (paths["run_name"] + "_config.yaml")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required; install Models/EasyTPP/requirements.txt"
        ) from exc
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def deterministic_validation(runner, data_loader, seed):
    """Evaluate fixed thinning samples without perturbing the training RNG."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return evaluate_loader(runner, data_loader)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_loader(runner, data_loader, prediction_path=None):
    """Stream evaluation so long datasets do not retain all batch predictions."""
    total_loss = 0.0
    total_num_events = 0
    squared_error = 0.0
    time_num_events = 0
    num_correct = 0
    type_num_events = 0

    prediction_handle = gzip.open(prediction_path, "wt", encoding="utf-8") if prediction_path else None
    event_offset = 0
    for batch in data_loader:
        loss, num_events, predictions, labels, masks = (
            runner.model_wrapper.run_batch(batch, phase=RunnerPhase.VALIDATE)
        )
        total_loss += float(loss)
        total_num_events += int(num_events)
        pred_time, pred_type = predictions
        label_time, label_type = labels
        if len(masks) == 3:
            _, time_mask, type_mask = masks
        else:
            time_mask = type_mask = masks[0]
        time_mask = np.asarray(time_mask, dtype=bool)
        type_mask = np.asarray(type_mask, dtype=bool)
        time_error = np.asarray(pred_time)[time_mask] - np.asarray(label_time)[time_mask]
        squared_error += float(np.square(time_error).sum())
        time_num_events += int(time_error.size)
        num_correct += int(
            (np.asarray(pred_type)[type_mask] == np.asarray(label_type)[type_mask]).sum()
        )
        type_num_events += int(type_mask.sum())
        if prediction_handle is not None:
            common = time_mask & type_mask
            flat_true_time = np.asarray(label_time)[common]
            flat_pred_time = np.asarray(pred_time)[common]
            flat_true_type = np.asarray(label_type)[common]
            flat_pred_type = np.asarray(pred_type)[common]
            for index, (true_time, predicted_time, true_type, predicted_type) in enumerate(zip(flat_true_time, flat_pred_time, flat_true_type, flat_pred_type)):
                prediction_handle.write(json.dumps({
                    "sequence_id": None,
                    "event_index": event_offset + index,
                    "true_type": int(true_type),
                    "predicted_type": int(predicted_type),
                    "type_probabilities": None,
                    "true_delta_time": float(true_time),
                    "predicted_delta_time": float(predicted_time),
                    "event_nll": None,
                }) + "\n")
            event_offset += int(common.sum())

    if total_num_events == 0 or time_num_events == 0 or type_num_events == 0:
        raise RuntimeError("Evaluation found no target events")
    if prediction_handle is not None:
        prediction_handle.close()
    return {
        "loglike": -total_loss / total_num_events,
        "rmse": math.sqrt(squared_error / time_num_events),
        "acc": num_correct / type_num_events,
        "num_events": total_num_events,
        "rmse_num_events": time_num_events,
    }


def metric_row(epoch, split, metrics, learning_rate):
    return {
        "Epoch": epoch,
        "Split": split,
        "Log-likelihood": float(metrics["loglike"]),
        "RMSE": float(metrics["rmse"]),
        "Accuracy": float(metrics["acc"]),
        "NumEvents": int(metrics["num_events"]),
        "RMSE NumEvents": int(metrics["rmse_num_events"]),
        "LearningRate": float(learning_rate),
    }


METRIC_FIELDS = (
    "Epoch", "Split", "Log-likelihood", "RMSE", "Accuracy",
    "NumEvents", "RMSE NumEvents", "LearningRate",
)


def write_metrics(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def restore_checkpoint(runner, checkpoint_path):
    kwargs = {"map_location": runner.model_wrapper.device}
    try:
        state = torch.load(checkpoint_path, weights_only=True, **kwargs)
    except TypeError:
        state = torch.load(checkpoint_path, **kwargs)
    runner.model.load_state_dict(state, strict=False)


def train_and_test(args, paths, config_path):
    pipeline = Config.build_from_yaml_file(
        str(config_path), experiment_id="RMTPP_train"
    )
    runner = Runner.build_from_config(pipeline)
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_file():
            raise FileNotFoundError(args.initial_checkpoint)
        restore_checkpoint(runner, args.initial_checkpoint)
    train_loader = runner._data_loader.train_loader(shuffle=True)
    train_eval_loader = runner._data_loader.train_loader(shuffle=False)
    valid_loader = runner._data_loader.valid_loader(shuffle=False)
    test_loader = runner._data_loader.test_loader()

    metrics_path = paths["csv"] / (paths["run_name"] + "_metrics.csv")
    checkpoint_path = paths["checkpoints"] / (paths["run_name"] + "_best.pt")
    rows = []
    best_epoch = None
    minimize_selection = args.selection_metric == "rmse"
    best_selection = float("inf") if minimize_selection else float("-inf")
    stale_epochs = 0
    lr_stale_epochs = 0

    if args.evaluate_only:
        test_metrics = evaluate_loader(
            runner, test_loader, paths["output"] / "predictions.jsonl.gz"
        )
        current_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
        test_row = metric_row(0, "test", test_metrics, current_lr)
        test_row["SelectionMetric"] = "evaluate_only"
        test_path = paths["csv"] / (paths["run_name"] + "_test.csv")
        with test_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "Epoch", "Split", "SelectionMetric", "Log-likelihood",
                "RMSE", "Accuracy", "NumEvents", "RMSE NumEvents",
                "LearningRate",
            ))
            writer.writeheader()
            writer.writerow(test_row)
        runner.model_wrapper.close_summary()
        return [], test_row

    for epoch in range(1, args.epochs + 1):
        runner.run_one_epoch(train_loader, RunnerPhase.TRAIN)
        train_metrics = deterministic_validation(
            runner, train_eval_loader, args.seed + 100000
        )
        valid_metrics = deterministic_validation(
            runner, valid_loader, args.seed + 200000
        )
        current_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
        rows.append(metric_row(epoch, "train", train_metrics, current_lr))
        rows.append(metric_row(epoch, "valid", valid_metrics, current_lr))
        write_metrics(metrics_path, rows)
        logger.info(
            "[Epoch %d/%d] lr=%.2e | train LL=%.6f, RMSE=%.6f, "
            "accuracy=%.6f | valid LL=%.6f, RMSE=%.6f, accuracy=%.6f",
            epoch,
            args.epochs,
            current_lr,
            train_metrics["loglike"],
            train_metrics["rmse"],
            train_metrics["acc"],
            valid_metrics["loglike"],
            valid_metrics["rmse"],
            valid_metrics["acc"],
        )

        selection_value = float(valid_metrics[args.selection_metric])
        improved = (
            selection_value < best_selection - args.min_delta
            if minimize_selection else
            selection_value > best_selection + args.min_delta
        )
        if improved:
            best_epoch = epoch
            best_selection = selection_value
            stale_epochs = 0
            lr_stale_epochs = 0
            runner.model_wrapper.save(str(checkpoint_path))
            logger.info(
                "Saved new best checkpoint at epoch %d (%s=%.6f)",
                epoch, args.selection_metric, selection_value,
            )
        else:
            stale_epochs += 1
            lr_stale_epochs += 1

        if lr_stale_epochs >= args.lr_patience:
            old_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
            new_lr = old_lr * args.lr_factor
            for group in runner.model_wrapper.opt.param_groups:
                group["lr"] = new_lr
            lr_stale_epochs = 0
            logger.info("Reduced learning rate from %.2e to %.2e", old_lr, new_lr)

        if stale_epochs >= args.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d after %d epochs without %s improvement",
                epoch, stale_epochs, args.selection_metric,
            )
            break

    if best_epoch is None:
        raise RuntimeError("Training produced no validation checkpoint")

    restore_checkpoint(runner, checkpoint_path)
    test_metrics = evaluate_loader(
        runner, test_loader, paths["output"] / "predictions.jsonl.gz"
    )
    runner.model_wrapper.close_summary()
    current_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
    test_row = metric_row(best_epoch, "test", test_metrics, current_lr)
    test_row["SelectionMetric"] = "validation_{}".format(args.selection_metric)
    test_path = paths["csv"] / (paths["run_name"] + "_test.csv")
    with test_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Epoch", "Split", "SelectionMetric", "Log-likelihood",
                "RMSE", "Accuracy", "NumEvents", "RMSE NumEvents",
                "LearningRate",
            ),
        )
        writer.writeheader()
        writer.writerow(test_row)
    logger.info(
        "Final test from best epoch %d: LL=%.6f, RMSE=%.6f, accuracy=%.6f",
        best_epoch,
        test_metrics["loglike"],
        test_metrics["rmse"],
        test_metrics["acc"],
    )
    persistent_checkpoint = paths["output"] / "checkpoint" / "best.pt"
    persistent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, persistent_checkpoint)
    return rows, test_row


def plot_metrics(paths, rows, test_row):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required for the final plots") from exc

    by_split = {
        split: [row for row in rows if row["Split"] == split]
        for split in ("train", "valid")
    }
    panels = (
        ("Log-likelihood", "Log-likelihood", "Log-likelihood per event", "likelihood.png"),
        ("RMSE", "Time RMSE", "RMSE", "rmse.png"),
        ("Accuracy", "Event Accuracy", "Accuracy", "accuracy.png"),
    )
    colors = {"train": "#2563EB", "valid": "#EA580C"}

    def draw(axis, metric, title, ylabel):
        for split in ("train", "valid"):
            split_rows = by_split[split]
            axis.plot(
                [row["Epoch"] for row in split_rows],
                [row[metric] for row in split_rows],
                color=colors[split],
                linewidth=2.1,
                label="Validation" if split == "valid" else "Train",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="best", fontsize=8, frameon=False)
        if metric == "Accuracy":
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    for metric, title, ylabel, filename in panels:
        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        draw(axis, metric, "RMTPP on {} - {}".format(
            paths["run_name"].upper(), title
        ), ylabel)
        figure.savefig(paths["plot"] / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, (metric, title, ylabel, _) in zip(axes, panels):
        draw(axis, metric, title, ylabel)
    figure.suptitle(
        "RMTPP on {} | test at best epoch {}: LL={:.4f}, RMSE={:.4f}, "
        "Accuracy={:.2%}".format(
            paths["run_name"].upper(),
            test_row["Epoch"],
            test_row["Log-likelihood"],
            test_row["RMSE"],
            test_row["Accuracy"],
        ),
        fontsize=14,
        fontweight="semibold",
    )
    combined_path = paths["plot"] / (paths["run_name"] + "_curves.png")
    figure.savefig(combined_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved plots to %s", paths["plot"])


def create_archive(paths):
    """Package only log, csv, and plot, as required by the baseline protocol."""
    with tarfile.open(paths["archive"], "w:gz") as archive:
        for folder_name in ("log", "csv", "plot"):
            folder = paths[folder_name]
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                archive.add(
                    path,
                    arcname=str(
                        Path(paths["output"].name)
                        / folder_name
                        / path.relative_to(folder)
                    ),
                )
    logger.info("Saved archive to %s", paths["archive"])


def adapt_dataset(args, paths):
    if args.prepared_data_dir is not None:
        target = args.prepared_data_dir.expanduser().resolve()
        for split in ("train", "dev", "test"):
            if not (target / (split + ".json")).is_file():
                raise FileNotFoundError(target / (split + ".json"))
        return target
    adapter = DataConfiguration(seed=args.seed)
    private_root = paths["output"] / "prepared_data"
    if args.dataset == "dws":
        adapter.dws(output_dir=private_root, variants=[args.variant])
        return private_root / "dws_{}".format(args.variant)
    target = private_root / args.dataset
    getattr(adapter, args.dataset)(output_dir=target)
    return target


def main():
    args = parse_args()
    validate_args(args)
    paths = prepare_paths(args)
    console_log = paths["log"] / (paths["run_name"] + "_console.log")
    file_handler = logging.FileHandler(console_log, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)
    success = False
    try:
        logger.info("Preparing %s with seed=%d", paths["run_name"], args.seed)
        adapted_dir = adapt_dataset(args, paths)
        num_event_types = read_num_event_types(adapted_dir / "train.json")
        training_stats = compute_training_stats(adapted_dir / "train.json")
        write_json(
            paths["log"] / (paths["run_name"] + "_training_stats.json"),
            training_stats,
        )
        logger.info(
            "Time stats: median=%.6f, mean=%.6f, p99=%.6f, max=%.6f",
            training_stats["median"],
            training_stats["mean"],
            training_stats["p99"],
            training_stats["max"],
        )
        config_path = write_config(
            args, paths, adapted_dir, num_event_types
        )
        rows, test_row = train_and_test(args, paths, config_path)
        if not args.evaluate_only:
            plot_metrics(paths, rows, test_row)
        logger.info("RMTPP %s experiment complete", paths["run_name"])
        logger.info("Archive contains only log/, csv/, and plot/")
        success = True
    except Exception:
        logger.exception("RMTPP experiment failed")
        raise
    finally:
        file_handler.flush()
        logger.removeHandler(file_handler)
        file_handler.close()
        shutil.rmtree(paths["scratch"], ignore_errors=True)

    if success:
        create_archive(paths)
        print("Result directory: {}".format(paths["output"]))
        print("Archive: {}".format(paths["archive"]))


if __name__ == "__main__":
    main()
