#!/usr/bin/env python3
"""Run a complete EasyTPP FullyNN experiment on supported benchmark datasets.

The driver adapts the CSV data, trains FullyNN, selects the best checkpoint by
validation log-likelihood, evaluates it on the test split, plots validation
log-likelihood/RMSE/accuracy, and creates a reproducibility tar.gz bundle.
"""

import argparse
import csv
import gzip
import json
import logging
import os
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch


FULLYNN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FULLYNN_ROOT.parents[1]
EASYTPP_ROOT = PROJECT_ROOT / "Models" / "EasyTPP"
LOCAL_CACHE_ROOT = Path(tempfile.gettempdir()) / "fullynn_dws_cache"
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
            "Train FullyNN, evaluate the best validation-likelihood "
            "checkpoint, plot all three metrics, and create tar.gz."
        )
    )
    parser.add_argument(
        "--dataset", choices=(
            "dws", "amazon", "covid_policy_tracker", "mimic", "taxi",
            "taobao", "retweet", "mobike", "stackoverflow"
        ),
        default="dws"
    )
    parser.add_argument("--variant", choices=("8", "13", "15", "17", "20"), default="13")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-mlp-layers", type=int, default=2)
    parser.add_argument(
        "--history-window", type=int, default=20,
        help="Number of preceding intervals per truncated sequence (paper: 5/10/20/40).",
    )
    parser.add_argument("--thinning-num-sample", type=int, default=20)
    parser.add_argument("--thinning-num-exp", type=int, default=500)
    parser.add_argument("--dtime-max", type=float, default=120.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--selection-metric", choices=("loglike", "acc", "rmse"), default=None
    )
    parser.add_argument(
        "--no-log-time", action="store_false", dest="log_time",
        help="Disable log(delta-t) standardization (enabled by default for DWS).",
    )
    parser.set_defaults(log_time=True)
    parser.add_argument(
        "--joint-marked-intensity", action="store_false", dest="factorized_marks",
        help="Use the old coupled per-mark intensity instead of the DWS factorized mark head.",
    )
    parser.set_defaults(factorized_marks=True)
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA index; use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: Models/FullyNN/Result/DWS/"
            "dws_<variant>_training_results"
        ),
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Output tar.gz. Default: <output-dir>.tar.gz",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory/archive for this variant.",
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        default=None,
        help=(
            "Use a benchmark-prepared train.json/dev.json/test.json view "
            "instead of re-splitting/adapting the raw dataset."
        ),
    )
    return parser.parse_args()


def prepare_paths(args):
    if args.dataset == "dws":
        run_name, result_group = "dws_{}".format(args.variant), "DWS"
    elif args.dataset == "amazon":
        run_name, result_group = "amazon", "Amazon"
    elif args.dataset == "mimic":
        run_name, result_group = "mimic", "MIMIC"
    elif args.dataset == "taxi":
        run_name, result_group = "taxi", "Taxi"
    elif args.dataset == "taobao":
        run_name, result_group = "taobao", "Taobao"
    elif args.dataset == "retweet":
        run_name, result_group = "retweet", "Retweet"
    elif args.dataset == "mobike":
        run_name, result_group = "mobike", "MobikeData"
    elif args.dataset == "stackoverflow":
        run_name, result_group = "stackoverflow", "StackOverflow"
    else:
        run_name, result_group = "covid_policy_tracker", "Covid-Policy-Tracker"
    default_dir = (
        FULLYNN_ROOT / "Result" / result_group / (run_name + "_training_results")
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
        "plot": output / "plot",
        "scratch": scratch,
        "checkpoints": scratch / "checkpoints",
        "config": scratch / "config",
        "runtime": scratch / "runtime",
    }
    for key in ("log", "plot", "checkpoints", "config", "runtime"):
        paths[key].mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    return paths


def read_num_event_types(train_json):
    with train_json.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError("{} is empty".format(train_json))
    return int(records[0]["dim_process"])


def load_records(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def observed_intervals(records):
    return np.asarray(
        [
            float(delta)
            for record in records
            for delta in record["time_since_last_event"][1:]
        ],
        dtype=np.float64,
    )


def compute_training_stats(train_json, epsilon=1e-6):
    intervals = observed_intervals(load_records(train_json))
    if intervals.size == 0 or np.any(intervals < 0):
        raise ValueError("Training data must contain non-negative inter-event times")
    log_intervals = np.log(intervals + epsilon)
    return {
        "time_epsilon": epsilon,
        "num_intervals": int(intervals.size),
        "mean": float(np.mean(intervals)),
        "std": float(np.std(intervals)),
        "median": float(np.median(intervals)),
        "p90": float(np.quantile(intervals, 0.90)),
        "p99": float(np.quantile(intervals, 0.99)),
        "max": float(np.max(intervals)),
        "log_time_mean": float(np.mean(log_intervals)),
        "log_time_std": float(max(np.std(log_intervals), 1e-6)),
    }


def write_config(args, paths, adapted_dir, num_event_types, training_stats):
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
        "FullyNN_train": {
            "base_config": {
                "stage": "train",
                "backend": "torch",
                "dataset_id": paths["run_name"],
                "runner_id": "std_tpp",
                "model_id": "FullyNN",
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
                "rnn_type": "LSTM",
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout_rate": 0.0,
                "model_specs": {
                    "num_mlp_layers": args.num_mlp_layers,
                    "proper_marked_intensities": True,
                    "factorized_marks": args.factorized_marks,
                    "log_time": args.log_time,
                    "time_epsilon": training_stats["time_epsilon"],
                    "log_time_mean": training_stats["log_time_mean"],
                    "log_time_std": training_stats["log_time_std"],
                    "history_window": args.history_window,
                    "target_predicate_ids": [51, 61]
                    if args.dataset == "mimic" else [],
                },
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


def metric_row(epoch, split, metrics, learning_rate):
    return {
        "Epoch": epoch,
        "Split": split,
        "Log-likelihood": float(metrics["loglike"]),
        "Time Log-likelihood": float(metrics.get("time_loglike", float("nan"))),
        "Mark Log-likelihood": float(metrics.get("mark_loglike", float("nan"))),
        "RMSE": float(metrics.get("rmse", float("nan"))),
        "Accuracy": float(metrics.get("acc", float("nan"))),
        "NumEvents": int(metrics["num_events"]),
        "LearningRate": float(learning_rate),
        "Target 51 Accuracy": float(metrics.get(
            "target_51_accuracy", float("nan")
        )),
        "Target 61 Accuracy": float(metrics.get(
            "target_61_accuracy", float("nan")
        )),
    }


def write_metrics(path, rows):
    fields = (
        "Epoch", "Split", "Log-likelihood", "Time Log-likelihood",
        "Mark Log-likelihood", "RMSE", "Accuracy", "NumEvents", "LearningRate",
        "Target 51 Accuracy", "Target 61 Accuracy"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
        return runner.run_one_epoch(data_loader, RunnerPhase.VALIDATE)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def write_event_predictions(path, metrics):
    """Export EasyTPP validation predictions when supplied by the wrapper."""
    predictions, labels = metrics.get("pred"), metrics.get("label")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        if not predictions or not labels:
            return
        pred_time, pred_type = map(np.asarray, predictions[:2])
        true_time, true_type = map(np.asarray, labels[:2])
        count = min(pred_time.size, pred_type.size, true_time.size, true_type.size)
        for index in range(count):
            handle.write(json.dumps({
                "sequence_id": None,
                "event_index": index,
                "true_type": int(true_type.reshape(-1)[index]),
                "predicted_type": int(pred_type.reshape(-1)[index]),
                "type_probabilities": None,
                "true_delta_time": float(true_time.reshape(-1)[index]),
                "predicted_delta_time": float(pred_time.reshape(-1)[index]),
                "event_nll": None,
            }) + "\n")


def train_and_test(args, paths, config_path):
    pipeline = Config.build_from_yaml_file(
        str(config_path), experiment_id="FullyNN_train"
    )
    runner = Runner.build_from_config(pipeline)
    train_loader = runner._data_loader.train_loader()
    valid_loader = runner._data_loader.valid_loader()
    test_loader = runner._data_loader.test_loader()

    metrics_path = paths["log"] / (paths["run_name"] + "_metrics.csv")
    checkpoint_path = paths["checkpoints"] / (paths["run_name"] + "_best.pt")
    rows = []
    best_epoch = None
    minimize_selection = args.selection_metric == "rmse"
    best_selection = float("inf") if minimize_selection else float("-inf")
    stale_epochs = 0
    lr_stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = runner.run_one_epoch(train_loader, RunnerPhase.TRAIN)
        valid_metrics = deterministic_validation(
            runner, valid_loader, args.seed + 100000
        )
        current_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
        rows.append(metric_row(epoch, "train", train_metrics, current_lr))
        rows.append(metric_row(epoch, "valid", valid_metrics, current_lr))
        write_metrics(metrics_path, rows)
        logger.info(
            "[Epoch %d/%d] lr=%.2e | train LL=%.6f | valid LL=%.6f "
            "(time=%.6f, mark=%.6f), RMSE=%.6f, accuracy=%.6f",
            epoch,
            args.epochs,
            current_lr,
            train_metrics["loglike"],
            valid_metrics["loglike"],
            valid_metrics.get("time_loglike", float("nan")),
            valid_metrics.get("mark_loglike", float("nan")),
            valid_metrics["rmse"],
            valid_metrics["acc"],
        )
        if "target_51_accuracy" in valid_metrics:
            logger.info(
                "[Epoch %d/%d] MIMIC target accuracy: predicate 51=%.6f, "
                "predicate 61=%.6f, combined=%.6f",
                epoch,
                args.epochs,
                valid_metrics["target_51_accuracy"],
                valid_metrics.get("target_61_accuracy", float("nan")),
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
                epoch,
                stale_epochs,
                args.selection_metric,
            )
            break

    if best_epoch is None:
        raise RuntimeError("Training produced no validation checkpoint")

    runner.model_wrapper.restore(str(checkpoint_path))
    test_metrics = deterministic_validation(runner, test_loader, args.seed + 200000)
    write_event_predictions(paths["output"] / "predictions.jsonl.gz", test_metrics)
    runner.model_wrapper.close_summary()
    current_lr = runner.model_wrapper.opt.param_groups[0]["lr"]
    test_row = metric_row(best_epoch, "test", test_metrics, current_lr)
    test_row["SelectionMetric"] = "validation_{}".format(args.selection_metric)
    test_path = paths["log"] / (paths["run_name"] + "_test.csv")
    fields = (
        "Epoch", "Split", "SelectionMetric", "Log-likelihood",
        "Time Log-likelihood", "Mark Log-likelihood", "RMSE",
        "Accuracy", "NumEvents", "LearningRate", "Target 51 Accuracy",
        "Target 61 Accuracy"
    )
    with test_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(test_row)
    logger.info(
        "Final test from best epoch %d: LL=%.6f (time=%.6f, mark=%.6f), "
        "RMSE=%.6f, accuracy=%.6f",
        best_epoch,
        test_metrics["loglike"],
        test_metrics.get("time_loglike", float("nan")),
        test_metrics.get("mark_loglike", float("nan")),
        test_metrics["rmse"],
        test_metrics["acc"],
    )
    if "target_51_accuracy" in test_metrics:
        logger.info(
            "Final MIMIC targets: predicate 51=%.6f, predicate 61=%.6f, "
            "combined=%.6f",
            test_metrics["target_51_accuracy"],
            test_metrics.get("target_61_accuracy", float("nan")),
            test_metrics["acc"],
        )
    persistent_checkpoint = paths["output"] / "checkpoint" / "best.pt"
    persistent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, persistent_checkpoint)
    return metrics_path, test_row


def compute_baselines(adapted_dir, num_event_types):
    train_records = load_records(adapted_dir / "train.json")
    test_records = load_records(adapted_dir / "test.json")
    if train_records and "type_loss_mask" in train_records[0]:
        return compute_mimic_baselines(
            train_records, test_records, num_event_types
        )
    train_dt = observed_intervals(train_records)
    test_dt = observed_intervals(test_records)
    mark_counts = np.zeros(num_event_types, dtype=np.int64)
    transition_counts = np.zeros((num_event_types, num_event_types), dtype=np.int64)
    for record in train_records:
        marks = record["type_event"]
        for previous_mark, next_mark in zip(marks[:-1], marks[1:]):
            previous_mark, next_mark = int(previous_mark), int(next_mark)
            mark_counts[next_mark] += 1
            transition_counts[previous_mark, next_mark] += 1
    majority_mark = int(np.argmax(mark_counts))
    transition_prediction = np.argmax(transition_counts, axis=1)
    empty_rows = transition_counts.sum(axis=1) == 0
    transition_prediction[empty_rows] = majority_mark
    test_marks = []
    transition_correct = 0
    previous_correct = 0
    for record in test_records:
        marks = record["type_event"]
        for previous_mark, next_mark in zip(marks[:-1], marks[1:]):
            previous_mark, next_mark = int(previous_mark), int(next_mark)
            test_marks.append(next_mark)
            transition_correct += int(
                transition_prediction[previous_mark] == next_mark
            )
            previous_correct += int(previous_mark == next_mark)
    test_marks = np.asarray(test_marks, dtype=int)
    num_test_pairs = len(test_marks)

    train_mean = float(np.mean(train_dt))
    train_median = float(np.median(train_dt))
    total_rate = float(len(train_dt) / max(np.sum(train_dt), 1e-12))
    mark_prob = (mark_counts + 1e-12) / (mark_counts.sum() + 1e-12 * num_event_types)
    poisson_ll = np.log(total_rate) + np.log(mark_prob[test_marks]) - total_rate * test_dt
    return {
        "num_test_intervals": int(len(test_dt)),
        "uniform_mark_accuracy": float(1.0 / num_event_types),
        "majority_mark": majority_mark,
        "majority_mark_accuracy": float(np.mean(test_marks == majority_mark)),
        "repeat_previous_mark_accuracy": float(previous_correct / num_test_pairs),
        "first_order_transition_accuracy": float(transition_correct / num_test_pairs),
        "train_mean_time": train_mean,
        "train_mean_time_test_rmse": float(np.sqrt(np.mean((test_dt - train_mean) ** 2))),
        "train_median_time": train_median,
        "train_median_time_test_rmse": float(np.sqrt(np.mean((test_dt - train_median) ** 2))),
        "homogeneous_marked_poisson_test_loglike": float(np.mean(poisson_ll)),
    }


def compute_mimic_baselines(train_records, test_records, num_event_types):
    def selected_times(records):
        return np.asarray([
            float(record["time_since_last_event"][index])
            for record in records
            for index in range(1, len(record["type_event"]))
            if record["time_loss_mask"][index]
        ], dtype=np.float64)

    def selected_targets(records):
        return [
            int(record["type_event"][index])
            for record in records
            for index in range(1, len(record["type_event"]))
            if record["type_loss_mask"][index]
        ]

    train_dt = selected_times(train_records)
    test_dt = selected_times(test_records)
    train_targets = selected_targets(train_records)
    test_targets = selected_targets(test_records)
    target_predicates = sorted(set(mark // 2 for mark in train_targets))
    majority_state = {}
    for predicate in target_predicates:
        states = [mark % 2 for mark in train_targets if mark // 2 == predicate]
        majority_state[predicate] = int(np.mean(states) >= 0.5)
    majority_accuracy = np.mean([
        majority_state[mark // 2] == mark % 2 for mark in test_targets
    ])

    previous_correct = []
    for record in test_records:
        latest_state = {}
        for index, mark in enumerate(record["type_event"]):
            predicate, state = int(mark) // 2, int(mark) % 2
            if record["type_loss_mask"][index]:
                prediction = latest_state.get(
                    predicate, majority_state[predicate]
                )
                previous_correct.append(prediction == state)
            latest_state[predicate] = state

    all_train_dt = observed_intervals(train_records)
    all_test_dt = observed_intervals(test_records)
    train_marks = np.asarray([
        int(record["type_event"][index])
        for record in train_records
        for index in range(1, len(record["type_event"]))
    ], dtype=int)
    test_marks = np.asarray([
        int(record["type_event"][index])
        for record in test_records
        for index in range(1, len(record["type_event"]))
    ], dtype=int)
    mark_counts = np.bincount(train_marks, minlength=num_event_types)
    total_rate = float(len(all_train_dt) / max(np.sum(all_train_dt), 1e-12))
    mark_prob = (mark_counts + 1e-12) / (
        mark_counts.sum() + 1e-12 * num_event_types
    )
    poisson_ll = (
        np.log(total_rate) + np.log(mark_prob[test_marks])
        - total_rate * all_test_dt
    )
    train_mean = float(np.mean(train_dt))
    train_median = float(np.median(train_dt))
    return {
        "num_test_intervals": int(len(test_dt)),
        "num_test_target_labels": int(len(test_targets)),
        "uniform_mark_accuracy": 0.5,
        "majority_mark": majority_state,
        "majority_mark_accuracy": float(majority_accuracy),
        "repeat_previous_mark_accuracy": float(np.mean(previous_correct)),
        "first_order_transition_accuracy": float(np.mean(previous_correct)),
        "train_mean_time": train_mean,
        "train_mean_time_test_rmse": float(np.sqrt(np.mean(
            (test_dt - train_mean) ** 2
        ))),
        "train_median_time": train_median,
        "train_median_time_test_rmse": float(np.sqrt(np.mean(
            (test_dt - train_median) ** 2
        ))),
        "homogeneous_marked_poisson_test_loglike": float(np.mean(poisson_ll)),
    }


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_diagnostic_summary(paths, test_row, baselines, num_event_types):
    model_ll = float(test_row["Log-likelihood"])
    model_rmse = float(test_row["RMSE"])
    model_accuracy = float(test_row["Accuracy"])
    mark_ll = float(test_row["Mark Log-likelihood"])
    summary = {
        "test": test_row,
        "comparisons": {
            "loglike_gain_over_poisson": model_ll - baselines[
                "homogeneous_marked_poisson_test_loglike"
            ],
            "rmse_reduction_vs_train_mean": baselines[
                "train_mean_time_test_rmse"
            ] - model_rmse,
            "accuracy_gain_over_majority": model_accuracy - baselines[
                "majority_mark_accuracy"
            ],
            "accuracy_gain_over_first_order_transition": model_accuracy - baselines[
                "first_order_transition_accuracy"
            ],
            "mark_loglike_gain_over_uniform": mark_ll + float(np.log(num_event_types)),
        },
        "passes": {
            "better_loglike_than_poisson": model_ll > baselines[
                "homogeneous_marked_poisson_test_loglike"
            ],
            "better_rmse_than_train_mean": model_rmse < baselines[
                "train_mean_time_test_rmse"
            ],
            "better_accuracy_than_majority": model_accuracy > baselines[
                "majority_mark_accuracy"
            ],
            "better_accuracy_than_transition": model_accuracy > baselines[
                "first_order_transition_accuracy"
            ],
            "better_mark_loglike_than_uniform": mark_ll > -float(np.log(num_event_types)),
        },
    }
    write_json(paths["log"] / (paths["run_name"] + "_diagnostic.json"), summary)


def plot_metrics(paths, metrics_path, test_row, baselines):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required for the final plot") from exc

    epochs, likelihood, rmse, accuracy = [], [], [], []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Split"] != "valid":
                continue
            epochs.append(int(row["Epoch"]))
            likelihood.append(float(row["Log-likelihood"]))
            rmse.append(float(row["RMSE"]))
            accuracy.append(float(row["Accuracy"]))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = (
        (likelihood, "Validation Log-likelihood", "Log-likelihood per event", "#2563EB"),
        (rmse, "Validation Time RMSE", "RMSE", "#EA580C"),
        (accuracy, "Validation Event Accuracy", "Accuracy", "#059669"),
    )
    for axis, (values, title, ylabel, color) in zip(axes, panels):
        axis.plot(epochs, values, color=color, linewidth=2.1)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].axhline(
        baselines["homogeneous_marked_poisson_test_loglike"],
        color="#64748B", linestyle="--", linewidth=1.2, label="Poisson baseline"
    )
    axes[1].axhline(
        baselines["train_mean_time_test_rmse"],
        color="#64748B", linestyle="--", linewidth=1.2, label="Mean-time baseline"
    )
    axes[2].axhline(
        baselines["majority_mark_accuracy"],
        color="#64748B", linestyle="--", linewidth=1.2, label="Majority baseline"
    )
    axes[2].axhline(
        baselines["first_order_transition_accuracy"],
        color="#7C3AED", linestyle=":", linewidth=1.2, label="Transition baseline"
    )
    for axis in axes:
        axis.legend(loc="best", fontsize=8, frameon=False)
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    figure.suptitle(
        "FullyNN on {} | test at best epoch {}: LL={:.4f}, RMSE={:.4f}, "
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
    plot_path = paths["plot"] / (paths["run_name"] + "_curves.png")
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved plot to %s", plot_path)


def create_archive(paths):
    with tarfile.open(paths["archive"], "w:gz") as archive:
        for path in sorted(paths["output"].rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(paths["output"])
            archive.add(
                path, arcname=str(Path(paths["output"].name) / relative)
            )
    logger.info("Saved archive to %s", paths["archive"])


def main():
    args = parse_args()
    if args.selection_metric is None:
        args.selection_metric = "acc" if args.dataset == "mimic" else "loglike"
    positive = (
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.num_layers,
        args.num_mlp_layers,
        args.history_window,
        args.thinning_num_sample,
        args.thinning_num_exp,
        args.early_stop_patience,
        args.lr_patience,
    )
    if any(value < 1 for value in positive):
        raise ValueError("epoch, batch/model sizes, patience, and thinning sizes must be positive")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr-factor must be between 0 and 1")
    if args.learning_rate <= 0 or args.dtime_max <= 0 or args.min_delta < 0:
        raise ValueError("learning-rate/dtime-max must be positive and min-delta non-negative")

    paths = prepare_paths(args)
    console_log = paths["log"] / (paths["run_name"] + "_console.log")
    file_handler = logging.FileHandler(console_log, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    try:
        logger.info("Preparing %s with seed=%d", paths["run_name"], args.seed)
        if args.prepared_data_dir is not None:
            adapted_dir = args.prepared_data_dir.expanduser().resolve()
            for split in ("train", "dev", "test"):
                if not (adapted_dir / f"{split}.json").is_file():
                    raise FileNotFoundError(adapted_dir / f"{split}.json")
        else:
            adapter = DataConfiguration(seed=args.seed)
            private_root = paths["output"] / "prepared_data"
            if args.dataset == "dws":
                adapter.dws(output_dir=private_root, variants=[args.variant], history_window=args.history_window)
                adapted_dir = private_root / paths["run_name"]
            elif args.dataset == "amazon":
                adapted_dir = private_root / "amazon"
                adapter.amazon(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "mimic":
                adapted_dir = private_root / "mimic"
                adapter.mimic(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "taxi":
                adapted_dir = private_root / "taxi"
                adapter.taxi(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "taobao":
                adapted_dir = private_root / "taobao"
                adapter.taobao(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "retweet":
                adapted_dir = private_root / "retweet"
                adapter.retweet(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "mobike":
                adapted_dir = private_root / "mobike"
                adapter.mobike(output_dir=adapted_dir, history_window=args.history_window)
            elif args.dataset == "stackoverflow":
                adapted_dir = private_root / "stackoverflow"
                adapter.stackoverflow(output_dir=adapted_dir, history_window=args.history_window)
            else:
                adapted_dir = private_root / "covid_policy_tracker"
                adapter.covid_policy_tracker(output_dir=adapted_dir, history_window=args.history_window)
        num_event_types = read_num_event_types(adapted_dir / "train.json")
        training_stats = compute_training_stats(adapted_dir / "train.json")
        baselines = compute_baselines(adapted_dir, num_event_types)
        write_json(paths["log"] / (paths["run_name"] + "_training_stats.json"), training_stats)
        write_json(paths["log"] / (paths["run_name"] + "_baselines.json"), baselines)
        logger.info(
            "DWS time stats: median=%.6f, mean=%.6f, p99=%.6f, max=%.6f",
            training_stats["median"], training_stats["mean"],
            training_stats["p99"], training_stats["max"],
        )
        logger.info(
            "Baselines: Poisson LL=%.6f, mean-time RMSE=%.6f, majority acc=%.6f, "
            "transition acc=%.6f",
            baselines["homogeneous_marked_poisson_test_loglike"],
            baselines["train_mean_time_test_rmse"],
            baselines["majority_mark_accuracy"],
            baselines["first_order_transition_accuracy"],
        )
        config_path = write_config(
            args, paths, adapted_dir, num_event_types, training_stats
        )
        metrics_path, test_row = train_and_test(args, paths, config_path)
        plot_metrics(paths, metrics_path, test_row, baselines)
        write_diagnostic_summary(paths, test_row, baselines, num_event_types)
        logger.info("FullyNN %s experiment complete", paths["run_name"])
    finally:
        file_handler.flush()
        logger.removeHandler(file_handler)
        file_handler.close()
        shutil.rmtree(paths["scratch"], ignore_errors=True)

    create_archive(paths)

    print("Result directory: {}".format(paths["output"]))
    print("Archive: {}".format(paths["archive"]))


if __name__ == "__main__":
    main()
