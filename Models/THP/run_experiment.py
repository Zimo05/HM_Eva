#!/usr/bin/env python3
"""Run one complete Transformer Hawkes Process baseline experiment.

The experiment adapts a named dataset, trains THP, evaluates the checkpoint
selected on validation data, plots train/validation likelihood, RMSE and
accuracy, and creates a compact archive containing only log/, csv/, and plot/.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path


THP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = THP_ROOT.parents[1]
SUPPORTED_DATASETS = (
    "amazon",
    "retweet",
    "taxi",
    "stackoverflow",
    "taobao",
    "mobike",
    "mimic",
    "covid_policy_tracker",
    "dws_8",
    "dws_10",
    "dws_13",
    "dws_15",
    "dws_17",
    "dws_20",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train THP, plot train/validation likelihood, RMSE and accuracy, "
            "evaluate test data, then package only log/csv/plot as tar.gz."
        )
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional display name for plots/logs when using prepared data.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-rnn", type=int, default=128)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--integral-method", choices=("trapezoid", "mc"), default="trapezoid"
    )
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--event-loss-weight", type=float, default=1.0)
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--time-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric", choices=("ll", "accuracy", "rmse"), default="ll"
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, or cuda:N. auto uses CUDA when available.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepared-data-dir", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    positive_integers = (
        args.epochs,
        args.batch_size,
        args.d_model,
        args.d_rnn,
        args.d_inner,
        args.d_k,
        args.d_v,
        args.num_heads,
        args.num_layers,
        args.mc_samples,
    )
    if any(value < 1 for value in positive_integers):
        raise ValueError("epoch, batch, model and sampling sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if min(
            args.event_loss_weight,
            args.type_loss_weight,
            args.time_loss_weight,
            args.grad_clip) < 0:
        raise ValueError("loss weights and --grad-clip cannot be negative")


def prepare_paths(args):
    output = Path(args.output_dir).expanduser().resolve()
    archive = Path(args.archive).expanduser().resolve()
    if archive.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("--archive must end with .tar.gz")
    if output in (THP_ROOT, PROJECT_ROOT):
        raise ValueError("--output-dir must be a dedicated result directory")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                "{} already contains files; pass --overwrite".format(output)
            )
        shutil.rmtree(output)
    if archive.exists():
        if not args.overwrite:
            raise FileExistsError(
                "{} already exists; pass --overwrite".format(archive)
            )
        archive.unlink()

    paths = {
        "output": output,
        "archive": archive,
        "log": output / "log",
        "csv": output / "csv",
        "plot": output / "plot",
        "scratch": Path(tempfile.mkdtemp(prefix="thp_experiment_")),
    }
    for name in ("log", "csv", "plot"):
        paths[name].mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    return paths


class ConsoleLog:
    def __init__(self, path):
        self.handle = path.open("w", encoding="utf-8")

    def write(self, message):
        sys.stdout.write(message)
        sys.stdout.flush()
        self.handle.write(message)
        self.handle.flush()

    def line(self, message):
        self.write(message.rstrip("\n") + "\n")

    def close(self):
        self.handle.close()


def adapt_dataset(dataset, seed, output_root):
    sys.path.insert(0, str(THP_ROOT))
    from data_configuration import DataConfiguration

    adapter = DataConfiguration(seed=seed)
    if dataset.startswith("dws_"):
        variant = dataset.split("_", 1)[1]
        adapter.dws(output_dir=output_root, variants=[variant])
        adapted_dir = output_root / dataset
    else:
        adapted_dir = output_root / dataset
        getattr(adapter, dataset)(output_dir=adapted_dir)

    missing = [
        adapted_dir / "{}.pkl".format(split)
        for split in ("train", "dev", "test")
        if not (adapted_dir / "{}.pkl".format(split)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "data adaptation did not create: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return adapted_dir


def resolve_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to train THP") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def training_command(args, paths, adapted_dir, device):
    command = [
        sys.executable,
        str(THP_ROOT / "Main.py"),
        "-data", str(adapted_dir),
        "-epoch", str(args.epochs),
        "-batch_size", str(args.batch_size),
        "-num_workers", str(args.num_workers),
        "-d_model", str(args.d_model),
        "-d_rnn", str(args.d_rnn),
        "-d_inner_hid", str(args.d_inner),
        "-d_k", str(args.d_k),
        "-d_v", str(args.d_v),
        "-n_head", str(args.num_heads),
        "-n_layers", str(args.num_layers),
        "-dropout", str(args.dropout),
        "-lr", str(args.learning_rate),
        "-smooth", str(args.label_smoothing),
        "-seed", str(args.seed),
        "-device", device,
        "-integral_method", args.integral_method,
        "-mc_samples", str(args.mc_samples),
        "-event_loss_weight", str(args.event_loss_weight),
        "-type_loss_weight", str(args.type_loss_weight),
        "-time_loss_weight", str(args.time_loss_weight),
        "-grad_clip", str(args.grad_clip),
        "-weight_decay", str(args.weight_decay),
        "-selection_metric", args.selection_metric,
        "-log", str(paths["csv"] / "epoch_metrics.csv"),
        "-test_log", str(paths["csv"] / "test_metrics.csv"),
        "-prediction_log", str(paths["output"] / "predictions.jsonl.gz"),
        "-save", str(paths["scratch"] / "best_model.pt"),
    ]
    if args.initial_checkpoint is not None:
        command += ["-load", str(args.initial_checkpoint)]
    if args.evaluate_only:
        command += ["-evaluate_only"]
    return command


def run_training(command, console):
    console.line("Training command: {}".format(" ".join(command)))
    process = subprocess.Popen(
        command,
        cwd=str(THP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        console.write(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def read_epoch_metrics(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append({
                    "epoch": int(row["Epoch"]),
                    "split": row["Split"].strip().lower(),
                    "likelihood": float(row["Log-likelihood"]),
                    "rmse": float(row["RMSE"]),
                    "accuracy": float(row["Accuracy"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid epoch metrics at {} line {}: {}".format(
                        path, row_number, exc
                    )
                ) from exc
    expected = {"train", "validation"}
    if not rows or {row["split"] for row in rows} != expected:
        raise ValueError("epoch metrics must contain train and validation rows")
    return rows


def read_test_metrics(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, skipinitialspace=True), None)
    if row is None:
        raise ValueError("{} contains no test metrics".format(path))
    return {
        "epoch": int(row["BestEpoch"]),
        "selection_metric": row["SelectionMetric"].strip(),
        "likelihood": float(row["Log-likelihood"]),
        "rmse": float(row["RMSE"]),
        "accuracy": float(row["Accuracy"]),
    }


def plot_metrics(rows, test_row, paths, dataset):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thp_matplotlib_cache")
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit("matplotlib is required to draw THP curves") from exc

    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation")
    }
    panels = (
        ("likelihood", "Log-likelihood", "Log-likelihood per event", "likelihood.png"),
        ("rmse", "Time RMSE", "RMSE", "rmse.png"),
        ("accuracy", "Event Accuracy", "Accuracy", "accuracy.png"),
    )
    colors = {"train": "#2563EB", "validation": "#EA580C"}

    def draw(axis, metric, title, ylabel):
        for split in ("train", "validation"):
            split_rows = by_split[split]
            axis.plot(
                [row["epoch"] for row in split_rows],
                [row[metric] for row in split_rows],
                color=colors[split],
                linewidth=2.0,
                label="Train" if split == "train" else "Validation",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False)
        if metric == "accuracy":
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    for metric, title, ylabel, filename in panels:
        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        draw(axis, metric, "THP on {} - {}".format(dataset, title), ylabel)
        figure.savefig(paths["plot"] / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for axis, (metric, title, ylabel, _) in zip(axes, panels):
        draw(axis, metric, title, ylabel)
    figure.suptitle(
        "THP on {} | test from best epoch {} ({}): LL={:.4f}, "
        "RMSE={:.4f}, Accuracy={:.2%}".format(
            dataset,
            test_row["epoch"],
            test_row["selection_metric"],
            test_row["likelihood"],
            test_row["rmse"],
            test_row["accuracy"],
        ),
        fontsize=13,
    )
    figure.savefig(paths["plot"] / "all_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def create_archive(paths):
    """Package exactly the requested log, csv, and plot directories."""
    with tarfile.open(paths["archive"], "w:gz") as archive:
        for folder_name in ("log", "csv", "plot"):
            folder = paths[folder_name]
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=str(
                            Path(paths["output"].name)
                            / folder_name
                            / path.relative_to(folder)
                        ),
                    )


def main():
    args = parse_args()
    validate_args(args)
    paths = prepare_paths(args)
    console = ConsoleLog(paths["log"] / "console.log")
    success = False
    try:
        device = resolve_device(args.device)
        dataset_label = args.dataset_label or args.dataset
        console.line("THP dataset: {}".format(dataset_label))
        console.line("Device: {}".format(device))
        console.line("Adapting dataset with seed={}".format(args.seed))
        adapted_dir = (
            args.prepared_data_dir.expanduser().resolve()
            if args.prepared_data_dir is not None
            else adapt_dataset(
                args.dataset, args.seed, paths["output"] / "prepared_data"
            )
        )
        command = training_command(args, paths, adapted_dir, device)
        config = vars(args).copy()
        config.update({
            "resolved_device": device,
            "adapted_data": str(adapted_dir),
            "python": sys.executable,
            "training_command": command,
        })
        with (paths["log"] / "run_config.json").open(
                "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")

        run_training(command, console)
        rows = (
            [] if args.evaluate_only
            else read_epoch_metrics(paths["csv"] / "epoch_metrics.csv")
        )
        test_row = read_test_metrics(paths["csv"] / "test_metrics.csv")
        if not args.evaluate_only:
            checkpoint_dir = paths["output"] / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                paths["scratch"] / "best_model.pt",
                checkpoint_dir / "best.pt",
            )
            plot_metrics(rows, test_row, paths, dataset_label)
            console.line(
                "Saved likelihood, RMSE and accuracy plots for train/validation"
            )
        console.line("Archive contains only log/, csv/, and plot/")
        success = True
    except Exception:
        console.line(traceback.format_exc())
        shutil.rmtree(paths["scratch"], ignore_errors=True)
        raise
    finally:
        console.close()

    try:
        if success:
            create_archive(paths)
    finally:
        shutil.rmtree(paths["scratch"], ignore_errors=True)

    print("Result directory: {}".format(paths["output"]))
    print("Archive: {}".format(paths["archive"]))


if __name__ == "__main__":
    main()
