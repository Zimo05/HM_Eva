#!/usr/bin/env python3
"""Plot one clean line per THP development metric.

Defaults match the Covid-Policy-Tracker training command documented for this
repository.  Repeated epoch rows are deduplicated before plotting, so an old
CSV accidentally containing multiple runs will not produce overlapping lines.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean


THP_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else THP_ROOT / path


def read_dev_metrics(path):
    by_epoch = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(
                csv.DictReader(handle, skipinitialspace=True), start=2):
            try:
                epoch = int(row["Epoch"])
                by_epoch[epoch] = {
                    "ll": float(row["Log-likelihood"]),
                    "accuracy": float(row["Accuracy"]),
                    "rmse": float(row["RMSE"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid development metrics at {} line {}: {}".format(
                        path, row_number, exc
                    )
                ) from exc
    if not by_epoch:
        raise ValueError("{} contains no development metric rows".format(path))
    epochs = sorted(by_epoch)
    return epochs, {
        name: [by_epoch[epoch][name] for epoch in epochs]
        for name in ("ll", "accuracy", "rmse")
    }


def read_test_metrics(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, skipinitialspace=True), None)
    if row is None:
        return None
    try:
        return {
            "epoch": int(row["BestEpoch"]),
            "selection_metric": row["SelectionMetric"].strip(),
            "ll": float(row["Log-likelihood"]),
            "accuracy": float(row["Accuracy"]),
            "rmse": float(row["RMSE"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid final test metrics in {}: {}".format(path, exc)) from exc


def moving_average(values, window):
    if window == 1:
        return list(values)
    return [mean(values[max(0, index - window + 1):index + 1])
            for index in range(len(values))]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot clean THP development curves with one line per metric."
    )
    parser.add_argument(
        "--metrics", default="logs/covid_policy_tracker_dev_metrics.csv",
        help="Development CSV written by Main.py, relative to Models/THP by default.",
    )
    parser.add_argument(
        "--test-metrics", default="logs/covid_policy_tracker_test.csv",
        help="Optional final-test CSV used only for the figure subtitle.",
    )
    parser.add_argument(
        "--output", default="Result/covid_policy_tracker_curves.png",
        help="Output PNG, relative to Models/THP by default.",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=1,
        help="Trailing moving-average window. Only the resulting single line is drawn.",
    )
    parser.add_argument(
        "--title", default="THP on Covid-Policy-Tracker",
        help="Figure title.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1")

    metrics_path = resolve_path(args.metrics)
    test_path = resolve_path(args.test_metrics)
    output_path = resolve_path(args.output)
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))

    epochs, metrics = read_dev_metrics(metrics_path)
    test = read_test_metrics(test_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it in the THP environment first."
        ) from exc

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": "#B7C0CA",
        "axes.linewidth": 0.8,
        "grid.color": "#D8DEE6",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
    })

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = (
        ("ll", "Development Log-likelihood", "Log-likelihood", "#2563EB"),
        ("accuracy", "Development Event Accuracy", "Accuracy", "#059669"),
        ("rmse", "Development Time RMSE", "RMSE (days)", "#EA580C"),
    )
    for axis, (name, title, ylabel, color) in zip(axes, panels):
        values = moving_average(metrics[name], args.smooth_window)
        axis.plot(epochs, values, color=color, linewidth=2.2)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True)
        axis.margins(x=0.01)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    subtitle = "Development metrics only"
    if test is not None:
        subtitle = (
            "Final test from best epoch {epoch} ({metric}): "
            "LL={ll:.4f}, Accuracy={accuracy:.2%}, RMSE={rmse:.4f} days"
        ).format(metric=test["selection_metric"], **test)
    if args.smooth_window > 1:
        subtitle += " | single {}-epoch moving-average line".format(
            args.smooth_window
        )
    figure.suptitle(args.title + "\n" + subtitle, fontsize=15, fontweight="semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print("Saved plot: {}".format(output_path))


if __name__ == "__main__":
    main()
