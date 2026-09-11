#!/usr/bin/env python3
"""Backward-compatible entry point for the former DWS-only RMTPP runner."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run RMTPP on one DWS variant.")
    parser.add_argument("--variant", choices=("13", "15", "17", "20"), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--exclude-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = "dws_{}".format(args.variant)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        ROOT / "Result" / "DWS" / "{}_training_results".format(dataset)
    )
    archive = Path(args.archive).expanduser().resolve() if args.archive else Path(
        str(output) + ".tar.gz"
    )
    command = [
        sys.executable,
        str(ROOT / "run_experiment.py"),
        "--dataset", "dws",
        "--variant", args.variant,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-size", str(args.hidden_size),
        "--mc-samples", str(args.mc_samples),
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
        "--output-dir", str(output),
        "--archive", str(archive),
    ]
    if args.overwrite:
        command.append("--overwrite")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
