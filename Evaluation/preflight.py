from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the evaluation runtime without changing data or models")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    errors = []
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ is required; found {sys.version.split()[0]}")
    for package in ("numpy", "pandas", "torch"):
        if importlib.util.find_spec(package) is None:
            errors.append(f"missing Python package: {package}")
    if args.require_cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA was requested but is unavailable")
        except ImportError:
            pass
    project_root = Path(__file__).resolve().parents[1]
    required_directories = [
        project_root / "Models",
        project_root / "Datasets",
        *(project_root / "Models" / name for name in (
            "RMTPP", "THP", "TPP-LLM", "FullyNN", "EasyTPP",
        )),
        project_root / "Models" / "HawkesMemory",
        project_root / "Models" / "HawkesMemory" / "Memory",
    ]
    errors.extend(
        f"missing project directory: {path}"
        for path in required_directories
        if not path.is_dir()
    )
    required_files = [
        project_root / "_data_configuration_common.py",
        project_root / "Models" / "RMTPP" / "run_experiment.py",
        project_root / "Models" / "THP" / "run_experiment.py",
        project_root / "Models" / "TPP-LLM" / "scripts" / "train_tpp_llm.py",
        project_root / "Models" / "FullyNN" / "run_experiment.py",
        project_root / "Models" / "HawkesMemory" / "Memory" / "Train" / "Train.py",
        project_root / "Models" / "HawkesMemory" / "Memory" / "Train" / "Inference.py",
        project_root / "Models" / "HawkesMemory" / "Memory" / "Evaluate.py",
        project_root / "Models" / "HawkesMemory" / "Memory" / "EvaluateCL.py",
    ]
    errors.extend(
        f"missing baseline entrypoint: {path}"
        for path in required_files
        if not path.is_file()
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Evaluation preflight passed")


if __name__ == "__main__":
    main()
