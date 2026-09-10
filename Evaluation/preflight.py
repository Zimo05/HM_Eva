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
    required = [Path(__file__).resolve().parents[1] / "Models", Path(__file__).resolve().parents[1] / "Datasets"]
    errors.extend(f"missing project directory: {path}" for path in required if not path.is_dir())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Evaluation preflight passed")


if __name__ == "__main__":
    main()
