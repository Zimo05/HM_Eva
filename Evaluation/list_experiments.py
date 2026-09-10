from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="List independently runnable evaluation jobs")
    parser.add_argument("--status", default=None)
    parser.add_argument("--registry", type=Path, default=ROOT / "experiment_registry.csv")
    args = parser.parse_args()
    with args.registry.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if args.status is not None and row["status"] != args.status:
            continue
        extra = row.get("arguments", "").strip()
        print(f"{row['job_id']}: python {row['script']} --seed {row['seed']} {extra}".rstrip())


if __name__ == "__main__":
    main()
