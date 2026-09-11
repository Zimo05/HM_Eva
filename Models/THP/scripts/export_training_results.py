#!/usr/bin/env python3
"""Export a reproducible THP experiment bundle as a timestamped tar.gz."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path


THP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = THP_ROOT.parent.parent


def resolve_thp_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else THP_ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_display_path(path):
    for root in (PROJECT_ROOT, THP_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def unique_output_path(path):
    if not path.exists():
        return path
    stem = path.name[:-len(".tar.gz")] if path.name.endswith(".tar.gz") else path.stem
    suffix = ".tar.gz" if path.name.endswith(".tar.gz") else path.suffix
    index = 2
    while True:
        candidate = path.with_name("{}_{}{}".format(stem, index, suffix))
        if not candidate.exists():
            return candidate
        index += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package THP logs, plot, checkpoint, adapted data, code, and documentation."
    )
    parser.add_argument("--run-name", default="covid_policy_tracker")
    parser.add_argument(
        "--dev-metrics", default="logs/covid_policy_tracker_dev_metrics.csv"
    )
    parser.add_argument("--test-metrics", default="logs/covid_policy_tracker_test.csv")
    parser.add_argument("--console-log", default="logs/covid_policy_tracker_console.log")
    parser.add_argument("--checkpoint", default="logs/covid_policy_tracker_best.pt")
    parser.add_argument("--pid", default="logs/covid_policy_tracker.pid")
    parser.add_argument("--plot", default="Result/covid_policy_tracker_curves.png")
    parser.add_argument(
        "--data-dir", default="data_adapted/covid_policy_tracker"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional output .tar.gz. Default: Result/<run>_training_results_<timestamp>.tar.gz",
    )
    parser.add_argument(
        "--exclude-checkpoint", action="store_true",
        help="Create a smaller archive without the best .pt checkpoint.",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Skip missing files instead of stopping. Not recommended for final archives.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    output = (
        resolve_thp_path(args.output)
        if args.output
        else THP_ROOT / "Result" / (
            "{}_training_results_{}.tar.gz".format(args.run_name, timestamp)
        )
    )
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("--output must end with .tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = unique_output_path(output)

    requested = [
        (resolve_thp_path(args.dev_metrics), "logs/" + Path(args.dev_metrics).name, True),
        (resolve_thp_path(args.test_metrics), "logs/" + Path(args.test_metrics).name, True),
        (resolve_thp_path(args.console_log), "logs/" + Path(args.console_log).name, True),
        (resolve_thp_path(args.plot), "plots/" + Path(args.plot).name, True),
        (resolve_thp_path(args.pid), "logs/" + Path(args.pid).name, False),
    ]
    if not args.exclude_checkpoint:
        requested.append((
            resolve_thp_path(args.checkpoint),
            "logs/" + Path(args.checkpoint).name,
            True,
        ))

    data_dir = resolve_thp_path(args.data_dir)
    for split in ("train.pkl", "dev.pkl", "test.pkl"):
        requested.append((
            data_dir / split,
            "data_adapted/{}/{}".format(data_dir.name, split),
            True,
        ))

    code_files = [
        THP_ROOT / "Main.py",
        THP_ROOT / "Utils.py",
        THP_ROOT / "data_configuration.py",
        THP_ROOT / "preprocess" / "Dataset.py",
        PROJECT_ROOT / "_data_configuration_common.py",
    ]
    for path in code_files:
        requested.append((path, "code/" + path.name, True))

    documentation = (
        PROJECT_ROOT / "Datasets" / "Covid-Policy-Tracker"
        / "README_THP_DATA_PREPARATION.md"
    )
    requested.append((
        documentation,
        "documentation/README_THP_DATA_PREPARATION.md",
        True,
    ))
    for script_name in ("plot_training_curves.py", "export_training_results.py"):
        requested.append((
            THP_ROOT / "scripts" / script_name,
            "scripts/" + script_name,
            True,
        ))

    missing = [
        path for path, _, required in requested
        if required and not path.is_file()
    ]
    if missing and not args.allow_missing:
        formatted = "\n".join("  - {}".format(path) for path in missing)
        raise FileNotFoundError(
            "Required result files are missing:\n{}\n"
            "Run plotting after training, or pass --allow-missing for a partial archive."
            .format(formatted)
        )

    archive_root_name = "{}_training_results".format(args.run_name)
    manifest_entries = []
    with tempfile.TemporaryDirectory(prefix="thp_export_") as temporary:
        staging_root = Path(temporary) / archive_root_name
        staging_root.mkdir(parents=True)

        for source, relative, _ in requested:
            if not source.is_file():
                continue
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest_entries.append({
                "archive_path": relative,
                "source": source_display_path(source),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            })

        manifest = {
            "run_name": args.run_name,
            "generated_at": datetime.now().astimezone().isoformat(),
            "format": "THP reproducibility bundle",
            "checkpoint_included": not args.exclude_checkpoint,
            "partial_archive": bool(missing),
            "file_count": len(manifest_entries),
            "files": sorted(manifest_entries, key=lambda item: item["archive_path"]),
        }
        manifest_path = staging_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging_root, arcname=archive_root_name)

    print("Saved archive: {}".format(output))
    print("Included files: {}".format(len(manifest_entries) + 1))


if __name__ == "__main__":
    main()
