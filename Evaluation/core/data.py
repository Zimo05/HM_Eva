from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path
from typing import Any

from .io import sha256, write_json
from .paths import DATASETS_ROOT


def _read_standard(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _times_types(record: dict[str, Any]) -> tuple[list[float], list[int]]:
    times = record.get("time_since_start", record.get("event_times"))
    types = record.get("type_event", record.get("event_types"))
    if times is None or types is None or len(times) != len(types) or not times:
        raise ValueError("invalid sequence record")
    return [float(v) for v in times], [int(v) for v in types]


def prepare_hm_dataset(dataset: str, seed: int, work_dir: Path, variant: str | None = None) -> tuple[Path, Path]:
    """Create one model-facing CSV plus an immutable split manifest.

    Oracle columns may remain in the physical CSV for evaluation, but the
    Memory loader consumes only event_times/event_types/source_index.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "canonical.csv"
    manifest_path = work_dir / "split_manifest.json"
    rows: list[dict[str, Any]] = []
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    if dataset == "dws":
        source = DATASETS_ROOT / "DWS" / f"hawkes_dataset_{variant}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(str(row.get("cluster", "_")), []).append(index)
        rng = random.Random(seed)
        for indices in groups.values():
            rng.shuffle(indices)
            n = len(indices)
            n_test = max(1, round(0.2 * n))
            n_val = max(1, round(0.1 * n))
            splits["test"].extend(indices[:n_test])
            splits["validation"].extend(indices[n_test:n_test + n_val])
            splits["train"].extend(indices[n_test + n_val:])
    else:
        base = DATASETS_ROOT / dataset
        names = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
        for split, filename in names.items():
            for record in _read_standard(base / filename):
                times, types = _times_types(record)
                source_index = len(rows)
                rows.append({"event_times": json.dumps(times), "event_types": json.dumps(types)})
                splits[split].append(source_index)
    if not rows or any(not values for values in splits.values()):
        raise ValueError(f"empty data or split for {dataset}")
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for values in splits.values():
        values.sort()
    manifest = {
        "format_version": 1,
        "seed": seed,
        "dataset": dataset,
        "data_path": str(output.resolve()),
        "data_sha256": sha256(output),
        "source_row_count": len(rows),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "oracle_fields_hidden_from_model": ["cluster", "ground_truth", "regime_id"],
    }
    write_json(manifest_path, manifest)
    return output, manifest_path


def dataset_inputs(dataset: str, variant: str | None = None) -> list[Path]:
    if dataset == "dws":
        return [DATASETS_ROOT / "DWS" / f"hawkes_dataset_{variant}.csv"]
    return [DATASETS_ROOT / dataset / name for name in ("train.json", "dev.json", "test.json")]


def continual_root(value: Path | None = None) -> Path:
    root = value or DATASETS_ROOT / "Data" / "CL" / "hm_continual_v1"
    return root.expanduser().resolve()


def _read_continual_csv(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            times = [float(value) for value in json.loads(row["event_times"])]
            types = [int(value) for value in json.loads(row["event_types"])]
            if len(times) != len(types) or len(times) < 2:
                raise ValueError(f"invalid continual sequence in {path} row {index + 2}")
            deltas = [0.0] + [right - left for left, right in zip(times, times[1:])]
            records.append({
                "dim_process": 8,
                "seq_idx": index,
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": types,
            })
    if not records:
        raise ValueError(f"empty continual split: {path}")
    return records


def prepare_continual_baseline_dataset(
        model: str, data_root: Path, output: Path, train_tasks: list[int],
        eval_csv: Path | None = None, replay_csvs: list[Path] | None = None,
) -> Path:
    """Create private task data in each baseline's native on-disk schema."""
    output.mkdir(parents=True, exist_ok=True)
    train = []
    for task in train_tasks:
        train.extend(_read_continual_csv(data_root / f"task_{task:02d}" / "train.csv"))
    for replay_path in replay_csvs or []:
        train.extend(_read_continual_csv(replay_path))
    current = train_tasks[-1]
    dev = _read_continual_csv(data_root / f"task_{current:02d}" / "val.csv")
    test = _read_continual_csv(eval_csv or data_root / f"task_{current:02d}" / "test.csv")
    for split_records in (train, dev, test):
        for index, record in enumerate(split_records):
            record["seq_idx"] = index
    if model in {"RMTPP", "TPP_LLM"}:
        for split, records in (("train", train), ("dev", dev), ("test", test)):
            payload = records
            if model == "TPP_LLM":
                payload = [dict(record, type_text=[f"event_{value}" for value in record["type_event"]]) for record in records]
            write_json(output / f"{split}.json", payload)
    elif model == "THP":
        for split, records in (("train", train), ("dev", dev), ("test", test)):
            streams = []
            for record in records:
                streams.append([{
                    "time_since_start": time,
                    "time_since_last_event": delta,
                    "type_event": event_type,
                    "loss_mask": True,
                    "event_loss_mask": True,
                    "type_loss_mask": True,
                    "time_loss_mask": True,
                } for time, delta, event_type in zip(
                    record["time_since_start"],
                    record["time_since_last_event"],
                    record["type_event"],
                )])
            with (output / f"{split}.pkl").open("wb") as handle:
                pickle.dump({"dim_process": 8, split: streams}, handle)
    else:
        raise KeyError(model)
    return output
