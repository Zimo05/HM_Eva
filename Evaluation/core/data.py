from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from _data_configuration_common import DWS_SPLIT_RATIOS, dws_split_indices, load_dws_file

from .cl_protocol import CLProtocol
from .io import sha256, write_json
from .paths import DATASETS_ROOT


def _dws_source_path(variant: str | None) -> Path:
    if variant is None:
        raise ValueError("DWS requires a tree variant")
    variant = str(variant)
    nested = DATASETS_ROOT / "DWS" / f"tree_{variant}" / f"hawkes_dataset_{variant}.csv"
    legacy = DATASETS_ROOT / "DWS" / f"hawkes_dataset_{variant}.csv"
    return nested if nested.is_file() else legacy


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


def _dws_model_record(record: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    """Return a baseline-facing record without DWS cluster metadata."""

    times = [float(value) for value in record["time_since_start"]]
    deltas = [float(value) for value in record["time_since_last_event"]]
    types = [int(value) for value in record["type_event"]]
    return {
        "dim_process": int(record["dim_process"]),
        "seq_idx": int(source_index),
        "seq_len": len(types),
        "time_since_start": times,
        "time_since_last_event": deltas,
        "type_event": types,
        "source_index": int(source_index),
    }


def _write_dws_native_views(
    output: Path,
    records: list[Mapping[str, Any]],
    split_indices: Mapping[str, list[int]],
    dim_process: int,
) -> None:
    """Write all standalone baseline schemas from the canonical source IDs."""

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        source_indices = sorted(int(value) for value in split_indices[split])
        base_records = [
            _dws_model_record(records[source_index], source_index)
            for source_index in source_indices
        ]
        for local_index, record in enumerate(base_records):
            record["seq_idx"] = local_index
        tpp_records = [
            dict(record, type_text=[f"event_{value}" for value in record["type_event"]])
            for record in base_records
        ]
        write_json(output / f"{split}.json", tpp_records)

        streams = []
        for record in base_records:
            streams.append([
                {
                    "time_since_start": time,
                    "time_since_last_event": delta,
                    "type_event": event_type,
                    "loss_mask": True,
                    "event_loss_mask": True,
                    "type_loss_mask": True,
                    "time_loss_mask": True,
                }
                for time, delta, event_type in zip(
                    record["time_since_start"],
                    record["time_since_last_event"],
                    record["type_event"],
                )
            ])
        with (output / f"{split}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "dim_process": int(dim_process),
                    split: streams,
                    "source_index_by_seq": source_indices,
                },
                handle,
            )

    write_json(
        output / "source_index_manifest.json",
        {
            "format_version": 1,
            "source_index_field": "source_index",
            "splits": {
                split: sorted(int(value) for value in split_indices[split])
                for split in ("train", "dev", "test")
            },
        },
    )


def prepare_hm_dataset(dataset: str, seed: int, work_dir: Path, variant: str | None = None) -> tuple[Path, Path]:
    """Create one model-facing CSV plus an immutable split manifest.

    For DWS, this function is the benchmark-construction boundary: it creates
    one cluster-stratified source-index split and derives every model-facing
    view from it.  The canonical CSV may retain cluster as an evaluation-only
    oracle column; all formal model views omit it and use source_index instead.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "canonical.csv"
    manifest_path = work_dir / "split_manifest.json"
    rows: list[dict[str, Any]] = []
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    dws_records: list[dict[str, Any]] | None = None
    dws_source_rows: list[dict[str, Any]] | None = None
    dws_split_indices_by_name: dict[str, list[int]] | None = None
    if dataset == "dws":
        source = _dws_source_path(variant)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            dws_source_rows = list(csv.DictReader(handle))
        if not dws_source_rows:
            raise ValueError(f"empty DWS source: {source}")
        _, dws_records = load_dws_file(source)
        dws_split_indices_by_name = dws_split_indices(dws_source_rows, seed=seed)
        splits = {
            "train": sorted(dws_split_indices_by_name["train"]),
            "validation": sorted(dws_split_indices_by_name["dev"]),
            "test": sorted(dws_split_indices_by_name["test"]),
        }
        rows = [
            {
                "event_times": json.dumps(record["time_since_start"], separators=(",", ":")),
                "event_types": json.dumps(record["type_event"], separators=(",", ":")),
                "source_index": int(record["seq_idx"]),
                "cluster": int(record["cluster"]),
            }
            for record in dws_records
        ]
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
        "source_index_field": "source_index" if dataset == "dws" else None,
    }
    if dataset == "dws":
        assert dws_records is not None
        assert dws_source_rows is not None
        assert dws_split_indices_by_name is not None
        manifest.update({
            "split_protocol": {
                "name": "DWS-cluster-stratified-v1",
                "ratios": dict(DWS_SPLIT_RATIOS),
                "seed": int(seed),
                "cluster_source_field": "cluster",
            },
            "source_data_path": str(source.resolve()),
            "source_data_sha256": sha256(source),
            "cluster_distribution": {
                split: {
                    str(cluster): sum(
                        1
                        for source_index in split_indices
                        if int(dws_source_rows[source_index]["cluster"]) == cluster
                    )
                    for cluster in sorted({
                        int(dws_source_rows[source_index]["cluster"])
                        for source_index in split_indices
                    })
                }
                for split, split_indices in (
                    ("train", splits["train"]),
                    ("validation", splits["validation"]),
                    ("test", splits["test"]),
                )
            },
        })
        _write_dws_native_views(
            work_dir,
            dws_records,
            dws_split_indices_by_name,
            max(int(record["dim_process"]) for record in dws_records),
        )
    write_json(manifest_path, manifest)
    return output, manifest_path


def prepare_continual_hm_dataset(
    data_root: Path,
    output: Path,
    task: int,
    protocol: CLProtocol | None = None,
) -> tuple[Path, Path]:
    """Stage one continual task for the current HM training CLI.

    The current HM trainer accepts one CSV and a split manifest.  Benchmark CL
    keeps train/validation/test in separate files, so this adapter combines
    the three declared splits while preserving their source rows in disjoint
    manifest partitions.  The trainer must not infer protocol membership from
    a synthetic cluster column.
    """

    data_root = Path(data_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    protocol = protocol or continual_protocol(data_root)
    split_paths = {
        "train": protocol.split_path(task, "train"),
        "validation": protocol.split_path(task, "validation"),
        "test": protocol.split_path(task, "test"),
    }
    rows: list[dict[str, str | int]] = []
    splits: dict[str, list[int]] = {name: [] for name in split_paths}
    source_counts: dict[str, int] = {}

    for split, source in split_paths.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {"event_times", "event_types"}
            missing = required - fields
            if missing:
                raise ValueError(
                    f"HM continual split {source} is missing {sorted(missing)}"
                )
            start = len(rows)
            for row in reader:
                event_times = str(row.get("event_times", "")).strip()
                event_types = str(row.get("event_types", "")).strip()
                if not event_times or not event_types:
                    raise ValueError(f"empty event sequence in {source}")
                rows.append({
                    "event_times": event_times,
                    "event_types": event_types,
                    "source_index": len(rows),
                })
                splits[split].append(len(rows) - 1)
            source_counts[split] = len(rows) - start

    if not rows or any(not values for values in splits.values()):
        raise ValueError(
            f"empty continual HM data or split for task {task}: {source_counts}"
        )

    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "canonical.csv"
    manifest_path = output / "split_manifest.json"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("event_times", "event_types", "source_index"),
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "format_version": 1,
        "seed": int(protocol.raw_manifest.get("seed", 0)),
        "dataset": f"{protocol.benchmark_id}/task_{int(task):02d}",
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "source_row_count": len(rows),
        "counts": {name: len(values) for name, values in splits.items()},
        "splits": splits,
        "source_splits": {
            name: str(path) for name, path in split_paths.items()
        },
        "source_index_field": "source_index",
    }
    write_json(manifest_path, manifest)
    return data_path, manifest_path


def dataset_inputs(dataset: str, variant: str | None = None) -> list[Path]:
    if dataset == "dws":
        return [_dws_source_path(variant)]
    return [DATASETS_ROOT / dataset / name for name in ("train.json", "dev.json", "test.json")]


def continual_root(value: Path | None = None) -> Path:
    root = value or DATASETS_ROOT / "CL" / "hm_continual_v2"
    return root.expanduser().resolve()


def benchmark_manifest_path(data_root: Path) -> Path:
    """Return the immutable protocol manifest for a continual benchmark."""

    return Path(data_root).expanduser().resolve() / "benchmark_manifest.json"


def continual_protocol(value: Path | None = None) -> CLProtocol:
    """Load the canonical continual benchmark protocol."""

    return CLProtocol.load(continual_root(value))


def load_benchmark_manifest(data_root: Path) -> dict[str, Any]:
    """Compatibility view backed by :class:`CLProtocol`."""

    return continual_protocol(data_root).raw_manifest


def benchmark_first_seen(
    manifest: Mapping[str, Any] | CLProtocol, *, persistent_only: bool = True
) -> dict[str, int]:
    """Return manifest first-seen tasks, optionally excluding diagnostics."""

    if isinstance(manifest, CLProtocol):
        regime_ids = (
            manifest.persistent_regimes if persistent_only else manifest.first_seen.keys()
        )
        return {
            str(regime_id): int(manifest.first_seen[regime_id])
            for regime_id in regime_ids
            if regime_id in manifest.first_seen
        }
    first_seen = manifest.get("first_seen", {})
    if not isinstance(first_seen, Mapping):
        raise ValueError("benchmark manifest first_seen must be an object")
    if persistent_only:
        regime_ids = manifest.get("persistent_regimes", ())
    else:
        regime_ids = first_seen.keys()
    return {str(regime_id): int(first_seen[regime_id]) for regime_id in regime_ids}


def _continual_split_path(
    data_root: Path,
    task: int,
    split: str,
    protocol: CLProtocol,
) -> Path:
    return protocol.split_path(task, split)


def _read_continual_csv(
    path: Path,
    event_dim: int,
    *,
    min_events: int = 2,
) -> list[dict[str, Any]]:
    if min_events < 1:
        raise ValueError("min_events must be positive")
    records = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            times = [float(value) for value in json.loads(row["event_times"])]
            types = [int(value) for value in json.loads(row["event_types"])]
            if len(times) != len(types) or len(times) < min_events:
                raise ValueError(f"invalid continual sequence in {path} row {index + 2}")
            if any(event_type < 0 or event_type >= event_dim for event_type in types):
                raise ValueError(
                    f"continual sequence in {path} row {index + 2} contains an "
                    f"event type outside [0, {event_dim - 1}]"
                )
            deltas = [times[0]] + [right - left for left, right in zip(times, times[1:])]
            records.append({
                "dim_process": event_dim,
                "seq_idx": index,
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": types,
            })
    if not records:
        raise ValueError(f"empty continual split: {path}")
    return records


def prepare_continual_baseline_dataset(
    model: str,
    data_root: Path,
    output: Path,
    train_tasks: list[int],
    eval_csv: Path | None = None,
    replay_csvs: list[Path] | None = None,
    protocol: CLProtocol | None = None,
    train_csvs: list[Path] | None = None,
    current_task: int | None = None,
    validation_csv: Path | None = None,
    validation_csvs: list[Path] | None = None,
    train_min_events: int = 2,
    validation_min_events: int = 2,
) -> Path:
    """Create private task data in each baseline's native on-disk schema."""
    if protocol is None:
        protocol = continual_protocol(data_root)
    event_dim = protocol.event_dim
    output.mkdir(parents=True, exist_ok=True)
    train = []
    if train_csvs is None:
        if not train_tasks:
            raise ValueError("train_tasks cannot be empty when train_csvs is omitted")
        train_paths = [
            _continual_split_path(data_root, task, "train", protocol)
            for task in train_tasks
        ]
    else:
        train_paths = [Path(path).expanduser().resolve() for path in train_csvs]
        if not train_paths:
            raise ValueError("train_csvs cannot be empty")
    for train_path in train_paths:
        train.extend(
            _read_continual_csv(
                train_path,
                event_dim,
                min_events=train_min_events,
            )
        )
    for replay_path in replay_csvs or []:
        train.extend(_read_continual_csv(replay_path, event_dim))
    if current_task is None:
        if not train_tasks:
            raise ValueError("current_task is required when train_tasks is empty")
        current = train_tasks[-1]
    else:
        current = int(current_task)
    if validation_csv is not None and validation_csvs is not None:
        raise ValueError("validation_csv and validation_csvs are mutually exclusive")
    if validation_csvs is None:
        validation_paths = [
            validation_csv
            or _continual_split_path(data_root, current, "val", protocol)
        ]
    else:
        validation_paths = [Path(path).expanduser().resolve() for path in validation_csvs]
        if not validation_paths:
            raise ValueError("validation_csvs cannot be empty")
    dev = []
    for validation_path in validation_paths:
        dev.extend(
            _read_continual_csv(
                validation_path,
                event_dim,
                min_events=validation_min_events,
            )
        )
    test = _read_continual_csv(
        eval_csv or _continual_split_path(
            data_root, current, "test", protocol
        ),
        event_dim,
    )
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
                pickle.dump({"dim_process": event_dim, split: streams}, handle)
    else:
        raise KeyError(model)
    return output
