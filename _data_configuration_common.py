"""Shared, standard-library-only helpers for the model data adapters."""

import csv
import json
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path


STANDARD_DATASETS = ("retweet", "taxi", "stackoverflow", "taobao", "amazon")
CL_TASK_PATTERN = re.compile(r"^task_(\d+)$")
DWS_SPLIT_RATIOS = {"train": 0.70, "dev": 0.10, "test": 0.20}


def infer_dataset_root(model_file):
    return Path(model_file).resolve().parents[2] / "Datasets"


def infer_output_root(model_file):
    return Path(model_file).resolve().parent / "data_adapted"


def infer_cl_dataset_root():
    """Return the repository's model-facing continual-learning data root."""
    return Path(__file__).resolve().parent / "Datasets" / "CL" / "hm_continual_v2"


def resolve_cl_dataset_root(dataset_root=None):
    """Resolve either ``Data/CL`` or ``Data/CL/Data`` to the task directory."""
    root = Path(dataset_root).expanduser().resolve() if dataset_root else infer_cl_dataset_root()
    if (root / "Data").is_dir() and not any(root.glob("task_*")):
        root = root / "Data"
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    return root


def normalize_cl_task_id(task_id):
    """Normalize 0, ``"0"`` and ``"task_00"`` to ``"task_00"``."""
    if isinstance(task_id, str):
        value = task_id.strip()
        match = CL_TASK_PATTERN.fullmatch(value)
        if match:
            value = match.group(1)
    else:
        value = task_id
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CL task_id must be an integer or task_XX") from exc
    if numeric < 0:
        raise ValueError("CL task_id cannot be negative")
    return "task_{:02d}".format(numeric)


def discover_cl_tasks(dataset_root=None):
    """Discover model-facing task directories without consulting oracle files."""
    root = resolve_cl_dataset_root(dataset_root)
    tasks = []
    for path in root.iterdir():
        match = CL_TASK_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            tasks.append((int(match.group(1)), path.name))
    if not tasks:
        raise FileNotFoundError("No task_XX directories found in {}".format(root))
    return [name for _, name in sorted(tasks)]


def _parse_json_list(value, field, source):
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("{} has invalid JSON in {}".format(source, field)) from exc
    if not isinstance(parsed, list):
        raise ValueError("{} {} must be a JSON list".format(source, field))
    return parsed


def _read_cl_csv(path):
    """Read one CL CSV split and return raw (times, marks) sequences."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    sequences = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event_times", "event_types"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(
                "{} must contain columns: event_times,event_types".format(path)
            )
        for row_index, row in enumerate(reader, start=2):
            source = "{} row {}".format(path, row_index)
            raw_times = _parse_json_list(row["event_times"], "event_times", source)
            raw_types = _parse_json_list(row["event_types"], "event_types", source)
            if not raw_times:
                raise ValueError("{} contains an empty event sequence".format(source))
            if len(raw_times) != len(raw_types):
                raise ValueError("{} has mismatched event arrays".format(source))
            try:
                times = [float(value) for value in raw_times]
                event_types = [int(value) for value in raw_types]
            except (TypeError, ValueError) as exc:
                raise ValueError("{} contains a non-numeric event value".format(source)) from exc
            if any(not math.isfinite(value) or value < 0 for value in times):
                raise ValueError("{} contains an invalid timestamp".format(source))
            if any(right < left for left, right in zip(times, times[1:])):
                raise ValueError("{} contains decreasing timestamps".format(source))
            if any(value < 0 for value in event_types):
                raise ValueError("{} contains a negative event type".format(source))
            if any(float(raw) != value for raw, value in zip(raw_types, event_types)):
                raise ValueError("{} contains a non-integral event type".format(source))
            sequences.append((times, event_types))
    if not sequences:
        raise ValueError("{} contains no event sequences".format(path))
    return sequences


def infer_cl_num_event_types(dataset_root=None):
    """Infer the global mark vocabulary from all model-facing task splits."""
    root = resolve_cl_dataset_root(dataset_root)
    maximum = -1
    for task_name in discover_cl_tasks(root):
        for source_split in ("train", "val", "test"):
            for _, event_types in _read_cl_csv(root / task_name / (source_split + ".csv")):
                maximum = max(maximum, max(event_types))
    if maximum < 0:
        raise ValueError("No event types found under {}".format(root))
    return maximum + 1


def load_cl_task_splits(
        dataset_root=None, task_id=0, epsilon=1e-8, num_event_types=None):
    """Load one CL task into the shared EasyTPP-style sequence schema.

    The source contains absolute timestamps.  Those timestamps are preserved;
    the first inter-event duration is the waiting time from the observation
    origin to the first event.  Equal timestamps, if present in a replacement
    dataset, receive deterministic ``epsilon`` separation because the bundled
    THP implementation requires an order.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    root = resolve_cl_dataset_root(dataset_root)
    task_name = normalize_cl_task_id(task_id)
    task_dir = root / task_name
    if not task_dir.is_dir():
        available = ", ".join(discover_cl_tasks(root))
        raise FileNotFoundError(
            "{} does not exist; available tasks: {}".format(task_dir, available)
        )
    dim_process = int(
        num_event_types
        if num_event_types is not None
        else infer_cl_num_event_types(root)
    )
    if dim_process < 1:
        raise ValueError("num_event_types must be positive")

    splits = {}
    counts = {}
    for output_split, source_split in (("train", "train"), ("dev", "val"), ("test", "test")):
        path = task_dir / (source_split + ".csv")
        records = []
        for seq_idx, (raw_times, event_types) in enumerate(_read_cl_csv(path)):
            if any(value >= dim_process for value in event_types):
                raise ValueError(
                    "{} sequence {} contains an event type outside [0, {})".format(
                        path, seq_idx, dim_process
                    )
                )
            ordered_times = _strictly_increasing(raw_times, epsilon)
            times = ordered_times
            deltas = [times[0]] + [
                right - left for left, right in zip(times, times[1:])
            ]
            records.append(validate_sequence({
                "dim_process": dim_process,
                "seq_idx": seq_idx,
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": event_types,
            }, dim_process, "{} sequence {}".format(path, seq_idx)))
        splits[output_split] = records
        counts[output_split] = {
            "sequences": len(records),
            "events": sum(record["seq_len"] for record in records),
        }

    metadata = {
        "dataset": "CL",
        "task_id": int(task_name.split("_", 1)[1]),
        "task_name": task_name,
        "source_directory": str(task_dir),
        "dim_process": dim_process,
        "source_columns": ["event_times", "event_types"],
        "split_mapping": {"train": "train", "val": "dev", "test": "test"},
        "time_transform": "preserve source timestamps; first duration equals first timestamp",
        "timestamp_epsilon": float(epsilon),
        "counts": counts,
        "oracle_files_used": False,
    }
    return dim_process, splits, metadata


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    return str(path)


def write_pickle(path, value):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


def validate_sequence(record, dim_process, source):
    required = ("time_since_start", "time_since_last_event", "type_event")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError("{} is missing fields: {}".format(source, ", ".join(missing)))

    times = [float(value) for value in record["time_since_start"]]
    deltas = [float(value) for value in record["time_since_last_event"]]
    types = [int(value) for value in record["type_event"]]
    if not (len(times) == len(deltas) == len(types)):
        raise ValueError("{} has arrays with different lengths".format(source))
    if not times:
        raise ValueError("{} contains an empty event sequence".format(source))
    if any(not math.isfinite(value) for value in times + deltas):
        raise ValueError("{} contains a non-finite timestamp".format(source))
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError("{} contains decreasing timestamps".format(source))
    if any(value < 0 or value >= dim_process for value in types):
        raise ValueError("{} contains an event type outside [0, {})".format(source, dim_process))

    normalized = {
        "dim_process": int(dim_process),
        "seq_idx": int(record.get("seq_idx", 0)),
        "seq_len": len(times),
        "time_since_start": times,
        "time_since_last_event": deltas,
        "type_event": types,
    }
    for mask_name in (
            "loss_mask", "event_loss_mask", "type_loss_mask", "time_loss_mask"):
        if mask_name not in record:
            continue
        loss_mask = record[mask_name]
        if len(loss_mask) != len(times):
            raise ValueError(
                "{} has a {} with the wrong length".format(source, mask_name)
            )
        if any(value not in (False, True, 0, 1) for value in loss_mask):
            raise ValueError(
                "{} has a non-boolean {}".format(source, mask_name)
            )
        normalized[mask_name] = [bool(value) for value in loss_mask]
    if "cluster" in record:
        normalized["cluster"] = int(record["cluster"])
    return normalized


def load_standard_splits(dataset_root, dataset_name):
    if dataset_name not in STANDARD_DATASETS:
        raise ValueError("Unknown standard dataset: {}".format(dataset_name))
    dataset_dir = Path(dataset_root) / dataset_name
    result = {}
    dim_process = None
    for split in ("train", "dev", "test"):
        path = dataset_dir / "{}.json".format(split)
        if not path.exists():
            raise FileNotFoundError(str(path))
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list) or not raw:
            raise ValueError("{} must contain a non-empty JSON list".format(path))
        split_dim = int(raw[0]["dim_process"])
        if dim_process is None:
            dim_process = split_dim
        elif dim_process != split_dim:
            raise ValueError("dim_process differs across splits in {}".format(dataset_name))
        result[split] = [
            validate_sequence(record, dim_process, "{} record {}".format(path, index))
            for index, record in enumerate(raw)
        ]
    return dim_process, result


def _strictly_increasing(values, epsilon):
    fixed = []
    for value in values:
        value = float(value)
        if fixed and value <= fixed[-1]:
            value = fixed[-1] + epsilon
        fixed.append(value)
    return fixed


def load_dws_file(path, epsilon=1e-6):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for seq_idx, row in enumerate(csv.DictReader(handle)):
            raw_times = [float(value) for value in row["event_times"].split(",") if value]
            event_types = [int(value) for value in row["event_types"].split(",") if value]
            if len(raw_times) != len(event_types):
                raise ValueError("{} row {} has mismatched arrays".format(path, seq_idx + 2))
            if not raw_times:
                raise ValueError("{} row {} is empty".format(path, seq_idx + 2))
            times = _strictly_increasing(raw_times, epsilon)
            deltas = [times[0]] + [
                right - left for left, right in zip(times, times[1:])
            ]
            record = {
                "dim_process": max(event_types) + 1,
                "seq_idx": seq_idx,
                "seq_len": len(times),
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": event_types,
                "cluster": int(row["cluster"]),
            }
            records.append(record)
    if not records:
        raise ValueError("{} contains no event sequences".format(path))
    dim_process = max(max(record["type_event"]) for record in records) + 1
    return dim_process, [
        validate_sequence(record, dim_process, "{} record {}".format(path, index))
        for index, record in enumerate(records)
    ]


def dws_split_indices(
        rows, seed=2024, train_ratio=None, dev_ratio=None):
    """Build the canonical cluster-stratified DWS split by source row.

    This is the only DWS partitioning rule used by the standalone adapters.
    The returned IDs refer to rows in the immutable raw-data order, so every
    model can consume the same source indices without consulting ``cluster``
    at runtime.
    """
    train_ratio = DWS_SPLIT_RATIOS["train"] if train_ratio is None else float(train_ratio)
    dev_ratio = DWS_SPLIT_RATIOS["dev"] if dev_ratio is None else float(dev_ratio)
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError(
            "Ratios must satisfy train > 0, dev >= 0, and train + dev < 1"
        )

    grouped = defaultdict(list)
    for source_index, row in enumerate(rows):
        try:
            cluster = int(row["cluster"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("DWS rows must contain an integer cluster") from exc
        grouped[cluster].append(source_index)

    result = {"train": [], "dev": [], "test": []}
    for cluster in sorted(grouped):
        group = list(grouped[cluster])
        random.Random(int(seed) + cluster).shuffle(group)
        train_end = int(round(len(group) * train_ratio))
        dev_end = train_end + int(round(len(group) * dev_ratio))
        # The bundled DWS variants have 100 sequences per cluster.  Keep the
        # helper useful for small fixtures too, while never inventing a row.
        if len(group) >= 3:
            train_end = min(max(train_end, 1), len(group) - 2)
            dev_end = min(max(dev_end, train_end + 1), len(group) - 1)
        result["train"].extend(group[:train_end])
        result["dev"].extend(group[train_end:dev_end])
        result["test"].extend(group[dev_end:])

    for offset, split in enumerate(("train", "dev", "test")):
        random.Random(int(seed) + 10000 + offset).shuffle(result[split])
    return result


def stratified_dws_splits(records, seed=2024, train_ratio=0.7, dev_ratio=0.1):
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, dev >= 0, and train + dev < 1")
    indices = dws_split_indices(
        records,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
    )
    result = {
        split: [records[source_index] for source_index in source_indices]
        for split, source_indices in indices.items()
    }
    for split in ("train", "dev", "test"):
        for seq_idx, record in enumerate(result[split]):
            record["source_index"] = int(record.get("seq_idx", seq_idx))
            record["seq_idx"] = seq_idx
    return result


def load_all_dws(dataset_root, variants=None, seed=2024, epsilon=1e-6):
    dws_dir = Path(dataset_root) / "DWS"
    if variants is None:
        paths = sorted(dws_dir.glob("hawkes_dataset_*.csv"))
        paths.extend(sorted(dws_dir.glob("tree_*/hawkes_dataset_*.csv")))
    else:
        paths = []
        for value in variants:
            nested = dws_dir / "tree_{}".format(value) / "hawkes_dataset_{}.csv".format(value)
            legacy = dws_dir / "hawkes_dataset_{}.csv".format(value)
            paths.append(nested if nested.exists() else legacy)
    if not paths:
        raise FileNotFoundError("No DWS CSV files found in {}".format(dws_dir))

    result = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        variant = path.stem.rsplit("_", 1)[-1]
        dim_process, records = load_dws_file(path, epsilon=epsilon)
        result[variant] = (
            dim_process,
            stratified_dws_splits(
                records,
                seed=seed,
                train_ratio=DWS_SPLIT_RATIOS["train"],
                dev_ratio=DWS_SPLIT_RATIOS["dev"],
            ),
        )
    return result


class _DataOnlyUnpickler(pickle.Unpickler):
    """Load legacy pickle containers without allowing global constructors."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            "global constructor {}.{} is not allowed".format(module, name)
        )


def _covid_policy_label_sort_key(label):
    if len(label) >= 2 and label[0] in "CEHV" and label[1:].isdigit():
        return ("CEHV".index(label[0]), int(label[1:]), label)
    return (4, 0 if label == "cc" else 1 if label == "cd" else 2, label)


def _policy_run_starts(values):
    """Collapse a daily policy-presence vector to starts of active runs."""
    starts = []
    previous = None
    for value in sorted(set(values)):
        if previous is None or value > previous + 1.0:
            starts.append(value)
        previous = value
    return starts


def _chronological_event_split(
        events, train_ratio, dev_ratio, min_events=2, seed=2024):
    """Split chronologically near requested ratios without dividing one day."""
    grouped = []
    for time, event_type in sorted(events):
        if not grouped or time != grouped[-1][0]:
            grouped.append([time, []])
        grouped[-1][1].append(event_type)

    # A numerical type sort would create a learnable but meaningless order for
    # simultaneous policies.  Shuffle ties once, deterministically, instead.
    tie_random = random.Random(seed)
    for _, event_types in grouped:
        tie_random.shuffle(event_types)

    if len(grouped) < 3 or len(events) < min_events * 3:
        raise ValueError("A country needs enough dated events for three non-empty splits")

    cumulative = [0]
    for _, event_types in grouped:
        cumulative.append(cumulative[-1] + len(event_types))
    total = cumulative[-1]
    target_train = total * train_ratio
    target_dev_end = total * (train_ratio + dev_ratio)

    candidates = []
    for train_end in range(1, len(grouped) - 1):
        train_count = cumulative[train_end]
        if train_count < min_events:
            continue
        for dev_end in range(train_end + 1, len(grouped)):
            dev_count = cumulative[dev_end] - train_count
            test_count = total - cumulative[dev_end]
            if dev_count < min_events or test_count < min_events:
                continue
            score = (
                abs(train_count - target_train)
                + abs(cumulative[dev_end] - target_dev_end)
            )
            candidates.append((score, train_end, dev_end))
    if not candidates:
        raise ValueError("Could not create chronological train/dev/test event splits")

    _, train_end, dev_end = min(candidates)

    def flatten(groups):
        return [
            (time, event_type)
            for time, event_types in groups
            for event_type in event_types
        ]

    return {
        "train": flatten(grouped[:train_end]),
        "dev": flatten(grouped[train_end:dev_end]),
        "test": flatten(grouped[dev_end:]),
    }


def load_covid_policy_tracker_splits(
        dataset_root, seed=2024, train_ratio=0.8, dev_ratio=0.1, epsilon=1e-4):
    """Convert the prepared Covid-Policy-Tracker files to marked TPP streams.

    Each source pickle maps a policy/outcome code to the day indices on which
    it is present.  Repeated daily policy states are collapsed to the first day
    of each contiguous active run.  The sparse ``cc`` and ``cd`` outcome lists
    are retained as individual events.  Every country is split chronologically
    and a single day is never divided across train, dev, and test.
    """
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, dev > 0, and train + dev < 1")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must be between zero and one day")

    dataset_dir = Path(dataset_root) / "Covid-Policy-Tracker"
    paths = sorted(
        path for path in dataset_dir.glob("Covid-19_*_decrease.pkl")
        if not path.name.startswith("._")
    )
    if not paths:
        raise FileNotFoundError(
            "No Covid-Policy-Tracker country pickle files found in {}".format(dataset_dir)
        )

    raw_by_country = []
    all_labels = set()
    for path in paths:
        with path.open("rb") as handle:
            raw = _DataOnlyUnpickler(handle).load()
        if not isinstance(raw, dict):
            raise ValueError("{} must contain a dictionary".format(path))
        country = path.stem[len("Covid-19_"):-len("_decrease")].replace("_", " ")
        normalized = {}
        for label, values in raw.items():
            if not isinstance(label, str) or not isinstance(values, (list, tuple)):
                raise ValueError("{} contains an invalid event entry".format(path))
            times = [float(value) for value in values]
            if any(not math.isfinite(value) or value < 0 for value in times):
                raise ValueError("{} {} contains an invalid day index".format(path, label))
            if len(times) != len(set(times)):
                raise ValueError("{} {} contains duplicate day indices".format(path, label))
            normalized[label] = times
            all_labels.add(label)
        raw_by_country.append((country, path.name, normalized))

    labels = sorted(all_labels, key=_covid_policy_label_sort_key)
    label_to_type = {label: index for index, label in enumerate(labels)}
    dim_process = len(labels)
    splits = {"train": [], "dev": [], "test": []}
    country_event_counts = {}

    for country_index, (country, _, raw) in enumerate(raw_by_country):
        events = []
        for label, values in raw.items():
            selected = values if label in {"cc", "cd"} else _policy_run_starts(values)
            events.extend((time, label_to_type[label]) for time in selected)
        if not events:
            raise ValueError("{} contains no converted events".format(country))

        country_parts = _chronological_event_split(
            events, train_ratio, dev_ratio, seed=int(seed) + country_index
        )
        country_event_counts[country] = {}
        for split, part in country_parts.items():
            origin = part[0][0]
            adjusted_times = []
            previous_raw_time = None
            tie_index = 0
            for raw_time, _ in part:
                if raw_time == previous_raw_time:
                    tie_index += 1
                else:
                    tie_index = 0
                adjusted_times.append(raw_time - origin + tie_index * epsilon)
                previous_raw_time = raw_time
            deltas = [0.0] + [
                right - left for left, right in zip(adjusted_times, adjusted_times[1:])
            ]
            record = validate_sequence({
                "dim_process": dim_process,
                "seq_idx": country_index,
                "time_since_start": adjusted_times,
                "time_since_last_event": deltas,
                "type_event": [event_type for _, event_type in part],
                "cluster": country_index,
            }, dim_process, "{} {} split".format(country, split))
            splits[split].append(record)
            country_event_counts[country][split] = len(part)

    metadata = {
        "dataset": "Covid-Policy-Tracker",
        "source_directory": dataset_dir.name,
        "source_files": [source_file for _, source_file, _ in raw_by_country],
        "countries": [country for country, _, _ in raw_by_country],
        "country_by_seq": {
            index: country for index, (country, _, _) in enumerate(raw_by_country)
        },
        "time_unit": "day",
        "event_type_mapping": {index: label for label, index in label_to_type.items()},
        "policy_event_encoding": "first day of each contiguous active run",
        "outcome_event_codes": [label for label in ("cc", "cd") if label in label_to_type],
        "simultaneous_event_epsilon_days": float(epsilon),
        "simultaneous_event_order": "fixed random permutation within day",
        "simultaneous_event_seed": int(seed),
        "split_strategy": "chronological within each country; same-day events kept together",
        "split_ratios": {
            "train": float(train_ratio),
            "dev": float(dev_ratio),
            "test": round(float(1.0 - train_ratio - dev_ratio), 10),
        },
        "event_counts": country_event_counts,
    }
    return dim_process, splits, metadata


def window_training_records(records, window_size=32, stride=16):
    """Create overlapping context windows while scoring each event once.

    ``loss_mask`` is true only for events not scored by an earlier window.  A
    compatible trainer can therefore use overlapping history without giving
    duplicated targets extra weight.
    """
    window_size = int(window_size)
    stride = int(stride)
    if window_size < 2:
        raise ValueError("window_size must be at least two events")
    if stride < 1 or stride > window_size:
        raise ValueError("stride must satisfy 1 <= stride <= window_size")

    windows = []
    windows_by_source = {}
    for source_index, record in enumerate(records):
        length = len(record["time_since_start"])
        if length < 2:
            raise ValueError("Training sequences must contain at least two events")
        if length <= window_size:
            starts = [0]
        else:
            starts = list(range(0, length - window_size + 1, stride))
            tail_start = length - window_size
            if starts[-1] != tail_start:
                starts.append(tail_start)

        covered_until = -1
        source_windows = []
        for start in starts:
            end = min(start + window_size, length)
            first_new = max(start, covered_until + 1)
            raw_times = record["time_since_start"][start:end]
            origin = raw_times[0]
            times = [value - origin for value in raw_times]
            deltas = [0.0] + [right - left for left, right in zip(times, times[1:])]
            window = validate_sequence({
                "dim_process": record["dim_process"],
                "seq_idx": len(windows),
                "time_since_start": times,
                "time_since_last_event": deltas,
                "type_event": record["type_event"][start:end],
                "loss_mask": [
                    global_index >= first_new for global_index in range(start, end)
                ],
                **({"cluster": record["cluster"]} if "cluster" in record else {}),
            }, record["dim_process"], "training window {}".format(len(windows)))
            windows.append(window)
            source_windows.append({
                "window_seq_idx": window["seq_idx"],
                "start_event": start,
                "end_event_exclusive": end,
                "first_scored_event": first_new,
            })
            covered_until = max(covered_until, end - 1)
        windows_by_source[str(source_index)] = source_windows

    return windows, {
        "window_size": window_size,
        "stride": stride,
        "number_of_windows": len(windows),
        "loss_mask_semantics": "score each original event once; earlier overlap is context only",
        "windows_by_source_seq": windows_by_source,
    }


def event_texts(dataset_name, event_types, labels=None):
    labels = labels or {}
    display_name = {
        "retweet": "Retweet",
        "taxi": "Taxi",
        "stackoverflow": "StackOverflow badge",
        "taobao": "Taobao behavior",
        "amazon": "Amazon category",
        "dws": "DWS simulated event",
        "covid_policy_tracker": "COVID policy/outcome",
    }.get(dataset_name, dataset_name)
    return [str(labels.get(value, "{} type {}".format(display_name, value))) for value in event_types]
