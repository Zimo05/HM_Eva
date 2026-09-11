"""Dataset adapters for the standalone Transformer Hawkes Process code."""

import bisect
import csv
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _data_configuration_common import (  # noqa: E402
    ensure_directory,
    infer_dataset_root,
    infer_output_root,
    load_all_dws,
    load_covid_policy_tracker_splits,
    load_standard_splits,
    validate_sequence,
    window_training_records,
    write_pickle,
)


def _load_mimic_splits(
        dataset_root, seed=2024, train_ratio=0.8, dev_ratio=0.1,
        epsilon=1e-6, stratify_pred_ids=(51, 61)):
    """Convert the Boolean MIMIC predicate histories to marked TPP streams.

    Each source observation is encoded as ``2 * predicate_id + state`` so the
    Boolean state is not discarded. Patients are split before conversion and
    stratified by the final states of the requested target predicates. Events
    sharing a timestamp receive one fixed random cross-predicate order and a
    tiny time offset. Source order is retained for multiple states of the same
    predicate at one timestamp. This avoids creating a learnable predicate-ID
    order for concurrent events without reversing Boolean transitions.
    """
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError(
            "Ratios must satisfy train > 0, dev > 0, and train + dev < 1"
        )
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    source = Path(dataset_root) / "mimic" / "mimic.npy"
    if not source.exists():
        raise FileNotFoundError(str(source))

    loaded = np.load(str(source), allow_pickle=True)
    if not isinstance(loaded, np.ndarray) or loaded.shape != ():
        raise ValueError("{} must contain one pickled dictionary".format(source))
    raw = loaded.item()
    if not isinstance(raw, dict) or not raw:
        raise ValueError("{} must contain a non-empty patient dictionary".format(source))

    patient_ids = sorted(raw)
    patient_seed_offsets = {
        patient_id: index for index, patient_id in enumerate(patient_ids)
    }
    first_patient = raw[patient_ids[0]]
    if not isinstance(first_patient, dict) or not first_patient:
        raise ValueError("MIMIC patient records must be non-empty dictionaries")
    predicate_ids = sorted(first_patient)
    if predicate_ids != list(range(len(predicate_ids))):
        raise ValueError("MIMIC predicate IDs must be contiguous and start at zero")

    num_predicates = len(predicate_ids)
    stratify_pred_ids = tuple(int(value) for value in stratify_pred_ids)
    if not stratify_pred_ids:
        raise ValueError("stratify_pred_ids must contain at least one predicate")
    if any(value < 0 or value >= num_predicates for value in stratify_pred_ids):
        raise ValueError("A MIMIC stratification predicate is out of range")

    validated = {}
    for patient_id in patient_ids:
        patient = raw[patient_id]
        if not isinstance(patient, dict) or sorted(patient) != predicate_ids:
            raise ValueError(
                "MIMIC patient {} has inconsistent predicate IDs".format(patient_id)
            )

        normalized = {}
        for predicate_id in predicate_ids:
            cell = patient[predicate_id]
            if not isinstance(cell, dict) or "time" not in cell or "state" not in cell:
                raise ValueError(
                    "MIMIC patient {} predicate {} is malformed".format(
                        patient_id, predicate_id
                    )
                )
            raw_times = list(cell["time"])
            raw_states = list(cell["state"])
            if not raw_times or len(raw_times) != len(raw_states):
                raise ValueError(
                    "MIMIC patient {} predicate {} has invalid arrays".format(
                        patient_id, predicate_id
                    )
                )
            observations = [
                (float(time), int(state))
                for time, state in zip(raw_times, raw_states)
            ]
            # Stable sorting repairs the small number of source cells whose
            # timestamps are out of order while retaining the source order of
            # same-time transitions such as 0 -> 1.
            observations.sort(key=lambda item: item[0])
            times = [time for time, _ in observations]
            states = [state for _, state in observations]
            if any(not math.isfinite(value) or value < 0 for value in times):
                raise ValueError(
                    "MIMIC patient {} predicate {} has an invalid time".format(
                        patient_id, predicate_id
                    )
                )
            if any(value not in (0, 1) for value in states):
                raise ValueError(
                    "MIMIC patient {} predicate {} has a non-Boolean state".format(
                        patient_id, predicate_id
                    )
                )
            normalized[predicate_id] = {"time": times, "state": states}
        validated[patient_id] = normalized

    stratified = defaultdict(list)
    for patient_id in patient_ids:
        label = tuple(
            validated[patient_id][predicate_id]["state"][-1]
            for predicate_id in stratify_pred_ids
        )
        stratified[label].append(patient_id)

    patient_splits = {"train": [], "dev": [], "test": []}
    for label, group in sorted(stratified.items()):
        group = list(group)
        label_seed = sum((index + 1) * value for index, value in enumerate(label))
        random.Random(int(seed) + label_seed).shuffle(group)
        train_end = int(len(group) * train_ratio)
        dev_end = int(len(group) * (train_ratio + dev_ratio))
        patient_splits["train"].extend(group[:train_end])
        patient_splits["dev"].extend(group[train_end:dev_end])
        patient_splits["test"].extend(group[dev_end:])

    for offset, split in enumerate(("train", "dev", "test")):
        random.Random(int(seed) + 10000 + offset).shuffle(patient_splits[split])
        if not patient_splits[split]:
            raise ValueError("MIMIC {} split is empty".format(split))

    dim_process = num_predicates * 2
    splits = {"train": [], "dev": [], "test": []}
    event_counts = {}
    for split, split_patient_ids in patient_splits.items():
        split_event_count = 0
        for sequence_index, patient_id in enumerate(split_patient_ids):
            by_time = defaultdict(lambda: defaultdict(list))
            for predicate_id, cell in validated[patient_id].items():
                for observation_index, (time, state) in enumerate(zip(
                        cell["time"], cell["state"])):
                    is_final_target = (
                        predicate_id in stratify_pred_ids
                        and observation_index == len(cell["time"]) - 1
                    )
                    by_time[time][predicate_id].append(
                        (2 * predicate_id + state, is_final_target)
                    )

            tie_random = random.Random(
                int(seed) + patient_seed_offsets[patient_id]
            )
            origin = min(by_time)
            adjusted_times = []
            event_types = []
            event_loss_mask = []
            type_loss_mask = []
            time_loss_mask = []
            previous = None
            previous_raw_time = None
            for raw_time in sorted(by_time):
                simultaneous = by_time[raw_time]
                simultaneous_predicates = list(simultaneous)
                tie_random.shuffle(simultaneous_predicates)
                first_at_timestamp = True
                for predicate_id in simultaneous_predicates:
                    for mark, is_final_target in simultaneous[predicate_id]:
                        adjusted = raw_time - origin
                        if previous is not None and adjusted <= previous:
                            adjusted = previous + epsilon
                        adjusted_times.append(adjusted)
                        event_types.append(mark)
                        event_loss_mask.append(previous is not None)
                        type_loss_mask.append(is_final_target)
                        time_loss_mask.append(
                            previous_raw_time is not None and first_at_timestamp
                        )
                        previous = adjusted
                        first_at_timestamp = False
                previous_raw_time = raw_time

            deltas = [0.0] + [
                right - left
                for left, right in zip(adjusted_times, adjusted_times[1:])
            ]
            record = validate_sequence({
                "dim_process": dim_process,
                "seq_idx": sequence_index,
                "time_since_start": adjusted_times,
                "time_since_last_event": deltas,
                "type_event": event_types,
                "event_loss_mask": event_loss_mask,
                "type_loss_mask": type_loss_mask,
                "time_loss_mask": time_loss_mask,
            }, dim_process, "MIMIC patient {}".format(patient_id))
            splits[split].append(record)
            split_event_count += len(event_types)
        event_counts[split] = split_event_count

    metadata = {
        "dataset": "mimic",
        "source_file": source.name,
        "num_patients": len(patient_ids),
        "num_predicates": num_predicates,
        "dim_process": dim_process,
        "event_encoding": "2 * predicate_id + Boolean state",
        "event_type_mapping": {
            2 * predicate_id + state: {
                "predicate_id": predicate_id,
                "state": state,
            }
            for predicate_id in predicate_ids
            for state in (0, 1)
        },
        "target_predicate_ids": list(stratify_pred_ids),
        "simultaneous_event_epsilon": float(epsilon),
        "simultaneous_event_order": (
            "fixed random predicate permutation within timestamp; "
            "same-predicate source order retained"
        ),
        "simultaneous_event_seed": int(seed),
        "split_strategy": "patient-level stratification by final target states",
        "split_ratios": {
            "train": float(train_ratio),
            "dev": float(dev_ratio),
            "test": round(float(1.0 - train_ratio - dev_ratio), 10),
        },
        "patient_ids_by_split": {
            split: list(values) for split, values in patient_splits.items()
        },
        "event_counts_by_split": event_counts,
        "event_loss_mask": "all events after the first event",
        "type_loss_mask": "final observation of each target predicate",
        "time_loss_mask": "first event at each strictly later source timestamp",
        "time_unit": "source units",
    }
    return dim_process, splits, metadata


class DataConfiguration:
    """Create the split-specific pickle dictionaries consumed by THP Main.py."""

    def __init__(self, dataset_root=None, output_root=None, seed=2024):
        self.dataset_root = Path(dataset_root or infer_dataset_root(__file__))
        self.output_root = Path(output_root or infer_output_root(__file__))
        self.seed = int(seed)

    def retweet(self, output_dir=None):
        return self._standard("retweet", output_dir)

    def taxi(self, output_dir=None):
        return self._standard("taxi", output_dir)

    def stackoverflow(self, output_dir=None):
        return self._standard("stackoverflow", output_dir)

    def taobao(self, output_dir=None):
        return self._standard("taobao", output_dir)

    def amazon(self, output_dir=None):
        return self._standard("amazon", output_dir)

    def mobike(
            self, output_dir=None, min_trips=5, grid_size=4,
            train_ratio=0.8, dev_ratio=0.1, epsilon_hours=1e-6):
        """Create user-level next-ride-time/next-origin Mobike streams."""
        source = (
            self.dataset_root / "Mobike_Data" /
            "mobike_shanghai_sample_updated.csv"
        )
        if not source.exists():
            raise FileNotFoundError(str(source))
        if grid_size < 2 or min_trips < 2:
            raise ValueError("grid_size and min_trips must be at least 2")
        if not 0 < train_ratio < train_ratio + dev_ratio < 1:
            raise ValueError("invalid train/dev ratios")

        rides_by_user = defaultdict(list)
        with source.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rides_by_user[str(row["userid"])].append((
                    datetime.fromisoformat(row["start_time"]),
                    float(row["start_location_x"]),
                    float(row["start_location_y"]),
                ))
        rides_by_user = {
            user: sorted(rides)
            for user, rides in rides_by_user.items()
            if len(rides) >= int(min_trips)
        }
        user_ids = sorted(rides_by_user)
        random.Random(self.seed).shuffle(user_ids)
        train_end = int(len(user_ids) * train_ratio)
        dev_end = int(len(user_ids) * (train_ratio + dev_ratio))
        users_by_split = {
            "train": user_ids[:train_end],
            "dev": user_ids[train_end:dev_end],
            "test": user_ids[dev_end:],
        }
        if any(not users for users in users_by_split.values()):
            raise ValueError("Mobike train/dev/test split contains an empty split")

        train_rides = [
            ride for user in users_by_split["train"]
            for ride in rides_by_user[user]
        ]

        def quantile_edges(values):
            values = sorted(values)
            return [
                values[round(index / float(grid_size) * (len(values) - 1))]
                for index in range(1, grid_size)
            ]

        x_edges = quantile_edges([ride[1] for ride in train_rides])
        y_edges = quantile_edges([ride[2] for ride in train_rides])
        dim_process = grid_size * grid_size
        splits = {}
        for split, split_users in users_by_split.items():
            records = []
            for user in split_users:
                rides = rides_by_user[user]
                origin = rides[0][0]
                times = []
                marks = []
                previous = None
                for start_time, longitude, latitude in rides:
                    elapsed = (start_time - origin).total_seconds() / 3600.0
                    if previous is not None and elapsed <= previous:
                        elapsed = previous + epsilon_hours
                    times.append(elapsed)
                    previous = elapsed
                    x_bin = bisect.bisect_right(x_edges, longitude)
                    y_bin = bisect.bisect_right(y_edges, latitude)
                    marks.append(x_bin * grid_size + y_bin)
                records.append(validate_sequence({
                    "dim_process": dim_process,
                    "seq_idx": len(records),
                    "time_since_start": times,
                    "time_since_last_event": [0.0] + [
                        right - left for left, right in zip(times, times[1:])
                    ],
                    "type_event": marks,
                }, dim_process, "Mobike user {}".format(user)))
            splits[split] = records

        metadata = {
            "dataset": "Mobike_Data",
            "source_file": source.name,
            "sequence_unit": "user",
            "event_time": "ride start time",
            "time_unit": "hour",
            "event_mark": "train-fitted start-location quantile grid",
            "grid_size": int(grid_size),
            "dim_process": dim_process,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "min_trips": int(min_trips),
            "split_strategy": "user-level seeded random split",
            "split_ratios": {
                "train": train_ratio,
                "dev": dev_ratio,
                "test": 1.0 - train_ratio - dev_ratio,
            },
            "seed": self.seed,
            "users_by_split": {
                split: len(users) for split, users in users_by_split.items()
            },
        }
        target = ensure_directory(output_dir or self.output_root / "mobike")
        return self._write_splits(dim_process, splits, target, metadata=metadata)

    def mimic(
            self, output_dir=None, epsilon=1e-6,
            train_ratio=0.8, dev_ratio=0.1,
            stratify_pred_ids=(51, 61)):
        """Create patient-level MIMIC train/dev/test pickle files."""
        dim_process, splits, metadata = _load_mimic_splits(
            self.dataset_root,
            seed=self.seed,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            epsilon=epsilon,
            stratify_pred_ids=stratify_pred_ids,
        )
        target = ensure_directory(output_dir or self.output_root / "mimic")
        return self._write_splits(
            dim_process, splits, target, metadata=metadata
        )

    def covid_policy_tracker(
            self, output_dir=None, epsilon=1e-4, window_size=32, stride=16):
        dim_process, splits, metadata = load_covid_policy_tracker_splits(
            self.dataset_root, seed=self.seed, epsilon=epsilon
        )
        splits["train"], window_metadata = window_training_records(
            splits["train"], window_size=window_size, stride=stride
        )
        metadata["training_windows"] = window_metadata
        target = ensure_directory(output_dir or self.output_root / "covid_policy_tracker")
        return self._write_splits(dim_process, splits, target, metadata=metadata)

    def dws(self, output_dir=None, variants=None, epsilon=1e-6):
        base = ensure_directory(output_dir or self.output_root / "dws")
        outputs = {}
        for variant, (dim_process, splits) in load_all_dws(
            self.dataset_root, variants=variants, seed=self.seed, epsilon=epsilon
        ).items():
            outputs[variant] = self._write_splits(
                dim_process, splits, ensure_directory(base / "dws_{}".format(variant))
            )
        return outputs

    def _standard(self, dataset_name, output_dir):
        dim_process, splits = load_standard_splits(self.dataset_root, dataset_name)
        target = ensure_directory(output_dir or self.output_root / dataset_name)
        return self._write_splits(dim_process, splits, target)

    def _write_splits(self, dim_process, splits, target, metadata=None):
        outputs = {}
        for split, records in splits.items():
            streams = []
            clusters = []
            for record in records:
                default_mask = [True] * len(record["type_event"])
                event_loss_mask = record.get("event_loss_mask", default_mask)
                # Keep the legacy field meaningful for older consumers while
                # exposing task-specific masks to the updated THP loader.
                loss_mask = record.get("loss_mask", event_loss_mask)
                type_loss_mask = record.get("type_loss_mask", loss_mask)
                time_loss_mask = record.get("time_loss_mask", loss_mask)
                streams.append([
                    {
                        "time_since_start": time,
                        "time_since_last_event": delta,
                        "type_event": event_type,
                        "loss_mask": score_event,
                        "event_loss_mask": score_event_ll,
                        "type_loss_mask": score_type,
                        "time_loss_mask": score_time,
                    }
                    for (
                        time, delta, event_type, score_event, score_event_ll,
                        score_type, score_time,
                    ) in zip(
                        record["time_since_start"],
                        record["time_since_last_event"],
                        record["type_event"],
                        loss_mask,
                        event_loss_mask,
                        type_loss_mask,
                        time_loss_mask,
                    )
                ])
                clusters.append(record.get("cluster"))
            payload = {"dim_process": int(dim_process), split: streams}
            if metadata:
                payload["metadata"] = dict(metadata)
            if any(value is not None for value in clusters):
                payload["cluster_by_seq"] = clusters
            outputs[split] = write_pickle(target / "{}.pkl".format(split), payload)
        return outputs
