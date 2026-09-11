"""Dataset adapters for the marked FullyNN implementation in EasyTPP."""

import bisect
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _data_configuration_common import (  # noqa: E402
    ensure_directory,
    infer_dataset_root,
    infer_output_root,
    load_all_dws,
    load_covid_policy_tracker_splits,
    load_standard_splits,
    write_json,
)
from Models.THP.data_configuration import _load_mimic_splits  # noqa: E402


class DataConfiguration:
    """Convert benchmarks to EasyTPP JSON for marked, multi-sequence FullyNN.

    The upstream FullyNN notebooks accept one unmarked timestamp vector.  They
    cannot preserve these datasets' event marks or independent sequences, so
    this adapter targets EasyTPP's faithful marked FullyNN implementation.
    """

    def __init__(self, dataset_root=None, output_root=None, seed=2024):
        self.dataset_root = Path(dataset_root or infer_dataset_root(__file__))
        self.output_root = Path(output_root or infer_output_root(__file__))
        self.seed = int(seed)

    def retweet(self, output_dir=None, history_window=None):
        return self._standard("retweet", output_dir, history_window=history_window)

    def taxi(self, output_dir=None, history_window=None):
        return self._standard("taxi", output_dir, history_window=history_window)

    def stackoverflow(self, output_dir=None, history_window=None):
        return self._standard(
            "stackoverflow", output_dir, history_window=history_window
        )

    def taobao(self, output_dir=None, history_window=None):
        return self._standard("taobao", output_dir, history_window=history_window)

    def amazon(self, output_dir=None, history_window=None):
        return self._standard("amazon", output_dir, history_window=history_window)

    def mimic(
            self, output_dir=None, epsilon=1e-6, history_window=None,
            train_ratio=0.8, dev_ratio=0.1,
            stratify_pred_ids=(51, 61)):
        """Create patient-level MIMIC JSON while preserving task masks."""
        _, splits, metadata = _load_mimic_splits(
            self.dataset_root,
            seed=self.seed,
            epsilon=epsilon,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            stratify_pred_ids=stratify_pred_ids,
        )
        if history_window is not None:
            splits = {
                split: self._window_sequences(records, history_window)
                for split, records in splits.items()
            }
        target = ensure_directory(output_dir or self.output_root / "mimic")
        write_json(target / "metadata.json", metadata)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }

    def mobike(
            self, output_dir=None, history_window=None, min_trips=5,
            grid_size=4, train_ratio=0.8, dev_ratio=0.1,
            epsilon_hours=1e-6):
        """Convert Mobike rides to user-level next-time/next-origin streams.

        Event time is ride start time. Event mark is a train-fitted quantile
        grid cell of the start location, so the task predicts the user's next
        ride time and next pickup area without using dev/test to define bins.
        """
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

        train_rides = [
            ride for user in users_by_split["train"]
            for ride in rides_by_user[user]
        ]

        def quantile_edges(values):
            values = sorted(values)
            return [
                values[round(fraction * (len(values) - 1))]
                for fraction in (
                    index / float(grid_size)
                    for index in range(1, grid_size)
                )
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
                records.append({
                    "dim_process": dim_process,
                    "seq_idx": len(records),
                    "seq_len": len(times),
                    "time_since_start": times,
                    "time_since_last_event": [0.0] + [
                        right - left for left, right in zip(times, times[1:])
                    ],
                    "type_event": marks,
                    "user_id": user,
                })
            if history_window is not None:
                records = self._window_sequences(records, history_window)
            splits[split] = records

        target = ensure_directory(output_dir or self.output_root / "mobike")
        metadata = {
            "dataset": "Mobike_Data",
            "source_file": source.name,
            "sequence_unit": "user",
            "event_time": "ride start time",
            "time_unit": "hour",
            "event_mark": "train-fitted start-location quantile grid",
            "grid_size": grid_size,
            "dim_process": dim_process,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "min_trips": min_trips,
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
        write_json(target / "metadata.json", metadata)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }

    def covid_policy_tracker(
            self, output_dir=None, epsilon=1e-4, history_window=None):
        _, splits, metadata = load_covid_policy_tracker_splits(
            self.dataset_root, seed=self.seed, epsilon=epsilon
        )
        if history_window is not None:
            splits = {
                split: self._window_sequences(records, history_window)
                for split, records in splits.items()
            }
        target = ensure_directory(output_dir or self.output_root / "covid_policy_tracker")
        write_json(target / "metadata.json", metadata)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }

    def dws(
            self, output_dir=None, variants=None, epsilon=1e-6,
            history_window=None):
        base = ensure_directory(output_dir or self.output_root / "dws")
        outputs = {}
        for variant, (_, splits) in load_all_dws(
            self.dataset_root, variants=variants, seed=self.seed, epsilon=epsilon
        ).items():
            if history_window is not None:
                splits = {
                    split: self._window_sequences(records, history_window)
                    for split, records in splits.items()
                }
            target = ensure_directory(base / "dws_{}".format(variant))
            outputs[variant] = {
                split: write_json(target / "{}.json".format(split), records)
                for split, records in splits.items()
            }
        return outputs

    @staticmethod
    def _window_sequences(records, history_window):
        """Split sequences into one-event-overlap truncated-BPTT windows.

        A window with history_window=20 contains at most 21 events and therefore
        20 prediction intervals. Adjacent windows overlap by the boundary event,
        so every original next-event interval is retained exactly once.
        """
        history_window = int(history_window)
        if history_window < 1:
            raise ValueError("history_window must be positive")

        windowed = []
        for record in records:
            sequence_length = len(record["type_event"])
            start = 0
            window_index = 0
            while start < sequence_length - 1:
                end = min(sequence_length, start + history_window + 1)
                absolute_times = record["time_since_start"][start:end]
                origin = absolute_times[0]
                item = {
                    "dim_process": record["dim_process"],
                    "seq_idx": len(windowed),
                    "seq_len": end - start,
                    "time_since_start": [value - origin for value in absolute_times],
                    "time_since_last_event": [0.0] + record[
                        "time_since_last_event"
                    ][start + 1:end],
                    "type_event": record["type_event"][start:end],
                    "cluster": record.get("cluster"),
                    "source_seq_idx": record.get("seq_idx"),
                    "window_idx": window_index,
                }
                for mask_name in (
                        "event_loss_mask", "type_loss_mask", "time_loss_mask"):
                    if mask_name in record:
                        values = list(record[mask_name][start:end])
                        values[0] = False
                        item[mask_name] = values
                windowed.append(item)
                start = end - 1
                window_index += 1
        return windowed

    def _standard(self, dataset_name, output_dir, history_window=None):
        _, splits = load_standard_splits(self.dataset_root, dataset_name)
        if history_window is not None:
            splits = {
                split: self._window_sequences(records, history_window)
                for split, records in splits.items()
            }
        target = ensure_directory(output_dir or self.output_root / dataset_name)
        return {
            split: write_json(target / "{}.json".format(split), records)
            for split, records in splits.items()
        }
