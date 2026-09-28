"""Build an auditable, globally chronological Taobao continual stream.

The repository's cached Taobao splits contain per-user relative clocks. This
builder therefore requires event-level rows with absolute timestamps and never
attempts to reconstruct calendar order from those split files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_source(path: Path) -> pd.DataFrame:
    if path.is_dir():
        cached = [path / name for name in ("train.json", "dev.json", "test.json")]
        existing = [candidate for candidate in cached if candidate.is_file()]
        if existing:
            fields = set()
            for candidate in existing:
                try:
                    rows = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    fields.update(rows[0])
            raise ValueError(
                "the supplied Taobao directory contains cached train/dev/test.json "
                "splits, but their per-user relative time_since_start values do "
                "not preserve one global clock. Supply the original event-level "
                "file with absolute timestamps instead. Detected fields: "
                f"{sorted(fields)}"
            )
        raise ValueError(f"no supported event-level source found in {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("raw event source must be CSV, JSON, JSONL, or Parquet")


def _absolute_utc_timestamps(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if len(numeric) and numeric.notna().all():
        magnitude = float(numeric.abs().median())
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        elif magnitude >= 1e8:
            unit = "s"
        else:
            raise ValueError(
                "numeric timestamps are too small to be absolute Unix time; "
                "relative per-user clocks cannot establish global chronology"
            )
        timestamps = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp column contains missing or invalid absolute times")
    years = timestamps.dt.year
    if bool(((years < 2000) | (years > 2100)).any()):
        raise ValueError(
            "timestamps must resolve to plausible absolute calendar dates (2000–2100)"
        )
    return timestamps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chronological_windows(
    frame: pd.DataFrame,
    min_events_per_window: int,
) -> list[tuple[list[str], pd.DataFrame]]:
    day_counts = frame.groupby("_day", sort=True).size().to_dict()
    windows: list[tuple[list[str], pd.DataFrame]] = []
    days: list[str] = []
    count = 0
    for day in sorted(day_counts):
        days.append(str(day))
        count += int(day_counts[day])
        if count >= min_events_per_window:
            windows.append((days, frame[frame["_day"].isin(days)].copy()))
            days, count = [], 0
    if days:
        if windows:
            previous_days, previous_frame = windows[-1]
            combined_days = previous_days + days
            combined_frame = frame[frame["_day"].isin(combined_days)].copy()
            windows[-1] = (combined_days, combined_frame)
        else:
            windows.append((days, frame.copy()))
    return windows


def _tie_safe_cuts(times_ns: list[int]) -> tuple[int, int]:
    count = len(times_ns)
    first = min(count, max(1, int(count * 0.70)))
    while first < count and times_ns[first] == times_ns[first - 1]:
        first += 1
    second = min(count, max(first + 1, int(count * 0.80)))
    while second < count and times_ns[second] == times_ns[second - 1]:
        second += 1
    return first, second


def _write_split(
    window_frame: pd.DataFrame,
    start: int,
    end: int,
    output_path: Path,
) -> dict[str, Any]:
    split = window_frame.iloc[start:end].copy()
    rows: list[dict[str, Any]] = []
    for user_id, group in split.groupby("_user", sort=False):
        group = group.sort_values(["_timestamp_ns", "_source_order"], kind="stable")
        absolute_ns = group["_timestamp_ns"].astype("int64").tolist()
        origin = absolute_ns[0]
        rows.append({
            "user_id": str(user_id),
            "absolute_start_time": pd.Timestamp(origin, tz="UTC").isoformat(),
            "absolute_end_time": pd.Timestamp(absolute_ns[-1], tz="UTC").isoformat(),
            "event_times": json.dumps(
                [(value - origin) / 1_000_000_000 for value in absolute_ns],
                separators=(",", ":"),
            ),
            "event_types": json.dumps(
                [int(value) for value in group["_event_type"]], separators=(",", ":")
            ),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "user_id", "absolute_start_time", "absolute_end_time",
                "event_times", "event_types",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "events": int(end - start),
        "sequences": len(rows),
        "file": str(output_path.name),
        "start_time": (
            pd.Timestamp(int(window_frame.iloc[start]["_timestamp_ns"]), tz="UTC").isoformat()
            if end > start else None
        ),
        "end_time": (
            pd.Timestamp(int(window_frame.iloc[end - 1]["_timestamp_ns"]), tz="UTC").isoformat()
            if end > start else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Taobao as contiguous UTC windows with chronological 70/10/20 splits"
    )
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--user-column", default="user_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--event-type-column", default="event_type")
    parser.add_argument("--min-events-per-window", type=int, default=20)
    args = parser.parse_args()

    if args.min_events_per_window < 3:
        raise ValueError("--min-events-per-window must be at least 3")
    source = args.raw_events.expanduser().resolve()
    frame = _read_source(source)
    required = {args.user_column, args.timestamp_column, args.event_type_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "raw event-level source is missing required columns "
            f"{sorted(missing)}; available columns are {list(frame.columns)}"
        )
    frame = frame[[args.user_column, args.timestamp_column, args.event_type_column]].copy()
    frame = frame.dropna(subset=[args.user_column, args.timestamp_column, args.event_type_column])
    if frame.empty:
        raise ValueError("raw event-level source has no complete event rows")
    frame["_source_order"] = range(len(frame))
    frame["_user"] = frame[args.user_column].astype(str)
    frame["_timestamp"] = _absolute_utc_timestamps(frame[args.timestamp_column])
    frame["_timestamp_ns"] = frame["_timestamp"].astype("int64")
    frame = frame.sort_values(["_timestamp_ns", "_source_order"], kind="stable").reset_index(drop=True)
    frame["_day"] = frame["_timestamp"].dt.strftime("%Y-%m-%d")

    raw_types = frame[args.event_type_column]
    numeric_types = pd.to_numeric(raw_types, errors="coerce")
    if numeric_types.notna().all() and bool((numeric_types >= 0).all()) and bool(
        (numeric_types == numeric_types.astype("int64")).all()
    ):
        # Preserve the dataset's native numeric event IDs so checkpoints see
        # the same type vocabulary used during training.
        labels = sorted({int(value) for value in numeric_types})
        event_type_map = {str(label): label for label in labels}
        frame["_event_type"] = numeric_types.astype("int64")
    else:
        labels = sorted(raw_types.astype(str).unique())
        event_type_map = {label: index for index, label in enumerate(labels)}
        frame["_event_type"] = raw_types.astype(str).map(event_type_map)
    event_type_count = max((int(value) for value in frame["_event_type"]), default=-1) + 1

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, Any]] = []
    for window_id, (days, subset) in enumerate(
        _chronological_windows(frame, args.min_events_per_window)
    ):
        subset = subset.sort_values(["_timestamp_ns", "_source_order"], kind="stable").reset_index(drop=True)
        times_ns = subset["_timestamp_ns"].astype("int64").tolist()
        train_end, validation_end = _tie_safe_cuts(times_ns)
        if not (0 < train_end < validation_end < len(subset)):
            raise ValueError(
                f"chronological window {days[0]}..{days[-1]} cannot support non-empty "
                "70/10/20 partitions without splitting tied timestamps; merge more days "
                "with --min-events-per-window"
            )
        window_dir = out / f"window_{window_id:03d}"
        splits = {
            "train": _write_split(subset, 0, train_end, window_dir / "train.csv"),
            "validation": _write_split(subset, train_end, validation_end, window_dir / "validation.csv"),
            "test": _write_split(subset, validation_end, len(subset), window_dir / "test.csv"),
        }
        train_end_time = pd.to_datetime(splits["train"]["end_time"], utc=True)
        validation_start_time = pd.to_datetime(
            splits["validation"]["start_time"], utc=True
        )
        validation_end_time = pd.to_datetime(
            splits["validation"]["end_time"], utc=True
        )
        test_start_time = pd.to_datetime(splits["test"]["start_time"], utc=True)
        if not (
            train_end_time < validation_start_time
            and validation_end_time < test_start_time
        ):
            raise RuntimeError(
                f"window {window_id} does not satisfy train < validation < test "
                "on the global UTC clock"
            )
        split_frames = {
            split_name: pd.read_csv(window_dir / f"{split_name}.csv")
            for split_name in ("train", "validation", "test")
        }
        combined_parts = []
        split_indices: dict[str, list[int]] = {}
        source_row = 0
        for split_name in ("train", "validation", "test"):
            part = split_frames[split_name]
            indices = list(range(source_row, source_row + len(part)))
            split_indices[split_name] = indices
            source_row += len(part)
            combined_parts.append(part)
        combined = pd.concat(combined_parts, ignore_index=True)
        combined_path = window_dir / "combined.csv"
        combined.to_csv(combined_path, index=False)
        split_manifest = {
            "format_version": 1,
            "strategy": "chronological_global_event_time",
            "data_path": str(combined_path.resolve()),
            "data_sha256": _sha256(combined_path),
            "source_row_count": int(len(combined)),
            "counts": {
                name: len(indices) for name, indices in split_indices.items()
            },
            "splits": split_indices,
        }
        (window_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )
        windows.append({
            "window_id": window_id,
            "start_date": days[0],
            "end_date": days[-1],
            "start_time": subset.iloc[0]["_timestamp"].isoformat(),
            "end_time": subset.iloc[-1]["_timestamp"].isoformat(),
            "days": days,
            "event_count": int(len(subset)),
            "splits": splits,
            "training_data": "combined.csv",
            "training_split_manifest": "split_manifest.json",
        })

    for previous_window, current_window in zip(windows, windows[1:]):
        if not pd.Timestamp(previous_window["end_time"]) < pd.Timestamp(
            current_window["start_time"]
        ):
            raise RuntimeError(
                "chronological windows overlap or are out of order: "
                f"{previous_window['window_id']} then {current_window['window_id']}"
            )

    manifest = {
        "benchmark_id": "taobao_real_chronological",
        "version": 1,
        "source_file": str(source),
        "source_sha256": _sha256(source) if source.is_file() else None,
        "global_clock": "UTC absolute event timestamp",
        "event_type_count": event_type_count,
        "event_type_map": event_type_map,
        "windowing": {
            "unit": "UTC calendar day; adjacent sparse days merged in chronological order",
            "min_events_per_window": args.min_events_per_window,
            "within_window_split": [0.70, 0.10, 0.20],
            "split_boundary": "global event-time order; tied timestamps stay together",
            "random_split": False,
        },
        "windows": windows,
    }
    (out / "chronological_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "event_type_map.json").write_text(
        json.dumps(event_type_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(windows)} chronological windows under {out}")


if __name__ == "__main__":
    main()
