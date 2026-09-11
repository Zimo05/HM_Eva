import csv
import json
from pathlib import Path

from core.runner import (
    _make_replay_buffer,
    _serialized_replay_row,
    _write_adaptation_prefix,
)


def test_replay_serialization_does_not_exceed_budget(tmpdir):
    root = Path(str(tmpdir)) / "data"
    task = root / "task_00"
    task.mkdir(parents=True)
    with (task / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
        writer.writeheader()
        for _ in range(3):
            writer.writerow({"event_times": "[0.1, 0.2]", "event_types": "[0, 1]"})
    output, actual = _make_replay_buffer(root, 0, 4096, root / "replay.csv")
    assert output.is_file()
    assert actual <= 4096


def test_replay_is_deterministic_and_task_balanced_without_oracle_labels(tmpdir):
    root = Path(str(tmpdir)) / "data"
    for task in (0, 1):
        task_dir = root / f"task_{task:02d}"
        task_dir.mkdir(parents=True)
        with (task_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
            writer.writeheader()
            for _ in range(4):
                writer.writerow({
                    "event_times": json.dumps([task + 0.1, task + 0.2]),
                    "event_types": "[0,1]",
                })

    row_bytes = len(_serialized_replay_row({
        "event_times": json.dumps([0.1, 0.2]),
        "event_types": "[0,1]",
    }))
    budget = len(b"event_times,event_types\r\n") + 4 * row_bytes
    metadata = {}
    first, actual = _make_replay_buffer(
        root, 1, budget, root / "replay_a.csv", selection_metadata=metadata
    )
    second, second_actual = _make_replay_buffer(
        root, 1, budget, root / "replay_b.csv"
    )

    assert actual == second_actual <= budget
    assert first.read_bytes() == second.read_bytes()
    with first.open("r", newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert len(selected) == 4
    assert sum(row["event_times"].startswith("[0.") for row in selected) == 2
    assert sum(row["event_times"].startswith("[1.") for row in selected) == 2
    assert set(metadata["task_bytes"]) == {"0", "1"}
    assert "task_law_bytes" not in metadata


def test_adaptation_prefix_uses_exact_observed_event_count(tmpdir):
    root = Path(str(tmpdir))
    source = root / "support.csv"
    source.write_text(
        "event_times,event_types\n"
        "\"[0.1, 0.2, 0.3]\",\"[0, 1, 0]\"\n",
        encoding="utf-8",
    )
    output = root / "prefix.csv"

    assert _write_adaptation_prefix(source, output, 1)
    with output.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["event_times"]) == [0.1]
    assert json.loads(row["event_types"]) == [0]
