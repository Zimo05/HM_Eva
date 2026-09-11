import csv
import json
from pathlib import Path

from core.runner import _make_replay_buffer, _serialized_replay_row


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


def test_replay_is_deterministic_and_task_law_balanced(tmpdir):
    root = Path(str(tmpdir)) / "data"
    rows = {
        0: ["A", "B", "A", "B"],
        1: ["C", "D", "C", "D"],
    }
    manifest = [
        "task_id,stage_label,split_index,regime_id,split,sequence_id",
    ]
    for task, laws in rows.items():
        task_dir = root / f"task_{task:02d}"
        task_dir.mkdir(parents=True)
        with (task_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("event_times", "event_types"))
            writer.writeheader()
            for index, _law in enumerate(laws):
                writer.writerow({
                    "event_times": json.dumps([task + 0.1, task + 0.2]),
                    "event_types": "[0,1]",
                })
                manifest.append(
                    f"{task},task, {index}, {_law},train,task_{task:02d}_train_{index:05d}".replace(" ", "")
                )
    (root / "ground_truth_manifest.csv").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

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
    assert set(metadata["task_law_bytes"]) == {"0:A", "0:B", "1:C", "1:D"}
