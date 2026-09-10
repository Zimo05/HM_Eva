import csv
from pathlib import Path

from core.runner import _make_replay_buffer


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
