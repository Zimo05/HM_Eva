from argparse import Namespace
from pathlib import Path

import pytest

from core.io import write_json
from core.manifest import begin, compatible


def test_resume_rejects_changed_identity(tmpdir):
    tmp_path = Path(str(tmpdir))
    base = {"job_key": "a", "dataset": "dws", "model": "HM", "condition": "full", "seed": 1, "variant": "13", "rank": None, "task_start": None, "task_end": None, "inputs": []}
    assert begin(tmp_path, base, False)
    write_json(tmp_path / "status.json", {"state": "complete"})
    assert not begin(tmp_path, base, True)
    changed = dict(base, seed=2)
    with pytest.raises(RuntimeError):
        begin(tmp_path, changed, True)


def test_resume_accepts_missing_historical_attention_gpu_default():
    base = {
        "job_key": "stationary:retweet:HM:full",
        "dataset": "retweet",
        "model": "HM",
        "condition": "full",
        "seed": 2025,
        "variant": None,
        "rank": None,
        "task_start": None,
        "task_end": None,
        "inputs": [],
        "arguments": {"hm_num_gpus": 2},
    }
    current = {
        **base,
        "arguments": {
            "hm_num_gpus": 2,
            "attention_num_gpus": 1,
            "resume": True,
        },
    }

    assert compatible(base, current)
    current["arguments"]["attention_num_gpus"] = 2
    assert not compatible(base, current)
