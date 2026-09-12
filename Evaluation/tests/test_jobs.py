import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_public_entry_points_exist():
    datasets = ("dws", "retweet", "taobao", "stackoverflow")
    models = ("HM", "RMTPP", "FullyNN", "THP", "S2P2", "AttNHP", "TPP_LLM")
    for dataset in datasets:
        for model in models:
            path = ROOT / f"evaluate_{dataset}_{model}.py"
            assert path.is_file()
            text = path.read_text(encoding="utf-8")
            assert f'dataset="{dataset}"' in text
            assert f'model="{model}"' in text


def test_easytpp_continual_entry_points_exist():
    for model in ("S2P2", "AttNHP"):
        for strategy in ("sequential", "joint", "replay"):
            path = ROOT / f"evaluate_continual_{model}_{strategy}.py"
            assert path.is_file()
            text = path.read_text(encoding="utf-8")
            assert f'model="{model}"' in text
            assert f'strategy="{strategy}"' in text


def test_registry_identities_are_unique():
    with (ROOT / "experiment_registry.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert len({row["job_id"] for row in rows}) == len(rows)
    assert len({row["identity"] for row in rows}) == len(rows)


def test_no_evaluate_all_entry_point():
    assert not (ROOT / "evaluate_all.py").exists()
