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


def test_easytpp_prediction_protocol_wiring():
    source = (ROOT.parent / "Models" / "EasyTPP" / "run_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "def _predict_attnhp_one_step(" in source
    assert "sample_times = relative_dtimes + prefix_times.unsqueeze(-1)" in source
    assert "collect_predictions=False" in source
    assert "def _plot_metrics(" in source
    assert 'plot_dir / "likelihood.png"' in source
    assert 'plot_dir / "test_metrics.png"' in source


def test_registry_identities_are_unique():
    with (ROOT / "experiment_registry.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert len({row["job_id"] for row in rows}) == len(rows)
    assert len({row["identity"] for row in rows}) == len(rows)


def test_easytpp_registry_coverage_and_replay_dependencies():
    with (ROOT / "experiment_registry.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    easy_rows = [row for row in rows if row["model"] in {"S2P2", "AttNHP"}]
    assert len(easy_rows) == 58
    by_job_id = {row["job_id"]: row for row in rows}
    for row in easy_rows:
        assert (ROOT / row["script"]).is_file()
        if row["condition"] == "replay":
            dependency = by_job_id[row["depends_on"]]
            assert dependency["model"] == "HM"
            assert dependency["condition"] == "full"
            assert dependency["seed"] == row["seed"]
            assert row["arguments"].startswith("--hm-resource-root ")


def test_no_evaluate_all_entry_point():
    assert not (ROOT / "evaluate_all.py").exists()
