"""Regression checks that batch evaluation does not retune training."""

from __future__ import annotations

import ast
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "Evaluation"
for import_root in (ROOT, EVALUATION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.runner import _baseline_command, _continual_cl_config


def _load_thp_runner():
    path = ROOT / "Models" / "THP" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("thp_run_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main_argument_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            not isinstance(function, ast.Attribute)
            or function.attr != "add_argument"
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    return defaults


def test_hm_and_baseline_runner_defaults_are_training_protocol_stable():
    args = Namespace(
        epochs=None,
        smoke=False,
        seed=27,
        batch_size=None,
        device="cpu",
        python_executable=sys.executable,
    )
    protocol = Namespace(benchmark_id="synthetic", version=1)
    assert _continual_cl_config(args, protocol, "full")["epochs_per_task"] == 50

    command, _cwd, _env = _baseline_command(
        "THP",
        args,
        ROOT / "prepared",
        ROOT / "output",
        None,
        False,
    )
    assert command[command.index("--epochs") + 1] == "20"
    assert "--learning-rate" not in command
    assert "--d-model" not in command
    assert "--label-smoothing" not in command

    rmtpp_command, _cwd, _env = _baseline_command(
        "RMTPP",
        args,
        ROOT / "prepared",
        ROOT / "output",
        None,
        True,
        dataset_label="CL-core-v2",
    )
    assert rmtpp_command[rmtpp_command.index("--dataset") + 1] == "taobao"
    assert (
        rmtpp_command[rmtpp_command.index("--dataset-label") + 1]
        == "CL-core-v2"
    )
    assert rmtpp_command[rmtpp_command.index("--epochs") + 1] == "80"
    assert rmtpp_command[rmtpp_command.index("--batch-size") + 1] == "16"
    assert rmtpp_command[rmtpp_command.index("--learning-rate") + 1] == "5e-4"
    assert rmtpp_command[rmtpp_command.index("--hidden-size") + 1] == "64"
    assert rmtpp_command[rmtpp_command.index("--mc-samples") + 1] == "32"
    assert (
        rmtpp_command[rmtpp_command.index("--early-stop-patience") + 1]
        == "20"
    )
    assert rmtpp_command[rmtpp_command.index("--lr-patience") + 1] == "6"
    assert rmtpp_command[rmtpp_command.index("--lr-factor") + 1] == "0.3"


def test_thp_defaults_match_the_pre_batch_training_protocol():
    module = _load_thp_runner()
    old_argv = sys.argv
    sys.argv = [
        "run_experiment.py",
        "--dataset",
        "taobao",
        "--output-dir",
        str(ROOT / "tmp-thp-output"),
        "--archive",
        str(ROOT / "tmp-thp-output.tar.gz"),
    ]
    try:
        args = module.parse_args()
    finally:
        sys.argv = old_argv

    assert {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "d_rnn": args.d_rnn,
        "d_inner": args.d_inner,
        "d_k": args.d_k,
        "d_v": args.d_v,
        "num_layers": args.num_layers,
        "label_smoothing": args.label_smoothing,
    } == {
        "epochs": 80,
        "learning_rate": 1e-4,
        "d_model": 64,
        "d_rnn": 256,
        "d_inner": 128,
        "d_k": 16,
        "d_v": 16,
        "num_layers": 4,
        "label_smoothing": 0.1,
    }

    main_defaults = _main_argument_defaults(ROOT / "Models" / "THP" / "Main.py")
    assert main_defaults["-epoch"] == 30
    assert main_defaults["-batch_size"] == 16
    assert main_defaults["-d_model"] == 64
    assert main_defaults["-d_rnn"] == 256
    assert main_defaults["-d_inner_hid"] == 128
    assert main_defaults["-d_k"] == 16
    assert main_defaults["-d_v"] == 16
    assert main_defaults["-n_layers"] == 4
    assert main_defaults["-lr"] == 1e-4
    assert main_defaults["-smooth"] == 0.1

    launcher = (ROOT / "Models" / "THP" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert 'LEARNING_RATE="${LEARNING_RATE:-0.0001}"' in launcher
    assert 'D_MODEL="${D_MODEL:-64}"' in launcher
    assert 'NUM_LAYERS="${NUM_LAYERS:-4}"' in launcher
    assert 'LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"' in launcher
    assert "WEIGHT_DECAY" not in launcher


def test_rmtpp_defaults_match_the_first_recommendation():
    defaults = _main_argument_defaults(ROOT / "Models" / "RMTPP" / "run_experiment.py")
    assert defaults["--epochs"] == 80
    assert defaults["--batch-size"] == 16
    assert defaults["--learning-rate"] == 5e-4
    assert defaults["--hidden-size"] == 64
    assert defaults["--mc-samples"] == 32
    assert defaults["--early-stop-patience"] == 20
    assert defaults["--lr-patience"] == 6
    assert defaults["--lr-factor"] == 0.3
