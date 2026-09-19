"""Regression checks that batch evaluation does not retune training."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "Evaluation"
for import_root in (ROOT, EVALUATION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.adapters import (
    evaluate_hm,
    resolve_hm_upstream_h_tree,
    stationary_command,
)
from core.data import prepare_hm_train_sequence_summary
from core.hm_bootstrap import expected_stationary_hm_upstream
from core.manifest import compatible
from core.runner import (
    _baseline_command,
    _continual_cl_config,
    _hm_continual_resume_checkpoint,
)
from core.specs import JobSpec


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
    cl_config = _continual_cl_config(args, protocol, "full")
    assert cl_config["epochs_per_task"] == 60
    assert cl_config["wake"]["retrieval_visit_chunk_size"] == 256

    command, _cwd, _env = _baseline_command(
        "THP",
        args,
        ROOT / "prepared",
        ROOT / "output",
        None,
        False,
    )
    assert command[command.index("--epochs") + 1] == "60"
    assert command[command.index("--batch-size") + 1] == "64"
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
    assert rmtpp_command[rmtpp_command.index("--epochs") + 1] == "60"
    assert rmtpp_command[rmtpp_command.index("--batch-size") + 1] == "64"
    assert rmtpp_command[rmtpp_command.index("--learning-rate") + 1] == "5e-4"
    assert rmtpp_command[rmtpp_command.index("--hidden-size") + 1] == "64"
    assert rmtpp_command[rmtpp_command.index("--mc-samples") + 1] == "32"
    assert (
        rmtpp_command[rmtpp_command.index("--early-stop-patience") + 1]
        == "20"
    )
    assert rmtpp_command[rmtpp_command.index("--lr-patience") + 1] == "6"
    assert rmtpp_command[rmtpp_command.index("--lr-factor") + 1] == "0.3"


def test_hm_continual_propagates_validation_best_state():
    protocol = Namespace(task_ids=(0, 1, 2))
    target = ROOT / "tmp-hm-continual"

    assert _hm_continual_resume_checkpoint(
        target,
        protocol,
        0,
        None,
    ) is None
    assert _hm_continual_resume_checkpoint(
        target,
        protocol,
        2,
        None,
    ) == target / "checkpoint" / "task_01_best.pt"

    explicit = target / "external_state.pt"
    assert _hm_continual_resume_checkpoint(
        target,
        protocol,
        2,
        explicit,
    ) == explicit


def test_continual_manifest_rejects_results_without_learner_protocol_binding():
    keys = (
        "job_key",
        "dataset",
        "model",
        "condition",
        "seed",
        "variant",
        "rank",
        "task_start",
        "task_end",
        "inputs",
    )
    existing = {key: None for key in keys}
    existing["arguments"] = {}
    existing["inputs"] = []
    current = {
        **existing,
        "learner_config": {
            "wake": {"retrieval_visit_chunk_size": 256},
        },
    }
    assert not compatible(existing, current)


def test_stationary_hm_batch_size_reaches_train_and_evaluate():
    args = Namespace(
        epochs=None,
        smoke=False,
        seed=27,
        batch_size=None,
        eval_batch_size=128,
        device="cpu",
        python_executable=sys.executable,
        variant="13",
    )
    spec = JobSpec(dataset="dws", model="HM")
    command, _cwd, _env = stationary_command(
        spec, args, ROOT / "prepared", prepared=ROOT / "prepared"
    )

    assert command[command.index("--validation-batch-size") + 1] == "128"
    assert command[command.index("--epochs") + 1] == "60"

    with patch("core.adapters.run_command") as run:
        evaluate_hm(spec, args, ROOT / "output", {})
    evaluate_command = run.call_args.args[0]
    assert evaluate_command[evaluate_command.index("--eval-batch-size") + 1] == "128"


def test_stationary_dws_hm_uses_variant_h_tree_and_depth_zero():
    args = Namespace(
        epochs=None,
        smoke=False,
        seed=27,
        batch_size=None,
        eval_batch_size=64,
        device="cpu",
        python_executable=sys.executable,
        variant="13",
    )
    spec = JobSpec(dataset="dws", model="HM")
    command, _cwd, _env = stationary_command(
        spec, args, ROOT / "prepared", prepared=ROOT / "prepared"
    )

    assert command[command.index("--tree-init-depth") + 1] == "0"
    assert command[command.index("--cold-start-epochs") + 1] == "5"
    assert command[command.index("--z-dim") + 1] == "50"
    assert command[command.index("--node-dim") + 1] == "128"
    assert command[command.index("--memory-key-dim") + 1] == "64"
    expected_stationary_hm = {
        "--frontier-min-experts": "2",
        "--frontier-budget": "7",
        "--frontier-routing-temperature": "1.10",
        "--frontier-exploration": "0",
        "--frontier-confidence-weight": "0.60",
        "--frontier-compute-cost": "0.005",
        "--frontier-posterior-temperature": "0.85",
        "--frontier-credible-mass": "0.30",
        "--frontier-owner-confidence": "0.50",
        "--max-writes-per-sequence": "8",
        "--semantic-blend": "0",
        "--leaf-symmetry-scale": "0",
        "--light-replay-budget": "128",
        "--route-mix-weight": "0",
        "--route-posterior-weight": "0",
        "--route-distill-weight": "0.25",
        "--route-mi-weight": "0.15",
        "--route-balance-weight": "0.10",
        "--route-energy-temperature": "1.0",
        "--route-encoder-warmup-epochs": "0",
        "--route-encoder-grad-scale": "0.08",
        "--route-encoder-reliability-decay": "0.80",
        "--route-teacher-temperature": "0.85",
        "--route-balance-batch-size": "32",
        "--prune-warmup-epochs": "12",
        "--merge-min-replay": "12",
        "--deep-min-interval": "3",
        "--deep-computation-cost": "0.05",
        "--deep-prior-probability": "0.10",
        "--deep-prior-weight": "0.01",
        "--deep-evidence-budget": "32",
        "--topology-inertia-strength": "0.03",
        "--topology-inertia-tau": "3.0",
        "--alignment-epochs": "5",
        "--alignment-batch-size": "16",
        "--alignment-lr": "0.001",
        "--alignment-weight-decay": "0.00001",
        "--alignment-temperature": "1.0",
        "--alignment-grad-clip": "5.0",
    }
    for option, expected in expected_stationary_hm.items():
        assert command[command.index(option) + 1] == expected
    h_tree = Path(command[command.index("--h-tree") + 1])
    original_root = (ROOT / "Datasets" / "DWS" / "../../../HawkesMemory_wfy").resolve()
    assert h_tree == (original_root / "Data" / "tree_13" / "h_tree_13.pt").resolve()
    sequence_summary = Path(
        command[command.index("--sequence-summary") + 1]
    )
    assert sequence_summary == (
        ROOT / "prepared" / "sequence_summary_train.csv"
    ).resolve()
    assert command[command.index("--residual-init-scale") + 1] == "0.08"
    assert command[command.index("--residual-init-rank") + 1] == "4"
    assert command[command.index("--residual-init-grad-clip") + 1] == "0"
    assert command[command.index("--leaf-symmetry-scale") + 1] == "0"
    assert "--stability-constrained-cold-start" not in command

    smoke_args = Namespace(**{**vars(args), "smoke": True})
    smoke_command, _cwd, _env = stationary_command(
        spec, smoke_args, ROOT / "prepared", prepared=ROOT / "prepared"
    )
    assert "--h-tree" in smoke_command
    assert "--sequence-summary" not in smoke_command

    resolved, metadata = resolve_hm_upstream_h_tree("20")
    assert resolved == (original_root / "Data" / "tree_20" / "h_tree_one_circle.pt").resolve()
    assert metadata["node_dim"] == 128


def test_stationary_discovery_hm_uses_one_generic_upstream_contract():
    args = Namespace(
        epochs=None,
        smoke=False,
        seed=27,
        batch_size=None,
        eval_batch_size=64,
        device="cpu",
        python_executable=sys.executable,
        variant=None,
    )
    expected_wavefront = {
        "retweet": "128",
        "taobao": "32",
        "stackoverflow": "64",
    }
    expected_cold_start = {
        "retweet": "20",
        "taobao": "40",
        "stackoverflow": "40",
    }
    for dataset, wavefront in expected_wavefront.items():
        spec = JobSpec(dataset=dataset, model="HM")
        command, _cwd, _env = stationary_command(
            spec,
            args,
            ROOT / "output" / dataset,
            prepared=ROOT / "prepared" / dataset,
        )
        descriptor = expected_stationary_hm_upstream(
            ROOT / "prepared" / dataset,
            dataset,
        )
        assert Path(command[command.index("--h-tree") + 1]) == descriptor.h_tree
        assert (
            Path(command[command.index("--sequence-summary") + 1])
            == descriptor.sequence_summary
        )
        assert command[command.index("--wake-wavefront-batch-size") + 1] == wavefront
        if dataset == "retweet":
            assert (
                command[command.index("--wake-transaction-mode") + 1]
                == "snapshot"
            )
        else:
            assert "--wake-transaction-mode" not in command
        assert "--residual-init-rank" in command
        assert command[command.index("--cold-start-epochs") + 1] == expected_cold_start[dataset]
        if dataset == "stackoverflow":
            assert command[command.index("--epochs") + 1] == "120"
            assert command[command.index("--frontier-budget") + 1] == "5"
            assert command[command.index("--frontier-routing-temperature") + 1] == "0.9"
            assert command[command.index("--residual-init-scale") + 1] == "0.06"
        else:
            assert command[command.index("--epochs") + 1] == "60"
            assert command[command.index("--frontier-budget") + 1] == "7"
            assert command[command.index("--frontier-routing-temperature") + 1] == "1.10"
            assert command[command.index("--residual-init-scale") + 1] == "0.08"
        assert "--stability-constrained-cold-start" in command
        assert (
            command[command.index("--cold-start-rho-base") + 1]
            == "0.88"
        )
        assert (
            command[command.index("--residual-rho-safe") + 1]
            == "0.95"
        )


def test_hm_train_sequence_summary_filters_only_train_source_indices():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        canonical = root / "canonical.csv"
        canonical.write_text(
            "event_times,event_types,source_index\n"
            "[1.0],[0],10\n"
            "[2.0],[0],11\n"
            "[3.0],[0],12\n"
            "[4.0],[0],13\n",
            encoding="utf-8",
        )
        split_manifest = root / "split_manifest.json"
        split_manifest.write_text(
            json.dumps(
                {
                    "data_path": str(canonical),
                    "splits": {
                        "train": [0, 2],
                        "validation": [1],
                        "test": [3],
                    },
                    "source_index_field": "source_index",
                }
            ),
            encoding="utf-8",
        )
        source_summary = root / "sequence_summary.csv"
        source_summary.write_text(
            "leaf_position,cluster_id,mu,A,decay,sequences\n"
            "left,0,mu-left,A-left,decay-left,\"[10, 11]\"\n"
            "right,1,mu-right,A-right,decay-right,\"[12, 13]\"\n",
            encoding="utf-8",
        )

        output = prepare_hm_train_sequence_summary(
            source_summary,
            canonical,
            split_manifest,
            root / "sequence_summary_train.csv",
        )

        with output.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["sequences"] for row in rows] == ["[10]", "[12]"]
        assert [row["mu"] for row in rows] == ["mu-left", "mu-right"]


def test_hawkes_backbone_matches_original_checkout_when_available():
    original_root = Path(
        os.environ.get("HM_ORIGINAL_ROOT", "/Volumes/shenzm/Shuang_RA/HawkesMemory_wfy")
    )
    original = original_root / "Memory" / "HawkesBackbone.py"
    if not original.is_file():
        return
    current = ROOT / "Models" / "HawkesMemory" / "Memory" / "HawkesBackbone.py"
    assert hashlib.sha256(current.read_bytes()).hexdigest() == hashlib.sha256(
        original.read_bytes()
    ).hexdigest()


def test_thp_defaults_match_shared_training_protocol():
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
        "selection_metric": args.selection_metric,
    } == {
        "epochs": 60,
        "learning_rate": 2e-4,
        "d_model": 512,
        "d_rnn": 256,
        "d_inner": 1024,
        "d_k": 128,
        "d_v": 128,
        "num_layers": 3,
        "label_smoothing": 0.01,
        "selection_metric": "ll",
    }

    main_defaults = _main_argument_defaults(ROOT / "Models" / "THP" / "Main.py")
    assert main_defaults["-epoch"] == 60
    assert main_defaults["-batch_size"] == 64
    assert main_defaults["-d_model"] == 512
    assert main_defaults["-d_rnn"] == 256
    assert main_defaults["-d_inner_hid"] == 1024
    assert main_defaults["-d_k"] == 128
    assert main_defaults["-d_v"] == 128
    assert main_defaults["-n_layers"] == 3
    assert main_defaults["-lr"] == 2e-4
    assert main_defaults["-smooth"] == 0.01
    assert main_defaults["-selection_metric"] == "ll"

    launcher = (ROOT / "Models" / "THP" / "run.sh").read_text(
        encoding="utf-8"
    )
    assert 'LEARNING_RATE="${LEARNING_RATE:-0.0002}"' in launcher
    assert 'D_MODEL="${D_MODEL:-512}"' in launcher
    assert 'D_RNN="${D_RNN:-256}"' in launcher
    assert 'D_INNER="${D_INNER:-1024}"' in launcher
    assert 'D_K="${D_K:-128}"' in launcher
    assert 'D_V="${D_V:-128}"' in launcher
    assert 'NUM_LAYERS="${NUM_LAYERS:-3}"' in launcher
    assert 'LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.01}"' in launcher
    assert 'SELECTION_METRIC="${SELECTION_METRIC:-accuracy}"' not in launcher
    assert "WEIGHT_DECAY" not in launcher


def test_rmtpp_defaults_match_shared_training_protocol():
    defaults = _main_argument_defaults(ROOT / "Models" / "RMTPP" / "run_experiment.py")
    assert defaults["--epochs"] == 60
    assert defaults["--batch-size"] == 64
    assert defaults["--learning-rate"] == 5e-4
    assert defaults["--hidden-size"] == 64
    assert defaults["--mc-samples"] == 32
    assert defaults["--early-stop-patience"] == 20
    assert defaults["--lr-patience"] == 6
    assert defaults["--lr-factor"] == 0.3
