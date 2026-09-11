import importlib.util
import json
import sys
from pathlib import Path


def load_generator():
    path = Path(__file__).resolve().parents[2] / "Datasets" / "CL" / "generate_continual_hawkes.py"
    spec = importlib.util.spec_from_file_location("hm_cl_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unified_curriculum_matches_instruction():
    module = load_generator()
    args = module.parse_args([])
    assert args.benchmark == "unified"
    assert args.output == Path("Datasets/CL/hm_continual_v2")
    stages = module._unified_schedule()
    assert [stage.task_id for stage in stages] == list(range(10))
    assert stages[4].mixture == {"B_prime_1": 1.0}
    assert stages[5].mixture == {"A_2": 1.0}
    assert stages[6].mixture == {"C_1": 0.9, "X_transient": 0.1}
    assert stages[7].mixture == {"A_merge": 1.0}
    assert stages[8].recurrence_of == "A_1"


def test_canonical_manifest_separates_persistent_and_transient_laws():
    root = Path(__file__).resolve().parents[2] / "Datasets" / "CL" / "hm_continual_v2"
    payload = json.loads((root / "benchmark_manifest.json").read_text())

    assert payload["format_version"] == 2
    assert payload["benchmark_id"] == "CL-core-v2"
    assert payload["benchmark"] == "CL-core-v2"
    assert payload["version"] == 2
    assert payload["persistent_regimes"] == [
        "A_1", "B_1", "C_1", "B_prime_1", "A_2", "A_merge", "E_1"
    ]
    assert payload["transient_regimes"] == ["X_transient"]
    assert payload["tasks"][6]["regime_weights"] == {
        "C_1": 0.9, "X_transient": 0.1
    }
    assert payload["controls"][0]["splits"]["test"].endswith(
        "controls/task_06_no_transient/test.csv"
    )
    assert payload["tasks"][6]["control"] == "controls/task_06_no_transient"
    assert payload["tasks"][6]["paired_control"] == "controls/task_06_no_transient"


def test_cl_protocol_is_the_manifest_boundary():
    from Evaluation.core.cl_protocol import CLProtocol

    root = Path(__file__).resolve().parents[2] / "Datasets" / "CL" / "hm_continual_v2"
    protocol = CLProtocol.load(root)

    assert protocol.task_ids == tuple(range(protocol.num_tasks))
    assert protocol.task(6).shift_type == "transient_anomaly"
    assert protocol.task(6).paired_control == "controls/task_06_no_transient"
    assert protocol.control_split_path(protocol.task(6).paired_control, "test").is_file()
    assert protocol.seen_persistent_regimes(6) == frozenset({
        "A_1", "B_1", "C_1", "B_prime_1", "A_2"
    })
    assert protocol.task_ids_between(4, None) == (4, 5, 6, 7, 8, 9)
