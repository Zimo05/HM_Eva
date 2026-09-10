import importlib.util
import sys
from pathlib import Path


def load_generator():
    path = Path(__file__).resolve().parents[2] / "Datasets" / "Data" / "CL" / "generate_continual_hawkes.py"
    spec = importlib.util.spec_from_file_location("hm_cl_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unified_curriculum_matches_instruction():
    module = load_generator()
    stages = module._unified_schedule()
    assert [stage.task_id for stage in stages] == list(range(10))
    assert stages[4].mixture == {"B_prime_1": 1.0}
    assert stages[5].mixture == {"A_2": 1.0}
    assert stages[6].mixture == {"C_1": 0.9, "X_transient": 0.1}
    assert stages[7].mixture == {"A_merge": 1.0}
    assert stages[8].recurrence_of == "A_1"
