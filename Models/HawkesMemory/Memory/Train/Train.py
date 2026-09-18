"""End-to-end Hawkes Memory Tree training pipeline.

This compatibility entry point re-exports the established training API while
composing the implementation from focused training modules.
"""

from __future__ import annotations

# When this file is executed as ``python Train/Train.py``, Python puts the
# ``Train`` directory itself first on sys.path. Some deployed worktrees also
# contain a legacy nested ``Train/Train/`` package, which can shadow the
# intended outer ``Train`` package and load stale CLI definitions. Pin the
# package root explicitly so direct-script and module execution use the same
# sources.
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
_MEMORY_ROOT = _TRAIN_DIR.parent
try:
    sys.path.remove(str(_TRAIN_DIR))
except ValueError:
    pass
sys.path.insert(0, str(_MEMORY_ROOT))

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import (
    _assert_finite_without_cuda_sync,
    _differentiable_merge_settings,
    _frontier_config_from_checkpoint,
)
from Train.TrainingCLI import (
    _leaf_spectral_radius_summary,
    _parse_args,
    _run_semantic_smoke_test,
    main,
)
from Train.TrainingTrainer import MemoryTreeTrainer


if __name__ == "__main__":
    main()
