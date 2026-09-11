"""Shared implementation used by the explicit experiment entry points."""

from .cl_protocol import CLAnchorSpec, CLProtocol, CLTaskSpec
from .cl_metrics import (
    AdaptationRecord,
    CLMetricEngine,
    FrozenAnchorRecord,
    FrozenLawMatrix,
    HMStateRecord,
    TaskBoundaryRecord,
    build_frozen_anchor_matrix,
    compute_adaptation_metrics,
    compute_fwt,
    compute_retention_metrics,
    compute_rrr,
    compute_task_boundary_metrics,
)
from .runner import run_continual_job, run_diagnostic_job, run_stationary_job

__all__ = [
    "CLAnchorSpec",
    "CLProtocol",
    "CLTaskSpec",
    "AdaptationRecord",
    "CLMetricEngine",
    "FrozenAnchorRecord",
    "FrozenLawMatrix",
    "HMStateRecord",
    "TaskBoundaryRecord",
    "build_frozen_anchor_matrix",
    "compute_adaptation_metrics",
    "compute_fwt",
    "compute_retention_metrics",
    "compute_rrr",
    "compute_task_boundary_metrics",
    "run_stationary_job",
    "run_continual_job",
    "run_diagnostic_job",
]
