"""The canonical continual-learning metric contract.

This module is intentionally independent of a model implementation.  Model
adapters reduce their raw observations to the records below and then call
``CLMetricEngine``.  Keeping the reductions here makes the frozen anchor
matrix, retention metrics, transfer metrics, and recurrence metrics identical
for HM and the flat temporal point-process baselines.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FrozenAnchorRecord:
    """One read-only NLL observation in the persistent anchor matrix."""

    checkpoint_task: int
    regime_id: str
    nll_per_event: float | None
    num_events: int
    evaluation_scope: str = "persistent"


@dataclass(frozen=True)
class TaskBoundaryRecord:
    """Pre/post/scratch observations for one task boundary."""

    task_id: int
    pre_nll: float | None
    post_nll: float | None
    scratch_nll: float | None
    shift_type: str | None = None
    recurrence_of: str | None = None


@dataclass(frozen=True)
class AdaptationRecord:
    """NLL after K support events from a fresh pre-task clone."""

    task_id: int
    K: int
    pre_nll: float | None
    adapted_nll: float | None
    # Optional output dimension for protocol-separated FAST_ADAPT and
    # ONLINE_WRITE curves.  The four fields above remain the canonical
    # record contract; callers that do not provide a protocol get one curve
    # per task as before.
    protocol: str | None = None


@dataclass(frozen=True)
class HMStateRecord:
    """HM-specific state emitted beside the generic CL metrics."""

    task_id: int
    node_count: int | None
    leaf_count: int | None
    episodic_rows: int | None
    episodic_bytes: int | None
    semantic_bytes: int | None
    split_count: int | None
    merge_count: int | None
    prune_count: int | None
    nise: float | None


@dataclass(frozen=True)
class FrozenLawMatrix:
    """Checkpoint x persistent-law NLL matrix used by all retention metrics."""

    values: dict[int, dict[str, float]]
    first_seen: dict[str, int]
    persistent_regimes: frozenset[str]

    def to_rows(self) -> list[dict[str, Any]]:
        regimes = sorted(self.persistent_regimes)
        rows: list[dict[str, Any]] = []
        for checkpoint_task in sorted(self.values):
            values = self.values[checkpoint_task]
            rows.append({
                "checkpoint_task": checkpoint_task,
                **{
                    regime_id: values.get(regime_id)
                    for regime_id in regimes
                },
            })
        return rows


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [numeric for value in values if (numeric := _finite(value)) is not None]
    return sum(clean) / len(clean) if clean else None


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def _normalise_persistent(
    persistent_regimes: Iterable[str] | None,
    first_seen: Mapping[str, int] | None,
    records: Sequence[Any] = (),
) -> frozenset[str]:
    if persistent_regimes is not None:
        return frozenset(str(regime_id) for regime_id in persistent_regimes)
    if first_seen is not None:
        return frozenset(str(regime_id) for regime_id in first_seen)
    return frozenset(
        str(_value(record, "regime_id"))
        for record in records
        if _value(record, "regime_id") is not None
    )


def build_frozen_anchor_matrix(
    records: Sequence[FrozenAnchorRecord | Mapping[str, Any]],
    *,
    first_seen: Mapping[str, int] | None = None,
    persistent_regimes: Iterable[str] | None = None,
) -> FrozenLawMatrix:
    """Build the strict persistent-law matrix from canonical records.

    Records marked as diagnostic/transient or belonging to a non-persistent
    regime are deliberately excluded.  Duplicate cells are accepted only
    when they contain the same value; silently overwriting a cell would hide
    an evaluation isolation bug.
    """

    source = list(records)
    first_seen_map = {
        str(regime_id): _as_int(task_id, f"first_seen[{regime_id!r}]")
        for regime_id, task_id in (first_seen or {}).items()
    }
    persistent = _normalise_persistent(
        persistent_regimes, first_seen_map or None, source
    )
    values: dict[int, dict[str, float]] = {}
    for record in source:
        scope = str(_value(record, "evaluation_scope", "persistent"))
        regime_id = _value(record, "regime_id")
        nll = _finite(_value(record, "nll_per_event"))
        if scope != "persistent" or regime_id is None:
            continue
        regime_id = str(regime_id)
        if regime_id not in persistent or nll is None:
            continue
        checkpoint_task = _as_int(
            _value(record, "checkpoint_task"), "checkpoint_task"
        )
        cell = values.setdefault(checkpoint_task, {})
        previous = cell.get(regime_id)
        if previous is not None and not math.isclose(
            previous, nll, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "duplicate frozen-anchor cell differs: "
                f"task={checkpoint_task}, regime={regime_id!r}, "
                f"values={previous} and {nll}"
            )
        cell[regime_id] = nll
    return FrozenLawMatrix(
        values=values,
        first_seen=first_seen_map,
        persistent_regimes=persistent,
    )


def _coerce_matrix(
    matrix: FrozenLawMatrix | Mapping[int, Mapping[str, float]],
    *,
    first_seen: Mapping[str, int] | None = None,
    persistent_regimes: Iterable[str] | None = None,
) -> FrozenLawMatrix:
    if isinstance(matrix, FrozenLawMatrix):
        return matrix
    values: dict[int, dict[str, float]] = {}
    for task, row in matrix.items():
        values[_as_int(task, "checkpoint_task")] = {
            str(regime_id): numeric
            for regime_id, value in row.items()
            if (numeric := _finite(value)) is not None
        }
    persistent = _normalise_persistent(
        persistent_regimes, first_seen, [
            {"regime_id": regime_id}
            for row in values.values()
            for regime_id in row
        ]
    )
    return FrozenLawMatrix(
        values=values,
        first_seen={
            str(regime_id): _as_int(task_id, f"first_seen[{regime_id!r}]")
            for regime_id, task_id in (first_seen or {}).items()
        },
        persistent_regimes=persistent,
    )


def compute_retention_metrics(
    matrix: FrozenLawMatrix | Mapping[int, Mapping[str, float]],
    *,
    first_seen: Mapping[str, int] | None = None,
    persistent_regimes: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute law-macro CLNLL, forgetting, and BWT from ``M`` only."""

    matrix = _coerce_matrix(
        matrix,
        first_seen=first_seen,
        persistent_regimes=persistent_regimes,
    )
    law_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint_task in sorted(matrix.values):
        current_values = matrix.values[checkpoint_task]
        seen = sorted(
            regime_id
            for regime_id in matrix.persistent_regimes
            if matrix.first_seen.get(regime_id, math.inf) <= checkpoint_task
            and regime_id in current_values
        )
        task_law_rows: list[dict[str, Any]] = []
        for regime_id in seen:
            first_task = matrix.first_seen.get(regime_id)
            if first_task is None:
                continue
            history = [
                (task, matrix.values[task][regime_id])
                for task in sorted(matrix.values)
                if first_task <= task <= checkpoint_task
                and regime_id in matrix.values[task]
            ]
            if not history:
                continue
            baseline = next(
                (
                    (task, value)
                    for task, value in history
                    if task == first_task
                ),
                history[0],
            )
            current_nll = current_values[regime_id]
            best_nll = min(value for _, value in history)
            bwt = (
                baseline[1] - current_nll
                if baseline[0] == first_task and first_task < checkpoint_task
                else None
            )
            row = {
                "checkpoint_task": checkpoint_task,
                "regime_id": regime_id,
                "first_seen_task": first_task,
                "baseline_checkpoint_task": baseline[0],
                "baseline_nll_per_event": baseline[1],
                "current_nll_per_event": current_nll,
                "best_nll_since_first_seen": best_nll,
                "forgetting_nll": current_nll - best_nll,
                "bwt_nll": bwt,
            }
            law_rows.append(row)
            task_law_rows.append(row)
        summary_rows.append({
            "checkpoint_task": checkpoint_task,
            "seen_law_count": len(seen),
            "clnll": _mean(current_values[regime_id] for regime_id in seen),
            "average_forgetting": _mean(
                row["forgetting_nll"] for row in task_law_rows
            ),
            "average_bwt": _mean(
                row["bwt_nll"] for row in task_law_rows
            ),
            "bwt_law_count": sum(
                row["bwt_nll"] is not None for row in task_law_rows
            ),
        })
    return law_rows, summary_rows


def _protocol_task(protocol: Any, task_id: int) -> Any:
    try:
        return protocol.task(task_id)
    except (AttributeError, KeyError):
        tasks = getattr(protocol, "tasks", {})
        return tasks.get(task_id) if isinstance(tasks, Mapping) else None


def _protocol_task_ids(protocol: Any) -> tuple[int, ...]:
    value = getattr(protocol, "task_ids", ())
    return tuple(int(task_id) for task_id in value)


def _protocol_first_seen(protocol: Any) -> Mapping[str, int]:
    return {
        str(regime_id): int(task_id)
        for regime_id, task_id in getattr(protocol, "first_seen", {}).items()
    }


def _protocol_persistent(protocol: Any) -> frozenset[str]:
    return frozenset(
        str(regime_id)
        for regime_id in getattr(protocol, "persistent_regimes", ())
    )


def _new_persistent_regimes(protocol: Any, task_id: int) -> list[str]:
    task = _protocol_task(protocol, task_id)
    if task is None:
        return []
    weights = getattr(task, "regime_weights", {}) or {}
    first_seen = _protocol_first_seen(protocol)
    persistent = _protocol_persistent(protocol)
    return sorted(
        str(regime_id)
        for regime_id, weight in weights.items()
        if float(weight) > 0.0
        and str(regime_id) in persistent
        and first_seen.get(str(regime_id)) == task_id
    )


RETURN_SHIFT_TYPES = frozenset({"exact_recurrence", "long_gap_recurrence"})


def compute_task_boundary_metrics(
    records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    protocol: Any | None = None,
) -> list[dict[str, Any]]:
    """Add adaptation gain and protocol-scoped FWT eligibility to boundaries."""

    output: list[dict[str, Any]] = []
    first_protocol_task = (
        min(_protocol_task_ids(protocol)) if protocol and _protocol_task_ids(protocol)
        else None
    )
    for record in sorted(records, key=lambda item: _as_int(_value(item, "task_id"), "task_id")):
        task_id = _as_int(_value(record, "task_id"), "task_id")
        pre_nll = _finite(_value(record, "pre_nll"))
        post_nll = _finite(_value(record, "post_nll"))
        scratch_nll = _finite(_value(record, "scratch_nll"))
        task = _protocol_task(protocol, task_id) if protocol else None
        shift_type = _value(record, "shift_type")
        recurrence_of = _value(record, "recurrence_of")
        if task is not None:
            shift_type = getattr(task, "shift_type", shift_type)
            recurrence_of = getattr(task, "recurrence_of", recurrence_of)
        new_regimes = _new_persistent_regimes(protocol, task_id) if protocol else []
        eligible = bool(
            protocol is not None
            and new_regimes
            and task_id != first_protocol_task
            and str(shift_type) not in RETURN_SHIFT_TYPES
        )
        if protocol is None:
            fwt_status = "not_available_no_protocol"
        elif not eligible:
            fwt_status = "excluded_not_unseen_law_task"
        elif pre_nll is None or scratch_nll is None:
            fwt_status = "not_available_missing_pre_or_scratch"
        else:
            fwt_status = "available"
        output.append({
            "task_id": task_id,
            "pre_nll": pre_nll,
            "post_nll": post_nll,
            "scratch_nll": scratch_nll,
            "shift_type": shift_type,
            "recurrence_of": recurrence_of,
            "new_persistent_regimes": new_regimes,
            "adaptation_gain_nll": (
                pre_nll - post_nll
                if pre_nll is not None and post_nll is not None else None
            ),
            "fwt_nll": (
                scratch_nll - pre_nll
                if eligible and scratch_nll is not None and pre_nll is not None
                else None
            ),
            "fwt_eligible": eligible,
            "fwt_status": fwt_status,
        })
    return output


def compute_fwt(
    records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    protocol: Any,
) -> dict[str, Any]:
    rows = compute_task_boundary_metrics(records, protocol)
    values = [row["fwt_nll"] for row in rows if row["fwt_nll"] is not None]
    return {
        "rows": rows,
        "average_fwt": _mean(values),
        "eligible_task_count": len(values),
        "status": "available" if values else "not_available",
    }


def _normalised_auc(points: Mapping[int, float]) -> float | None:
    ordered = sorted((int(key), float(value)) for key, value in points.items())
    if not ordered or ordered[0][0] != 0 or len(ordered) < 2:
        return None
    span = ordered[-1][0] - ordered[0][0]
    if span <= 0:
        return None
    area = sum(
        (right_k - left_k) * (left_v + right_v) / 2.0
        for (left_k, left_v), (right_k, right_v)
        in zip(ordered, ordered[1:])
    )
    return area / span


def compute_adaptation_metrics(
    records: Sequence[AdaptationRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute gain/AUC by K, with K=0 as the pre-adaptation baseline."""

    grouped: dict[tuple[str | None, int], dict[int, dict[str, Any]]] = {}
    for record in records:
        task_id = _as_int(_value(record, "task_id"), "task_id")
        k = _as_int(_value(record, "K"), "K")
        if k < 0:
            raise ValueError("adaptation K must be non-negative")
        protocol = _value(record, "protocol")
        protocol = None if protocol in (None, "") else str(protocol)
        pre_nll = _finite(_value(record, "pre_nll"))
        adapted_nll = _finite(_value(record, "adapted_nll"))
        target = grouped.setdefault((protocol, task_id), {})
        if k in target and target[k]["adapted_nll"] != adapted_nll:
            raise ValueError(
                "duplicate adaptation point for "
                f"protocol={protocol!r}, task={task_id}, K={k}"
            )
        target[k] = {
            "task_id": task_id,
            "K": k,
            "pre_nll": pre_nll,
            "adapted_nll": adapted_nll,
            "protocol": protocol,
        }

    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for protocol, task_id in sorted(
        grouped, key=lambda item: (str(item[0]), item[1])
    ):
        points = grouped[(protocol, task_id)]
        ordered = sorted(points.values(), key=lambda row: row["K"])
        baseline = points.get(0, {}).get("adapted_nll")
        if baseline is None:
            baseline = next(
                (row["pre_nll"] for row in ordered if row["pre_nll"] is not None),
                None,
            )
        gains: dict[int, float] = {}
        for row in ordered:
            pre_nll = row["pre_nll"] if row["pre_nll"] is not None else baseline
            gain = (
                pre_nll - row["adapted_nll"]
                if pre_nll is not None and row["adapted_nll"] is not None
                else None
            )
            output_row = {
                **row,
                "pre_nll": pre_nll,
                "gain_nll": gain,
            }
            point_rows.append(output_row)
            if gain is not None:
                gains[row["K"]] = gain
        # A K=0 row is only a valid baseline when it contains an actual
        # measured value.  A placeholder row must not make a later K-only
        # curve look like a normalized adaptation curve.
        has_zero = (
            0 in points
            and points[0].get("adapted_nll") is not None
            and 0 in gains
        )
        auc = _normalised_auc(gains) if has_zero else None
        if not has_zero:
            status = "not_available_missing_K0"
        elif auc is None:
            status = "not_available_insufficient_span"
        else:
            status = "available"
        summary_rows.append({
            "protocol": protocol,
            "task_id": task_id,
            "K_min": min(points) if points else None,
            "K_max": max(points) if points else None,
            "K_count": len(points),
            "adaptation_auc": auc,
            "status": status,
        })
    values = [row["adaptation_auc"] for row in summary_rows if row["adaptation_auc"] is not None]
    return {
        "points": point_rows,
        "summary": summary_rows,
        "average_adaptation_auc": _mean(values),
        "status": "available" if values else "not_available",
    }


def compute_rrr(
    records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    protocol: Any,
) -> dict[str, Any]:
    """Compute protocol-declared recurrence retention ratios.

    RRR is emitted once per returned law.  The returned law and its first
    occurrence are resolved from ``CLProtocol``; no task number is special.
    """

    boundaries = {
        _as_int(_value(record, "task_id"), "task_id"): record
        for record in records
    }
    first_seen = _protocol_first_seen(protocol)
    output: list[dict[str, Any]] = []
    for task_id in _protocol_task_ids(protocol):
        task = _protocol_task(protocol, task_id)
        if task is None:
            continue
        shift_type = str(getattr(task, "shift_type", ""))
        recurrence_of = getattr(task, "recurrence_of", None)
        if shift_type not in RETURN_SHIFT_TYPES or not recurrence_of:
            continue
        parents = [parent for parent in str(recurrence_of).split("|") if parent]
        for regime_id in parents:
            first_task = first_seen.get(regime_id)
            first = boundaries.get(first_task) if first_task is not None else None
            returned = boundaries.get(task_id)
            first_pre = _finite(_value(first, "pre_nll")) if first else None
            first_post = _finite(_value(first, "post_nll")) if first else None
            return_pre = _finite(_value(returned, "pre_nll")) if returned else None
            denominator = (
                first_pre - first_post
                if first_pre is not None and first_post is not None else None
            )
            if first_task is None or first is None or returned is None:
                status = "not_available_missing_boundary"
                ratio = None
            elif denominator is None:
                status = "not_available_missing_first_gain"
                ratio = None
            elif abs(denominator) <= 1e-12:
                status = "undefined_zero_first_gain"
                ratio = None
            elif return_pre is None:
                status = "not_available_missing_return_pre"
                ratio = None
            else:
                status = "available"
                ratio = (first_pre - return_pre) / denominator
            output.append({
                "regime_id": regime_id,
                "first_task": first_task,
                "return_task": task_id,
                "shift_type": shift_type,
                "first_pre_nll": first_pre,
                "first_post_nll": first_post,
                "return_pre_nll": return_pre,
                "first_gain_nll": denominator,
                "rrr": ratio,
                "status": status,
            })
    values = [row["rrr"] for row in output if row["rrr"] is not None]
    return {
        "rows": output,
        "average_rrr": _mean(values),
        "status": "available" if values else "not_available",
    }


class CLMetricEngine:
    """Apply the canonical CL contract for one validated protocol."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol

    def frozen_anchor_matrix(
        self, records: Sequence[FrozenAnchorRecord | Mapping[str, Any]]
    ) -> FrozenLawMatrix:
        return build_frozen_anchor_matrix(
            records,
            first_seen=_protocol_first_seen(self.protocol),
            persistent_regimes=_protocol_persistent(self.protocol),
        )

    def retention(
        self,
        records: Sequence[FrozenAnchorRecord | Mapping[str, Any]],
    ) -> dict[str, Any]:
        matrix = self.frozen_anchor_matrix(records)
        law_rows, summary_rows = compute_retention_metrics(matrix)
        return {
            "matrix": matrix,
            "matrix_rows": matrix.to_rows(),
            "law_rows": law_rows,
            "summary_rows": summary_rows,
        }

    def task_boundaries(
        self,
        records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return compute_task_boundary_metrics(records, self.protocol)

    def fwt(
        self,
        records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    ) -> dict[str, Any]:
        return compute_fwt(records, self.protocol)

    def adaptation(
        self,
        records: Sequence[AdaptationRecord | Mapping[str, Any]],
    ) -> dict[str, Any]:
        return compute_adaptation_metrics(records)

    def rrr(
        self,
        records: Sequence[TaskBoundaryRecord | Mapping[str, Any]],
    ) -> dict[str, Any]:
        return compute_rrr(records, self.protocol)

    def evaluate(
        self,
        *,
        frozen_anchor_records: Sequence[FrozenAnchorRecord | Mapping[str, Any]] = (),
        task_boundary_records: Sequence[TaskBoundaryRecord | Mapping[str, Any]] = (),
        adaptation_records: Sequence[AdaptationRecord | Mapping[str, Any]] = (),
        hm_state_records: Sequence[HMStateRecord | Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        retention = self.retention(frozen_anchor_records)
        boundaries = self.task_boundaries(task_boundary_records)
        fwt = compute_fwt(task_boundary_records, self.protocol)
        adaptation = compute_adaptation_metrics(adaptation_records)
        rrr = compute_rrr(task_boundary_records, self.protocol)
        hm_state = [
            asdict(record) if not isinstance(record, Mapping) else dict(record)
            for record in hm_state_records
        ]
        return {
            "frozen_anchor_matrix": retention["matrix_rows"],
            "law_metrics": retention["law_rows"],
            "continual_summary": retention["summary_rows"],
            "task_boundaries": boundaries,
            "fwt": fwt,
            "adaptation": adaptation,
            "rrr": rrr,
            "hm_state": hm_state,
            "metric_contract": {
                "name": "canonical_cl_metrics",
                "version": 1,
                "primary_matrix": "frozen_anchor",
                "persistent_regimes": sorted(_protocol_persistent(self.protocol)),
            },
        }


__all__ = [
    "AdaptationRecord",
    "CLMetricEngine",
    "FrozenAnchorRecord",
    "FrozenLawMatrix",
    "HMStateRecord",
    "TaskBoundaryRecord",
    "RETURN_SHIFT_TYPES",
    "build_frozen_anchor_matrix",
    "compute_adaptation_metrics",
    "compute_fwt",
    "compute_retention_metrics",
    "compute_rrr",
    "compute_task_boundary_metrics",
]
