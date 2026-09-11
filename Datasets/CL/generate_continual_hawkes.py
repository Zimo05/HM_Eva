#!/usr/bin/env python3
"""Generate reproducible continual Hawkes benchmarks.

The ``unified`` benchmark is the paper-facing CL-core-v2 protocol::

    A_1 -> B_1 -> C_1 -> A_1 -> B_prime_1 -> A_2
        -> (C_1 + X_transient) -> A_merge -> A_1 -> (E_1 + B_1)

The older recurrence, drift, hierarchy, and transient suites remain available
as auxiliary stress tests.  CSV files contain only model-facing event data;
the benchmark manifest and the ground-truth files are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BETAS = (0.5, 1.5)
EVENT_DIM = 8
UNIFIED_BENCHMARK = "CL-core-v2"
ADAPTATION_K_VALUES = (0, 1, 2, 4, 8, 16, 32)


@dataclass
class Regime:
    """Ground-truth parameters for one latent Hawkes law."""

    regime_id: str
    mu: np.ndarray
    W: np.ndarray
    parent_regime: str = ""
    kind: str = "base"
    parameter_distance: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StageSpec:
    task_id: int
    label: str
    mixture: Mapping[str, float]
    shift_type: str
    recurrence_of: str = ""


def _json_dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _stable_offset(value: str, modulus: int) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value)) % modulus


def spectral_radius(W: np.ndarray, betas: Sequence[float]) -> float:
    K = np.sum(W / np.asarray(betas, dtype=np.float64)[None, None, :], axis=2)
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def stabilize(W: np.ndarray, betas: Sequence[float], target: float) -> np.ndarray:
    if np.any(W < 0) or not np.all(np.isfinite(W)):
        raise ValueError("Hawkes excitation weights must be finite and non-negative")
    rho = spectral_radius(W, betas)
    return W.copy() if rho <= 1e-12 else W * (float(target) / rho)


def parameter_distance(a: Regime, b: Regime) -> float:
    numerator = float(np.linalg.norm(a.mu - b.mu) ** 2 + np.linalg.norm(a.W - b.W) ** 2) ** 0.5
    denominator = float(np.linalg.norm(a.mu) ** 2 + np.linalg.norm(a.W) ** 2) ** 0.5
    return numerator / max(denominator, 1e-12)


def _edge(W: np.ndarray, target: int, source: int, strength: float) -> None:
    W[target, source, 0] += strength * 0.72
    W[target, source, 1] += strength * 0.28


def make_shared_backbone(rng: np.random.Generator, dim: int, n_basis: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = rng.uniform(0.025, 0.055, size=dim)
    W = np.zeros((dim, dim, n_basis), dtype=np.float64)
    for target in range(dim):
        for source in range(dim):
            if rng.random() < 0.13:
                base = float(rng.uniform(0.004, 0.016))
                W[target, source, 0] = base * 0.65
                W[target, source, 1] = base * 0.35
    return mu, W


MOTIFS: Dict[str, Tuple[Tuple[int, int, float], ...]] = {
    "A": ((1, 0, 0.13), (0, 1, 0.12), (3, 2, 0.105), (2, 3, 0.055)),
    "B": ((5, 4, 0.14), (4, 5, 0.115), (6, 5, 0.125), (7, 6, 0.045)),
    "C": ((4, 0, 0.115), (5, 1, 0.11), (0, 4, 0.08), (1, 5, 0.08)),
    "D": ((6, 2, 0.14), (2, 6, 0.12), (7, 3, 0.13), (3, 7, 0.1)),
    "E": ((7, 0, 0.13), (0, 7, 0.1), (6, 1, 0.12), (1, 6, 0.085), (5, 2, 0.1)),
}


def make_motif_regime(
    regime_id: str,
    motif_name: str,
    shared_mu: np.ndarray,
    shared_W: np.ndarray,
    rng: np.random.Generator,
    betas: Sequence[float],
    *,
    target_rho: Optional[float] = None,
    parent_regime: str = "",
    kind: str = "base",
) -> Regime:
    mu = np.clip(shared_mu + rng.normal(0.0, 0.0025, size=shared_mu.shape), 0.012, 0.09)
    W = shared_W.copy()
    for target, source, strength in MOTIFS[motif_name]:
        _edge(W, target, source, strength * float(rng.uniform(0.88, 1.12)))
    active_nodes = sorted({x for edge in MOTIFS[motif_name] for x in edge[:2]})
    mu[active_nodes] *= rng.uniform(1.02, 1.12)
    W = stabilize(W, betas, target_rho if target_rho is not None else float(rng.uniform(0.66, 0.78)))
    regime = Regime(regime_id, mu, W, parent_regime=parent_regime, kind=kind)
    regime.metadata.update({"motif": motif_name, "spectral_radius": spectral_radius(W, betas)})
    return regime


def perturb_regime(
    base: Regime,
    regime_id: str,
    rng: np.random.Generator,
    betas: Sequence[float],
    relative_scale: float,
    *,
    kind: str,
    target_rho: Optional[float] = None,
) -> Regime:
    mu = np.clip(base.mu * (1.0 + rng.normal(0.0, relative_scale, size=base.mu.shape)), 0.008, 0.12)
    W = np.clip(base.W * (1.0 + rng.normal(0.0, relative_scale, size=base.W.shape)), 0.0, None)
    if kind == "specialization":
        node = int(rng.integers(0, base.W.shape[0]))
        _edge(W, (node + 1) % base.W.shape[0], node, 0.035 * relative_scale / 0.12)
    W = stabilize(W, betas, target_rho if target_rho is not None else float(rng.uniform(0.66, 0.78)))
    result = Regime(
        regime_id,
        mu,
        W,
        parent_regime=base.regime_id,
        kind=kind,
        metadata=dict(base.metadata),
    )
    result.parameter_distance = parameter_distance(base, result)
    result.metadata.update({"spectral_radius": spectral_radius(W, betas), "parent": base.regime_id})
    return result


def interpolate_regime(a: Regime, b: Regime, alpha: float, regime_id: str, betas: Sequence[float]) -> Regime:
    """Interpolate parameters so A_merge has shared A_1/A_2 semantics."""

    alpha = float(alpha)
    mu = (1.0 - alpha) * a.mu + alpha * b.mu
    W = (1.0 - alpha) * a.W + alpha * b.W
    result = Regime(
        regime_id,
        mu,
        W,
        parent_regime=a.regime_id,
        kind="drift",
        parameter_distance=parameter_distance(a, Regime("tmp", mu, W)),
        metadata={"from": a.regime_id, "to": b.regime_id, "alpha": alpha},
    )
    result.metadata["spectral_radius"] = spectral_radius(W, betas)
    return result


def simulate_hawkes(
    regime: Regime,
    rng: np.random.Generator,
    betas: Sequence[float],
    n_events: int,
    *,
    max_time: float = 100000.0,
) -> Tuple[List[float], List[int]]:
    if n_events < 1:
        raise ValueError("n_events must be positive")
    dim, _, n_basis = regime.W.shape
    beta_arr = np.asarray(betas, dtype=np.float64)
    mu = np.asarray(regime.mu, dtype=np.float64)
    W = np.asarray(regime.W, dtype=np.float64)
    if W.shape != (dim, dim, n_basis) or beta_arr.shape != (n_basis,):
        raise ValueError("inconsistent Hawkes parameter shapes")

    state = np.zeros((dim, n_basis), dtype=np.float64)
    times: List[float] = []
    types: List[int] = []
    time = 0.0
    intensity = mu.copy()
    upper = float(np.sum(intensity))
    if upper <= 0.0:
        raise ValueError("background intensity must be positive")

    attempts = 0
    while len(times) < n_events:
        attempts += 1
        if attempts > max(10000, n_events * 10000):
            raise RuntimeError("Ogata thinning exceeded safety limit; check Hawkes stability")
        wait = float(rng.exponential(1.0 / max(upper, 1e-12)))
        candidate_time = time + wait
        if candidate_time > max_time:
            raise RuntimeError("sequence exceeded max_time; increase max_time or background intensity")
        candidate_state = state * np.exp(-beta_arr * wait)[None, :]
        candidate_intensity = mu + np.einsum("ijm,jm->i", W, candidate_state)
        candidate_total = float(np.sum(candidate_intensity))
        if candidate_total > 0.0 and rng.random() <= min(1.0, candidate_total / max(upper, 1e-12)):
            event_type = int(rng.choice(dim, p=candidate_intensity / candidate_total))
            times.append(candidate_time)
            types.append(event_type)
            candidate_state[event_type, :] += 1.0
            state = candidate_state
            time = candidate_time
            intensity = mu + np.einsum("ijm,jm->i", W, state)
            upper = float(np.sum(intensity))
        else:
            state = candidate_state
            time = candidate_time
            intensity = candidate_intensity
            upper = max(candidate_total, float(np.sum(mu)), 1e-12)
    return times, types


def _write_sequences(path: Path, sequences: Iterable[Tuple[Sequence[float], Sequence[int]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_times", "event_types"])
        writer.writeheader()
        for times, types in sequences:
            if len(times) != len(types) or not times:
                raise ValueError("each sequence must contain equally sized non-empty times/types")
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("event_times must be strictly increasing")
            writer.writerow({
                "event_times": _json_dumps([round(float(value), 8) for value in times]),
                "event_types": _json_dumps([int(value) for value in types]),
            })
            count += 1
    return count


def _sequence_count(rng: np.random.Generator, min_events: int, max_events: int) -> int:
    return min_events if min_events == max_events else int(rng.integers(min_events, max_events + 1))


def _sample_stage(
    stage: StageSpec,
    regimes: Mapping[str, Regime],
    rng: np.random.Generator,
    betas: Sequence[float],
    count: int,
    min_events: int,
    max_events: int,
) -> Tuple[List[Tuple[List[float], List[int]]], List[Dict[str, object]]]:
    names = list(stage.mixture)
    if not names or any(name not in regimes for name in names):
        raise KeyError(f"stage {stage.task_id} references an unknown regime")
    weights = np.asarray([float(stage.mixture[name]) for name in names], dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError(f"stage {stage.task_id} has invalid regime weights")
    weights /= weights.sum()
    output: List[Tuple[List[float], List[int]]] = []
    rows: List[Dict[str, object]] = []
    for index in range(count):
        selected = str(rng.choice(names, p=weights))
        n_events = _sequence_count(rng, min_events, max_events)
        times, types = simulate_hawkes(regimes[selected], rng, betas, n_events)
        output.append((times, types))
        rows.append({
            "task_id": stage.task_id,
            "stage_label": stage.label,
            "split_index": index,
            "regime_id": selected,
            "regime_weights": _json_dumps(dict(stage.mixture)),
            "shift_type": stage.shift_type,
            "recurrence_of": stage.recurrence_of,
            "parameter_distance": regimes[selected].parameter_distance,
            "num_events": n_events,
        })
    return output, rows


def _build_regimes(seed: int, betas: Sequence[float]) -> Dict[str, Regime]:
    rng = np.random.default_rng(seed)
    shared_mu, shared_W = make_shared_backbone(rng, EVENT_DIM, len(betas))
    regimes: Dict[str, Regime] = {}
    for name in ("A", "B", "C", "D", "E"):
        regimes[f"{name}_1"] = make_motif_regime(
            f"{name}_1", name, shared_mu, shared_W, rng, betas
        )
    regimes["B_prime_1"] = perturb_regime(
        regimes["B_1"], "B_prime_1", rng, betas, 0.07, kind="near_recurrence"
    )
    regimes["A_2"] = perturb_regime(
        regimes["A_1"], "A_2", rng, betas, 0.14, kind="specialization"
    )
    regimes["X_transient"] = make_motif_regime(
        "X_transient", "E", shared_mu, shared_W, rng, betas,
        target_rho=0.68, kind="transient",
    )
    regimes["X_transient"].W = stabilize(
        np.roll(regimes["X_transient"].W, shift=2, axis=0), betas, 0.68
    )
    regimes["X_transient"].metadata.update({
        "spectral_radius": spectral_radius(regimes["X_transient"].W, betas),
        "transient": True,
        "never_reappears": True,
    })
    return regimes


def _unified_schedule() -> List[StageSpec]:
    return [
        StageSpec(0, "A_1_initial", {"A_1": 1.0}, "initial"),
        StageSpec(1, "B_1_novel", {"B_1": 1.0}, "novel"),
        StageSpec(2, "C_1_novel", {"C_1": 1.0}, "novel"),
        StageSpec(3, "A_1_exact_recurrence", {"A_1": 1.0}, "exact_recurrence", "A_1"),
        StageSpec(4, "B_prime_1_near_recurrence", {"B_prime_1": 1.0}, "near_recurrence", "B_1"),
        StageSpec(5, "A_2_specialization", {"A_2": 1.0}, "specialization", "A_1"),
        StageSpec(6, "C_1_with_transient_X", {"C_1": 0.9, "X_transient": 0.1}, "transient_anomaly", "C_1"),
        StageSpec(7, "A_merge", {"A_merge": 1.0}, "merge", "A_1|A_2"),
        StageSpec(8, "A_1_long_gap_recurrence", {"A_1": 1.0}, "long_gap_recurrence", "A_1"),
        StageSpec(9, "E_1_B_1_mixture", {"E_1": 0.7, "B_1": 0.3}, "mixture"),
    ]


def _recurrence_schedule() -> List[StageSpec]:
    return [
        StageSpec(0, "A_1_initial", {"A_1": 1.0}, "initial"),
        StageSpec(1, "B_1_novel", {"B_1": 1.0}, "novel"),
        StageSpec(2, "C_1_novel", {"C_1": 1.0}, "novel"),
        StageSpec(3, "A_1_exact_recurrence", {"A_1": 1.0}, "exact_recurrence", "A_1"),
        StageSpec(4, "D_1_novel", {"D_1": 1.0}, "novel"),
        StageSpec(5, "B_prime_1_near_recurrence", {"B_prime_1": 1.0}, "near_recurrence", "B_1"),
        StageSpec(6, "A_2_specialization", {"A_2": 1.0}, "specialization", "A_1"),
        StageSpec(7, "E_1_novel", {"E_1": 1.0}, "novel"),
        StageSpec(8, "A_1_long_gap_recurrence", {"A_1": 1.0}, "long_gap_recurrence", "A_1"),
        StageSpec(9, "E_B_mixture", {"E_1": 0.7, "B_1": 0.3}, "mixture"),
    ]


def _unified_regimes(seed: int, betas: Sequence[float]) -> Dict[str, Regime]:
    regimes = _build_regimes(seed, betas)
    merge = interpolate_regime(regimes["A_1"], regimes["A_2"], 0.5, "A_merge", betas)
    merge.kind = "merge"
    merge.parent_regime = "A_1|A_2"
    merge.metadata.update({"parents": ["A_1", "A_2"], "merge_target": True})
    regimes["A_merge"] = merge
    return regimes


def _drift_regimes(regimes: MutableMapping[str, Regime], betas: Sequence[float]) -> None:
    a, b = regimes["A_1"], regimes["B_1"]
    for alpha in (0.2, 0.4, 0.6, 0.8):
        regime_id = f"A_drift_{alpha:.1f}"
        regimes[regime_id] = interpolate_regime(a, b, alpha, regime_id, betas)
    regimes["B_2"] = perturb_regime(
        b, "B_2", np.random.default_rng(9001), betas, 0.10, kind="specialization"
    )


def _hierarchy_regimes(seed: int, betas: Sequence[float]) -> Dict[str, Regime]:
    rng = np.random.default_rng(seed + 17)
    shared_mu, shared_W = make_shared_backbone(rng, EVENT_DIM, len(betas))
    global_base = make_motif_regime(
        "global", "A", shared_mu, shared_W, rng, betas,
        target_rho=0.68, kind="global",
    )
    regimes: Dict[str, Regime] = {"Global": global_base}
    group_a = make_motif_regime(
        "Group_A", "A", global_base.mu, global_base.W, rng, betas,
        target_rho=0.72, parent_regime="Global", kind="group",
    )
    group_b = make_motif_regime(
        "Group_B", "B", global_base.mu, global_base.W, rng, betas,
        target_rho=0.72, parent_regime="Global", kind="group",
    )
    regimes.update({"Group_A": group_a, "Group_B": group_b})
    regimes["A_1"] = perturb_regime(group_a, "A_1", rng, betas, 0.035, kind="leaf", target_rho=0.73)
    regimes["A_2"] = perturb_regime(group_a, "A_2", rng, betas, 0.045, kind="leaf", target_rho=0.73)
    regimes["B_1"] = perturb_regime(group_b, "B_1", rng, betas, 0.04, kind="leaf", target_rho=0.74)
    regimes["B_2"] = perturb_regime(group_b, "B_2", rng, betas, 0.045, kind="leaf", target_rho=0.74)
    regimes["A_merge"] = interpolate_regime(regimes["A_1"], regimes["A_2"], 0.5, "A_merge", betas)
    regimes["A_merge"].kind = "merge"
    regimes["A_merge"].parent_regime = "A_1|A_2"
    regimes["A_merge"].metadata.update({"parents": ["A_1", "A_2"], "merge_target": True})
    return regimes


def build_benchmark(name: str, seed: int, betas: Sequence[float]) -> Tuple[Dict[str, Regime], List[StageSpec]]:
    if name == "unified":
        return _unified_regimes(seed, betas), _unified_schedule()
    if name == "hierarchy":
        regimes = _hierarchy_regimes(seed, betas)
        return regimes, [
            StageSpec(0, "A_1_initial", {"A_1": 1.0}, "initial"),
            StageSpec(1, "A_2_new_leaf", {"A_2": 1.0}, "new_specialization", "A_1"),
            StageSpec(2, "B_1_novel_group", {"B_1": 1.0}, "novel"),
            StageSpec(3, "A_1_recurrence", {"A_1": 1.0}, "exact_recurrence", "A_1"),
            StageSpec(4, "A_merge", {"A_merge": 1.0}, "merge", "A_1|A_2"),
            StageSpec(5, "A_2_recurrence", {"A_2": 1.0}, "exact_recurrence", "A_2"),
        ]
    regimes = _build_regimes(seed, betas)
    if name == "recurrence":
        return regimes, _recurrence_schedule()
    if name == "drift":
        _drift_regimes(regimes, betas)
        regimes["B_drift_0.3"] = interpolate_regime(regimes["B_1"], regimes["B_2"], 0.3, "B_drift_0.3", betas)
        return regimes, [
            StageSpec(0, "A_initial", {"A_1": 1.0}, "initial"),
            StageSpec(1, "A_drift_0.2", {"A_drift_0.2": 1.0}, "gradual_drift", "A_1"),
            StageSpec(2, "A_drift_0.4", {"A_drift_0.4": 1.0}, "gradual_drift", "A_1"),
            StageSpec(3, "B_novel", {"B_1": 1.0}, "abrupt_shift"),
            StageSpec(4, "B_drift_0.3", {"B_drift_0.3": 1.0}, "gradual_drift", "B_1"),
            StageSpec(5, "A_return", {"A_1": 1.0}, "exact_recurrence", "A_1"),
        ]
    if name == "transient":
        return regimes, [
            StageSpec(0, "A_persistent", {"A_1": 1.0}, "initial"),
            StageSpec(1, "B_persistent", {"B_1": 1.0}, "novel"),
            StageSpec(2, "C_with_transient_X", {"C_1": 0.9, "X_transient": 0.1}, "transient_anomaly"),
            StageSpec(3, "A_return", {"A_1": 1.0}, "exact_recurrence", "A_1"),
            StageSpec(4, "B_return", {"B_1": 1.0}, "exact_recurrence", "B_1"),
        ]
    raise ValueError(f"unknown benchmark: {name}")


def _used_regimes(stages: Sequence[StageSpec]) -> List[str]:
    used: List[str] = []
    for stage in stages:
        for regime_id in stage.mixture:
            if regime_id not in used:
                used.append(regime_id)
    return used


def _manifest_regime_groups(regimes: Mapping[str, Regime], used: Sequence[str]) -> Tuple[List[str], List[str]]:
    transient = [regime_id for regime_id in used if regimes[regime_id].kind == "transient"]
    persistent = [regime_id for regime_id in used if regime_id not in transient]
    return persistent, transient


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_regime_artifacts(root: Path, regimes: Mapping[str, Regime], betas: Sequence[float], seed: int) -> None:
    ground_truth = root / "ground_truth"
    ground_truth.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {
        "seed": seed,
        "event_dim": EVENT_DIM,
        "num_basis": len(betas),
        "betas": [float(value) for value in betas],
        "regimes": {},
    }
    arrays: Dict[str, np.ndarray] = {}
    for regime_id, regime in regimes.items():
        key = regime_id.replace(".", "p").replace("-", "_")
        metadata["regimes"][regime_id] = {
            "array_key": key,
            "parent_regime": regime.parent_regime,
            "kind": regime.kind,
            "parameter_distance": float(regime.parameter_distance),
            "spectral_radius": spectral_radius(regime.W, betas),
            "metadata": regime.metadata,
        }
        arrays[f"{key}__mu"] = regime.mu
        arrays[f"{key}__W"] = regime.W
    (ground_truth / "regimes.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(ground_truth / "regimes.npz", **arrays)
    try:
        import torch  # type: ignore

        torch.save(
            {regime_id: {"mu": torch.as_tensor(regime.mu), "W": torch.as_tensor(regime.W)}
             for regime_id, regime in regimes.items()},
            ground_truth / "regimes.pt",
        )
    except Exception:
        pass


def _task_manifest_entry(
    root: Path,
    stage: StageSpec,
    *,
    control: Optional[str] = None,
    adaptation: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    prefix = Path(control) if control else Path(f"task_{stage.task_id:02d}")
    entry: Dict[str, object] = {
        "task_id": stage.task_id,
        "stage_label": stage.label,
        "regime_weights": dict(stage.mixture),
        "shift_type": stage.shift_type,
        "recurrence_of": stage.recurrence_of or None,
        "paired_control": control,
        "splits": {
            "train": (prefix / "train.csv").as_posix(),
            "validation": (prefix / "val.csv").as_posix(),
            "test": (prefix / "test.csv").as_posix(),
        },
    }
    if adaptation is not None:
        entry["adaptation"] = {
            "support": str(adaptation["support"]),
            "query": str(adaptation["query"]),
            "K": list(adaptation.get("K", ADAPTATION_K_VALUES)),
        }
    if control:
        entry.update({
            "control_id": control.replace("/", "_"),
            "control_path": control,
            "control_for_task": stage.task_id,
        })
    return entry


def _write_readme(root: Path, benchmark: str, stages: Sequence[StageSpec], betas: Sequence[float], manifest_name: str) -> None:
    lines = [
        "# Continual Hawkes benchmark",
        "",
        f"Protocol: `{UNIFIED_BENCHMARK if benchmark == 'unified' else benchmark}`",
        f"Event dimension: `{EVENT_DIM}`; exponential decay bases: `{list(map(float, betas))}`",
        "",
        "Model-facing task and control CSVs contain only `event_times,event_types`.",
        f"`{manifest_name}` is the protocol source of truth; stream and ground-truth manifests are oracle-only.",
        "Anchors are independently sampled and are evaluated read-only after each checkpoint.",
        "Each task also has independent `adapt_support.csv` and `adapt_query.csv` files; the manifest records the fixed adaptation exposure values `K={0,1,2,4,8,16,32}`.",
        "Every K curve starts from a fresh clone of the same pre-task checkpoint and scores the same fixed query set.",
        "",
        "## Stage schedule",
        "",
        "| task_id | stage | regime mixture | shift | recurrence_of |",
        "|---:|---|---|---|---|",
    ]
    for stage in stages:
        mix = ", ".join(f"{key}:{value:g}" for key, value in stage.mixture.items())
        lines.append(f"| {stage.task_id} | {stage.label} | {mix} | {stage.shift_type} | {stage.recurrence_of} |")
    if benchmark == "unified":
        lines.extend([
            "",
            "Task 6 has a matched `controls/task_06_no_transient/` C_1-only stream.",
            "Persistent-law averages exclude `X_transient`; it is reported as diagnostic OOD evidence.",
            "The old recurrence-only suite belongs under `legacy/recurrence_v1/` when retained locally.",
        ])
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_benchmark_manifest(
    root: Path,
    benchmark: str,
    seed: int,
    betas: Sequence[float],
    stages: Sequence[StageSpec],
    regimes: Mapping[str, Regime],
    anchor_files: Mapping[str, str],
    controls: Sequence[Mapping[str, object]],
    adaptation_specs: Mapping[int, Mapping[str, object]] | None = None,
) -> None:
    used = _used_regimes(stages)
    persistent, transient = _manifest_regime_groups(regimes, used)
    first_seen: Dict[str, int] = {}
    for stage in stages:
        for regime_id in stage.mixture:
            first_seen.setdefault(regime_id, stage.task_id)
    adaptation_specs = adaptation_specs or {}
    tasks = [
        _task_manifest_entry(
            root,
            stage,
            adaptation=adaptation_specs.get(stage.task_id),
        )
        for stage in stages
    ]
    for control in controls:
        control_task = control.get("control_for_task")
        control_path = control.get("control_path")
        if control_task is None or control_path is None:
            continue
        for task in tasks:
            if task["task_id"] == control_task:
                task["control"] = control_path
                task["paired_control"] = control_path
    payload: Dict[str, object] = {
        "format_version": 2 if benchmark == "unified" else 1,
        "benchmark_id": UNIFIED_BENCHMARK if benchmark == "unified" else f"CL-{benchmark}-v1",
        # Keep these aliases for older result readers while all new consumers
        # use format_version and benchmark_id through CLProtocol.
        "benchmark": UNIFIED_BENCHMARK if benchmark == "unified" else f"CL-{benchmark}-v1",
        "version": 2 if benchmark == "unified" else 1,
        "protocol": "continual_hawkes",
        "schedule_name": benchmark,
        "seed": int(seed),
        "event_dim": EVENT_DIM,
        "betas": [float(value) for value in betas],
        "num_tasks": len(stages),
        "persistent_regimes": persistent,
        "transient_regimes": transient,
        "first_seen": first_seen,
        "tasks": tasks,
        "anchors": [
            {
                "regime_id": regime_id,
                "path": path,
                "evaluation_scope": "diagnostic_only" if regime_id in transient else "persistent",
            }
            for regime_id, path in anchor_files.items()
        ],
        "controls": list(controls),
        "oracle_files": {
            "stream_manifest": "stream_manifest.csv",
            "ground_truth_manifest": "ground_truth_manifest.csv",
            "ground_truth": "ground_truth",
        },
    }
    (root / "benchmark_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def generate_benchmark(
    output: Path,
    benchmark: str,
    seed: int,
    betas: Sequence[float],
    train_per_stage: int,
    val_per_stage: int,
    test_per_stage: int,
    anchor_count: int,
    min_events: int,
    max_events: int,
) -> None:
    if min_events < 1 or max_events < min_events:
        raise ValueError("event count range is invalid")
    if min(train_per_stage, val_per_stage, test_per_stage, anchor_count) < 1:
        raise ValueError("per-stage and anchor counts must be positive")
    regimes, stages = build_benchmark(benchmark, seed, betas)
    used = _used_regimes(stages)
    for regime_id in used:
        rho = spectral_radius(regimes[regime_id].W, betas)
        if not math.isfinite(rho) or rho >= 1.0:
            raise RuntimeError(f"unstable generated regime {regime_id}: spectral radius={rho}")
    output.mkdir(parents=True, exist_ok=True)

    # Ground truth contains only laws named by the schedule.  Support laws used
    # to construct a hierarchy are not silently turned into evaluation anchors.
    _write_regime_artifacts(output, {regime_id: regimes[regime_id] for regime_id in used}, betas, seed)

    manifest_rows: List[Dict[str, object]] = []
    adaptation_specs: Dict[int, Dict[str, object]] = {}
    split_counts = {"train": train_per_stage, "val": val_per_stage, "test": test_per_stage}
    for stage in stages:
        for split, count in split_counts.items():
            split_rng = np.random.default_rng(
                seed + 100003 * (stage.task_id + 1) + _stable_offset(split, 1000)
            )
            sequences, rows = _sample_stage(stage, regimes, split_rng, betas, count, min_events, max_events)
            task_dir = output / f"task_{stage.task_id:02d}"
            _write_sequences(task_dir / f"{split}.csv", sequences)
            for row_index, row in enumerate(rows):
                row.update({
                    "split": split,
                    "sequence_id": f"task_{stage.task_id:02d}_{split}_{row_index:05d}",
                    "source": "stream",
                })
                manifest_rows.append(row)

        # Support and query are independent from the ordinary train/val/test
        # stream.  The evaluator consumes the first K support events and then
        # scores the same fixed query file for every K from a fresh clone of
        # the pre-task checkpoint.
        support_count = max(
            1,
            math.ceil(max(ADAPTATION_K_VALUES) / max(min_events, 1)),
        )
        support_rng = np.random.default_rng(
            seed
            + 500003 * (stage.task_id + 1)
            + _stable_offset("adapt_support", 100000)
        )
        query_rng = np.random.default_rng(
            seed
            + 500003 * (stage.task_id + 1)
            + _stable_offset("adapt_query", 100000)
        )
        support_sequences, _ = _sample_stage(
            stage,
            regimes,
            support_rng,
            betas,
            support_count,
            min_events,
            max_events,
        )
        query_sequences, _ = _sample_stage(
            stage,
            regimes,
            query_rng,
            betas,
            max(1, min(test_per_stage, 16)),
            min_events,
            max_events,
        )
        task_dir = output / f"task_{stage.task_id:02d}"
        _write_sequences(task_dir / "adapt_support.csv", support_sequences)
        _write_sequences(task_dir / "adapt_query.csv", query_sequences)
        adaptation_specs[stage.task_id] = {
            "support": (task_dir / "adapt_support.csv").relative_to(output).as_posix(),
            "query": (task_dir / "adapt_query.csv").relative_to(output).as_posix(),
            "K": ADAPTATION_K_VALUES,
        }

    anchor_files: Dict[str, str] = {}
    for regime_id in used:
        regime = regimes[regime_id]
        anchor_rng = np.random.default_rng(seed + 700001 + _stable_offset(regime_id, 100000))
        anchor_sequences = [
            simulate_hawkes(regime, anchor_rng, betas, _sequence_count(anchor_rng, min_events, max_events))
            for _ in range(anchor_count)
        ]
        safe_name = regime_id.replace("/", "_").replace(".", "p")
        relative_path = (Path("anchors") / f"{safe_name}.csv").as_posix()
        anchor_files[regime_id] = relative_path
        _write_sequences(output / relative_path, anchor_sequences)
        for index, (times, types) in enumerate(anchor_sequences):
            manifest_rows.append({
                "task_id": "anchor",
                "stage_label": "frozen_anchor",
                "split_index": index,
                "regime_id": regime_id,
                "regime_weights": _json_dumps({regime_id: 1.0}),
                "shift_type": "frozen_evaluation",
                "recurrence_of": "",
                "parameter_distance": regime.parameter_distance,
                "num_events": len(times),
                "split": "anchor",
                "sequence_id": f"anchor_{safe_name}_{index:05d}",
                "source": "anchor",
            })

    controls: List[Mapping[str, object]] = []
    if benchmark == "unified":
        task = stages[6]
        control_id = "controls/task_06_no_transient"
        control_stage = StageSpec(
            task.task_id,
            "task_06_no_transient_control",
            {"C_1": 1.0},
            "matched_no_transient_control",
            "C_1",
        )
        for split, count in split_counts.items():
            control_rng = np.random.default_rng(
                seed + 910003 + _stable_offset(split, 1000)
            )
            sequences, _ = _sample_stage(
                control_stage, regimes, control_rng, betas, count, min_events, max_events
            )
            _write_sequences(output / control_id / f"{split}.csv", sequences)
        controls.append(_task_manifest_entry(output, control_stage, control=control_id))

    _write_manifest(output / "stream_manifest.csv", manifest_rows)
    _write_manifest(output / "ground_truth_manifest.csv", manifest_rows)
    _write_benchmark_manifest(
        output,
        benchmark,
        seed,
        betas,
        stages,
        regimes,
        anchor_files,
        controls,
        adaptation_specs=adaptation_specs,
    )
    _write_readme(output, benchmark, stages, betas, "benchmark_manifest.json")


def _parse_betas(value: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("betas must be a comma-separated list of numbers") from error
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("betas must be a comma-separated list of positive numbers")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("Datasets/CL/hm_continual_v2"))
    parser.add_argument(
        "--benchmark",
        choices=("unified", "recurrence", "drift", "hierarchy", "transient", "all"),
        default="unified",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--betas", type=_parse_betas, default=DEFAULT_BETAS)
    parser.add_argument("--train-per-stage", type=int, default=128)
    parser.add_argument("--val-per-stage", type=int, default=32)
    parser.add_argument("--test-per-stage", type=int, default=32)
    parser.add_argument("--anchor-count", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--min-events", type=int, default=48)
    parser.add_argument("--max-events", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    min_events = args.seq_len if args.seq_len is not None else args.min_events
    max_events = args.seq_len if args.seq_len is not None else args.max_events
    benchmarks = (
        ("unified", "recurrence", "drift", "hierarchy", "transient")
        if args.benchmark == "all" else (args.benchmark,)
    )
    for benchmark in benchmarks:
        output = args.output / (f"HM-{benchmark.title()}" if len(benchmarks) > 1 else "")
        generate_benchmark(
            output=output,
            benchmark=benchmark,
            seed=args.seed + benchmarks.index(benchmark),
            betas=args.betas,
            train_per_stage=args.train_per_stage,
            val_per_stage=args.val_per_stage,
            test_per_stage=args.test_per_stage,
            anchor_count=args.anchor_count,
            min_events=min_events,
            max_events=max_events,
        )
        print(f"Generated {benchmark} benchmark at {output}")


if __name__ == "__main__":
    main()
