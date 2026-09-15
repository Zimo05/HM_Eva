"""Train-only upstream construction for stationary HawkesMemory datasets.

The DWS H-tree is an external, pre-existing artifact.  Retweet has no such
oracle tree, so its stationary HM job must construct the upstream state before
Memory training:

    D_train -> THP pretrain -> THP train encoding -> residual signatures
             -> train-only hierarchical clustering -> Hawkes semantic laws
             -> attention encoder -> H-tree artifact

This module deliberately keeps the construction separate from the Memory
trainer.  The returned artifact descriptor is consumed by ``runner.py`` and
``adapters.py`` as the common HM contract used by DWS and Retweet.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import sha256
from .paths import MODELS_ROOT, PROJECT_ROOT


DEFAULT_NODE_DIM = 128
DEFAULT_THP_EPOCHS = 100
DEFAULT_ATTENTION_EPOCHS = 50
DEFAULT_HAWKES_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_RESIDUAL_RANK = 4
DEFAULT_SEMANTIC_BLEND = 0.0
DEFAULT_CLUSTER_MIN = 4
DEFAULT_CLUSTER_MAX = 8


@dataclass(frozen=True)
class HMUpstreamArtifacts:
    """Paths and metadata for one completed HM upstream construction."""

    h_tree: Path
    sequence_summary: Path
    node_dim: int = DEFAULT_NODE_DIM
    input_paths: tuple[Path, ...] = field(default_factory=tuple)
    manifest_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def expected_retweet_hm_upstream(prepared_dir: Path) -> HMUpstreamArtifacts:
    """Return deterministic output paths without running any training.

    ``--dry-run`` uses this descriptor to render the final Memory command.
    Its ``input_paths`` is intentionally empty because those artifacts have
    not been produced yet and therefore must not be hashed into a dry-run
    result manifest.
    """

    root = Path(prepared_dir).expanduser().resolve() / "hm_upstream"
    return HMUpstreamArtifacts(
        h_tree=root / "h_tree_train.pt",
        sequence_summary=root / "sequence_summary_train.csv",
        node_dim=DEFAULT_NODE_DIM,
        manifest_path=root / "hm_upstream_manifest.json",
        metadata={"dataset": "retweet", "status": "expected"},
    )


def _load_train_records(
    canonical_path: Path,
    split_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[int], dict[str, list[int]]]:
    """Load canonical events and select train rows from the immutable split."""

    canonical_path = Path(canonical_path).expanduser().resolve()
    split_manifest_path = Path(split_manifest_path).expanduser().resolve()
    with canonical_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"event_times", "event_types", "source_index"}
        missing = required - fields
        if missing:
            raise ValueError(
                f"canonical HM data is missing required fields: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("canonical HM data is empty")

    parsed_rows: list[dict[str, Any]] = []
    source_ids: list[int] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            times = [float(value) for value in json.loads(row["event_times"])]
            types = [int(value) for value in json.loads(row["event_types"])]
            source_index = int(row["source_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid canonical HM row {row_number}: {error}"
            ) from error
        if not times or len(times) != len(types):
            raise ValueError(
                f"canonical HM row {row_number} has mismatched/empty events"
            )
        if any(not math.isfinite(value) for value in times):
            raise ValueError(f"canonical HM row {row_number} has non-finite times")
        if any(current < previous for previous, current in zip(times, times[1:])):
            raise ValueError(
                f"canonical HM row {row_number} has non-monotone event times"
            )
        if any(value < 0 for value in types):
            raise ValueError(f"canonical HM row {row_number} has a negative event type")
        parsed_rows.append({
            "event_times": times,
            "event_types": types,
            "source_index": source_index,
        })
        source_ids.append(source_index)

    if len(set(source_ids)) != len(source_ids):
        raise ValueError("canonical HM source_index values must be unique")
    # The strict THP readers map numeric JSON keys to enumerated source IDs.
    # Standard stationary preparation intentionally creates this identity map.
    if sorted(source_ids) != list(range(len(source_ids))):
        raise ValueError(
            "Retweet HM upstream requires source_index values 0..N-1; "
            "prepare the canonical dataset through core.data first"
        )

    raw_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("HM split manifest must be a JSON object")
    declared_data_path = raw_manifest.get("data_path")
    if declared_data_path is not None:
        if Path(str(declared_data_path)).expanduser().resolve() != canonical_path:
            raise ValueError("HM split manifest data_path does not match canonical data")
    declared_data_sha = raw_manifest.get("data_sha256")
    if declared_data_sha is not None and str(declared_data_sha) != sha256(canonical_path):
        raise ValueError("HM split manifest data_sha256 does not match canonical data")
    splits = raw_manifest.get("splits")
    required_splits = ("train", "validation", "test")
    if not isinstance(splits, Mapping) or any(
        name not in splits for name in required_splits
    ):
        raise ValueError(
            "HM split manifest must contain train, validation, and test splits"
        )

    normalized_splits: dict[str, list[int]] = {}
    assigned: set[int] = set()
    for name in required_splits:
        values = splits[name]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"HM split {name!r} must be a list")
        normalized: list[int] = []
        for raw_value in values:
            try:
                row_index = int(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"HM split {name!r} contains a non-integer row ID"
                ) from error
            if row_index < 0 or row_index >= len(parsed_rows):
                raise ValueError(
                    f"HM split {name!r} references row {row_index} outside canonical data"
                )
            normalized.append(row_index)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"HM split {name!r} contains duplicate row IDs")
        overlap = assigned.intersection(normalized)
        if overlap:
            raise ValueError(
                f"HM split rows overlap between partitions: {sorted(overlap)[:5]}"
            )
        assigned.update(normalized)
        normalized_splits[name] = sorted(normalized)

    if assigned != set(range(len(parsed_rows))):
        raise ValueError("HM split manifest does not partition canonical data exactly")
    train_records = [parsed_rows[index] for index in normalized_splits["train"]]
    if not train_records:
        raise ValueError("HM train split is empty")
    train_source_ids = [int(row["source_index"]) for row in train_records]
    return parsed_rows, train_source_ids, normalized_splits


def _stream_from_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    times = [float(value) for value in record["event_times"]]
    types = [int(value) for value in record["event_types"]]
    gaps = [times[0]] + [current - previous for previous, current in zip(times, times[1:])]
    return [
        {
            "time_since_start": time,
            "time_since_last_event": gap,
            "type_event": event_type,
        }
        for time, gap, event_type in zip(times, gaps, types)
    ]


def _write_thp_json(
    records: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    payload = {
        str(int(record["source_index"])): _stream_from_record(record)
        for record in sorted(records, key=lambda item: int(item["source_index"]))
    }
    if not payload:
        raise ValueError(f"cannot write empty THP JSON: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - target env validates this
        raise RuntimeError(
            "Retweet HM upstream requires PyTorch in the experiment environment"
        ) from error
    return torch


def _to_hawkes_sequences(
    records: Sequence[Mapping[str, Any]],
    torch,
    type_to_index: Mapping[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int], int]:
    raw_types = sorted({
        int(event_type)
        for record in records
        for event_type in record["event_types"]
    })
    if not raw_types:
        raise ValueError("train-only HM data contains no event types")
    if type_to_index is None:
        type_to_index = {value: index for index, value in enumerate(raw_types)}
    missing_types = set(raw_types).difference(type_to_index)
    if missing_types:
        raise ValueError(
            "Hawkes event-type mapping is missing train types: "
            f"{sorted(missing_types)}"
        )
    ordered_indices = sorted(int(index) for index in type_to_index.values())
    if ordered_indices != list(range(len(type_to_index))):
        raise ValueError("Hawkes event-type mapping must use contiguous indices")
    sequences: list[dict[str, Any]] = []
    for record in records:
        times = torch.tensor(record["event_times"], dtype=torch.float32)
        types = torch.tensor(
            [type_to_index[int(value)] for value in record["event_types"]],
            dtype=torch.long,
        )
        sequences.append({
            "times": times,
            "types": types,
            "T": times[-1],
            "source_index": torch.tensor(
                int(record["source_index"]), dtype=torch.long
            ),
        })
    return sequences, dict(type_to_index), len(type_to_index)


def _memory_imports():
    """Import Memory-side Hawkes helpers without polluting core imports."""

    memory_root = MODELS_ROOT / "HawkesMemory" / "Memory"
    if str(memory_root) not in sys.path:
        sys.path.insert(0, str(memory_root))
    from HawkesBackbone import HawkesFamily
    from Train.ResidualInitialization import compute_sequence_residual_signatures

    return HawkesFamily, compute_sequence_residual_signatures


def _fit_hawkes(
    records: Sequence[Mapping[str, Any]],
    *,
    torch,
    hawkes_family,
    type_to_index: Mapping[int, int],
    output_path: Path,
    seed: int,
    device: str,
    epochs: int,
    verbose: bool,
) -> Any:
    sequences, _ignored_mapping, num_types = _to_hawkes_sequences(
        records,
        torch,
        type_to_index=type_to_index,
    )
    model = hawkes_family(
        num_types=num_types,
        num_basis=2,
        decays=torch.tensor([0.5, 1.5], dtype=torch.float32),
    ).to(torch.device(device))
    model.cold_start(
        dataset=sequences,
        num_epochs=max(1, int(epochs)),
        learning_rate=1e-3,
        weight_decay=1e-5,
        stability_weight=1e-3,
        grad_clip=5.0,
        # The upstream contract is train-only.  Do not hold out validation or
        # use any dev/test row to choose the semantic-law parameters.
        validation_fraction=0.0,
        patience=max(1, min(8, int(epochs))),
        min_delta=0.0,
        checkpoint_path=str(output_path),
        seed=int(seed),
        metadata={
            "evaluation_regime": "strict_inductive",
            "population": "D_train",
            "event_type_mapping": {str(key): int(value) for key, value in type_to_index.items()},
        },
        verbose=verbose,
    )
    return model, sequences


def _load_encoded(path: Path):
    torch = _load_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"THP encoding artifact must be a mapping: {path}")
    embeddings = payload.get("embeddings")
    raw_ids = payload.get("sequence_ids")
    if embeddings is None or raw_ids is None:
        raise ValueError(f"THP encoding artifact is missing embeddings/sequence_ids: {path}")
    embeddings = embeddings.detach().cpu().float().numpy()
    source_ids = [int(value) for value in raw_ids]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(source_ids):
        raise ValueError("THP encoding shape does not match its sequence ID index")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("THP encoding sequence IDs are not unique")
    return source_ids, embeddings


def _normalize_columns(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("clustering features must be a matrix")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return centered / scale


def _two_means(features, indices, *, iterations: int = 10):
    """Deterministic two-means split used by the scalable HAC scaffold."""

    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        return None
    values = features[indices]
    first = 0
    distances = ((values - values[first]) ** 2).sum(axis=1)
    second = int(np.argmax(distances))
    if second == first:
        second = 1
    centers = np.stack([values[first], values[second]], axis=0).astype(np.float32)
    labels = np.zeros(indices.size, dtype=np.int8)
    for _ in range(max(1, iterations)):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1).astype(np.int8)
        if labels.all() or (~labels.astype(bool)).all():
            # Pick the point farthest from the occupied center for the empty
            # side.  This keeps the split deterministic and non-empty.
            occupied = int(labels[0])
            distances_from = ((values - centers[occupied]) ** 2).sum(axis=1)
            move = int(np.argmax(distances_from))
            labels[move] = 1 - occupied
        new_centers = np.stack([
            values[labels == side].mean(axis=0)
            for side in (0, 1)
        ]).astype(np.float32)
        if np.allclose(new_centers, centers, rtol=0.0, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers

    left = indices[labels == 0]
    right = indices[labels == 1]
    if not left.size or not right.size:
        order = np.argsort(values[:, 0], kind="mergesort")
        midpoint = max(1, min(len(order) - 1, len(order) // 2))
        left = indices[order[:midpoint]]
        right = indices[order[midpoint:]]
    return np.sort(left), np.sort(right)


def _cluster_sse(features, indices) -> float:
    values = features[indices]
    center = values.mean(axis=0, keepdims=True)
    return float(((values - center) ** 2).sum())


def _hierarchical_clusters(features, source_ids):
    """Build a train-only binary hierarchy and select a coarse cut.

    The implementation is a scalable divisive hierarchical clustering:
    it repeatedly bisects the highest-variance active node.  Candidate cuts
    are scored with a BIC-style train-only
    reconstruction/complexity criterion, so the number of initial leaves is
    not supplied as an oracle label.
    """

    import numpy as np

    n_samples, feature_dim = features.shape
    if n_samples < 2:
        raise ValueError("hierarchical clustering needs at least two train sequences")
    max_k = min(DEFAULT_CLUSTER_MAX, n_samples)
    min_k = min(DEFAULT_CLUSTER_MIN, max_k)
    if min_k < 2:
        min_k = 2

    root = {
        "path": (),
        "indices": np.arange(n_samples, dtype=np.int64),
        "sse": _cluster_sse(features, np.arange(n_samples, dtype=np.int64)),
    }
    active = [root]
    snapshots: dict[int, list[dict[str, Any]]] = {1: list(active)}
    while len(active) < max_k:
        splittable = [cluster for cluster in active if len(cluster["indices"]) >= 2]
        if not splittable:
            break
        selected = sorted(
            splittable,
            key=lambda cluster: (
                -float(cluster["sse"]),
                len(cluster["path"]),
                tuple(cluster["path"]),
            ),
        )[0]
        split = _two_means(features, selected["indices"])
        if split is None:
            break
        left, right = split
        children = [
            {"path": selected["path"] + (0,), "indices": left},
            {"path": selected["path"] + (1,), "indices": right},
        ]
        # Left/right are structural names, not semantic labels.  Normalize
        # their orientation by the earliest source ID for reproducibility.
        if int(source_ids[children[0]["indices"]].min()) > int(
            source_ids[children[1]["indices"]].min()
        ):
            children.reverse()
            children[0]["path"] = selected["path"] + (0,)
            children[1]["path"] = selected["path"] + (1,)
        for child in children:
            child["sse"] = _cluster_sse(features, child["indices"])
        active = [cluster for cluster in active if cluster is not selected] + children
        active.sort(key=lambda cluster: tuple(cluster["path"]))
        snapshots[len(active)] = [dict(cluster) for cluster in active]

    available_k = sorted(k for k in snapshots if k >= min_k)
    if not available_k:
        raise ValueError("hierarchical clustering could not produce the requested cut")
    total = max(float(sum(cluster["sse"] for cluster in snapshots[1])), 1e-12)
    scores: dict[str, float] = {}
    for k in available_k:
        sse = max(float(sum(cluster["sse"] for cluster in snapshots[k])), 1e-12)
        # Gaussian BIC up to a common additive constant.  It penalizes extra
        # leaves using only D_train geometry and has no knowledge of labels.
        score = (
            n_samples * feature_dim * math.log(sse / (n_samples * feature_dim))
            + k * feature_dim * math.log(max(n_samples, 2))
        )
        scores[str(k)] = float(score)
    selected_k = min(available_k, key=lambda k: (scores[str(k)], k))
    selected = snapshots[selected_k]
    selected.sort(key=lambda cluster: int(source_ids[cluster["indices"]].min()))
    clusters = []
    for cluster_id, cluster in enumerate(selected):
        clusters.append({
            "cluster_id": cluster_id,
            "path": tuple(cluster["path"]),
            "indices": np.asarray(cluster["indices"], dtype=np.int64),
            "source_ids": sorted(int(source_ids[index]) for index in cluster["indices"]),
            "sse": float(cluster["sse"]),
        })
    return clusters, {
        "candidate_k": available_k,
        "selected_k": int(selected_k),
        "criterion": "train_only_bic_style_sse",
        "scores": scores,
        "feature_dim": int(feature_dim),
        "sample_count": int(n_samples),
        "total_sse_at_k1": total,
    }


def _leaf_position(path: Sequence[int]) -> str:
    if not path:
        return "root"
    return "_".join("l" if value == 0 else "r" for value in path)


def _write_sequence_summary(
    clusters: Sequence[Mapping[str, Any]],
    models: Mapping[int, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def as_list(value):
        return value.detach().cpu().tolist()

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("leaf_position", "cluster_id", "mu", "A", "decay", "sequences"),
        )
        writer.writeheader()
        for cluster in clusters:
            cluster_id = int(cluster["cluster_id"])
            model = models[cluster_id]
            with _load_torch().no_grad():
                mu = as_list(model.mu())
                excitation = as_list(model.integrated_excitation_matrix())
            decay = float(model.decays.detach().cpu().float().mean())
            writer.writerow({
                "leaf_position": _leaf_position(cluster["path"]),
                "cluster_id": cluster_id,
                "mu": repr(mu),
                "A": repr(excitation),
                "decay": decay,
                "sequences": repr(list(cluster["source_ids"])),
            })


def _device_environment(
    *,
    python_executable: str,
    device: str,
    batch_size: int,
    thp_epochs: int,
    attention_epochs: int,
    seed: int,
    paths: Mapping[str, Path],
) -> dict[str, str]:
    env = os.environ.copy()
    encoder_root = MODELS_ROOT / "HawkesMemory" / "MultiAttentionEncoder"
    python_paths = [
        str(PROJECT_ROOT),
        str(MODELS_ROOT / "HawkesMemory"),
        str(MODELS_ROOT / "HawkesMemory" / "Memory"),
        str(encoder_root),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.update({
        "PYTHON": str(python_executable),
        "THP_BATCH_SIZE": str(max(1, int(batch_size))),
        "THP_NUM_WORKERS": "0",
        "THP_EPOCHS": str(max(1, int(thp_epochs))),
        "THP_DATA_PARALLEL": "0",
        "THP_SEED": str(int(seed)),
        "ATTENTION_BATCH_SIZE": str(max(1, int(batch_size))),
        "ATTENTION_EPOCHS": str(max(1, int(attention_epochs))),
        "ATTENTION_SEED": str(int(seed)),
        "DATA_PATH": str(paths["all_json"]),
        "ENCODE_DATA_PATH": str(paths["train_json"]),
        "OUTPUT_DIR": str(paths["thp_checkpoint_dir"]),
        "TRAIN_LOG": str(paths["thp_model_log"]),
        "CHECKPOINT": str(paths["thp_checkpoint"]),
        "ENCODED_OUTPUT": str(paths["encoded_train"]),
        "SUMMARY_CSV": str(paths["summary"]),
        "TREE_CSV": str(paths["tree_csv"]),
        "FINAL_OUTPUT": str(paths["h_tree"]),
        "ATTENTION_WEIGHTS": str(paths["attention_weights"]),
        "SPLIT_MANIFEST": str(paths["split_manifest"]),
        "SPLIT_DATA_PATH": str(paths["canonical"]),
    })
    if str(device).startswith("cuda"):
        env["DEVICE_TYPE"] = "cuda"
        env["DEVICES"] = str(device).split(":", 1)[1] if ":" in str(device) else "0"
    else:
        env["DEVICE_TYPE"] = "cpu"
        env["DEVICES"] = ""
    return env


def _run_encoder_stage(
    stage: str,
    *,
    env: Mapping[str, str],
    log_path: Path,
) -> None:
    run_sh = MODELS_ROOT / "HawkesMemory" / "MultiAttentionEncoder" / "run.sh"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            ["bash", str(run_sh), stage],
            cwd=run_sh.parent,
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        raise RuntimeError(
            f"HM upstream stage {stage!r} failed with exit code "
            f"{process.returncode}; see {log_path}"
        )


def build_retweet_hm_upstream(
    *,
    canonical_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    python_executable: str,
    batch_size: int | None = None,
    epochs: int | None = None,
    smoke: bool = False,
) -> HMUpstreamArtifacts:
    """Build the Retweet H-tree using only the declared training split."""

    import numpy as np

    canonical_path = Path(canonical_path).expanduser().resolve()
    split_manifest_path = Path(split_manifest_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_records, train_source_ids, split_rows = _load_train_records(
        canonical_path,
        split_manifest_path,
    )
    train_id_set = set(train_source_ids)
    train_records = [
        record for record in all_records if int(record["source_index"]) in train_id_set
    ]
    if len(train_records) != len(train_source_ids):
        raise ValueError("train source IDs do not map one-to-one to canonical rows")

    paths = {
        "canonical": canonical_path,
        "split_manifest": split_manifest_path,
        "all_json": root / "thp_train_manifested.json",
        "train_json": root / "thp_train_only.json",
        "thp_checkpoint_dir": root / "thp_checkpoints",
        "thp_model_log": root / "thp_model.log",
        "thp_checkpoint": root / "thp_checkpoints" / "checkpoint_best.pt",
        "encoded_train": root / "thp_encoded_train.pt",
        "global_hawkes": root / "hawkes_global.pt",
        "residual_signatures": root / "residual_signatures.pt",
        "summary": root / "sequence_summary_train.csv",
        "tree_csv": root / "tree_node_sequences.csv",
        "attention_weights": root / "attention_weights.pt",
        "h_tree": root / "h_tree_train.pt",
        "manifest": root / "hm_upstream_manifest.json",
    }
    _write_thp_json(all_records, paths["all_json"])
    _write_thp_json(train_records, paths["train_json"])

    requested_epochs = int(epochs) if epochs is not None else DEFAULT_THP_EPOCHS
    if smoke:
        requested_epochs = 1
    attention_epochs = 1 if smoke else DEFAULT_ATTENTION_EPOCHS
    effective_batch = int(batch_size) if batch_size is not None else DEFAULT_BATCH_SIZE
    env = _device_environment(
        python_executable=python_executable,
        device=device,
        batch_size=effective_batch,
        thp_epochs=requested_epochs,
        attention_epochs=attention_epochs,
        seed=seed,
        paths=paths,
    )

    logs = root / "logs"
    _run_encoder_stage("train", env=env, log_path=logs / "01_thp_train.log")
    _run_encoder_stage("encode", env=env, log_path=logs / "02_thp_encode_train.log")

    source_ids_from_encoding, z_matrix = _load_encoded(paths["encoded_train"])
    if set(source_ids_from_encoding) != train_id_set:
        raise ValueError(
            "train-only THP encoding IDs do not match split_manifest.splits.train"
        )
    order = {source_id: index for index, source_id in enumerate(source_ids_from_encoding)}
    record_by_source_id = {
        int(record["source_index"]): record for record in train_records
    }
    ordered_train_records = [record_by_source_id[source_id] for source_id in source_ids_from_encoding]

    torch = _load_torch()
    HawkesFamily, compute_residuals = _memory_imports()
    type_to_index = {
        int(value): index
        for index, value in enumerate(sorted({
            int(event_type)
            for record in ordered_train_records
            for event_type in record["event_types"]
        }))
    }
    global_model, hawkes_sequences = _fit_hawkes(
        ordered_train_records,
        torch=torch,
        hawkes_family=HawkesFamily,
        type_to_index=type_to_index,
        output_path=paths["global_hawkes"],
        seed=seed,
        device=device,
        epochs=1 if smoke else DEFAULT_HAWKES_EPOCHS,
        verbose=not smoke,
    )
    residuals, residual_stats = compute_residuals(
        global_model,
        hawkes_sequences,
        lowrank_rank=DEFAULT_RESIDUAL_RANK,
        grad_clip=0.0,
        progress=not smoke,
    )
    if residuals.size(0) != len(source_ids_from_encoding):
        raise ValueError("residual signature count does not match THP encoding count")
    torch.save(
        {
            "signatures": residuals.detach().cpu(),
            "source_ids": source_ids_from_encoding,
            "stats": residual_stats,
            "evaluation_regime": "strict_inductive",
            "population": "D_train",
        },
        paths["residual_signatures"],
    )

    z = _normalize_columns(z_matrix)
    residual_matrix = _normalize_columns(residuals.detach().cpu().numpy())
    if not np.isfinite(z).all() or not np.isfinite(residual_matrix).all():
        raise FloatingPointError(
            "THP embeddings or Hawkes residual signatures contain NaN/Inf"
        )
    alpha = 0.7
    features = np.concatenate(
        [alpha * z, (1.0 - alpha) * residual_matrix],
        axis=1,
    ).astype(np.float32, copy=False)
    source_id_array = np.asarray(source_ids_from_encoding, dtype=np.int64)
    clusters, cluster_stats = _hierarchical_clusters(features, source_id_array)

    cluster_models: dict[int, Any] = {}
    for cluster in clusters:
        cluster_records = [ordered_train_records[order[source_id]] for source_id in cluster["source_ids"]]
        cluster_model, _cluster_sequences = _fit_hawkes(
            cluster_records,
            torch=torch,
            hawkes_family=HawkesFamily,
            type_to_index=type_to_index,
            output_path=root / f"hawkes_cluster_{int(cluster['cluster_id'])}.pt",
            seed=seed + int(cluster["cluster_id"]) + 1,
            device=device,
            epochs=1 if smoke else DEFAULT_HAWKES_EPOCHS,
            verbose=False,
        )
        cluster_models[int(cluster["cluster_id"])] = cluster_model
    _write_sequence_summary(clusters, cluster_models, paths["summary"])

    # The final two stages consume the exact original attention implementation:
    # summary -> Process_input -> attention training -> node-only final encode.
    _run_encoder_stage(
        "train_attention",
        env=env,
        log_path=logs / "03_attention_train.log",
    )
    _run_encoder_stage(
        "final_encode",
        env=env,
        log_path=logs / "04_attention_final_encode.log",
    )
    for required_path in (paths["summary"], paths["tree_csv"], paths["h_tree"]):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"HM upstream stage completed without artifact: {required_path}"
            )

    upstream_manifest = {
        "format_version": 1,
        "dataset": "retweet",
        "evaluation_regime": "strict_inductive",
        "population": "D_train",
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha256(canonical_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": sha256(split_manifest_path),
        "train_source_ids": sorted(train_source_ids),
        "split_row_counts": {
            name: len(values) for name, values in split_rows.items()
        },
        "pipeline": [
            "THP pretrain",
            "train-only sequence encoding",
            "Hawkes residual signatures",
            "train-only hierarchical clustering",
            "per-cluster Hawkes semantic law",
            "Attention Encoder",
            "H-tree node-only final encode",
        ],
        "cluster_selection": cluster_stats,
        "feature_contract": {
            "formula": "[alpha*z_i, (1-alpha)*vec(Delta_theta_i)]",
            "alpha": alpha,
            "residual_projection": "lowrank",
            "residual_rank": DEFAULT_RESIDUAL_RANK,
            "residual_stats": residual_stats,
        },
        "artifacts": {
            key: {
                "path": str(path),
                "sha256": sha256(path) if path.is_file() else None,
            }
            for key, path in paths.items()
            if key not in {"canonical", "split_manifest", "manifest"}
        },
        "node_dim": DEFAULT_NODE_DIM,
        "semantic_blend": DEFAULT_SEMANTIC_BLEND,
    }
    paths["manifest"].write_text(
        json.dumps(upstream_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    input_paths = tuple(
        path for path in (
            paths["manifest"],
            paths["all_json"],
            paths["train_json"],
            paths["thp_checkpoint"],
            paths["encoded_train"],
            paths["global_hawkes"],
            paths["residual_signatures"],
            paths["summary"],
            paths["tree_csv"],
            paths["attention_weights"],
            paths["h_tree"],
        )
        if path.is_file()
    )
    return HMUpstreamArtifacts(
        h_tree=paths["h_tree"],
        sequence_summary=paths["summary"],
        node_dim=DEFAULT_NODE_DIM,
        input_paths=input_paths,
        manifest_path=paths["manifest"],
        metadata={
            "dataset": "retweet",
            "evaluation_regime": "strict_inductive",
            "population": "D_train",
            "cluster_selection": cluster_stats,
            "upstream_manifest_path": str(paths["manifest"]),
        },
    )
