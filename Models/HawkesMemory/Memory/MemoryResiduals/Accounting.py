"""Persistent-memory tensor accounting and global episodic budget projection.

The byte count deliberately excludes the encoder, Hawkes backbone, controller,
shared router network, optimizers, and rebuildable Bank append caches. It
measures semantic-tree tensors, persistent node-routing prototypes, and the
tensors that would be serialized for episodic memory, including tensors
retained inside replay windows. A budget is optional: measurement is always
available, while projection/eviction only runs when a caller supplies a cap.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def _tensor_bytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if torch.is_tensor(value) else 0


def _window_tensor_bytes(window: Any) -> int:
    if window is None:
        return 0
    return sum(
        _tensor_bytes(getattr(window, name, None))
        for name in (
            "times",
            "types",
            "T",
            "event_time_features",
            "hawkes_history_stats",
            "hawkes_interval_stats",
        )
    )


def _bank_row_tensor_bytes(bank: Any, index: int) -> int:
    total = sum(
        _tensor_bytes(getattr(bank, field)[index])
        for field in bank._tensor_state_fields()
    )
    total += _window_tensor_bytes(bank.windows[index])
    return total


def _router_prototype_bytes(tree: Any) -> int:
    """Count only serialized per-node routing prototype tensors.

    ``ancestor_matrix`` and the routing network are topology/computation state,
    not persistent evidence. The Welford ``count``, ``mean``, and ``m2``
    buffers are included because they are saved with the tree checkpoint.
    """

    frontier = getattr(tree, "frontier_routing", None)
    prototypes = getattr(frontier, "prototypes", None)
    if prototypes is None:
        return 0
    return sum(
        _tensor_bytes(getattr(prototypes, name, None))
        for name in ("count", "mean", "m2")
    )


def _semantic_tree_tensor_bytes(tree: Any) -> int:
    return sum(
        _tensor_bytes(value)
        for parameter_group in (tree.node_emb, tree.semantic_offset)
        for value in parameter_group.values()
    )


def persistent_memory_nbytes(tree: Any) -> dict[str, int]:
    """Return structural, episodic, prototype-breakdown, and total bytes.

    ``semantic_bytes`` is the full non-episodic structural cost: node
    embeddings, semantic offsets, and persistent node-routing prototypes.
    ``router_prototype_bytes`` is a subset of ``semantic_bytes`` provided as a
    diagnostic breakdown, so it must not be added to that field a second time.
    """

    semantic_tree = _semantic_tree_tensor_bytes(tree)
    router_prototype = _router_prototype_bytes(tree)
    semantic = semantic_tree + router_prototype
    episodic = 0
    for bank in tree.episodic_memory.banks.values():
        bank._ensure_prototype_state()
        episodic += sum(
            _tensor_bytes(getattr(bank, field))
            for field in bank._tensor_state_fields()
        )
        episodic += sum(_window_tensor_bytes(window) for window in bank.windows)
        # Pending mode candidates are serialized as tensor-valued law keys in
        # EpisodicMemory.get_extra_state(), but are not aligned with a row.
        episodic += sum(
            _tensor_bytes(candidate.get("law_key"))
            for candidates in bank._mode_pending_candidates.values()
            for candidate in candidates
        )
    return {
        "semantic_bytes": int(semantic),
        "semantic_tree_tensor_bytes": int(semantic_tree),
        "episodic_bytes": int(episodic),
        "router_prototype_bytes": int(router_prototype),
        "total_memory_bytes": int(semantic + episodic),
    }


def projected_non_episodic_bytes_after_split(
    tree: Any,
    leaf_id: str,
) -> dict[str, int]:
    """Estimate fixed persistent bytes after splitting ``leaf_id``.

    A split adds two semantic nodes and two rows to the per-node routing
    prototype store. Episodic rows are omitted here because the budget
    projection can evict them after the Sleep transaction.
    """

    node = tree.nodes.get(leaf_id)
    if node is None or not node.is_leaf:
        raise ValueError(f"Split target is not an active leaf: {leaf_id!r}")

    semantic_tree_growth = 2 * (
        _tensor_bytes(tree.node_emb[leaf_id])
        + _tensor_bytes(tree.semantic_offset[leaf_id])
    )

    prototype_growth = 0
    frontier = getattr(tree, "frontier_routing", None)
    prototypes = getattr(frontier, "prototypes", None)
    if prototypes is not None:
        for name in ("count", "mean", "m2"):
            value = getattr(prototypes, name, None)
            if not torch.is_tensor(value):
                continue
            row_numel = 1
            for dimension in value.shape[1:]:
                row_numel *= int(dimension)
            # A Split keeps the former leaf as an internal node and adds two
            # child rows to the store.
            prototype_growth += 2 * row_numel * int(value.element_size())

    semantic_tree_after = (
        _semantic_tree_tensor_bytes(tree) + semantic_tree_growth
    )
    prototype_after = _router_prototype_bytes(tree) + prototype_growth
    semantic_after = semantic_tree_after + prototype_after
    return {
        "projected_semantic_bytes": int(semantic_after),
        "projected_semantic_tree_tensor_bytes": int(semantic_tree_after),
        "projected_router_prototype_bytes": int(prototype_after),
        "projected_non_episodic_bytes": int(semantic_after),
        "semantic_tree_growth_bytes": int(semantic_tree_growth),
        "router_prototype_growth_bytes": int(prototype_growth),
    }


@torch.no_grad()
def project_episodic_memory_budget(tree: Any, budget_bytes: int) -> dict[str, int]:
    """Evict the globally lowest-retention residual rows until under budget.

    Retention is ranked across every node using existing write quality,
    retrieval usage, staleness, and effective age evidence. The budget is
    global; no node receives its own derived capacity.
    """

    budget_bytes = int(budget_bytes)
    if budget_bytes <= 0:
        raise ValueError("persistent memory budget must be positive")
    before = persistent_memory_nbytes(tree)
    non_evictable_before = before["semantic_bytes"]
    if non_evictable_before > budget_bytes:
        raise ValueError(
            "persistent memory budget is smaller than non-evictable persistent "
            "state (semantic tree plus router prototypes): "
            f"{budget_bytes} < {non_evictable_before} bytes "
            f"(semantic={before['semantic_bytes']}, "
            f"router_prototype={before['router_prototype_bytes']})"
        )

    candidates: list[tuple[float, str, int, int]] = []
    for node_id, bank in tree.episodic_memory.banks.items():
        bank._ensure_prototype_state()
        if not len(bank):
            continue
        age = bank.effective_age(tree.episodic_memory._age_clock)
        retention = (
            torch.log1p(bank.usage.clamp_min(0.0) + bank.cycle_usage.clamp_min(0.0))
            + 0.25 * torch.log1p(bank.support.clamp_min(0.0))
            + 0.10 * torch.log1p(bank.quality_mass.clamp_min(0.0))
            + 0.25 * bank.write_quality.clamp_min(0.0)
            - bank.stale_cycles
            - 0.01 * age
        )
        for index, score in enumerate(retention.detach().cpu().tolist()):
            candidates.append(
                (float(score), str(node_id), int(index), _bank_row_tensor_bytes(bank, index))
            )

    bytes_to_remove = before["total_memory_bytes"] - budget_bytes
    removals: dict[str, set[int]] = defaultdict(set)
    if bytes_to_remove > 0:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _score, node_id, index, row_bytes in candidates:
            if bytes_to_remove <= 0:
                break
            removals[node_id].add(index)
            bytes_to_remove -= row_bytes
        for node_id, indices in removals.items():
            bank = tree.episodic_memory.banks[node_id]
            keep = torch.tensor(
                [index not in indices for index in range(len(bank))],
                dtype=torch.bool,
                device=bank.device,
            )
            bank.keep(torch.nonzero(keep, as_tuple=False).flatten())
        tree.episodic_memory.invalidate_packed_mirror()

    after = persistent_memory_nbytes(tree)
    if after["total_memory_bytes"] > budget_bytes:
        raise ValueError(
            "persistent memory budget cannot be met by episodic row eviction: "
            f"{after['total_memory_bytes']} > {budget_bytes} bytes"
        )
    return {
        "budget_bytes": budget_bytes,
        "before_bytes": before["total_memory_bytes"],
        "after_bytes": after["total_memory_bytes"],
        "semantic_bytes": after["semantic_bytes"],
        "semantic_tree_tensor_bytes": after["semantic_tree_tensor_bytes"],
        "episodic_bytes": after["episodic_bytes"],
        "router_prototype_bytes": after["router_prototype_bytes"],
        "evicted_rows": int(sum(len(value) for value in removals.values())),
    }


__all__ = [
    "persistent_memory_nbytes",
    "project_episodic_memory_budget",
    "projected_non_episodic_bytes_after_split",
]
