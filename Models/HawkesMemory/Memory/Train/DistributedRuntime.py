"""Small, explicit ``torch.distributed`` runtime for HawkesMemory training.

The memory tree has dynamic, non-parameter state, so this module deliberately
does not wrap the trainer in ``DistributedDataParallel``.  It provides the
collectives needed by the staged protocol instead: deterministic contiguous
wavefront sharding, object transaction gathering, event-weighted gradient
reduction, rank-zero execution, and state broadcast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor, nn


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedRuntime:
    """Process-group metadata plus narrowly scoped collective operations."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None
    device: torch.device = torch.device("cpu")
    owns_process_group: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(
        cls,
        *,
        device: Optional[str | torch.device] = None,
        backend: Optional[str] = None,
    ) -> "DistributedRuntime":
        """Initialize from ``torchrun`` variables, or return a local runtime."""

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 1:
            resolved = torch.device(
                device
                if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            return cls(device=resolved)

        if rank < 0 or rank >= world_size:
            raise ValueError("RANK must lie in [0, WORLD_SIZE)")
        if local_rank < 0:
            raise ValueError("LOCAL_RANK must be non-negative")
        if device is not None and torch.device(device).type == "cpu":
            resolved = torch.device("cpu")
        elif torch.cuda.is_available():
            resolved = torch.device("cuda", local_rank)
            torch.cuda.set_device(resolved)
        else:
            resolved = torch.device("cpu")

        selected_backend = backend or (
            "nccl" if resolved.type == "cuda" else "gloo"
        )
        owns_group = False
        if not dist.is_initialized():
            dist.init_process_group(
                backend=selected_backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
            owns_group = True
        else:
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            if (actual_rank, actual_world_size) != (rank, world_size):
                raise RuntimeError(
                    "initialized process group disagrees with torchrun "
                    "environment"
                )
            selected_backend = str(dist.get_backend())
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=selected_backend,
            device=resolved,
            owns_process_group=owns_group,
        )

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()

    def contiguous_shard(self, values: Sequence[T]) -> list[T]:
        """Return a stable, near-even contiguous shard of ``values``."""

        count = len(values)
        base, remainder = divmod(count, self.world_size)
        start = self.rank * base + min(self.rank, remainder)
        length = base + int(self.rank < remainder)
        return list(values[start : start + length])

    def gather_transactions(
        self,
        local_transactions: Sequence[T],
    ) -> Optional[list[T]]:
        """Gather rank-local sequence-ordered transactions onto rank zero."""

        if not self.enabled:
            return list(local_transactions)
        gathered: list[Any] | None = (
            [None for _ in range(self.world_size)] if self.is_rank0 else None
        )
        dist.gather_object(
            list(local_transactions),
            object_gather_list=gathered,
            dst=0,
        )
        if not self.is_rank0:
            return None
        flattened: list[T] = []
        for rank_rows in gathered or ():
            flattened.extend(rank_rows)
        return flattened

    def broadcast_object(self, value: T | None, *, source: int = 0) -> T:
        if not self.enabled:
            return value  # type: ignore[return-value]
        payload = [value]
        dist.broadcast_object_list(payload, src=source)
        return payload[0]

    def event_weighted_all_reduce_gradients(
        self,
        parameters: Iterable[nn.Parameter],
        local_event_count: int,
    ) -> int:
        """Reduce mean-loss gradients as ``sum(N_r g_r) / sum(N_r)``.

        Parameters without a local gradient participate with zeros so every
        rank issues collectives in the same parameter order.  The returned
        integer is the global event count and is useful for diagnostics.
        """

        if local_event_count < 0:
            raise ValueError("local_event_count must be non-negative")
        parameters = tuple(parameters)
        if not self.enabled:
            return local_event_count
        count = torch.tensor(
            float(local_event_count),
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_event_count = int(count.item())
        if global_event_count <= 0:
            raise ValueError("distributed gradient batch has no events")
        scale = float(local_event_count)
        denominator = float(global_event_count)
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
                parameter.grad = gradient
            gradient.mul_(scale)
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(denominator)
        return global_event_count

    def broadcast_module_state(
        self,
        modules: Iterable[nn.Module],
        *,
        source: int = 0,
    ) -> None:
        """Broadcast fixed-topology parameters and buffers in module order."""

        if not self.enabled:
            return
        for module in modules:
            for tensor in module.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=source)


__all__ = ["DistributedRuntime"]
