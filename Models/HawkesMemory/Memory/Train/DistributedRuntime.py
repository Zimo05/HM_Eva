"""Small, explicit distributed runtime used by the Retweet Wake path.

The memory tree is intentionally *not* a :class:`torch.nn.parallel.DistributedDataParallel`
module.  Wake mutates a persistent bank and a dynamic topology, so the useful
unit of synchronization is a tensor reduction or a small transaction log.  This
module keeps those operations in one place and remains a no-op for the normal
single-process training command.

The runtime is deliberately dependency-light.  Importing it does not initialize
``torch.distributed``; :meth:`DistributedRuntime.from_environment` is the only
entry point that may create a process group.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
import hashlib
import os
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist


def _plain(value: Any) -> Any:
    """Convert a transaction payload to CPU/Python values for object collectives."""

    if torch.is_tensor(value):
        return value.detach().cpu()
    if is_dataclass(value):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Memory proposals may contain lightweight dataclass-like objects.  Keep
    # their public attributes instead of relying on a process-local pickle.
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _hash_bytes(value: Any, digest: "hashlib._Hash") -> None:
    """Feed a stable representation into ``digest`` for debug state hashes."""

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"map{")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_bytes(str(key), digest)
            _hash_bytes(value[key], digest)
        digest.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"[")
        for item in value:
            _hash_bytes(item, digest)
        digest.update(b"]")
        return
    if is_dataclass(value):
        _hash_bytes(_plain(value), digest)
        return
    digest.update(repr(value).encode("utf-8"))


def stable_state_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for debug parity checks."""

    digest = hashlib.sha256()
    _hash_bytes(value, digest)
    return digest.hexdigest()


@dataclass
class WakeTransaction:
    """One ordered event-level Wake transaction.

    All fields are intentionally plain metadata/tensors.  The class does not
    commit anything by itself; rank 0 may sort these records and apply them to
    the persistent memory bank.
    """

    sequence_index: int
    event_index: int
    prediction_metrics: Mapping[str, Any] = field(default_factory=dict)
    responsibility: Any = None
    controller_actions: Any = field(default_factory=tuple)
    usage_credits: Mapping[str, Any] = field(default_factory=dict)
    write_proposals: Any = field(default_factory=tuple)
    refresh_proposals: Any = field(default_factory=tuple)
    queue_split_evidence: Any = field(default_factory=tuple)
    age_advance: int = 0
    controller_stat_delta: Mapping[str, Any] = field(default_factory=dict)
    # ``sequence_index`` identifies the sample; it is not the order in which
    # the sample was presented to Wake.  Snapshot Commit therefore carries an
    # explicit physical order key.  Older payloads omit this field and fall
    # back to the legacy identity order in :meth:`commit_key`.
    commit_order: tuple[int, int] | None = None

    def commit_key(self) -> tuple[int, int]:
        """Return the deterministic physical Commit order for this record.

        ``commit_order`` is deliberately separate from ``sequence_index``:
        the latter is a stable dataset identity while the former follows the
        epoch's shuffled wavefront.  Be permissive when reading old/object
        collective payloads so a missing or malformed key remains compatible
        with pre-ordering transaction logs.
        """

        if self.commit_order is not None:
            try:
                values = tuple(self.commit_order)
                if len(values) >= 2:
                    return int(values[0]), int(values[1])
            except (TypeError, ValueError):
                pass
        return int(self.sequence_index), int(self.event_index)

    def to_payload(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WakeTransaction":
        values: dict[str, Any] = {}
        for item in fields(cls):
            if item.name in payload:
                values[item.name] = payload[item.name]
            elif item.default_factory is not MISSING:
                values[item.name] = item.default_factory()
            elif item.default is not MISSING:
                values[item.name] = item.default
        return cls(**values)


@dataclass
class WakeTransactionBatch:
    """A serializable batch produced by one rank during snapshot Compute."""

    transactions: tuple[WakeTransaction, ...] = field(default_factory=tuple)
    wavefront_index: int = 0
    source_rank: int = 0
    snapshot_id: str = ""

    @property
    def sequence_index(self) -> tuple[int, ...]:
        return tuple(item.sequence_index for item in self.transactions)

    @property
    def event_index(self) -> tuple[int, ...]:
        return tuple(item.event_index for item in self.transactions)

    @property
    def commit_order(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.commit_key() for item in self.transactions)

    @property
    def prediction_metrics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(item.prediction_metrics for item in self.transactions)

    @property
    def responsibility(self) -> tuple[Any, ...]:
        return tuple(item.responsibility for item in self.transactions)

    @property
    def controller_actions(self) -> tuple[Any, ...]:
        return tuple(item.controller_actions for item in self.transactions)

    @property
    def usage_credits(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(item.usage_credits for item in self.transactions)

    @property
    def write_proposals(self) -> tuple[Any, ...]:
        return tuple(item.write_proposals for item in self.transactions)

    @property
    def refresh_proposals(self) -> tuple[Any, ...]:
        return tuple(item.refresh_proposals for item in self.transactions)

    @property
    def queue_split_evidence(self) -> tuple[Any, ...]:
        return tuple(item.queue_split_evidence for item in self.transactions)

    @property
    def age_advance(self) -> int:
        return sum(int(item.age_advance) for item in self.transactions)

    @property
    def controller_stat_delta(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in self.transactions:
            for key, value in item.controller_stat_delta.items():
                if torch.is_tensor(value):
                    result[key] = result.get(key, torch.zeros_like(value)) + value
                elif isinstance(value, (int, float)):
                    result[key] = result.get(key, 0) + value
                else:
                    result[key] = value
        return result

    def ordered(self) -> "WakeTransactionBatch":
        return WakeTransactionBatch(
            transactions=tuple(
                sorted(
                    self.transactions,
                    key=lambda item: item.commit_key(),
                )
            ),
            wavefront_index=self.wavefront_index,
            source_rank=self.source_rank,
            snapshot_id=self.snapshot_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "transactions": [item.to_payload() for item in self.transactions],
            "wavefront_index": int(self.wavefront_index),
            "source_rank": int(self.source_rank),
            "snapshot_id": str(self.snapshot_id),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WakeTransactionBatch":
        records = payload.get("transactions", ())
        return cls(
            transactions=tuple(
                WakeTransaction.from_payload(item)
                if not isinstance(item, WakeTransaction)
                else item
                for item in records
            ),
            wavefront_index=int(payload.get("wavefront_index", 0)),
            source_rank=int(payload.get("source_rank", 0)),
            snapshot_id=str(payload.get("snapshot_id", "")),
        )


@dataclass
class CommitLog:
    """Rank-0's deterministic delta, replayed by every rank."""

    accepted_appends: tuple[Any, ...] = field(default_factory=tuple)
    accepted_refreshes: tuple[Any, ...] = field(default_factory=tuple)
    usage_increments: Mapping[str, Any] = field(default_factory=dict)
    age_advance: int = 0
    controller_stat_delta: Any = field(default_factory=dict)
    structural_evidence: Any = field(default_factory=tuple)
    # Every proposal and its admission result, including ``queue``.  The
    # append/refresh arrays above are retained as compact result summaries,
    # while this stream is the authoritative deterministic Bank replay log.
    admission_results: Any = field(default_factory=tuple)
    state_hash: str = ""

    def to_payload(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CommitLog":
        return cls(
            accepted_appends=tuple(payload.get("accepted_appends", ())),
            accepted_refreshes=tuple(payload.get("accepted_refreshes", ())),
            usage_increments=payload.get("usage_increments", {}),
            age_advance=int(payload.get("age_advance", 0)),
            controller_stat_delta=payload.get("controller_stat_delta", {}),
            structural_evidence=payload.get("structural_evidence", ()),
            admission_results=payload.get("admission_results", ()),
            state_hash=str(payload.get("state_hash", "")),
        )


@dataclass
class DistributedRuntime:
    """Process-group wrapper with a safe single-process fallback."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device | str = field(default_factory=lambda: torch.device("cpu"))
    backend: str = "gloo"
    initialized_process_group: bool = False
    debug_hash: bool = False

    @classmethod
    def from_environment(
        cls,
        device: torch.device | str | None = None,
        *,
        backend: str | None = None,
        initialize: bool = True,
        debug_hash: bool = False,
    ) -> "DistributedRuntime":
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if rank < 0 or local_rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError("invalid torchrun rank environment")

        requested = None if device is None else torch.device(device)
        wants_cuda = bool(
            requested is not None and requested.type == "cuda"
        ) or (requested is None and torch.cuda.is_available())
        if world_size > 1 and wants_cuda and not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA training requested but CUDA is unavailable")
        selected_backend = backend or ("nccl" if wants_cuda else "gloo")
        if selected_backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL requires CUDA")
        if world_size > 1 and not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")

        if wants_cuda:
            # Under torchrun every process must use its LOCAL_RANK device.  An
            # explicit ``cuda:N`` is retained only for the non-distributed case.
            selected_device = torch.device(
                f"cuda:{local_rank}" if world_size > 1 else (requested or torch.device("cuda"))
            )
            torch.cuda.set_device(selected_device)
        else:
            selected_device = requested or torch.device("cpu")

        created = False
        if world_size > 1 and initialize and not dist.is_initialized():
            dist.init_process_group(
                backend=selected_backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
            created = True
        elif world_size > 1 and initialize and dist.is_initialized():
            actual_world = dist.get_world_size()
            if actual_world != world_size or dist.get_rank() != rank:
                raise RuntimeError("torchrun environment disagrees with process group")

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=selected_device,
            backend=selected_backend,
            initialized_process_group=created,
            debug_hash=bool(debug_hash),
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def contiguous_shard(self, size: int) -> tuple[int, int]:
        """Return the half-open contiguous shard ``[start, end)`` for rank."""

        size = int(size)
        if size < 0:
            raise ValueError("shard size must be non-negative")
        start = (size * self.rank) // self.world_size
        end = (size * (self.rank + 1)) // self.world_size
        return start, end

    def contiguous_indices(self, values: Sequence[Any]) -> list[Any]:
        start, end = self.contiguous_shard(len(values))
        return list(values[start:end])

    def contiguous_slice(self, size: int) -> slice:
        """Return the rank-local contiguous slice for a sequence length."""

        start, end = self.contiguous_shard(size)
        return slice(start, end)

    def shard_indices(self, size: int) -> range:
        """Return a lazy range over the rank-local contiguous indices."""

        start, end = self.contiguous_shard(size)
        return range(start, end)

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, *, op: Any = None) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError("all_reduce expects a Tensor")
        if self.is_distributed:
            dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
        return tensor

    def all_reduce_gradients(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        average: bool = False,
    ) -> None:
        """SUM gradients in-place; optionally divide by world size."""

        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                if not self.is_distributed:
                    continue
                # A branch can be locally inactive while another rank
                # produces a gradient for the same parameter.  Every rank
                # must still enter the collective and retain the reduced
                # value for the optimizer step.
                parameter.grad = torch.zeros_like(parameter)
            self.all_reduce(parameter.grad)
            if average and self.is_distributed:
                parameter.grad.div_(float(self.world_size))

    def gather_object(self, value: Any, *, dst: int = 0) -> list[Any] | None:
        if not self.is_distributed:
            return [value] if self.rank == dst else None
        gathered: list[Any] | None = [None] * self.world_size if self.rank == dst else None
        dist.gather_object(value, gathered, dst=dst)
        return gathered

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        payload = [value]
        if self.is_distributed:
            dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def state_hash(self, value: Any) -> str:
        return stable_state_hash(value)

    def close(self) -> None:
        if self.initialized_process_group and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized_process_group = False


__all__ = [
    "CommitLog",
    "DistributedRuntime",
    "WakeTransaction",
    "WakeTransactionBatch",
    "stable_state_hash",
]
