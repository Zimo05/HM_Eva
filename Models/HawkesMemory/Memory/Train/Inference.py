"""Wake-only online inference for a trained Hawkes Memory Tree."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from HawkesBackbone import (
    EVENT_TIME_FEATURES_KEY,
    HAWKES_CACHE_SIGNATURE_KEY,
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from LatentHawkesTree import HawkesTree
from MemoryResiduals.ProbationMemory import (
    ProbationCandidate,
    WriteProbationBuffer,
)
from Train.Train import (
    CausalPrefixEncoder,
    WakeObjectiveConfig,
    _frontier_config_from_checkpoint,
)
from Wake.HawkesParams import HawkesParams
from Wake.SequentialController import Action, Controller


@dataclass
class InferenceConfig:
    adapt_working_memory: bool = True
    allow_memory_writes: bool = True
    update_memory_usage: bool = True
    probe_write_counterfactuals: bool = False
    write_probe_random_count: int = 16
    write_probe_seed: int = 42
    prototype_duplicate_threshold: Optional[float] = None
    prototype_mode_threshold: Optional[float] = None
    prototype_context_alias_capacity: Optional[int] = None
    # Local Write acceptance only creates an invisible probation candidate.
    # Persistent admission requires reuse evidence from independent sequences.
    probation_capacity: int = 1024
    probation_match_threshold: float = 0.5
    probation_match_temperature: float = 0.1
    probation_min_effective_samples: float = 2.0
    probation_lcb_kappa: float = 1.96
    probation_persist_threshold: float = 0.0
    # Diagnostic-only causal replay.  When set, no event outside the allowlist
    # may write; listed events bypass the learned Write threshold but still use
    # the normal candidate construction and virtual retrieval path.
    write_event_allowlist: Optional[tuple[int, ...]] = None


class EvaluationProtocol(str, Enum):
    """State-transition contract used by the benchmark evaluator."""

    FROZEN = "frozen"
    FAST_ADAPT = "fast_adapt"
    ONLINE_WRITE = "online_write"


def inference_config_for_protocol(
    protocol: EvaluationProtocol | str,
    **overrides: Any,
) -> InferenceConfig:
    """Build the inference switches for one canonical evaluation protocol.

    The latest HM runtime exposes independent working-memory, persistent-write,
    and usage-update switches.  Keeping their mapping here makes the benchmark
    evaluator describe state transitions instead of reaching into HM internals.
    """

    if isinstance(protocol, str):
        protocol = EvaluationProtocol(protocol)
    defaults = {
        EvaluationProtocol.FROZEN: {
            "adapt_working_memory": False,
            "allow_memory_writes": False,
            "update_memory_usage": False,
        },
        EvaluationProtocol.FAST_ADAPT: {
            "adapt_working_memory": True,
            "allow_memory_writes": False,
            "update_memory_usage": False,
        },
        EvaluationProtocol.ONLINE_WRITE: {
            "adapt_working_memory": True,
            "allow_memory_writes": True,
            "update_memory_usage": True,
        },
    }[protocol].copy()
    defaults.update(overrides)
    return InferenceConfig(**defaults)


class MemoryTreeInference:
    def _controller_effective_parameters(
        self,
        memory_output: Mapping[str, Any],
        working_delta: Tensor,
        retrieval_gate: Tensor,
    ):
        semantic = memory_output["frontier_semantic_theta"]
        episodic = memory_output["frontier_episodic_delta"]
        gate = torch.as_tensor(retrieval_gate).to(episodic)
        while gate.ndim < episodic.ndim - 1:
            gate = gate.unsqueeze(-1)
        D = self.hawkes.num_types
        return self.tree.episodic_memory.parameter_update.compose_effective_parameters(
            semantic_mu=semantic[..., :D],
            semantic_W=semantic[..., D:].reshape(
                *semantic.shape[:-1], D, D, self.hawkes.num_basis
            ),
            episodic_delta=episodic * gate.unsqueeze(-1),
            routing_weights=memory_output["r"],
            working_delta=working_delta,
            decays=self.hawkes.decays,
        )

    """Route, retrieve, adapt working memory, and predict without sleep updates."""

    def __init__(
        self,
        tree: HawkesTree,
        hawkes: HawkesFamily,
        encoder: nn.Module,
        *,
        wake_config: Optional[WakeObjectiveConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.wake_config = (
            WakeObjectiveConfig() if wake_config is None else wake_config
        )
        self.config = InferenceConfig() if inference_config is None else inference_config
        if self.config.probation_match_temperature <= 0.0:
            raise ValueError("probation match temperature must be positive")
        if self.config.probation_min_effective_samples < 0.0:
            raise ValueError(
                "probation minimum effective samples must be non-negative"
            )
        if self.config.probation_lcb_kappa < 0.0:
            raise ValueError("probation LCB kappa must be non-negative")
        if device is None:
            device = next(tree.parameters()).device
        self.device = torch.device(device)
        self.tree = tree.to(self.device).eval()
        self.hawkes = hawkes.to(self.device).eval()
        self.encoder = encoder.to(self.device).eval()
        self.tree.episodic_memory.configure_prototype_memory(
            duplicate_threshold=(
                self.wake_config.prototype_duplicate_threshold
                if self.config.prototype_duplicate_threshold is None
                else self.config.prototype_duplicate_threshold
            ),
            mode_threshold=(
                self.wake_config.prototype_mode_threshold
                if self.config.prototype_mode_threshold is None
                else self.config.prototype_mode_threshold
            ),
            duplicate_quantile=self.wake_config.prototype_duplicate_quantile,
            mode_capacity=self.wake_config.prototype_mode_capacity,
            context_alias_capacity=(
                self.wake_config.prototype_context_alias_capacity
                if self.config.prototype_context_alias_capacity is None
                else self.config.prototype_context_alias_capacity
            ),
        )
        self.tree.episodic_memory.rebuild_law_keys(
            self.tree.semantic_theta,
            self.hawkes.decays,
        )
        self.write_probation = WriteProbationBuffer(
            capacity=self.config.probation_capacity
        )
        self._anonymous_sequence_counter = 0
        self.controller = Controller(
            nll_fn=self.hawkes,
            tau_s=self.wake_config.tau_surprise,
            tau_n=self.wake_config.tau_novelty,
            tau_c=self.wake_config.tau_count,
            tau_sim=self.wake_config.tau_similarity,
            eta_mem=self.wake_config.eta_memory_write,
            memory_write_grad_clip=self.wake_config.memory_write_grad_clip,
            write_horizon=self.wake_config.write_horizon,
            episodic_memory=self.tree.episodic_memory,
            working_memory=self.tree.working_memory,
            action_temperature=self.wake_config.action_temperature,
            novelty_temperature=self.wake_config.novelty_temperature,
            count_exponent=self.wake_config.count_exponent,
            count_similarity_low=self.wake_config.count_similarity_low,
            count_similarity_high=self.wake_config.count_similarity_high,
            count_topk=self.wake_config.count_topk,
            count_saturation=self.wake_config.count_saturation,
            surprise_ema_decay=self.wake_config.surprise_ema_decay,
            write_candidate_threshold=(
                self.wake_config.controller_write_admission_threshold
            ),
            # Inference is deterministic and never creates exploration writes.
            exploration_rate=0.0,
            utility_topc_multiplier=(
                self.wake_config.controller_utility_topc_multiplier
            ),
            utility_stage_enabled=(
                self.wake_config.controller_utility_stage_enabled
            ),
            utility_temperature=self.wake_config.controller_utility_temperature,
            utility_cost_margin=self.wake_config.controller_utility_cost_margin,
        ).to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: Optional[torch.device | str] = None,
        inference_config: Optional[InferenceConfig] = None,
        encoder: Optional[nn.Module] = None,
    ) -> "MemoryTreeInference":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        config = checkpoint["model_config"]
        if config.get("router_kind") not in {
            "node_semantic_compat_v1",
            "active_frontier_v1",
            "posterior_frontier_v2",
        }:
            raise ValueError(
                "checkpoint uses the retired z-only Linear Router; the current "
                "model requires Compat(z_t, u_n). Retrain before inference."
            )
        decays = torch.as_tensor(config["decays"], dtype=torch.float32)
        hawkes = HawkesFamily(
            num_types=config["num_event_types"],
            num_basis=config["num_basis"],
            decays=decays,
        ).to(device)
        tree = HawkesTree(
            z_dim=config["z_dim"],
            node_dim=config["node_dim"],
            num_event_types=config["num_event_types"],
            num_basis=config["num_basis"],
            init_depth=0,
            temperature=config.get("tree_temperature", 1.0),
            hyper_hidden_dim=config.get("hyper_hidden_dim", 256),
            memory_key_dim=config["memory_key_dim"],
            memory_capacity_per_node=config.get(
                "memory_capacity_per_node", 128
            ),
            working_rho=config.get("working_rho", 0.8),
            working_eta=config.get("working_eta", 1e-2),
        ).to(device)
        tree.configure_frontier_routing(
            config=_frontier_config_from_checkpoint(config)
        )
        if encoder is None:
            encoder_config = config.get("encoder_config", {"kind": "causal_prefix"})
            kind = encoder_config.get("kind", "causal_prefix")
            if kind == "attention_memory":
                raise ValueError(
                    "checkpoint encoder_config.kind='attention_memory' is no longer "
                    "supported; retrain with CausalPrefixEncoder and optional "
                    "--h-tree semantic initialization"
                )
            type_dim = config.get("encoder_type_dim") or 32
            hidden_dim = config.get("encoder_hidden_dim") or 128
            encoder = CausalPrefixEncoder(
                num_event_types=config["num_event_types"],
                z_dim=config["z_dim"],
                type_dim=type_dim,
                hidden_dim=hidden_dim,
            )
        encoder = encoder.to(device)
        hawkes.load_state_dict(checkpoint["hawkes_state_dict"])
        incompatible = tree.load_state_dict(
            checkpoint["tree_state_dict"],
            strict=False,
        )
        legacy_missing = {
            key
            for key in incompatible.missing_keys
            if key.startswith("frontier_routing.prototypes.")
        }
        if set(incompatible.missing_keys).difference(legacy_missing):
            raise RuntimeError(
                "checkpoint is missing model tensors: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint contains unexpected model tensors: "
                f"{incompatible.unexpected_keys}"
            )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        wake_config = WakeObjectiveConfig(**checkpoint.get("wake_config", {}))
        inference = cls(
            tree=tree,
            hawkes=hawkes,
            encoder=encoder,
            wake_config=wake_config,
            inference_config=inference_config,
            device=device,
        )
        controller_state = checkpoint.get("controller_state", {})
        module_state = controller_state.get("module_state_dict")
        if module_state is not None:
            incompatible_controller = inference.controller.load_state_dict(
                module_state, strict=False
            )
            allowed_missing = {
                "controller_version",
                "context_gate.weight",
                "bias_assimilate",
                "utility_mean",
                "utility_variance",
                "utility_observations",
                "utility_temperatures",
                "calibration_thresholds",
                "split_enabled",
            }
            if set(incompatible_controller.missing_keys).difference(
                allowed_missing
            ) or incompatible_controller.unexpected_keys:
                raise RuntimeError(
                    "checkpoint controller state is incompatible: "
                    f"missing={incompatible_controller.missing_keys}, "
                    f"unexpected={incompatible_controller.unexpected_keys}"
                )
            if controller_state.get("controller_version", 1) < 2:
                inference.controller.migrate_legacy_policy()
            if controller_state.get("controller_version", 1) < 5:
                # v2-v4 checkpoints predate the conservative v5 Split policy.
                inference.controller.split_enabled.fill_(True)
        inference.controller.utility_stage_enabled = bool(
            controller_state.get(
                "utility_stage_enabled",
                inference.controller.utility_stage_enabled,
            )
        )
        inference.controller.split_queues.update(
            controller_state.get("split_queues", {})
        )
        return inference

    def _move_sequence(self, sequence: Mapping[str, Tensor]) -> Dict[str, Any]:
        cached = self.hawkes.prepare_sequence_cache(sequence, inplace=True)
        result = {
            "times": cached["times"].to(self.device),
            "types": cached["types"].to(self.device).long(),
            EVENT_TIME_FEATURES_KEY: cached[EVENT_TIME_FEATURES_KEY].to(
                self.device
            ),
            HAWKES_HISTORY_STATS_KEY: cached[HAWKES_HISTORY_STATS_KEY].to(
                self.device
            ),
            HAWKES_INTERVAL_STATS_KEY: cached[HAWKES_INTERVAL_STATS_KEY].to(
                self.device
            ),
            HAWKES_CACHE_SIGNATURE_KEY: cached[HAWKES_CACHE_SIGNATURE_KEY],
        }
        if "T" in sequence:
            result["T"] = sequence["T"].to(self.device)
        return result

    @torch.no_grad()
    def _forecast_history_statistics(
        self,
        sequence: Mapping[str, Tensor],
    ) -> Tensor:
        """Cache prefix statistics at the causal forecast origin.

        ``event_NLL`` uses the strict-time history cache.  Forecasting in
        ``run_sequence`` instead evaluates the local rate at
        ``t[k - 1] + 1e-6``.  For the next event, every prefix event is at or
        before ``t[k - 1]``, so the cached history at ``t[k]`` can be rescaled
        to that forecast origin; tied events are then added explicitly.  This
        keeps the batched path algebraically identical to the scalar path
        without putting a Python event loop around ``intensity_at_event``.
        """
        times = sequence["times"]
        types = sequence["types"].long()
        event_count = int(times.numel())
        if event_count == 0:
            return times.new_empty(
                (0, self.hawkes.num_types, self.hawkes.num_basis)
            )
        expected_shape = (
            event_count,
            self.hawkes.num_types,
            self.hawkes.num_basis,
        )
        strict_history = sequence.get(HAWKES_HISTORY_STATS_KEY)
        if (
            not isinstance(strict_history, Tensor)
            or sequence.get(HAWKES_CACHE_SIGNATURE_KEY)
            != self.hawkes.cache_signature
            or strict_history.shape != expected_shape
        ):
            cached = self.hawkes.prepare_sequence_cache(sequence, inplace=False)
            strict_history = cached[HAWKES_HISTORY_STATS_KEY]
        strict_history = strict_history.to(
            device=times.device,
            dtype=times.dtype,
        )

        # ``event_NLL``'s history cache is evaluated at times[k], while the
        # scalar forecast is evaluated at times[k - 1] + 1e-6.  Since event k
        # is the next event in a non-decreasing sequence, the contribution of
        # every strict-time prefix event is changed by the same per-basis
        # decay factor.  The remaining tied predecessors are not present in
        # strict_history and are added from cumulative type counts.  No
        # [L, L, M] tensor is needed.
        if event_count == 1:
            return strict_history
        one_hot_types = F.one_hot(
            types,
            num_classes=self.hawkes.num_types,
        ).to(dtype=times.dtype)
        run_start = torch.cat([
            torch.ones(1, dtype=torch.bool, device=times.device),
            times[1:] != times[:-1],
        ])
        run_ids = run_start.to(torch.long).cumsum(dim=0) - 1
        cumulative = one_hot_types.cumsum(dim=0)
        run_starts = torch.nonzero(run_start, as_tuple=False).reshape(-1)
        run_base = cumulative.index_select(0, run_starts) - (
            one_hot_types.index_select(0, run_starts)
        )
        same_timestamp_counts = cumulative[:-1] - run_base.index_select(
            0, run_ids[:-1]
        )
        tied_with_previous = times[1:] == times[:-1]
        decays = self.hawkes.decays.to(device=times.device, dtype=times.dtype)
        forecast_epsilon = times.new_tensor(1e-6)
        strict_scale = torch.exp(
            (
                times[1:] - times[:-1] - forecast_epsilon
            ).reshape(-1, 1, 1)
            * decays.reshape(1, 1, -1)
        )
        strict_forecast = strict_history[1:] * strict_scale
        same_timestamp_kernel = torch.exp(-forecast_epsilon * decays)
        tied_correction = (
            same_timestamp_counts.unsqueeze(-1)
            * same_timestamp_kernel.reshape(1, 1, -1)
            * tied_with_previous.reshape(-1, 1, 1).to(times.dtype)
        )
        return torch.cat([
            strict_history[:1],
            strict_forecast + tied_correction,
        ], dim=0)

    @torch.no_grad()
    def prepare_sequence_batch(
        self,
        sequences: Sequence[Mapping[str, Any]],
        *,
        frontier_static_cache: Any = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Prepare a padded sequence minibatch for packed inference.

        The returned ``z`` tensors contain one strict-prefix embedding per
        observed event.  ``projected_z`` and ``memory_query`` are prepared in
        the same flattened order, so routing and retrieval can consume all
        valid prefixes in one call.  Sequence metadata is retained for the
        evaluator, while all model tensors live on ``self.device``.
        """
        if frontier_static_cache is None:
            frontier_static_cache = self.tree.frontier_routing.build_static_cache(
                detach=True
            )
        if not sequences:
            return [], frontier_static_cache

        moved_sequences: list[dict[str, Any]] = []
        lengths: list[int] = []
        for sequence in sequences:
            # Do not mutate the evaluator's source mapping while adding the
            # resident Hawkes cache.  Keep labels such as eval_set_id beside
            # the device-resident tensors for downstream reporting.
            source = dict(sequence)
            moved = self._move_sequence(source)
            moved_sequence = {**source, **moved}
            moved_sequences.append(moved_sequence)
            lengths.append(int(moved_sequence["times"].numel()))

        # The benchmark loader normally filters empty sequences.  Returning a
        # compatibility marker here lets callers use the established scalar
        # fallback for unusual/custom inputs instead of producing an invalid
        # packed GRU batch.
        if any(length <= 0 for length in lengths):
            return [
                {
                    "sequence": moved,
                    "z": None,
                    "projected_z": None,
                    "memory_query": None,
                    "length": length,
                }
                for moved, length in zip(moved_sequences, lengths)
            ], frontier_static_cache

        z_by_sequence: list[Tensor]
        if hasattr(self.encoder, "forward_padded_prefix"):
            batch_size = len(moved_sequences)
            max_length = max(lengths)
            reference_times = moved_sequences[0]["times"]
            padded_times = reference_times.new_zeros(batch_size, max_length)
            padded_types = torch.zeros(
                batch_size,
                max_length,
                dtype=torch.long,
                device=reference_times.device,
            )
            padded_features = reference_times.new_zeros(
                batch_size,
                max_length,
                2,
            )
            valid_mask = torch.zeros(
                batch_size,
                max_length,
                dtype=torch.bool,
                device=reference_times.device,
            )
            for row, (sequence, length) in enumerate(
                zip(moved_sequences, lengths)
            ):
                padded_times[row, :length] = sequence["times"]
                padded_types[row, :length] = sequence["types"]
                padded_features[row, :length] = sequence[
                    EVENT_TIME_FEATURES_KEY
                ]
                valid_mask[row, :length] = True
            z_padded, prefix_mask = self.encoder.forward_padded_prefix(
                padded_times,
                padded_types,
                valid_mask,
                time_features=padded_features,
                lengths_cpu=lengths,
            )
            z_by_sequence = [
                z_padded[row, :length].contiguous()
                for row, length in enumerate(lengths)
            ]
            z_flat = z_padded[prefix_mask]
        elif hasattr(self.encoder, "forward_all_prefix"):
            # This is a compatibility path for prefix encoders that already
            # expose a vectorized single-sequence API but not the padded API.
            z_by_sequence = [
                self.encoder.forward_all_prefix(
                    sequence["times"],
                    sequence["types"],
                    time_features=sequence.get(EVENT_TIME_FEATURES_KEY),
                )
                for sequence in moved_sequences
            ]
            z_flat = torch.cat(z_by_sequence, dim=0)
        else:
            return [
                {
                    "sequence": moved,
                    "z": None,
                    "projected_z": None,
                    "memory_query": None,
                    "length": length,
                }
                for moved, length in zip(moved_sequences, lengths)
            ], frontier_static_cache

        projected_flat = self.tree.router_compat.project_z(z_flat)
        if self.tree.episodic_memory.query_net is None:
            raise RuntimeError("the wrapped tree must construct memory.query_net")
        query_flat = self.tree.episodic_memory.query_net(z_flat)

        prepared: list[dict[str, Any]] = []
        start = 0
        for sequence, length, z in zip(moved_sequences, lengths, z_by_sequence):
            stop = start + length
            prepared.append({
                "sequence": sequence,
                "z": z,
                "projected_z": projected_flat[start:stop],
                "memory_query": query_flat[start:stop],
                "forecast_history_stats": self._forecast_history_statistics(
                    sequence
                ),
                "length": length,
            })
            start = stop
        return prepared, frontier_static_cache

    def _batched_hawkes_intensity(
        self,
        theta: Tensor,
        history_stats: Tensor,
    ) -> Tensor:
        """Evaluate Hawkes intensities for ``[N]`` or ``[N, K]`` rows."""
        if theta.ndim not in (2, 3):
            raise ValueError("theta must have shape [N, P] or [N, K, P]")
        if history_stats.ndim != 3:
            raise ValueError("history_stats must have shape [N, D, M]")
        history_stats = history_stats.to(device=theta.device, dtype=theta.dtype)
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        raw_mu = theta[..., :D]
        raw_W = theta[..., D:].reshape(*theta.shape[:-1], D, D, M)
        mu = F.softplus(raw_mu)
        W = F.softplus(raw_W)
        if theta.ndim == 2:
            excitation = (
                W * history_stats.unsqueeze(1)
            ).sum(dim=(-1, -2))
        else:
            excitation = (
                W * history_stats.unsqueeze(1).unsqueeze(1)
            ).sum(dim=(-1, -2))
        return (mu + excitation).clamp_min(1e-8)

    def _batched_event_nll(
        self,
        theta: Tensor,
        history_stats: Tensor,
        interval_stats: Tensor,
        event_types: Tensor,
        durations: Tensor,
    ) -> Tensor:
        """Vectorized equivalent of ``HawkesFamily.event_NLL``."""
        if theta.ndim not in (2, 3):
            raise ValueError("theta must have shape [N, P] or [N, K, P]")
        history_stats = history_stats.to(device=theta.device, dtype=theta.dtype)
        interval_stats = interval_stats.to(device=theta.device, dtype=theta.dtype)
        durations = durations.to(device=theta.device, dtype=theta.dtype)
        event_types = event_types.to(device=theta.device, dtype=torch.long)
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        intensity = self._batched_hawkes_intensity(theta, history_stats)
        raw_mu = theta[..., :D]
        raw_W = theta[..., D:].reshape(*theta.shape[:-1], D, D, M)
        mu = F.softplus(raw_mu)
        W = F.softplus(raw_W)
        if theta.ndim == 2:
            # W is [N, D_out, D_src, M] and interval_stats is
            # [N, D_src, M].  The interval integral is a scalar per event:
            # all output types, source types, and basis functions must be
            # reduced.  Reducing only (-1, -2) leaves an erroneous [N, D_out]
            # tensor that broadcasts into the event loss.
            target = intensity.gather(
                1,
                event_types.reshape(-1, 1),
            ).squeeze(1)
            integral = (
                mu.sum(dim=-1) * durations
                + (W * interval_stats.unsqueeze(1)).sum(dim=(-3, -2, -1))
            )
        else:
            width = theta.size(1)
            target = intensity.gather(
                2,
                event_types.reshape(-1, 1, 1).expand(-1, width, 1),
            ).squeeze(-1)
            integral = (
                mu.sum(dim=-1) * durations.unsqueeze(1)
                + (
                    W * interval_stats.unsqueeze(1).unsqueeze(1)
                ).sum(dim=(-3, -2, -1))
            )
        return -torch.log(target.clamp_min(1e-8)) + integral

    def _lca_index_table(self, reference: Tensor) -> Tensor:
        """Return a topology-versioned device table for pairwise LCA folds."""
        signature = tuple(
            (
                node_id,
                self.tree.nodes[node_id].parent,
                self.tree.nodes[node_id].left,
                self.tree.nodes[node_id].right,
            )
            for node_id in self.tree.all_node_ids
        )
        cached = getattr(self, "_inference_lca_table_cache", None)
        if (
            cached is not None
            and cached[0] == signature
            and cached[1].device == reference.device
        ):
            return cached[1]

        node_ids = tuple(self.tree.all_node_ids)
        node_index = {
            node_id: index for index, node_id in enumerate(node_ids)
        }
        values = [
            [
                node_index[self._lowest_common_ancestor((left, right))]
                for right in node_ids
            ]
            for left in node_ids
        ]
        table = torch.tensor(
            values,
            device=reference.device,
            dtype=torch.long,
        )
        self._inference_lca_table_cache = (signature, table)
        return table

    def _posterior_owner_indices_batch(
        self,
        frontier_node_indices: Tensor,
        frontier_mask: Tensor,
        posterior: Tensor,
    ) -> Tensor:
        """Resolve posterior owners for a whole wavefront on the device.

        The returned indices are consumed by packed retrieval/controller
        calculations.  String IDs are intentionally materialized separately
        at the diagnostic boundary so the numerical path has no per-event
        ``.item()``/``.cpu()`` synchronization.
        """
        if (
            frontier_node_indices.ndim != 2
            or frontier_mask.shape != frontier_node_indices.shape
            or posterior.shape != frontier_node_indices.shape
        ):
            raise ValueError(
                "frontier indices, mask, and posterior must align as [N, K]"
            )
        safe_posterior = posterior.masked_fill(~frontier_mask, 0.0)
        sort_posterior = posterior.masked_fill(~frontier_mask, -torch.inf)
        order = sort_posterior.argsort(
            dim=-1,
            descending=True,
            stable=True,
        )
        sorted_mass = safe_posterior.gather(1, order)
        credible_count = (
            sorted_mass.cumsum(dim=-1)
            < self.tree.frontier_routing.config.credible_mass
        ).sum(dim=-1) + 1
        credible_count = torch.minimum(
            credible_count,
            frontier_mask.sum(dim=-1),
        )
        sorted_nodes = frontier_node_indices.clamp_min(0).gather(1, order)
        owner = sorted_nodes[:, 0]
        lca_table = self._lca_index_table(posterior)
        for slot in range(1, posterior.size(1)):
            combined = lca_table[owner, sorted_nodes[:, slot]]
            owner = torch.where(slot < credible_count, combined, owner)
        return owner

    def _materialize_batched_frontier_diagnostics(
        self,
        frontier_node_indices: Tensor,
        frontier_mask: Tensor,
        owner_indices: Tensor,
    ) -> tuple[list[tuple[str, ...]], list[str]]:
        """Convert a completed owner batch into human-readable diagnostic IDs."""
        width = frontier_node_indices.size(1)
        packed = torch.cat(
            [
                frontier_node_indices.detach().to(dtype=torch.long),
                frontier_mask.detach().to(dtype=torch.long),
                owner_indices.detach().to(dtype=torch.long).unsqueeze(1),
            ],
            dim=1,
        ).cpu().tolist()
        node_ids = tuple(self.tree.all_node_ids)
        frontier_ids: list[tuple[str, ...]] = []
        owner_ids: list[str] = []
        for row in packed:
            indices = row[:width]
            mask = row[width:2 * width]
            owner_index = int(row[2 * width])
            active_indices = [
                int(index)
                for index, active in zip(indices, mask)
                if active
            ]
            if not active_indices:
                raise RuntimeError("packed frontier contains no active node")
            if any(index < 0 or index >= len(node_ids) for index in active_indices):
                raise RuntimeError("packed frontier contains an invalid node index")
            if owner_index < 0 or owner_index >= len(node_ids):
                raise RuntimeError("packed owner contains an invalid node index")
            frontier_ids.append(tuple(node_ids[index] for index in active_indices))
            owner_ids.append(node_ids[owner_index])
        return frontier_ids, owner_ids

    def _batched_frontier_owners(
        self,
        frontier_node_indices: Tensor,
        frontier_mask: Tensor,
        posterior: Tensor,
    ) -> tuple[list[tuple[str, ...]], list[str], Tensor]:
        """Compatibility wrapper returning IDs plus device owner indices."""
        owner_indices = self._posterior_owner_indices_batch(
            frontier_node_indices,
            frontier_mask,
            posterior,
        )
        frontier_ids, owner_ids = self._materialize_batched_frontier_diagnostics(
            frontier_node_indices,
            frontier_mask,
            owner_indices,
        )
        return frontier_ids, owner_ids, owner_indices

    def run_sequence_batch_compact(
        self,
        prepared: Sequence[Mapping[str, Any]],
        *,
        frontier_static_cache: Any = None,
        capture_event_predictions: bool | Sequence[bool] = False,
        capture_prediction_theta: bool = False,
    ) -> list[dict[str, Any]]:
        """Run a read-only sequence minibatch through packed HM inference.

        Prefix encoding, routing, memory retrieval, frontier energies, and
        Hawkes terms are evaluated on flattened valid prefixes.  The only
        remaining time loop advances independent Working Memory rows, which
        preserves the causal recurrence for ``FAST_ADAPT`` while allowing all
        sequences in one minibatch to share each wavefront computation.
        """
        if not prepared:
            return []
        if self.config.allow_memory_writes or self.config.update_memory_usage:
            raise ValueError(
                "packed inference is restricted to read-only protocols"
            )
        if any(item.get("z") is None for item in prepared):
            raise ValueError(
                "packed inference requires prepared prefix embeddings"
            )
        if isinstance(capture_event_predictions, bool):
            capture_by_sequence = [
                capture_event_predictions for _ in prepared
            ]
        else:
            capture_by_sequence = [
                bool(value) for value in capture_event_predictions
            ]
            if len(capture_by_sequence) != len(prepared):
                raise ValueError(
                    "capture_event_predictions must be a bool or have one "
                    "value per prepared sequence"
                )
        # Working Memory is sequence-local.  The packed implementation keeps
        # its recurrence in ``working_state`` rather than the adapter's scalar
        # slot, so start from the same clean boundary as ``run_sequence``.
        self.tree.reset_working_memory()
        if frontier_static_cache is None:
            frontier_static_cache = self.tree.frontier_routing.build_static_cache(
                detach=True
            )

        lengths = [
            int(item.get("length", item["sequence"]["times"].numel()))
            for item in prepared
        ]
        if any(length <= 0 for length in lengths):
            raise ValueError("packed inference cannot contain empty sequences")
        z_flat = torch.cat([
            item["z"][:length]
            for item, length in zip(prepared, lengths)
        ], dim=0)
        if any(item.get("projected_z") is None for item in prepared):
            with torch.no_grad():
                projected_flat = self.tree.router_compat.project_z(z_flat)
        else:
            projected_flat = torch.cat([
                item["projected_z"][:length]
                for item, length in zip(prepared, lengths)
            ], dim=0)
        if any(item.get("memory_query") is None for item in prepared):
            if self.tree.episodic_memory.query_net is None:
                raise RuntimeError("the wrapped tree must construct memory.query_net")
            with torch.no_grad():
                query_flat = self.tree.episodic_memory.query_net(z_flat)
        else:
            query_flat = torch.cat([
                item["memory_query"][:length]
                for item, length in zip(prepared, lengths)
            ], dim=0)

        device = z_flat.device
        sequence_index = torch.cat([
            torch.full(
                (length,),
                row,
                dtype=torch.long,
                device=device,
            )
            for row, length in enumerate(lengths)
        ])
        time_index = torch.cat([
            torch.arange(length, dtype=torch.long, device=device)
            for length in lengths
        ])
        times_flat = torch.cat([
            item["sequence"]["times"][:length]
            for item, length in zip(prepared, lengths)
        ], dim=0)
        types_flat = torch.cat([
            item["sequence"]["types"][:length].long()
            for item, length in zip(prepared, lengths)
        ], dim=0)
        history_flat = torch.cat([
            item["sequence"][HAWKES_HISTORY_STATS_KEY][:length]
            for item, length in zip(prepared, lengths)
        ], dim=0)
        interval_flat = torch.cat([
            item["sequence"][HAWKES_INTERVAL_STATS_KEY][:length]
            for item, length in zip(prepared, lengths)
        ], dim=0)
        forecast_stats_by_sequence = []
        duration_by_sequence = []
        forecast_origin_by_sequence = []
        for item, length in zip(prepared, lengths):
            sequence = item["sequence"]
            forecast_stats = item.get("forecast_history_stats")
            if forecast_stats is None:
                forecast_stats = self._forecast_history_statistics(sequence)
            forecast_stats_by_sequence.append(forecast_stats[:length])
            times = sequence["times"][:length]
            duration_by_sequence.append(
                torch.cat([
                    # event_NLL integrates the baseline over [0, t_0] for
                    # the first event; only later rows use inter-event gaps.
                    times[:1].clamp_min(0.0),
                    (times[1:] - times[:-1]).clamp_min(0.0),
                ])
            )
            forecast_origin_by_sequence.append(
                torch.cat([
                    times.new_zeros(1),
                    times[:-1],
                ])
            )
        forecast_flat = torch.cat(forecast_stats_by_sequence, dim=0)
        duration_flat = torch.cat(duration_by_sequence, dim=0)
        forecast_origin_flat = torch.cat(forecast_origin_by_sequence, dim=0)

        parameter_dim = self.tree.param_dim
        with torch.no_grad():
            static_memory_output = self.tree(
                z_flat,
                working_delta=z_flat.new_zeros(z_flat.size(0), parameter_dim),
                decays=self.hawkes.decays,
                frontier_static_cache=frontier_static_cache,
                frontier_projected_z=projected_flat,
                frontier_query=query_flat,
                update_memory_state=False,
                # Frozen/fast-adapt evaluation is read-only, including the
                # routing diagnostic counters.  This keeps checkpoint-level
                # results independent of evaluation order and batch schedule.
                update_search_state=False,
                materialize_diagnostics=False,
            )
            frontier_energy = self._batched_event_nll(
                static_memory_output["frontier_theta"],
                history_flat,
                interval_flat,
                types_flat,
                duration_flat,
            )
            frontier_energy = frontier_energy.masked_fill(
                ~static_memory_output["frontier_mask"],
                torch.inf,
            )
            temperature = self.tree.frontier_routing.config.posterior_temperature
            frontier_logits = (
                static_memory_output["frontier_mass"].clamp_min(1e-12).log()
                - frontier_energy / temperature
            )
            posterior = torch.softmax(
                frontier_logits.masked_fill(
                    ~static_memory_output["frontier_mask"],
                    -torch.inf,
                ),
                dim=-1,
            ).masked_fill(~static_memory_output["frontier_mask"], 0.0)

            owner_indices = self._posterior_owner_indices_batch(
                static_memory_output["frontier_node_indices"],
                static_memory_output["frontier_mask"],
                posterior,
            )
            novelty, count, weighted_similarity = (
                self.tree.episodic_memory.novelty_count_packed(
                    query_flat,
                    owner_indices,
                    tuple(self.tree.all_node_ids),
                    temperature=self.controller.novelty_temperature,
                    count_exponent=self.controller.count_exponent,
                    eps=self.controller.controller_eps,
                    count_similarity_low=self.controller.count_similarity_low,
                    count_similarity_high=self.controller.count_similarity_high,
                    count_topk=self.controller.count_topk,
                    count_saturation=self.controller.count_saturation,
                )
            )

            packed_info = static_memory_output["packed_memory_info"]
            visited_mask = static_memory_output["visited_node_mask"]
            retrieval_alpha_mass = (
                packed_info["alpha"]
                * visited_mask.unsqueeze(-1).to(packed_info["alpha"].dtype)
            ).sum(dim=(-1, -2))
            retrieval_effective_k = (
                packed_info["effective_k"]
                * visited_mask.to(packed_info["effective_k"].dtype)
            ).sum(dim=-1)
            visited_count = visited_mask.sum(dim=-1)
            retrieval_null_alpha = (
                packed_info["null_alpha"] * visited_mask.to(
                    packed_info["null_alpha"].dtype
                )
            ).sum(dim=-1) / visited_count.clamp_min(1).to(
                packed_info["null_alpha"].dtype
            )
            retrieval_null_alpha = torch.where(
                visited_count > 0,
                retrieval_null_alpha,
                retrieval_null_alpha.new_ones(()),
            )
            visited_nonempty = (
                packed_info["valid_mask"].any(dim=-1) & visited_mask
            ).sum(dim=-1)
            raw_episodic_residual_norm = (
                static_memory_output["frontier_episodic_delta"]
                .detach()
                .norm(dim=-1)
                .mean(dim=-1)
            )
            episodic_residual_norm = (
                static_memory_output["episodic_delta"]
                .detach()
                .norm(dim=(-2, -1))
            )
            owner_on_retrieval_path = (
                static_memory_output["visited_node_indices"]
                == owner_indices.unsqueeze(1)
            ) & visited_mask
            owner_on_retrieval_path = owner_on_retrieval_path.any(dim=-1)

        batch_size = len(prepared)
        nll_by_sequence = z_flat.new_zeros(batch_size)
        correct_by_sequence = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        time_abs_by_sequence = z_flat.new_zeros(batch_size)
        working_state = self.tree.working_memory.new_batch_state(batch_size)
        adapt_working_memory = bool(self.config.adapt_working_memory)
        need_events = bool(
            capture_prediction_theta or any(capture_by_sequence)
        )
        event_results: list[Optional[dict[str, Any]]] = (
            [None] * z_flat.size(0) if need_events else []
        )
        if need_events:
            frontier_ids, owner_ids = self._materialize_batched_frontier_diagnostics(
                static_memory_output["frontier_node_indices"],
                static_memory_output["frontier_mask"],
                owner_indices,
            )

        # Static tensors are copied to CPU only when the caller asks for event
        # rows. The numerical hot path remains entirely device-side.
        if need_events:
            posterior_cpu = posterior.detach().cpu()
            responsibility_cpu = static_memory_output["r"].detach().cpu()
            raw_residual_cpu = raw_episodic_residual_norm.detach().cpu()
            episodic_residual_cpu = episodic_residual_norm.detach().cpu()
            alpha_mass_cpu = retrieval_alpha_mass.detach().cpu()
            effective_k_cpu = retrieval_effective_k.detach().cpu()
            null_alpha_cpu = retrieval_null_alpha.detach().cpu()
            visited_count_cpu = visited_count.detach().cpu()
            visited_nonempty_cpu = visited_nonempty.detach().cpu()
            owner_on_path_cpu = owner_on_retrieval_path.detach().cpu()
            sequence_index_cpu = sequence_index.detach().cpu().tolist()
            time_index_cpu = time_index.detach().cpu().tolist()
            times_flat_cpu = times_flat.detach().cpu().tolist()
            weighted_similarity_cpu = weighted_similarity.detach().cpu()
            source_index_by_sequence = []
            for item in prepared:
                source_value = item["sequence"].get("source_index", -1)
                source_index_by_sequence.append(
                    int(
                        source_value.item()
                        if hasattr(source_value, "item")
                        else source_value
                    )
                )
            calibrated_write_threshold = float(
                self.controller.calibration_thresholds[2].detach().cpu()
            )

        if adapt_working_memory:
            active_row_waves = [
                torch.nonzero(
                    time_index == position,
                    as_tuple=False,
                ).reshape(-1)
                for position in range(max(lengths))
            ]
        else:
            # Frozen evaluation has no sequence-dependent recurrence. All
            # valid events can therefore share one controller/Hawkes wave,
            # avoiding an otherwise unnecessary Python loop over positions.
            active_row_waves = [
                torch.arange(z_flat.size(0), dtype=torch.long, device=device)
            ]
        for active_rows in active_row_waves:
            if active_rows.numel() == 0:
                continue
            sequence_rows = sequence_index.index_select(0, active_rows)
            state_before = working_state.index_select(0, sequence_rows)
            if adapt_working_memory:
                working_used = state_before.detach().clone().requires_grad_(True)
            else:
                working_used = state_before

            wave_memory = {
                key: static_memory_output[key].index_select(0, active_rows)
                for key in (
                    "r",
                    "frontier_semantic_theta",
                    "frontier_episodic_delta",
                )
            }
            active_history = history_flat.index_select(0, active_rows)
            active_interval = interval_flat.index_select(0, active_rows)
            active_types = types_flat.index_select(0, active_rows)
            active_durations = duration_flat.index_select(0, active_rows)

            with torch.no_grad():
                pre_params = self._controller_effective_parameters(
                    wave_memory,
                    working_used,
                    working_used.new_zeros(active_rows.numel()),
                )
                pre_nll = self._batched_event_nll(
                    pre_params.theta,
                    active_history,
                    active_interval,
                    active_types,
                    active_durations,
                )
                controller_output = self.controller.action_distribution_batch(
                    surprise=pre_nll,
                    novelty=novelty.index_select(0, active_rows),
                    count=count.index_select(0, active_rows),
                    update_statistics=False,
                    owner_confidence=posterior.index_select(
                        0, active_rows
                    ).max(dim=-1).values,
                    retrieval_similarity=weighted_similarity.index_select(
                        0, active_rows
                    ),
                    retrieval_residual_norm=raw_episodic_residual_norm.index_select(
                        0, active_rows
                    ),
                )
                action_probabilities = controller_output["probabilities"]
                raw_action_probabilities = controller_output[
                    "raw_probabilities"
                ]

            with torch.set_grad_enabled(adapt_working_memory):
                final_params = self._controller_effective_parameters(
                    wave_memory,
                    working_used,
                    action_probabilities[:, 1],
                )
                final_nll = self._batched_event_nll(
                    final_params.theta,
                    active_history,
                    active_interval,
                    active_types,
                    active_durations,
                )

            if adapt_working_memory:
                working_gradient = torch.autograd.grad(
                    final_nll.sum(),
                    working_used,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                self.tree.working_memory.update_batch_rows(
                    working_state,
                    sequence_rows,
                    working_gradient,
                    adaptation_probability=action_probabilities[:, 0],
                )

            with torch.no_grad():
                active_forecast = forecast_flat.index_select(0, active_rows)
                current_intensity = self._batched_hawkes_intensity(
                    final_params.theta,
                    active_history,
                )
                forecast_intensity = self._batched_hawkes_intensity(
                    final_params.theta,
                    active_forecast,
                )
                type_probabilities = current_intensity / current_intensity.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
                forecast_rate = forecast_intensity.sum(dim=-1).clamp_min(1e-8)
                forecast_type_probabilities = (
                    forecast_intensity / forecast_rate.unsqueeze(-1)
                )
                predicted_delta = forecast_rate.reciprocal()
                active_origins = forecast_origin_flat.index_select(
                    0, active_rows
                )
                predicted_time = active_origins + predicted_delta
                predicted_type = current_intensity.argmax(dim=-1)

                nll_by_sequence.index_add_(
                    0,
                    sequence_rows,
                    final_nll.detach(),
                )
                correct_by_sequence.index_add_(
                    0,
                    sequence_rows,
                    predicted_type.eq(active_types).long(),
                )
                time_abs_by_sequence.index_add_(
                    0,
                    sequence_rows,
                    (predicted_time - times_flat.index_select(0, active_rows))
                    .abs(),
                )

            if not need_events:
                continue

            active_global = active_rows.detach().cpu().tolist()
            nll_cpu = final_nll.detach().cpu().tolist()
            type_probabilities_cpu = type_probabilities.detach().cpu()
            forecast_probabilities_cpu = (
                forecast_type_probabilities.detach().cpu()
            )
            predicted_delta_cpu = predicted_delta.detach().cpu().tolist()
            predicted_time_cpu = predicted_time.detach().cpu().tolist()
            active_types_cpu = active_types.detach().cpu().tolist()
            predicted_type_cpu = predicted_type.detach().cpu().tolist()
            action_probabilities_cpu = action_probabilities.detach().cpu()
            raw_action_probabilities_cpu = (
                raw_action_probabilities.detach().cpu()
            )
            pre_theta_cpu = pre_params.theta.detach().cpu()
            current_intensity_cpu = current_intensity.detach().cpu()
            for local_index, global_index in enumerate(active_global):
                sequence_row = int(sequence_index_cpu[global_index])
                if not (
                    capture_prediction_theta
                    or capture_by_sequence[sequence_row]
                ):
                    continue
                event_index = int(time_index_cpu[global_index])
                source_index = source_index_by_sequence[sequence_row]
                action_index = int(
                    action_probabilities_cpu[local_index].argmax().item()
                )
                action = str(tuple(Action)[action_index])
                retrieve_gate = float(
                    action_probabilities_cpu[local_index, 1]
                )
                raw_write_gate = float(
                    raw_action_probabilities_cpu[local_index, 2]
                )
                action_write_gate = float(
                    action_probabilities_cpu[local_index, 2]
                )
                event_results[global_index] = {
                    "event_index": event_index,
                    "nll": float(nll_cpu[local_index]),
                    "predicted_type": int(predicted_type_cpu[local_index]),
                    "true_type": int(active_types_cpu[local_index]),
                    "prediction_theta": pre_theta_cpu[local_index],
                    "intensity": current_intensity_cpu[local_index],
                    "type_probabilities_at_event_time": type_probabilities_cpu[
                        local_index
                    ],
                    "forecast_type_probabilities": forecast_probabilities_cpu[
                        local_index
                    ],
                    "predicted_delta": float(predicted_delta_cpu[local_index]),
                    "predicted_time": float(predicted_time_cpu[local_index]),
                    "true_time": float(
                        times_flat_cpu[global_index]
                    ),
                    "responsibility": responsibility_cpu[global_index],
                    "owner_id": owner_ids[global_index],
                    "frontier_node_ids": frontier_ids[global_index],
                    "frontier_posterior": posterior_cpu[global_index],
                    "action": action,
                    "memorize_argmax": action_index == 2,
                    "action_probabilities": action_probabilities_cpu[
                        local_index
                    ],
                    "raw_action_probabilities": raw_action_probabilities_cpu[
                        local_index
                    ],
                    "retrieval_alpha_mass": float(
                        alpha_mass_cpu[global_index]
                    ),
                    "retrieval_alpha_per_visited_node": float(
                        alpha_mass_cpu[global_index]
                        / max(int(visited_count_cpu[global_index]), 1)
                    ),
                    "retrieval_similarity": float(
                        weighted_similarity_cpu[global_index]
                    ),
                    "retrieval_effective_k": int(
                        effective_k_cpu[global_index]
                    ),
                    "retrieval_null_alpha": float(
                        null_alpha_cpu[global_index]
                    ),
                    "visited_bank_count": int(visited_count_cpu[global_index]),
                    "visited_nonempty_bank_count": int(
                        visited_nonempty_cpu[global_index]
                    ),
                    "raw_episodic_residual_norm": float(
                        raw_residual_cpu[global_index]
                    ),
                    "retrieve_gate": retrieve_gate,
                    "gated_episodic_residual_norm": float(
                        raw_residual_cpu[global_index]
                    ) * abs(retrieve_gate),
                    "owner_on_retrieval_path": bool(
                        owner_on_path_cpu[global_index]
                    ),
                    "retrieval_counterfactual_gain": None,
                    "retrieval_counterfactual_unavailable_reason": (
                        "requires paired no_episodic evaluation"
                    ),
                    "episodic_residual_norm": float(
                        episodic_residual_cpu[global_index]
                    ),
                    "write_token": f"{source_index}:{event_index}",
                    "write_candidate": bool(
                        raw_write_gate >= self.controller.write_candidate_threshold
                    ),
                    "write_gate_active": bool(action_write_gate > 0.0),
                    "write_probed": False,
                    "write_utility": None,
                    "write_priority": None,
                    "write_accepted": False,
                    "write_local_accepted": False,
                    "write_probation_enqueued": False,
                    "write_promotion_count": 0,
                    "write_retrieved_later": False,
                    "write_beneficial": False,
                    "write_gate_passed": bool(
                        raw_write_gate >= calibrated_write_threshold
                    ),
                    "write_priority_passed": False,
                    "write_window_complete": False,
                    "write_utility_passed": False,
                    "write_owner_on_score_path": False,
                    "write_virtual_candidate_alpha": None,
                }

        results: list[dict[str, Any]] = []
        offset = 0
        for row, length in enumerate(lengths):
            capture_row = bool(
                capture_prediction_theta or capture_by_sequence[row]
            )
            if capture_row:
                row_events = [
                    event_results[index]
                    for index in range(offset, offset + length)
                ]
                if any(event is None for event in row_events):
                    raise RuntimeError(
                        "packed inference did not materialize all event rows"
                    )
                materialized_events = [event for event in row_events if event is not None]
            else:
                materialized_events = []
            total_nll = float(nll_by_sequence[row].detach().cpu())
            results.append({
                "events": materialized_events,
                "total_nll": total_nll,
                "nll_per_event": total_nll / max(length, 1),
                "pending_write_count": 0,
                "accepted_write_count": 0,
                "local_accepted_write_count": 0,
                "probation_validation_count": 0,
                "probation_validations": [],
                "promoted_write_count": 0,
                "promotions": [],
                "probation_size": 0,
                "write_probe_count": 0,
                "leaf_ids": list(self.tree.leaf_ids),
                "scalar_metrics": {
                    "events": length,
                    "nll_sum": total_nll,
                    "correct": int(correct_by_sequence[row].detach().cpu()),
                    "time_abs_sum": float(
                        time_abs_by_sequence[row].detach().cpu()
                    ),
                },
            })
            offset += length
        return results

    def _action(
        self,
        memory_output: Mapping[str, Any],
        nll: Tensor,
        frontier_energy: Tensor,
    ) -> tuple[Action, str, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mask = memory_output["frontier_mask"][0]
        prior = memory_output["frontier_mass"][0]
        temperature = (
            self.tree.frontier_routing.config.posterior_temperature
        )
        posterior = torch.softmax(
            (
                prior.clamp_min(1e-12).log()
                - frontier_energy / temperature
            ).masked_fill(~mask, -torch.inf),
            dim=-1,
        ).masked_fill(~mask, 0.0)
        frontier_ids = memory_output["frontier_node_ids"][0]
        owner_id = self._posterior_owner(frontier_ids, posterior)
        query = memory_output["memory_query"][0].detach()
        novelty, count, retrieval_similarity = (
            self.controller.leaf_novelty_count(
            query, owner_id
            )
        )
        controller_output = self.controller.action_distribution(
            surprise=nll.detach(),
            novelty=novelty,
            count=count,
            update_statistics=False,
            owner_confidence=posterior.max(),
            retrieval_similarity=retrieval_similarity,
            retrieval_residual_norm=memory_output[
                "frontier_episodic_delta"
            ][0].norm(dim=-1).mean(),
        )
        probabilities = controller_output["probabilities"]
        raw_probabilities = controller_output.get("raw_probabilities", probabilities)
        action = tuple(Action)[
            int(probabilities.detach().argmax().item())
        ]
        return (
            action,
            owner_id,
            query,
            probabilities,
            raw_probabilities,
            posterior,
            novelty.detach(),
            retrieval_similarity.detach(),
        )

    def _lowest_common_ancestor(self, node_ids: Sequence[str]) -> str:
        paths = [self.tree.path_to_node(node_id) for node_id in node_ids]
        common = "root"
        for values in zip(*paths):
            if len(set(values)) != 1:
                break
            common = values[0]
        return common

    def _posterior_owner(
        self,
        frontier_ids: Sequence[str],
        posterior: Tensor,
    ) -> str:
        config = self.tree.frontier_routing.config
        order = posterior[: len(frontier_ids)].argsort(
            descending=True,
            stable=True,
        )
        cumulative = posterior.index_select(0, order).cumsum(dim=0)
        count = min(
            int((cumulative < config.credible_mass).sum().item()) + 1,
            len(frontier_ids),
        )
        top = int(order[0].item())
        if (
            count == 1
            and float(posterior[top]) >= config.owner_confidence_threshold
        ):
            return frontier_ids[top]
        return self._lowest_common_ancestor([
            frontier_ids[index]
            for index in order[:count].cpu().tolist()
        ])

    def _frontier_event_energy(
        self,
        sequence: Mapping[str, Tensor],
        memory_output: Mapping[str, Any],
        event_index: int,
    ) -> Tensor:
        theta = memory_output["frontier_theta"][0]
        D = self.hawkes.num_types
        energies = []
        for row in theta:
            params = HawkesParams(
                row[:D],
                row[D:].reshape(D, D, self.hawkes.num_basis),
            )
            energies.append(
                self.hawkes.event_NLL(sequence, params, event_index)
            )
        result = torch.stack(energies)
        return result.masked_fill(
            ~memory_output["frontier_mask"][0], torch.inf
        )

    def _delayed_write_evidence(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if int(self.controller.controller_version.detach().cpu()) >= 6:
            return self._delayed_write_evidence_v6(sequence, request)
        D = self.hawkes.num_types
        start = int(request["event_index"])
        end = min(
            int(sequence["times"].numel()),
            start + self.wake_config.write_horizon + 1,
        )
        energy = []
        for theta in request["frontier_theta"]:
            params = HawkesParams(
                theta[:D],
                theta[D:].reshape(D, D, self.hawkes.num_basis),
            )
            value = theta.new_zeros(())
            for event_index in range(start, end):
                value += self.hawkes.event_NLL(
                    sequence, params, event_index
                )
            energy.append(value / max(end - start, 1))
        energy_tensor = torch.stack(energy)
        prior = request["frontier_mass"].clamp_min(1e-12)
        temperature = (
            self.tree.frontier_routing.config.posterior_temperature
        )
        posterior = torch.softmax(
            prior.log() - energy_tensor / temperature,
            dim=-1,
        )
        owner_id = self._posterior_owner(
            request["frontier_node_ids"], posterior
        )
        mixture_energy = -torch.logsumexp(
            prior.log() - energy_tensor / temperature,
            dim=0,
        )
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            theta_owner[:D].clone(),
            theta_owner[D:].reshape(D, D, self.hawkes.num_basis).clone(),
        )
        candidate_item = self.controller.write_residual_memory(
            q_t=request["query"],
            theta_sem_leaf=owner_params,
            times=sequence["times"],
            types=sequence["types"],
            k=start,
            node_id=owner_id,
            cached_sequence=sequence,
            write_quality=request["write_gate"],
            queue_weight=request["queue_weight"],
        )
        delta = candidate_item.delta_theta
        if int(self.controller.controller_version.detach().cpu()) < 5:
            before = energy_tensor.new_zeros(())
            after = energy_tensor.new_zeros(())
            candidate_theta = theta_owner + delta
            candidate_params = HawkesParams(
                candidate_theta[:D],
                candidate_theta[D:].reshape(D, D, self.hawkes.num_basis),
            )
            for future_index in range(start, end):
                before += self.hawkes.event_NLL(sequence, owner_params, future_index)
                after += self.hawkes.event_NLL(sequence, candidate_params, future_index)
            raw_write_gain = before - after
            write_utility = raw_write_gain / max(end - start, 1) - self.wake_config.lambda_write
            write_gain = raw_write_gain.clamp_min(0.0)
            bounded_gain = -torch.expm1(
                -write_gain / self.wake_config.controller_gain_reference
            )
            return {
                "owner_id": owner_id,
                "posterior": posterior,
                "write_gain": write_gain,
                "write_utility": write_utility,
                "owner_on_score_path": True,
                "virtual_candidate_alpha": 0.0,
                "bounded_gain": bounded_gain,
                "candidate_item": candidate_item,
                "priority": (
                    request["write_gate"] * posterior.max()
                    * write_utility.clamp_min(0.0)
                    * request["novelty"].clamp(0.0, 1.0)
                ),
            }
        score_start = start + self.wake_config.write_horizon + 1
        score_end = start + 2 * self.wake_config.write_horizon + 1
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("Write probe score window is incomplete")
        before = energy_tensor.new_zeros(())
        after = energy_tensor.new_zeros(())
        virtual_usage = 1.0
        virtual_alpha = []
        owner_on_path = False
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            base_delta, _ = self.tree.episodic_memory.read_nodes(
                query, [owner_id], update_state=False
            )
            virtual_delta, virtual_info = self.tree.episodic_memory.read_node_with_virtual_item(
                query, owner_id, key=candidate_item.key, delta=delta,
                write_quality=request["write_gate"], virtual_usage=virtual_usage,
                virtual_age=float(age),
            )
            candidate_alpha = float(virtual_info["alpha"][-1].detach().cpu())
            virtual_alpha.append(candidate_alpha)
            virtual_usage += candidate_alpha
            episodic = context["frontier_episodic_delta"].clone()
            difference = virtual_delta - base_delta[owner_id]
            for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                if owner_id in self.tree.path_to_node(frontier_id):
                    episodic[slot] = episodic[slot] + difference
                    owner_on_path = True
            virtual_output = {
                "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                "frontier_episodic_delta": episodic.unsqueeze(0),
                "r": context["posterior"].unsqueeze(0),
            }
            with_params = self._controller_effective_parameters(
                virtual_output, context["working_delta"], context["retrieve_gate"]
            ).select(0)
            theta = context["no_write_theta"]
            without_params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            event_index = int(context["event_index"])
            before += self.hawkes.event_NLL(sequence, without_params, event_index)
            after += self.hawkes.event_NLL(sequence, with_params, event_index)
        raw_write_gain = before - after
        if not owner_on_path:
            raw_write_gain = raw_write_gain * 0.0
        write_utility = (
            raw_write_gain / self.wake_config.write_horizon
            - self.wake_config.lambda_write
        )
        write_gain = raw_write_gain.clamp_min(0.0)
        bounded_gain = -torch.expm1(
            -write_gain / self.wake_config.controller_gain_reference
        )
        priority = (
            request["write_gate"]
            * posterior.max()
            * write_utility.clamp_min(0.0)
            * request["novelty"].clamp(0.0, 1.0)
        )
        return {
            "owner_id": owner_id,
            "posterior": posterior,
            "write_gain": write_gain,
            "write_utility": write_utility,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": sum(virtual_alpha) / max(len(virtual_alpha), 1),
            "bounded_gain": bounded_gain,
            "candidate_item": candidate_item,
            "priority": priority,
        }

    def _causal_write_evidence_v6(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build a candidate from exactly F=[t,t+h), without future labels."""
        D = self.hawkes.num_types
        h = self.wake_config.write_horizon
        start = int(request["event_index"])
        build_end = start + h
        if build_end > int(sequence["times"].numel()):
            raise RuntimeError("v6 construction window is incomplete")
        energy = []
        for theta in request["frontier_theta"]:
            params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            value = theta.new_zeros(())
            for event_index in range(start, build_end):
                value = value + self.hawkes.event_NLL(sequence, params, event_index)
            energy.append(value / h)
        energy_tensor = torch.stack(energy)
        prior = request["frontier_mass"].clamp_min(1e-12)
        posterior = torch.softmax(
            prior.log() - energy_tensor / self.tree.frontier_routing.config.posterior_temperature,
            dim=-1,
        )
        owner_id = self._posterior_owner(request["frontier_node_ids"], posterior)
        theta_owner = self.tree.semantic_theta(owner_id).detach()
        owner_params = HawkesParams(
            theta_owner[:D].clone(),
            theta_owner[D:].reshape(D, D, self.hawkes.num_basis).clone(),
        )
        candidate_item = self.controller.write_residual_memory(
            q_t=request["query"], theta_sem_leaf=owner_params,
            times=sequence["times"], types=sequence["types"], k=start,
            node_id=owner_id, cached_sequence=sequence,
            write_quality=request["write_gate"], queue_weight=request["queue_weight"],
            window_events=h,
        )
        threshold = self.controller.calibration_thresholds[2].to(request["write_gate"])
        priority = (
            (request["write_gate"] - threshold).clamp_min(0.0)
            * posterior.max() * request["novelty"].clamp(0.0, 1.0)
        )
        return {
            "owner_id": owner_id,
            "posterior": posterior,
            "candidate_item": candidate_item,
            "priority": priority,
            "bounded_gain": request["write_gate"].clamp(0.0, 1.0),
            "construction_window": [start, build_end],
        }

    def _delayed_write_evidence_v6(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Score v6 candidate on C=[t+h,t+2h), after causal admission."""
        base = dict(request.get("admission_evidence") or self._causal_write_evidence_v6(sequence, request))
        D = self.hawkes.num_types
        h = self.wake_config.write_horizon
        start = int(request["event_index"])
        score_start, score_end = start + h, start + 2 * h
        contexts = request.get("future_contexts") or []
        if score_end > len(contexts):
            raise RuntimeError("v6 score window C is incomplete")
        owner_id = base["owner_id"]
        item = base["candidate_item"]
        before = request["frontier_theta"].new_zeros(())
        after = before.clone()
        virtual_usage = 1.0
        virtual_alpha: list[float] = []
        owner_on_path = False
        committed = bool(request.get("committed", False))
        for age, context in enumerate(contexts[score_start:score_end]):
            query = context["query"]
            stored_episodic = context["frontier_episodic_delta"].clone()
            if committed:
                with_delta, with_info = self.tree.episodic_memory.read_nodes(
                    query, [owner_id], update_state=False
                )
                without_delta, without_info = self.tree.episodic_memory.read_node_without_item(
                    query, owner_id, key=item.key, delta=item.delta_theta
                )
                difference = with_delta[owner_id] - without_delta
                excluded = without_info.get("excluded_index")
                alpha = 0.0
                if excluded is not None and with_info[owner_id].get("alpha") is not None:
                    index = int(excluded.detach().cpu())
                    values = with_info[owner_id]["alpha"]
                    if 0 <= index < values.numel():
                        alpha = float(values[index].detach().cpu())
                virtual_alpha.append(alpha)
                baseline_episodic = stored_episodic.clone()
                for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                    if owner_id in self.tree.path_to_node(frontier_id):
                        baseline_episodic[slot] = baseline_episodic[slot] - difference
                        owner_on_path = True
                baseline_output = {
                    "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                    "frontier_episodic_delta": baseline_episodic.unsqueeze(0),
                    "r": context["posterior"].unsqueeze(0),
                }
                before_params = self._controller_effective_parameters(
                    baseline_output, context["working_delta"], context["retrieve_gate"]
                ).select(0)
                theta = context["no_write_theta"]
                after_params = HawkesParams(
                    theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
                )
            else:
                # Training/read-only probes use the production add_memory and
                # sparse read path on an isolated bank snapshot.  This avoids
                # a second, approximate "virtual retriever" implementation.
                memory = self.tree.episodic_memory
                if age == 0:
                    baseline_memory = copy.deepcopy(memory)
                    treatment_memory = copy.deepcopy(memory)
                    base_by_age = []
                    original_clock = baseline_memory._age_clock
                    for base_age, base_context in enumerate(
                        contexts[score_start:score_end]
                    ):
                        baseline_memory._age_clock = original_clock + base_age
                        values, _ = baseline_memory.read_nodes(
                            base_context["query"], [owner_id], update_state=False
                        )
                        base_by_age.append(values[owner_id].detach().clone())
                    treatment_memory.add_memory(
                        owner_id, item.key, item.delta_theta,
                        write_quality=base["bounded_gain"],
                        queue_weight=request["queue_weight"],
                        semantic_theta=self.tree.semantic_theta(owner_id).detach(),
                        decays=self.hawkes.decays.detach(),
                        force_new_mode_confirmation=True,
                    )
                    request["_physical_probe_memory"] = treatment_memory
                    request["_physical_probe_clock"] = original_clock
                    request["_physical_probe_base_delta"] = base_by_age
                probe_memory = request["_physical_probe_memory"]
                probe_memory._age_clock = request["_physical_probe_clock"] + age
                with_delta, info_by_node = probe_memory.read_nodes(
                    query, [owner_id], update_state=False
                )
                info = info_by_node[owner_id]
                alpha_values = info.get("alpha")
                alpha = 0.0
                if alpha_values is not None and alpha_values.numel():
                    alpha = float(alpha_values[-1].detach().cpu())
                virtual_alpha.append(alpha)
                difference = (
                    with_delta[owner_id]
                    - request["_physical_probe_base_delta"][age]
                )
                virtual_episodic = stored_episodic.clone()
                for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                    if owner_id in self.tree.path_to_node(frontier_id):
                        virtual_episodic[slot] = virtual_episodic[slot] + difference
                        owner_on_path = True
                virtual_output = {
                    "frontier_semantic_theta": context["frontier_semantic_theta"].unsqueeze(0),
                    "frontier_episodic_delta": virtual_episodic.unsqueeze(0),
                    "r": context["posterior"].unsqueeze(0),
                }
                after_params = self._controller_effective_parameters(
                    virtual_output, context["working_delta"], context["retrieve_gate"]
                ).select(0)
                theta = context["no_write_theta"]
                before_params = HawkesParams(
                    theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
                )
            event_index = int(context["event_index"])
            before = before + self.hawkes.event_NLL(sequence, before_params, event_index)
            after = after + self.hawkes.event_NLL(sequence, after_params, event_index)
        if not committed and "_physical_probe_memory" in request:
            request.pop("_physical_probe_memory")
            request.pop("_physical_probe_clock", None)
            request.pop("_physical_probe_base_delta", None)
        raw_gain = before - after
        if not owner_on_path:
            raw_gain = raw_gain * 0.0
        utility = raw_gain / h - self.wake_config.lambda_write
        return {
            **base,
            "write_gain": raw_gain,
            "write_utility": utility,
            "owner_on_score_path": owner_on_path,
            "virtual_candidate_alpha": sum(virtual_alpha) / max(len(virtual_alpha), 1),
            "score_window": [score_start, score_end],
        }

    def _commit_delayed_write(
        self,
        sequence: Mapping[str, Tensor],
        request: Mapping[str, Any],
    ) -> bool:
        persistent = self._commit_delayed_writes_batch(sequence, [request])
        return bool(persistent[0]) if persistent else False

    def _commit_delayed_writes_batch(
        self,
        sequence: Mapping[str, Tensor],
        requests: Sequence[Mapping[str, Any]],
    ) -> list[bool]:
        """Commit ready delayed writes and report persistent admissions.

        A ``queue`` result is a pending-law candidate, not a committed
        physical row.  Returning one flag per request prevents the caller
        from marking queued candidates as accepted writes.
        """
        if not requests:
            return []
        records = []
        for request in requests:
            evidence = request.get("window_evidence")
            if evidence is None:
                evidence = self._delayed_write_evidence(sequence, request)
            owner_id = evidence["owner_id"]
            item = evidence.get("candidate_item")
            if item is None:
                raise RuntimeError("write evidence is missing its residual candidate")
            item.write_quality = float(evidence["bounded_gain"].detach().cpu())
            item.queue_weight = float(request["queue_weight"])
            item.prediction_gain = float(evidence["write_gain"].detach().cpu())
            records.append((owner_id, item))

        grouped_items = {}
        for owner_id, item in records:
            grouped_items.setdefault(owner_id, []).append(item)
        persistent_by_item: dict[int, bool] = {}
        for owner_id, owner_items in grouped_items.items():
            reference = owner_items[0].key
            admission = self.tree.episodic_memory.add_memory_batch(
                owner_id,
                torch.stack([
                    item.key.reshape(-1).to(
                        device=reference.device, dtype=reference.dtype
                    )
                    for item in owner_items
                ]),
                torch.stack([
                    item.delta_theta.reshape(-1).to(device=reference.device)
                    for item in owner_items
                ]),
                windows=[item.window for item in owner_items],
                write_quality=torch.as_tensor(
                    [item.write_quality for item in owner_items],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                queue_weight=torch.as_tensor(
                    [item.queue_weight for item in owner_items],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                prediction_gain=torch.as_tensor(
                    [item.prediction_gain for item in owner_items],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                semantic_theta=self.tree.semantic_theta(owner_id).detach(),
                decays=self.hawkes.decays.detach(),
            )
            for item, result in zip(owner_items, admission):
                persistent = result["action"] != "queue"
                persistent_by_item[id(item)] = persistent
                if persistent:
                    self.controller.split_queues[owner_id] += item.queue_weight
        return [persistent_by_item.get(id(item), False) for _, item in records]

    def _enqueue_probation_candidate(
        self,
        request: Mapping[str, Any],
        *,
        source_sequence_id: Any,
    ) -> ProbationCandidate:
        """Store local evidence without exposing the residual to retrieval."""
        evidence = request.get("window_evidence")
        if evidence is None:
            raise RuntimeError("probation candidate requires local window evidence")
        item = evidence.get("candidate_item")
        if item is None:
            raise RuntimeError("write evidence is missing its residual candidate")
        local_utility = float(evidence["write_utility"].detach().cpu())
        gain_reference = self.wake_config.controller_gain_reference
        local_quality = 1.0 - torch.exp(torch.tensor(
            -max(local_utility, 0.0) / gain_reference
        )).item()
        probabilities = request.get("raw_action_probabilities")
        write_probability = float(request["write_gate"].detach().cpu())
        split_probability = (
            float(probabilities[3].detach().cpu())
            if probabilities is not None and probabilities.numel() > 3
            else 0.0
        )
        token = (source_sequence_id, int(request["event_index"]))
        candidate = ProbationCandidate(
            token=token,
            source_sequence_id=source_sequence_id,
            event_index=int(request["event_index"]),
            owner_id=str(evidence["owner_id"]),
            key=item.key.detach().clone(),
            delta_theta=item.delta_theta.detach().clone(),
            window=item.window,
            local_utility=local_utility,
            local_quality=float(local_quality),
            write_probability=write_probability,
            split_probability=split_probability,
            # Probation preserves the pure structural vote.  The legacy
            # request queue_weight is p_write * p_split and must not dilute
            # q_struct once cross-sequence persistence has been established.
            queue_weight=split_probability,
        )
        self.write_probation.add(candidate)
        return candidate

    def _candidate_match(
        self,
        candidate: ProbationCandidate,
        contexts: Sequence[Mapping[str, Any]],
    ) -> Optional[tuple[int, float]]:
        """Return the best topology-compatible context with a full CF window."""
        horizon = self.wake_config.write_horizon
        best: Optional[tuple[int, float]] = None
        for index in range(max(len(contexts) - horizon + 1, 0)):
            context = contexts[index]
            compatible = any(
                candidate.owner_id in self.tree.path_to_node(frontier_id)
                for frontier_id in context["frontier_node_ids"]
            )
            if not compatible:
                continue
            query = context["query"]
            similarity = float(F.cosine_similarity(
                candidate.key.to(query).reshape(1, -1),
                query.reshape(1, -1),
                dim=-1,
            )[0].detach().cpu())
            if best is None or similarity > best[1]:
                best = (index, similarity)
        return best

    def _cross_sequence_gain(
        self,
        sequence: Mapping[str, Tensor],
        contexts: Sequence[Mapping[str, Any]],
        candidate: ProbationCandidate,
        match_index: int,
    ) -> tuple[float, float]:
        """Evaluate a probation residual through the production read path."""
        horizon = self.wake_config.write_horizon
        D = self.hawkes.num_types
        baseline_loss = candidate.key.new_zeros(())
        counterfactual_loss = baseline_loss.clone()
        virtual_usage = 1.0
        alpha_values: list[float] = []
        for age, context in enumerate(
            contexts[match_index:match_index + horizon]
        ):
            query = context["query"]
            baseline_delta, _ = self.tree.episodic_memory.read_nodes(
                query, [candidate.owner_id], update_state=False
            )
            virtual_delta, info = (
                self.tree.episodic_memory.read_node_with_virtual_item(
                    query,
                    candidate.owner_id,
                    key=candidate.key,
                    delta=candidate.delta_theta,
                    write_quality=candidate.local_quality,
                    virtual_usage=virtual_usage,
                    virtual_age=float(age),
                )
            )
            alpha = 0.0
            values = info.get("alpha")
            if values is not None and values.numel():
                alpha = float(values[-1].detach().cpu())
            alpha_values.append(alpha)
            virtual_usage += alpha

            difference = virtual_delta - baseline_delta[candidate.owner_id]
            episodic = context["frontier_episodic_delta"].clone()
            for slot, frontier_id in enumerate(context["frontier_node_ids"]):
                if candidate.owner_id in self.tree.path_to_node(frontier_id):
                    episodic[slot] = episodic[slot] + difference
            virtual_output = {
                "frontier_semantic_theta": (
                    context["frontier_semantic_theta"].unsqueeze(0)
                ),
                "frontier_episodic_delta": episodic.unsqueeze(0),
                "r": context["posterior"].unsqueeze(0),
            }
            counterfactual_params = self._controller_effective_parameters(
                virtual_output,
                context["working_delta"],
                context["retrieve_gate"],
            ).select(0)
            theta = context["no_write_theta"]
            baseline_params = HawkesParams(
                theta[:D], theta[D:].reshape(D, D, self.hawkes.num_basis)
            )
            event_index = int(context["event_index"])
            baseline_loss = baseline_loss + self.hawkes.event_NLL(
                sequence, baseline_params, event_index
            )
            counterfactual_loss = counterfactual_loss + self.hawkes.event_NLL(
                sequence, counterfactual_params, event_index
            )
        gain = float(
            ((baseline_loss - counterfactual_loss) / horizon).detach().cpu()
        )
        mean_alpha = sum(alpha_values) / max(len(alpha_values), 1)
        return gain, mean_alpha

    def _validate_probation_candidates(
        self,
        sequence: Mapping[str, Tensor],
        contexts: Sequence[Mapping[str, Any]],
        *,
        sequence_id: Any,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        """Validate once per independent sequence, then promote by ESS + LCB."""
        validations: list[Dict[str, Any]] = []
        for candidate in self.write_probation.candidates_for(sequence_id):
            match = self._candidate_match(candidate, contexts)
            # Even an unmatched sequence is consumed: rerunning it must not
            # fabricate another independent opportunity for this candidate.
            if match is None:
                candidate.record_validation(
                    sequence_id, gain=0.0, weight=0.0
                )
                validations.append({
                    "token": candidate.token,
                    "matched": False,
                })
                continue
            match_index, similarity = match
            gain, virtual_alpha = self._cross_sequence_gain(
                sequence, contexts, candidate, match_index
            )
            weight = float(torch.sigmoid(torch.tensor(
                (
                    similarity - self.config.probation_match_threshold
                ) / self.config.probation_match_temperature
            )).item())
            candidate.record_validation(
                sequence_id, gain=gain, weight=weight
            )
            validations.append({
                "token": candidate.token,
                "matched": True,
                "match_index": match_index,
                "similarity": similarity,
                "weight": weight,
                "gain": gain,
                "virtual_candidate_alpha": virtual_alpha,
                "effective_sample_size": candidate.effective_sample_size,
                "gain_mean": candidate.gain_mean,
                "gain_variance": candidate.gain_variance,
                "lcb": candidate.lower_confidence_bound(
                    self.config.probation_lcb_kappa
                ),
            })

        promotions: list[Dict[str, Any]] = []
        promotable: list[tuple[ProbationCandidate, float]] = []
        for candidate in list(self.write_probation):
            if not candidate.promotion_ready(
                minimum_effective_samples=(
                    self.config.probation_min_effective_samples
                ),
                kappa=self.config.probation_lcb_kappa,
                persist_threshold=self.config.probation_persist_threshold,
            ):
                continue
            quality = candidate.promoted_quality(
                self.wake_config.controller_gain_reference
            )
            promotable.append((candidate, quality))

        # Promotion is the other persistent-write path outside the Wake
        # batch. Candidates remain in insertion order inside each owner group;
        # grouping only removes repeated law-key construction and admission
        # synchronizations, while the metadata report below retains the
        # original global order.
        grouped_promotions = {}
        admission_by_token: Dict[Any, Dict[str, Any]] = {}
        for candidate, quality in promotable:
            grouped_promotions.setdefault(candidate.owner_id, []).append(
                (candidate, quality)
            )
        for owner_id, owner_candidates in grouped_promotions.items():
            reference = owner_candidates[0][0].key
            admission = self.tree.episodic_memory.add_memory_batch(
                owner_id,
                torch.stack([
                    candidate.key.reshape(-1).to(
                        device=reference.device, dtype=reference.dtype
                    )
                    for candidate, _ in owner_candidates
                ]),
                torch.stack([
                    candidate.delta_theta.reshape(-1).to(
                        device=reference.device
                    )
                    for candidate, _ in owner_candidates
                ]),
                windows=[candidate.window for candidate, _ in owner_candidates],
                write_quality=torch.as_tensor(
                    [quality for _, quality in owner_candidates],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                queue_weight=torch.as_tensor(
                    [candidate.queue_weight for candidate, _ in owner_candidates],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                prediction_gain=torch.as_tensor(
                    [candidate.gain_mean for candidate, _ in owner_candidates],
                    device=reference.device,
                    dtype=reference.dtype,
                ),
                semantic_theta=self.tree.semantic_theta(owner_id).detach(),
                decays=self.hawkes.decays.detach(),
            )
            for (candidate, _), result in zip(owner_candidates, admission):
                admission_by_token[candidate.token] = result

        for candidate, quality in promotable:
            admission = admission_by_token[candidate.token]
            if admission["action"] == "queue":
                # Keep the independently validated candidate in probation;
                # a later recurrence supplies the persistence confirmation.
                continue
            # This is the sole probation-to-Split bridge. Before promotion the
            # candidate is absent from both the bank and the trigger queue.
            self.controller.split_queues[
                candidate.owner_id
            ] += candidate.queue_weight
            promotions.append({
                "token": candidate.token,
                "owner_id": candidate.owner_id,
                "write_quality": quality,
                "structural_weight": candidate.queue_weight,
                "gain_mean": candidate.gain_mean,
                "effective_sample_size": candidate.effective_sample_size,
                "lcb": candidate.lower_confidence_bound(
                    self.config.probation_lcb_kappa
                ),
            })
            self.write_probation.remove(candidate.token)
        return validations, promotions

    def run_sequence(
        self,
        cpu_sequence: Mapping[str, Any],
        *,
        precomputed_z: Optional[Tensor] = None,
        frontier_static_cache: Any = None,
        precomputed_projected_z: Optional[Tensor] = None,
        precomputed_memory_query: Optional[Tensor] = None,
        compact: bool = False,
        capture_event_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Process observed events causally using only the wake mechanism.

        Local candidates wait for their complete causal write horizon, then
        remain retrieval-invisible until independent sequences pass ESS + LCB.
        """
        # The ordinary API keeps its historical event-row contract.  Compact
        # callers can explicitly request those rows for diagnostics; otherwise
        # keep only the tiny mutable write state needed by the causal wake
        # logic and return scalar accumulators below.
        materialize_events = (not compact) or bool(capture_event_predictions)
        source_value = cpu_sequence.get("source_index", -1)
        source_index = int(
            source_value.item() if hasattr(source_value, "item") else source_value
        )
        if source_index >= 0:
            source_sequence_id: Any = source_index
        else:
            source_sequence_id = (
                "anonymous_inference_sequence",
                self._anonymous_sequence_counter,
            )
            self._anonymous_sequence_counter += 1
        sequence = self._move_sequence(cpu_sequence)
        event_count = int(sequence["times"].numel())
        if precomputed_z is not None:
            precomputed_z = precomputed_z.to(self.device)
            if (
                precomputed_z.ndim != 2
                or precomputed_z.shape != (event_count, self.tree.z_dim)
            ):
                raise ValueError(
                    "precomputed_z must have shape "
                    f"[{event_count}, {self.tree.z_dim}]"
                )
        if precomputed_projected_z is not None:
            precomputed_projected_z = precomputed_projected_z.to(self.device)
            if (
                precomputed_projected_z.ndim != 2
                or precomputed_projected_z.size(0) != event_count
            ):
                raise ValueError(
                    "precomputed_projected_z must have one row per event"
                )
        if precomputed_memory_query is not None:
            precomputed_memory_query = precomputed_memory_query.to(self.device)
            if (
                precomputed_memory_query.ndim != 2
                or precomputed_memory_query.size(0) != event_count
            ):
                raise ValueError(
                    "precomputed_memory_query must have one row per event"
                )
        self.tree.reset_working_memory()
        pending_writes: list[Dict[str, Any]] = []
        write_probe_contexts: list[Dict[str, Any]] = []
        outputs: list[Dict[str, Any]] = []
        total_nll = 0.0
        scalar_correct = 0
        scalar_time_abs_sum = 0.0
        accepted_write_count = 0
        local_accepted_write_count = 0
        accepted_write_requests: list[Dict[str, Any]] = []

        for event_index in range(sequence["times"].numel()):
            with torch.no_grad():
                if precomputed_z is not None:
                    z_t = precomputed_z[event_index].reshape(1, -1)
                elif isinstance(self.encoder, CausalPrefixEncoder):
                    z_t = self.encoder(
                        sequence["times"],
                        sequence["types"],
                        event_index,
                        time_features=sequence.get(
                            EVENT_TIME_FEATURES_KEY
                        ),
                    ).reshape(1, -1)
                else:
                    z_t = self.encoder(
                        sequence["times"],
                        sequence["types"],
                        event_index,
                    ).reshape(1, -1)
            working_delta = self.tree.working_memory.make_trainable_delta()
            if not self.config.adapt_working_memory:
                working_delta = working_delta.detach()

            with torch.set_grad_enabled(self.config.adapt_working_memory):
                memory_output = self.tree(
                    z_t=z_t,
                    working_delta=working_delta,
                    decays=self.hawkes.decays,
                    frontier_static_cache=frontier_static_cache,
                    frontier_projected_z=(
                        None
                        if precomputed_projected_z is None
                        else precomputed_projected_z[event_index].reshape(1, -1)
                    ),
                    frontier_query=(
                        None
                        if precomputed_memory_query is None
                        else precomputed_memory_query[event_index].reshape(1, -1)
                    ),
                    update_memory_state=False,
                    update_search_state=(
                        self.config.allow_memory_writes
                        or self.config.update_memory_usage
                    ),
                )
                pre_action_params = self._controller_effective_parameters(
                    memory_output,
                    working_delta,
                    working_delta.new_zeros(()),
                ).select(0)
                nll = self.hawkes.event_NLL(
                    sequence, pre_action_params, event_index
                )
                # The working-memory gradient is evaluated after the retrieval
                # gate has recomposed the final effective parameters below.
            if self.config.update_memory_usage:
                # Retrieval uses age in its differentiable scores; mutate it
                # only after the event's working-memory gradient is complete.
                self.tree.episodic_memory.step_age()

            with torch.no_grad():
                # A causal local-rate forecast made at the end of the prefix.
                # This is distinct from ``intensity_at_cached_event`` below,
                # which conditions mark prediction on the observed event time.
                forecast_origin = (
                    sequence["times"].new_tensor(0.0)
                    if event_index == 0
                    else sequence["times"][event_index - 1]
                )
                frontier_energy = self._frontier_event_energy(
                    sequence, memory_output, event_index
                )
                (
                    action,
                    owner_id,
                    query,
                    action_probabilities,
                    raw_action_probabilities,
                    posterior,
                    novelty,
                    retrieval_similarity,
                ) = self._action(
                    memory_output, nll, frontier_energy
                )
                with torch.set_grad_enabled(self.config.adapt_working_memory):
                    params = self._controller_effective_parameters(
                        memory_output,
                        working_delta,
                        action_probabilities[1],
                    ).select(0)
                    nll = self.hawkes.event_NLL(
                        sequence, params, event_index
                    )
                    if self.config.adapt_working_memory:
                        working_grad = torch.autograd.grad(
                            nll, working_delta
                        )[0]
                forecast_intensity = self.hawkes.intensity_at_event(
                    {
                        "times": sequence["times"][:event_index],
                        "types": sequence["types"][:event_index],
                    },
                    forecast_origin + forecast_origin.new_tensor(1e-6),
                    params,
                )
                forecast_rate = forecast_intensity.sum().clamp_min(1e-8)
                forecast_type_probabilities = forecast_intensity / forecast_rate
                predicted_delta = forecast_rate.reciprocal()
                predicted_time = forecast_origin + predicted_delta
                intensity = self.hawkes.intensity_at_cached_event(
                    sequence, event_index, params
                )
                type_probabilities_at_event_time = (
                    intensity / intensity.sum().clamp_min(1e-8)
                )
                predicted_type = int(intensity.argmax().item())
                if self.config.adapt_working_memory:
                    self.tree.working_memory.update_from_gradient(
                        working_grad,
                        adaptation_probability=action_probabilities[0],
                    )
                if self.config.update_memory_usage:
                    self.tree.episodic_memory.credit_retrieval(
                        info_by_batch=memory_output["memory_info"],
                        leaf_paths=[
                            self.tree.path_to_node(node_id)
                            for node_id in memory_output[
                                "frontier_node_ids"
                            ][0]
                        ],
                        routing_weights=posterior.unsqueeze(0),
                        retrieval_probability=action_probabilities[1],
                    )
                selected_memory_info = (
                    memory_output["memory_info"][0]
                    if memory_output.get("memory_info")
                    else {}
                )
                for accepted_request in accepted_write_requests:
                    if accepted_request.get("retrieved_later", False):
                        continue
                    accepted_event = int(accepted_request["event_index"])
                    if event_index <= accepted_event:
                        continue
                    evidence = accepted_request.get("admission_evidence") or {}
                    owner = evidence.get("owner_id")
                    item = evidence.get("candidate_item")
                    info = selected_memory_info.get(owner, {})
                    bank = self.tree.episodic_memory.banks.get(owner)
                    alpha = info.get("alpha")
                    if item is None or bank is None or alpha is None or not len(bank):
                        continue
                    bank._ensure_prototype_state()
                    key = F.normalize(
                        item.key.to(bank.context_keys).reshape(1, -1), dim=-1
                    ).reshape(-1)
                    delta = item.delta_theta.to(bank.deltas)
                    alias_matches = torch.isclose(
                        bank.context_keys,
                        key.unsqueeze(0).unsqueeze(1),
                        rtol=1e-5,
                        atol=1e-7,
                    ).all(dim=-1) & bank.context_valid
                    matches = alias_matches.any(dim=-1) & torch.isclose(
                        bank.deltas, delta.unsqueeze(0), rtol=1e-5, atol=1e-7
                    ).all(dim=-1)
                    indices = torch.nonzero(matches, as_tuple=False).flatten()
                    if indices.numel() and bool(
                        (alpha.index_select(0, indices.to(alpha.device)) > 1e-6)
                        .any().detach().cpu()
                    ):
                        accepted_request["retrieved_later"] = True
                        outputs[accepted_event]["write_retrieved_later"] = True
                packed_memory_info = memory_output.get(
                    "packed_memory_info", {}
                )
                retrieval_alpha_mass = sum(
                    float(info["alpha"].sum().detach().cpu())
                    for info in selected_memory_info.values()
                    if "alpha" in info
                )
                retrieval_effective_k = sum(
                    int(info["effective_k"].sum().detach().cpu())
                    for info in selected_memory_info.values()
                    if "effective_k" in info
                )
                packed_null_alpha = packed_memory_info.get("null_alpha")
                if packed_null_alpha is not None:
                    packed_null_alpha = packed_null_alpha[0]
                    visited_mask = memory_output["visited_node_mask"][0]
                    null_values = packed_null_alpha[visited_mask]
                    retrieval_null_alpha = (
                        float(null_values.mean().detach().cpu())
                        if null_values.numel() else 1.0
                    )
                else:
                    retrieval_null_alpha = 1.0

                visited_mask = memory_output["visited_node_mask"][0]
                visited_indices = memory_output["visited_node_indices"][0][
                    visited_mask
                ]
                visited_node_ids = tuple(
                    memory_output["memory_node_ids"][int(index)]
                    for index in visited_indices.detach().cpu().tolist()
                )
                visited_bank_count = len(visited_node_ids)
                visited_nonempty_bank_count = sum(
                    bool(
                        node_id in self.tree.episodic_memory.banks
                        and len(self.tree.episodic_memory.banks[node_id]) > 0
                    )
                    for node_id in visited_node_ids
                )
                raw_episodic_residual_norm = float(
                    memory_output["frontier_episodic_delta"][0]
                    .detach().norm(dim=-1).mean().cpu()
                )
                retrieve_gate = float(action_probabilities[1].detach().cpu())
                gated_episodic_residual_norm = (
                    raw_episodic_residual_norm * abs(retrieve_gate)
                )
                owner_on_retrieval_path = owner_id in set(visited_node_ids)

            frontier_ids = tuple(memory_output["frontier_node_ids"][0])
            write_probe_contexts.append({
                "event_index": int(event_index),
                "query": query.detach().clone(),
                "frontier_node_ids": frontier_ids,
                "frontier_semantic_theta": memory_output["frontier_semantic_theta"][
                    0, :len(frontier_ids)
                ].detach().clone(),
                "frontier_episodic_delta": memory_output["frontier_episodic_delta"][
                    0, :len(frontier_ids)
                ].detach().clone(),
                "posterior": posterior[:len(frontier_ids)].detach().clone(),
                "working_delta": working_delta.detach().clone(),
                "retrieve_gate": action_probabilities[1].detach().clone(),
                "no_write_theta": params.theta.detach().clone(),
            })

            if (
                self.config.allow_memory_writes
                or self.config.probe_write_counterfactuals
            ):
                pending_writes.append({
                    "event_index": event_index,
                    "ready_index": (
                        event_index + 2 * self.wake_config.write_horizon - 1
                        if int(self.controller.controller_version.detach().cpu()) >= 6
                        else event_index + (
                            2 if int(self.controller.controller_version.detach().cpu()) >= 5 else 1
                        ) * self.wake_config.write_horizon
                    ),
                    "admission_index": event_index + self.wake_config.write_horizon - 1,
                    "query": query,
                    "frontier_node_ids": frontier_ids,
                    "frontier_mass": memory_output["frontier_mass"][
                        0, : len(frontier_ids)
                    ].detach().clone(),
                    "frontier_theta": memory_output["frontier_theta"][
                        0, : len(frontier_ids)
                    ].detach().clone(),
                    "action": action,
                    "write_gate": raw_action_probabilities[2].detach(),
                    "raw_action_probabilities": (
                        raw_action_probabilities.detach().clone()
                    ),
                    "source_sequence_id": source_sequence_id,
                    "exploration": False,
                    "novelty": novelty,
                    "queue_weight": float(
                        self.controller.queue_weight(
                            raw_action_probabilities
                        ).detach().cpu()
                    ),
                    "future_contexts": write_probe_contexts,
                })

            event_nll = float(nll.detach().cpu())
            true_type = int(sequence["types"][event_index].item())
            true_time = float(sequence["times"][event_index].detach().cpu())
            total_nll += event_nll
            if compact:
                scalar_correct += int(predicted_type == true_type)
                scalar_time_abs_sum += abs(
                    float(predicted_time.detach().cpu()) - true_time
                )

            if materialize_events:
                outputs.append({
                    "event_index": event_index,
                    "nll": event_nll,
                    "predicted_type": predicted_type,
                    "true_type": true_type,
                    # ``pre_action_theta`` is the parameter state formed from
                    # the strict prefix events[:event_index].  CL law-recovery uses it
                    # to draw a genuinely causal intensity curve; the ordinary
                    # event rows intentionally keep this diagnostic internal.
                    "prediction_theta": pre_action_params.theta.detach().cpu(),
                    "intensity": intensity.detach().cpu(),
                    "type_probabilities_at_event_time": (
                        type_probabilities_at_event_time.detach().cpu()
                    ),
                    "forecast_type_probabilities": (
                        forecast_type_probabilities.detach().cpu()
                    ),
                    "predicted_delta": float(predicted_delta.detach().cpu()),
                    "predicted_time": float(predicted_time.detach().cpu()),
                    "true_time": true_time,
                    "responsibility": memory_output["r"][0].detach().cpu(),
                    "owner_id": owner_id,
                    "frontier_node_ids": tuple(
                        memory_output["frontier_node_ids"][0]
                    ),
                    "frontier_posterior": posterior.detach().cpu(),
                    "action": str(action),
                    "memorize_argmax": action == Action.MEMORIZE,
                    "action_probabilities": (
                        action_probabilities.detach().cpu()
                    ),
                    "raw_action_probabilities": (
                        raw_action_probabilities.detach().cpu()
                    ),
                    "retrieval_alpha_mass": retrieval_alpha_mass,
                    "retrieval_alpha_per_visited_node": (
                        retrieval_alpha_mass / visited_bank_count
                        if visited_bank_count else 0.0
                    ),
                    "retrieval_similarity": float(
                        retrieval_similarity.detach().cpu()
                    ),
                    "retrieval_effective_k": retrieval_effective_k,
                    "retrieval_null_alpha": retrieval_null_alpha,
                    "visited_bank_count": visited_bank_count,
                    "visited_nonempty_bank_count": visited_nonempty_bank_count,
                    "raw_episodic_residual_norm": raw_episodic_residual_norm,
                    "retrieve_gate": retrieve_gate,
                    "gated_episodic_residual_norm": (
                        gated_episodic_residual_norm
                    ),
                    "owner_on_retrieval_path": owner_on_retrieval_path,
                    "retrieval_counterfactual_gain": None,
                    "retrieval_counterfactual_unavailable_reason": (
                        "requires paired no_episodic evaluation"
                    ),
                    "episodic_residual_norm": float(
                        memory_output["episodic_delta"][0]
                        .detach().norm().cpu()
                    ),
                    "write_token": (
                        f"{source_index}:{int(event_index)}"
                    ),
                    "write_candidate": bool(
                        float(raw_action_probabilities[2].detach().cpu())
                        >= self.controller.write_candidate_threshold
                    ),
                    "write_gate_active": bool(
                        float(action_probabilities[2].detach().cpu()) > 0.0
                    ),
                    "write_gate_passed": bool(
                        float(raw_action_probabilities[2].detach().cpu())
                        >= float(self.controller.calibration_thresholds[2].detach().cpu())
                    ),
                    "write_priority_passed": False,
                    "write_window_complete": False,
                    "write_accepted": False,
                    "write_local_accepted": False,
                    "write_probation_enqueued": False,
                    "write_promotion_count": 0,
                    "write_retrieved_later": False,
                    "write_beneficial": False,
                })
            else:
                # These are the only fields mutated by delayed write/admission
                # bookkeeping after the event itself has been scored.
                outputs.append({
                    "predicted_type": predicted_type,
                    "true_type": true_type,
                    "predicted_time": float(predicted_time.detach().cpu()),
                    "true_time": true_time,
                    "write_probed": False,
                    "write_priority_passed": False,
                    "write_window_complete": False,
                    "write_accepted": False,
                    "write_local_accepted": False,
                    "write_probation_enqueued": False,
                    "write_promotion_count": 0,
                    "write_retrieved_later": False,
                    "write_beneficial": False,
                })

            # F=[t,t+h) still performs the existing causal local admission.
            # It no longer mutates EpisodicMemory: C-window local utility must
            # first create an invisible probation candidate, which later needs
            # evidence from independent sequences.
            if (
                int(self.controller.controller_version.detach().cpu()) >= 6
                and self.config.allow_memory_writes
            ):
                due = [
                    request for request in pending_writes
                    if request["admission_index"] == event_index
                    and not request.get("admission_evaluated", False)
                ]
                for request in due:
                    evidence = self._causal_write_evidence_v6(sequence, request)
                    allowlist = self.config.write_event_allowlist
                    if (
                        allowlist is not None
                        and int(request["event_index"]) in allowlist
                        and float(evidence["priority"].detach().cpu()) <= 0.0
                    ):
                        evidence["priority"] = (
                            request["write_gate"].clamp_min(1e-6)
                            * evidence["posterior"].max().clamp_min(1e-6)
                            * request["novelty"].clamp_min(1e-6)
                        )
                    request["admission_evidence"] = evidence
                    request["admission_evaluated"] = True
                    origin = outputs[request["event_index"]]
                    origin["write_window_complete"] = True
                    origin["write_priority_passed"] = bool(
                        float(evidence["priority"].detach().cpu())
                        > self.wake_config.controller_priority_threshold
                    )
                    allowlist = self.config.write_event_allowlist
                    request["local_admission_passed"] = bool(
                        (
                            allowlist is not None
                            and int(request["event_index"]) in allowlist
                        )
                        or (
                            allowlist is None
                            and self.controller.write_admissible(
                                request["write_gate"],
                                0.0,
                                evidence["priority"],
                                future_window_complete=True,
                                priority_threshold=(
                                    self.wake_config.controller_priority_threshold
                                ),
                            )
                        )
                    )

        event_count = int(sequence["times"].numel())
        probation_validations: list[Dict[str, Any]] = []
        promotions: list[Dict[str, Any]] = []
        eligible = [
            request
            for request in pending_writes
            if request["ready_index"] < event_count
        ]
        top_c = self.wake_config.controller_write_probe_topc
        ranked = sorted(
            eligible,
            key=lambda request: float(
                request["write_gate"].detach().cpu()
            ),
            reverse=True,
        )
        if self.config.write_event_allowlist is not None:
            allowed_events = set(self.config.write_event_allowlist)
            top = [
                request for request in ranked
                if int(request["event_index"]) in allowed_events
            ]
            remaining = []
        else:
            top = ranked[:top_c]
            remaining = ranked[top_c:]
        explored = []
        if self.config.probe_write_counterfactuals and remaining:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.write_probe_seed + source_index)
            count = min(self.config.write_probe_random_count, len(remaining))
            indices = torch.randperm(len(remaining), generator=generator)[:count].tolist()
            explored = [remaining[index] for index in indices]
        eligible = [*top, *explored]
        top_ids = {id(request) for request in top}
        explored_ids = {id(request) for request in explored}
        for request in eligible:
            outputs[request["event_index"]]["write_probed"] = True
            outputs[request["event_index"]]["write_probe_top"] = id(request) in top_ids
            outputs[request["event_index"]]["write_probe_exploration"] = id(request) in explored_ids
            request["write_probe_exploration"] = id(request) in explored_ids
            outputs[request["event_index"]]["write_probe_propensity"] = (
                1.0 if id(request) in top_ids else len(explored) / max(len(remaining), 1)
            )
        for request in eligible:
            request["window_evidence"] = self._delayed_write_evidence(
                sequence, request
            )
            outputs[request["event_index"]].update({
                "write_utility": float(
                    request["window_evidence"]["write_utility"].detach().cpu()
                ),
                "write_priority": float(
                    request["window_evidence"]["priority"].detach().cpu()
                ),
                "write_owner_on_score_path": bool(
                    request["window_evidence"].get("owner_on_score_path", False)
                ),
                "write_virtual_candidate_alpha": float(
                    request["window_evidence"].get("virtual_candidate_alpha", 0.0)
                ),
                "write_gate_passed": bool(
                    float(request["write_gate"])
                    >= float(self.controller.calibration_thresholds[2])
                ),
                "write_utility_passed": bool(
                    float(
                        request["window_evidence"]["write_utility"]
                        .detach().cpu()
                    ) > self.wake_config.controller_write_gain_threshold
                ),
                "write_beneficial": bool(
                    float(
                        request["window_evidence"]["write_utility"]
                        .detach().cpu()
                    ) > 0.0
                ),
                "write_priority_passed": bool(
                    float(
                        request["window_evidence"]["priority"].detach().cpu()
                    )
                    > self.wake_config.controller_priority_threshold
                ),
                "write_window_complete": True,
                "write_accepted": bool(request.get("committed", False)),
            })
        if (
            self.config.allow_memory_writes
            and int(self.controller.controller_version.detach().cpu()) >= 6
        ):
            probation_validations, promotions = (
                self._validate_probation_candidates(
                    sequence,
                    write_probe_contexts,
                    sequence_id=source_sequence_id,
                )
            )
            accepted_write_count = len(promotions)
            if outputs and promotions:
                outputs[0]["write_promotion_count"] = len(promotions)
                outputs[0]["promoted_write_tokens"] = [
                    row["token"] for row in promotions
                ]
        if int(self.controller.controller_version.detach().cpu()) >= 6:
            local_candidates = [
                request
                for request in eligible
                if (
                    not request.get("write_probe_exploration", False)
                    and request.get("local_admission_passed", False)
                    and float(
                        request["window_evidence"]["write_utility"]
                        .detach().cpu()
                    )
                    > self.wake_config.controller_write_gain_threshold
                )
            ]
            local_candidates.sort(
                key=lambda request: float(
                    request["window_evidence"]["priority"].detach().cpu()
                ),
                reverse=True,
            )
            local_budget = min(
                4, self.tree.frontier_routing.config.max_writes_per_sequence
            )
            for request in local_candidates[:local_budget]:
                candidate = self._enqueue_probation_candidate(
                    request,
                    source_sequence_id=source_sequence_id,
                )
                local_accepted_write_count += 1
                outputs[request["event_index"]].update({
                    "write_local_accepted": True,
                    "write_probation_enqueued": True,
                    "write_probation_token": candidate.token,
                    # Persistent acceptance is deliberately false until a
                    # later independent sequence passes the ESS + LCB gate.
                    "write_accepted": False,
                })
        else:
            eligible = [
                request for request in eligible
                if not request.get("write_probe_exploration", False)
                and self.controller.write_admissible(
                    request["write_gate"],
                    request["window_evidence"].get("write_utility", 0.0),
                    request["window_evidence"]["priority"],
                    future_window_complete=True,
                    priority_threshold=self.wake_config.controller_priority_threshold,
                )
            ]
            if eligible:
                priorities = torch.stack([
                    request["window_evidence"]["priority"]
                    for request in eligible
                ])
                selected = priorities.topk(
                    min(
                        len(eligible),
                        min(4, self.tree.frontier_routing.config.max_writes_per_sequence),
                    )
                ).indices.cpu().tolist()
            else:
                selected = []
            selected_requests = [eligible[index] for index in selected]
            if self.config.allow_memory_writes:
                persistent_flags = self._commit_delayed_writes_batch(
                    sequence, selected_requests
                )
                for request, persistent in zip(
                    selected_requests, persistent_flags
                ):
                    if not persistent:
                        continue
                    request["committed"] = True
                    accepted_write_requests.append(request)
                    outputs[request["event_index"]]["write_accepted"] = True
            if not self.config.allow_memory_writes:
                selected = []
        pending_writes = [
            request
            for request in pending_writes
            if request["ready_index"] >= event_count
        ]
        result = {
            "events": outputs if materialize_events else [],
            "total_nll": total_nll,
            "nll_per_event": total_nll / max(event_count, 1),
            "pending_write_count": len(pending_writes),
            "accepted_write_count": (
                accepted_write_count
                if int(self.controller.controller_version.detach().cpu()) >= 6
                else len(accepted_write_requests)
            ),
            "local_accepted_write_count": local_accepted_write_count,
            "probation_validation_count": sum(
                bool(row.get("matched", False))
                for row in probation_validations
            ),
            "probation_validations": probation_validations,
            "promoted_write_count": len(promotions),
            "promotions": promotions,
            "probation_size": len(self.write_probation),
            "write_probe_count": sum(
                bool(event.get("write_probed", False)) for event in outputs
            ),
            "leaf_ids": list(self.tree.leaf_ids),
        }
        if compact:
            result["scalar_metrics"] = {
                "events": event_count,
                "nll_sum": total_nll,
                "correct": scalar_correct,
                "time_abs_sum": scalar_time_abs_sum,
            }
        return result

    @torch.no_grad()
    def predict_next_event(
        self,
        cpu_prefix: Mapping[str, Tensor],
    ) -> Dict[str, Any]:
        """Return a local-rate next-event forecast from the observed prefix.

        The event type distribution is the normalized current Hawkes intensity.
        The reported time uses the standard locally constant-rate expectation
        ``1 / sum_d lambda_d``; it is deterministic and is not an exact Hawkes
        sample. Exact simulation can be added with Ogata thinning if required.
        """
        prefix = self._move_sequence(cpu_prefix)
        event_index = int(prefix["times"].numel())
        if isinstance(self.encoder, CausalPrefixEncoder):
            z_t = self.encoder(
                prefix["times"],
                prefix["types"],
                event_index,
                time_features=prefix.get(EVENT_TIME_FEATURES_KEY),
            ).reshape(1, -1)
        else:
            z_t = self.encoder(
                prefix["times"], prefix["types"], event_index
            ).reshape(1, -1)
        memory_output = self.tree(
            z_t=z_t,
            working_delta=self.tree.working_memory.delta,
            decays=self.hawkes.decays,
            update_memory_state=False,
        )
        params = memory_output["effective_params"].select(0)
        if event_index == 0:
            current_time = prefix["times"].new_tensor(0.0)
        else:
            current_time = prefix["times"][-1]
        evaluation_time = current_time + current_time.new_tensor(1e-6)
        intensity = self.hawkes.intensity_at_event(
            {"times": prefix["times"], "types": prefix["types"]},
            evaluation_time,
            params,
        )
        total_rate = intensity.sum().clamp_min(1e-8)
        probabilities = intensity / total_rate
        expected_delta = total_rate.reciprocal()
        return {
            "predicted_time": float((current_time + expected_delta).cpu()),
            "expected_delta": float(expected_delta.cpu()),
            "predicted_type": int(probabilities.argmax().item()),
            "type_probabilities": probabilities.cpu(),
            "intensity": intensity.cpu(),
            "responsibility": memory_output["r"][0].cpu(),
        }


def _parse_args():
    parser = argparse.ArgumentParser(description="Wake-only Memory Tree inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--types", type=int, nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inference = MemoryTreeInference.from_checkpoint(
        args.checkpoint,
        device=args.device,
        inference_config=InferenceConfig(allow_memory_writes=not args.no_write),
    )
    sequence = {
        "times": torch.tensor(args.times, dtype=torch.float32),
        "types": torch.tensor(args.types, dtype=torch.long),
    }
    result = inference.run_sequence(sequence)
    forecast = inference.predict_next_event(sequence)
    print(result)
    print(forecast)


if __name__ == "__main__":
    main()
