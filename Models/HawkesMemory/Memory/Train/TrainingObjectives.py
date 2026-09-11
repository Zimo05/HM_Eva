"""Cross-sequence routing and regional-probe objectives."""

from __future__ import annotations

from dataclasses import dataclass

from Train.TrainingComponents import *  # noqa: F403
from Train.TrainingComponents import _assert_finite_without_cuda_sync


@dataclass(frozen=True)
class RegionalProbePlan:
    """Device-resident static topology for Regional Probe supervision."""

    signature: tuple[Any, ...]
    coarse_ids: tuple[str, ...]
    descendant_leaf_ids: tuple[tuple[str, ...], ...]
    pair_index_by_leaf: tuple[Mapping[str, int], ...]
    pair_leaf_ids: tuple[str, ...]
    coarse_indices: Tensor
    region_pair_offsets: Tensor
    pair_leaf_node_indices: Tensor
    pair_leaf_priors: Tensor
    pair_path_offsets: Tensor
    path_router_nodes: Tensor
    path_targets: Tensor
    path_steps: Tensor


class TrainingObjectivesMixin:
    @staticmethod
    def _segment_sum(
        values: Tensor,
        segment_index: Tensor,
        segment_count: int,
    ) -> Tensor:
        """Differentiable GPU segment sum implemented with ``index_add_``."""
        shape = (segment_count, *values.shape[1:])
        result = values.new_zeros(shape)
        if values.numel():
            result.index_add_(0, segment_index, values)
        return result

    @classmethod
    def _segment_mean(
        cls,
        values: Tensor,
        segment_index: Tensor,
        segment_count: int,
    ) -> tuple[Tensor, Tensor]:
        sums = cls._segment_sum(values, segment_index, segment_count)
        counts = cls._segment_sum(
            values.new_ones(values.size(0)),
            segment_index,
            segment_count,
        )
        count_shape = (segment_count,) + (1,) * (values.ndim - 1)
        means = sums / counts.clamp_min(1.0).reshape(count_shape)
        return means, counts

    def _probe_leaf_local_theta(self, leaf_id: str) -> Tensor:
        """Leaf parameters whose only trainable term is its local offset."""
        fixed = self.tree.base_semantic_theta(leaf_id).detach()
        for ancestor in self.tree.node_paths[leaf_id][:-1]:
            fixed = fixed + self.tree.semantic_offset[ancestor].detach()
        return fixed + self.tree.semantic_offset[leaf_id]

    def _probe_sequence_energy(
        self,
        flat: Mapping[str, Tensor],
        sequence_index: Tensor,
        sequence_count: int,
        raw_theta: Tensor,
    ) -> Tensor:
        """Mean event NLL for ``[sequence, candidate, theta]`` in one pass."""
        if (
            raw_theta.ndim != 3
            or raw_theta.shape[0] != sequence_count
            or raw_theta.shape[-1] != self.tree.param_dim
        ):
            raise ValueError("probe theta must have shape [S, C, P]")
        event_theta = raw_theta.index_select(0, sequence_index)
        D = self.hawkes.num_types
        M = self.hawkes.num_basis
        mu = F.softplus(event_theta[..., :D])
        W = F.softplus(
            event_theta[..., D:].reshape(
                event_theta.size(0), event_theta.size(1), D, D, M
            )
        )
        history = flat[HAWKES_HISTORY_STATS_KEY]
        interval = flat[HAWKES_INTERVAL_STATS_KEY]
        intensity = (
            mu + torch.einsum("ncdem,nem->ncd", W, history)
        ).clamp_min(1e-8)
        selected = intensity.gather(
            2,
            flat["types"][:, None, None].expand(
                -1, raw_theta.size(1), 1
            ),
        ).squeeze(2)
        event_energy = (
            -selected.log()
            + mu.sum(dim=-1) * flat["duration"][:, None]
            + torch.einsum("ncdem,nem->nc", W, interval)
        )
        energy_sum = self._segment_sum(
            event_energy, sequence_index, sequence_count
        )
        event_count = self._segment_sum(
            event_energy.new_ones(event_energy.size(0)),
            sequence_index,
            sequence_count,
        )
        return energy_sum / event_count.clamp_min(1.0)[:, None]

    @staticmethod
    def _regional_probe_leaf_count(descendant_count: int) -> int:
        if descendant_count <= 0:
            raise ValueError("descendant_count must be positive")
        return (descendant_count + 1) // 2

    def _least_probed_leaves(
        self,
        descendant_leaves: Sequence[str],
    ) -> list[str]:
        """Probe half the region, with two-round coverage for odd counts."""
        visits = self.tree.frontier_routing.probe_leaf_visits
        order = {leaf_id: index for index, leaf_id in enumerate(descendant_leaves)}
        probe_count = self._regional_probe_leaf_count(
            len(descendant_leaves)
        )
        selected = sorted(
            descendant_leaves,
            key=lambda leaf_id: (visits.get(leaf_id, 0), order[leaf_id]),
        )[:probe_count]
        # Coverage counters are Router training state. Controller-only v5 may
        # read them to preserve the fixed routing policy, but must never advance
        # them (they are serialized in frontier_routing._extra_state).
        if not self.training_config.controller_only_finetune:
            for leaf_id in selected:
                visits[leaf_id] = visits.get(leaf_id, 0) + 1
        return selected

    def _regional_probe_topology_plan(self) -> RegionalProbePlan:
        """Cache Regional Probe topology as device-side CSR tensors.

        Leaf coverage remains dynamic because ``probe_leaf_visits`` changes
        after each Global batch.  Coarse nodes, descendant pairs, priors, and
        every router path are topology state and are rebuilt only after Sleep,
        a prior change, or a device move.
        """
        topology_signature = tuple(
            (
                node_id,
                self.tree.nodes[node_id].parent,
                self.tree.nodes[node_id].left,
                self.tree.nodes[node_id].right,
            )
            for node_id in self.tree.all_node_ids
        )
        prior_signature = tuple(
            (
                leaf_id,
                float(
                    self.tree.frontier_routing._target_leaf_mass_by_id.get(
                        leaf_id, 1.0
                    )
                ),
            )
            for leaf_id in self.tree.leaf_ids
        )
        device = self.tree._device_anchor.device
        signature = (topology_signature, prior_signature, str(device))
        cached_signature = getattr(
            self, "_regional_probe_plan_signature", None
        )
        cached_plan = getattr(self, "_regional_probe_plan", None)
        if cached_signature == signature and cached_plan is not None:
            return cached_plan

        node_index = {
            node_id: index
            for index, node_id in enumerate(self.tree.all_node_ids)
        }
        coarse_ids: list[str] = []
        coarse_indices: list[int] = []
        descendant_leaf_ids: list[tuple[str, ...]] = []
        pair_index_by_leaf: list[Mapping[str, int]] = []
        pair_leaf_ids: list[str] = []
        pair_leaf_node_indices: list[int] = []
        pair_leaf_priors: list[float] = []
        region_pair_offsets = [0]
        pair_path_offsets = [0]
        path_router_nodes: list[int] = []
        path_targets: list[int] = []
        for coarse_id in self.tree.internal_ids:
            descendants = tuple(
                leaf_id
                for leaf_id in self.tree.leaf_ids
                if coarse_id in self.tree.node_paths[leaf_id]
            )
            if len(descendants) < 2:
                continue
            coarse_ids.append(coarse_id)
            coarse_indices.append(node_index[coarse_id])
            descendant_leaf_ids.append(descendants)
            pair_lookup: Dict[str, int] = {}
            for leaf_id in descendants:
                pair_index = len(pair_leaf_ids)
                pair_lookup[leaf_id] = pair_index
                pair_leaf_ids.append(leaf_id)
                pair_leaf_node_indices.append(node_index[leaf_id])
                pair_leaf_priors.append(
                    self.tree.frontier_routing._target_leaf_mass_by_id.get(
                        leaf_id, 1.0
                    )
                )
                path = self.tree.node_paths[leaf_id]
                start = path.index(coarse_id)
                for path_position in range(start, len(path) - 1):
                    router_node_id = path[path_position]
                    next_node_id = path[path_position + 1]
                    router_node = self.tree.nodes[router_node_id]
                    if router_node.left == next_node_id:
                        target = 0
                    elif router_node.right == next_node_id:
                        target = 1
                    else:
                        raise RuntimeError("invalid descendant routing path")
                    path_router_nodes.append(node_index[router_node_id])
                    path_targets.append(target)
                pair_path_offsets.append(len(path_router_nodes))
            pair_index_by_leaf.append(pair_lookup)
            region_pair_offsets.append(len(pair_leaf_ids))

        max_path_length = max(
            (
                pair_path_offsets[index + 1] - pair_path_offsets[index]
                for index in range(len(pair_leaf_ids))
            ),
            default=0,
        )
        plan = RegionalProbePlan(
            signature=signature,
            coarse_ids=tuple(coarse_ids),
            descendant_leaf_ids=tuple(descendant_leaf_ids),
            pair_index_by_leaf=tuple(pair_index_by_leaf),
            pair_leaf_ids=tuple(pair_leaf_ids),
            coarse_indices=torch.tensor(
                coarse_indices, device=device, dtype=torch.long
            ),
            region_pair_offsets=torch.tensor(
                region_pair_offsets, device=device, dtype=torch.long
            ),
            pair_leaf_node_indices=torch.tensor(
                pair_leaf_node_indices, device=device, dtype=torch.long
            ),
            pair_leaf_priors=torch.tensor(
                pair_leaf_priors, device=device, dtype=torch.float32
            ),
            pair_path_offsets=torch.tensor(
                pair_path_offsets, device=device, dtype=torch.long
            ),
            path_router_nodes=torch.tensor(
                path_router_nodes, device=device, dtype=torch.long
            ),
            path_targets=torch.tensor(
                path_targets, device=device, dtype=torch.long
            ),
            path_steps=torch.arange(
                max_path_length, device=device, dtype=torch.long
            ),
        )

        self._regional_probe_plan_signature = signature
        self._regional_probe_plan = plan
        return plan

    def _regional_probe_objective(
        self,
        memory_output: Mapping[str, Any],
        frontier_posterior: Tensor,
        sequence_index: Tensor,
        sequence_count: int,
        sequence_event_embeddings: Tensor,
        flat: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Counterfactually compare stop vs. a covered subset of deep leaves.

        Wake routing is unchanged.  For every coarse final-frontier region the
        probe evaluates its own Hawkes energy plus ``Kp`` least-probed leaves.
        The detached energy teacher independently supervises expansion,
        dormant paths, and leaf-local predictive calibration.
        """
        zero = sequence_event_embeddings.sum() * 0.0
        empty_long = torch.empty(0, dtype=torch.long, device=self.device)
        empty_float = sequence_event_embeddings.new_empty(0)

        def empty_result() -> Dict[str, Tensor]:
            return {
                "loss": zero,
                "router_loss": zero,
                "expand_loss": zero,
                "leaf_loss": zero,
                "node_indices": empty_long,
                "refinement_gain": empty_float,
                "expand_probability": empty_float,
                "expand_target": empty_float,
                "stop_distortion": empty_float,
                "leaf_distortion": empty_float,
                "assignment_confidence": empty_float,
                "regions": zero.detach(),
                "router_rows": zero.detach(),
                "probe_leaves": zero.detach(),
            }

        if self.wake_config.lambda_route_probe == 0.0:
            return empty_result()
        frontier_nodes = memory_output["frontier_node_indices"]
        frontier_mask = memory_output["frontier_mask"]
        if frontier_posterior.shape != frontier_nodes.shape:
            raise ValueError("frontier posterior and node slots must align")

        node_count = len(self.tree.all_node_ids)
        event_node_mass = frontier_posterior.detach().new_zeros(
            frontier_nodes.size(0), node_count
        )
        event_node_mass.scatter_add_(
            1,
            frontier_nodes.clamp_min(0),
            frontier_posterior.detach().masked_fill(~frontier_mask, 0.0),
        )
        sequence_node_mass, _ = self._segment_mean(
            event_node_mass, sequence_index, sequence_count
        )
        sequence_embedding, _ = self._segment_mean(
            sequence_event_embeddings, sequence_index, sequence_count
        )
        cumulative_node = self.tree._node_embedding_table()
        probe_plan = self._regional_probe_topology_plan()
        if probe_plan.coarse_indices.numel() == 0:
            return empty_result()

        # Select active coarse regions with one vector reduction and one
        # boundary transfer.  The previous implementation synchronized once
        # per internal node via ``float(total_weight)``.
        coarse_weights = sequence_node_mass.index_select(
            1, probe_plan.coarse_indices
        ).detach()
        total_weights = coarse_weights.sum(dim=0)
        active_position_tensor = torch.nonzero(
            total_weights > 1e-12,
            as_tuple=False,
        ).reshape(-1)
        active_positions = active_position_tensor.detach().cpu().tolist()
        if not active_positions:
            return empty_result()

        # Runtime work is intentionally limited to the stateful least-probed
        # choice.  Every selected (region, leaf) pair maps into cached CSR
        # topology, so no node-path parsing or route-row construction occurs
        # in the Global hot path.
        selected_pair_rows: list[list[int]] = []
        selected_by_region: list[list[str]] = []
        unique_leaf_ids: list[str] = []
        seen_leaf_ids: set[str] = set()
        for region_position in active_positions:
            selected = self._least_probed_leaves(
                probe_plan.descendant_leaf_ids[region_position]
            )
            selected_by_region.append(selected)
            selected_pair_rows.append([
                probe_plan.pair_index_by_leaf[region_position][leaf_id]
                for leaf_id in selected
            ])
            for leaf_id in selected:
                if leaf_id not in seen_leaf_ids:
                    seen_leaf_ids.add(leaf_id)
                    unique_leaf_ids.append(leaf_id)

        active_count = len(active_positions)
        selected_leaf_count = sum(map(len, selected_pair_rows))
        max_selected = max(map(len, selected_pair_rows))
        pair_matrix_rows = [
            row + [-1] * (max_selected - len(row))
            for row in selected_pair_rows
        ]
        selected_pairs = torch.tensor(
            pair_matrix_rows,
            device=sequence_node_mass.device,
            dtype=torch.long,
        )
        selected_mask = selected_pairs >= 0
        safe_pairs = selected_pairs.clamp_min(0)

        # Every selected coarse/leaf candidate shares the same flattened
        # event statistics.  Evaluate all unique candidates once, then gather
        # the columns needed by each region.  The leaf columns retain their
        # gradients; coarse columns remain the detached stop teacher.
        candidate_ids = [
            probe_plan.coarse_ids[position] for position in active_positions
        ] + unique_leaf_ids
        candidate_theta = torch.stack([
            self.tree.semantic_theta(candidate_id).detach()
            if index < active_count
            else self._probe_leaf_local_theta(candidate_id)
            for index, candidate_id in enumerate(candidate_ids)
        ]).unsqueeze(0).expand(sequence_count, -1, -1)
        candidate_energy = self._probe_sequence_energy(
            flat,
            sequence_index,
            sequence_count,
            candidate_theta,
        )
        candidate_index = {
            candidate_id: index
            for index, candidate_id in enumerate(candidate_ids)
        }
        selected_candidate_columns = torch.tensor(
            [
                [
                    candidate_index[leaf_id] if leaf_id else 0
                    for leaf_id in (
                        row + [""] * (max_selected - len(row))
                    )
                ]
                for row in selected_by_region
            ],
            device=candidate_energy.device,
            dtype=torch.long,
        )
        gathered_energy = candidate_energy.gather(
            1,
            selected_candidate_columns.reshape(1, -1).expand(
                sequence_count, -1
            ),
        ).reshape(sequence_count, active_count, max_selected)
        leaf_energy = gathered_energy.masked_fill(
            ~selected_mask.unsqueeze(0), 0.0
        )
        evidence_leaf_energy = gathered_energy.detach().masked_fill(
            ~selected_mask.unsqueeze(0), torch.inf
        )
        coarse_energy = candidate_energy[:, :active_count]
        active_coarse_weights = coarse_weights.index_select(
            1, active_position_tensor
        )
        active_total_weights = total_weights.index_select(
            0, active_position_tensor
        )
        leaf_prior = probe_plan.pair_leaf_priors.to(candidate_energy).index_select(
            0, safe_pairs.reshape(-1)
        ).reshape(active_count, max_selected).masked_fill(~selected_mask, 0.0)

        # Batched, masked form of counterfactual_energy_probe.  It is
        # algebraically identical for each region, including variable Kp,
        # while avoiding one Python call and several host checks per region.
        teacher_temperature = float(
            self.wake_config.route_probe_residual_temperature
        )
        gain_temperature = float(
            self.wake_config.route_probe_gain_temperature
        )
        leaf_smoothing = float(self.wake_config.route_probe_leaf_smoothing)
        with torch.no_grad():
            all_energy = torch.cat(
                (coarse_energy.detach().unsqueeze(2), evidence_leaf_energy),
                dim=2,
            )
            teacher = F.softmax(
                -all_energy / teacher_temperature, dim=2
            )
            expand_target = 1.0 - teacher[:, :, 0]
            conditional_leaf_credit = F.softmax(
                -evidence_leaf_energy / teacher_temperature, dim=2
            ).masked_fill(~selected_mask.unsqueeze(0), 0.0)
            leaf_counts = selected_mask.sum(dim=1).clamp_min(1)
            smoothed_leaf_credit = (
                (1.0 - leaf_smoothing) * conditional_leaf_credit
                + selected_mask.unsqueeze(0).to(candidate_energy)
                * (leaf_smoothing / leaf_counts.to(candidate_energy))[None, :, None]
            )
            entropy = -(
                conditional_leaf_credit.clamp_min(1e-12)
                * conditional_leaf_credit.clamp_min(1e-12).log()
                * selected_mask.unsqueeze(0)
            ).sum(dim=2)
            assignment_confidence = torch.where(
                leaf_counts.unsqueeze(0) == 1,
                torch.ones_like(entropy),
                (
                    1.0
                    - entropy
                    / leaf_counts.to(candidate_energy).log().unsqueeze(0)
                ).clamp(0.0, 1.0),
            )
            normalized_prior = leaf_prior / leaf_prior.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            fine_energy = -gain_temperature * torch.logsumexp(
                normalized_prior.clamp_min(1e-12).log().unsqueeze(0)
                - evidence_leaf_energy / gain_temperature,
                dim=2,
            )
            observed_gain = (
                active_coarse_weights
                * (coarse_energy.detach() - fine_energy).clamp_min(0.0)
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12)

        active_coarse_indices = probe_plan.coarse_indices.index_select(
            0, active_position_tensor
        )
        active_node_embedding = cumulative_node.index_select(
            0, active_coarse_indices
        ).detach()
        expanded_sequence = sequence_embedding.detach()[:, None, :].expand(
            -1, active_count, -1
        ).reshape(-1, sequence_embedding.size(1))
        expanded_node = active_node_embedding[None, :, :].expand(
            sequence_count, -1, -1
        ).reshape(-1, active_node_embedding.size(1))
        expansion_logit = self.tree.expansion_predictor(
            expanded_sequence, expanded_node
        ).reshape(sequence_count, active_count)
        expansion_loss = (
            active_coarse_weights
            * F.binary_cross_entropy_with_logits(
                expansion_logit, expand_target, reduction="none"
            )
        ).sum(dim=0) / active_total_weights.clamp_min(1e-12)
        expand_loss = expansion_loss.mean()
        leaf_loss = (
            active_coarse_weights
            * (smoothed_leaf_credit * leaf_energy).sum(dim=2)
        ).sum(dim=0) / active_total_weights.clamp_min(1e-12)
        leaf_loss = leaf_loss.mean()

        pooling_weight = (
            active_coarse_weights.unsqueeze(2) * teacher[:, :, 1:]
        )
        leaf_mass = pooling_weight.sum(dim=0)
        pooled_query = torch.einsum(
            "sak,sz->akz", pooling_weight, sequence_embedding
        ) / leaf_mass.clamp_min(1e-12).unsqueeze(2)
        leaf_fraction = leaf_mass / leaf_mass.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

        # Expand selected CSR pairs into router rows on-device.  The rectangular
        # step grid preserves region, selected-leaf, and path order exactly.
        selected_pair_flat = selected_pairs.masked_select(selected_mask)
        selected_slot = torch.nonzero(
            selected_mask.reshape(-1), as_tuple=False
        ).reshape(-1)
        path_start = probe_plan.pair_path_offsets.index_select(
            0, selected_pair_flat
        )
        path_end = probe_plan.pair_path_offsets.index_select(
            0, selected_pair_flat + 1
        )
        path_length = path_end - path_start
        path_grid = path_start[:, None] + probe_plan.path_steps[None, :]
        path_mask = probe_plan.path_steps[None, :] < path_length[:, None]
        path_indices = path_grid.masked_select(path_mask)
        query_owner = selected_slot[:, None].expand_as(path_grid).masked_select(
            path_mask
        )
        query = pooled_query.reshape(
            active_count * max_selected, -1
        ).index_select(0, query_owner)
        weights = leaf_fraction.detach().reshape(-1).index_select(
            0, query_owner
        )
        router_nodes = probe_plan.path_router_nodes.index_select(
            0, path_indices
        )
        targets = probe_plan.path_targets.index_select(0, path_indices)
        child_index = self.tree.frontier_routing._topology_tensors[
            "child_index"
        ].index_select(0, router_nodes)
        normalized_node = self.tree.router_compat.normalize_nodes(
            cumulative_node.detach()
        )
        child_embedding = normalized_node.index_select(
            0, child_index.reshape(-1)
        ).reshape(query.size(0), 2, -1)
        semantic_score = self.tree.router_compat.score_normalized(
            self.tree.router_compat.project_z(query), child_embedding
        )
        child_prior = self.tree.frontier_routing._topology_tensors[
            "child_prior"
        ].to(semantic_score).index_select(0, router_nodes)
        logits = (
            self.tree.frontier_routing.config.semantic_weight
            * semantic_score
            / self.tree.frontier_routing.config.routing_temperature
            + child_prior.clamp_min(1e-12).log()
        )
        router_loss = (
            weights * F.cross_entropy(logits, targets, reduction="none")
        ).sum() / max(active_count, 1)
        loss = (
            self.wake_config.route_probe_router_weight * router_loss
            + self.wake_config.route_probe_expand_weight * expand_loss
            + self.wake_config.route_probe_leaf_weight * leaf_loss
        )
        return {
            "loss": loss,
            "router_loss": router_loss,
            "expand_loss": expand_loss,
            "leaf_loss": leaf_loss,
            "node_indices": active_coarse_indices,
            "refinement_gain": observed_gain,
            "expand_probability": (
                active_coarse_weights * expansion_logit.sigmoid().detach()
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12),
            "expand_target": (
                active_coarse_weights * expand_target
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12),
            "stop_distortion": (
                active_coarse_weights * coarse_energy.detach()
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12),
            "leaf_distortion": (
                active_coarse_weights * fine_energy
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12),
            "assignment_confidence": (
                active_coarse_weights * assignment_confidence
            ).sum(dim=0) / active_total_weights.clamp_min(1e-12),
            "regions": loss.new_tensor(float(active_count)).detach(),
            "router_rows": loss.new_tensor(float(query.size(0))).detach(),
            "probe_leaves": loss.new_tensor(float(selected_leaf_count)).detach(),
        }

    def _batched_local_frontier_objective(
        self,
        memory_output: Mapping[str, Any],
        child_energy: Tensor,
        sequence_index: Tensor,
        sequence_count: int,
    ) -> Dict[str, Tensor]:
        """Local teacher/MI loss with exact sequence and node normalization.

        The old implementation first averaged rows within each node, averaged
        nodes within each sequence, then averaged sequences. Two-level segment
        reductions reproduce that weighting without a Python node loop.
        """
        probability = memory_output["expanded_probability"]
        expanded_mask = memory_output["expanded_mask"]
        expanded_nodes = memory_output["expanded_node_indices"]
        zero = probability.sum() * 0.0

        topology = self.tree.frontier_routing._topology_tensors
        node_count = len(self.tree.all_node_ids)
        safe_expanded = expanded_nodes.clamp_min(0)
        child_prior = topology["child_prior"].to(probability)
        log_child_prior = topology.get("log_child_prior")
        if log_child_prior is None:
            log_child_prior = child_prior.clamp_min(1e-12).log()
        else:
            log_child_prior = log_child_prior.to(probability)
        safe_energy = child_energy.detach().masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        teacher_logits = (
            log_child_prior.index_select(
                0, safe_expanded.reshape(-1)
            ).reshape_as(probability)
            - safe_energy / self.wake_config.route_teacher_temperature
        )
        branch_target = F.softmax(teacher_logits, dim=-1).masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        energy_teacher = F.softmax(
            -safe_energy / self.wake_config.route_teacher_temperature,
            dim=-1,
        ).masked_fill(~expanded_mask.unsqueeze(-1), 0.0)
        reliability_rows = (
            1.0
            + (
                energy_teacher.clamp_min(1e-12)
                * energy_teacher.clamp_min(1e-12).log()
            ).sum(dim=-1)
            / math.log(2.0)
        ).clamp(0.0, 1.0).masked_fill(~expanded_mask, 0.0)
        # Distill the distribution used by the actual search.  It already
        # contains the same fixed topology prior as ``branch_target``.
        route_student = probability.masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        distill_rows = (
            branch_target.detach()
            * (
                branch_target.detach().clamp_min(1e-12).log()
                - route_student.clamp_min(1e-12).log()
            )
        ).sum(dim=-1)

        # Keep a static ``[event, round]`` layout.  Boolean indexing here used
        # to create a dynamic ``[active_round, 2]`` tensor and blocked graph
        # capture/compilation.  Invalid rounds are represented by zero mask
        # weights and a safe node index; all reductions below honor them.
        flat_mask = expanded_mask.reshape(-1)
        flat_weight = flat_mask.to(probability.dtype)
        flat_sequence = (
            sequence_index[:, None]
            .expand_as(expanded_mask)
            .reshape(-1)
        )
        flat_node = expanded_nodes.clamp_min(0).reshape(-1)
        flat_probability = probability.reshape(-1, probability.size(-1))
        flat_distill = distill_rows.reshape(-1)
        flat_reliability = reliability_rows.reshape(-1)

        # L_router = sum rho KL / (sum rho + eps). Equal child energies give
        # rho=0, so topology prior alone cannot manufacture supervision.
        distill = (
            flat_weight * flat_reliability * flat_distill
        ).sum() / (flat_weight * flat_reliability).sum().clamp_min(1e-12)

        combined_segment = (
            flat_sequence * node_count + flat_node
        )
        combined_count = sequence_count * node_count
        weighted_probability = flat_probability * flat_weight[:, None]
        marginal_sum = self._segment_sum(
            weighted_probability,
            combined_segment,
            combined_count,
        )
        row_counts = self._segment_sum(
            flat_weight,
            combined_segment,
            combined_count,
        )
        marginal = marginal_sum / row_counts.clamp_min(1.0)[:, None]
        row_entropy = -(
            flat_probability.clamp_min(1e-12)
            * flat_probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        conditional_sum = self._segment_sum(
            row_entropy * flat_weight,
            combined_segment,
            combined_count,
        )
        conditional = conditional_sum / row_counts.clamp_min(1.0)
        marginal_entropy = -(
            marginal.clamp_min(1e-12)
            * marginal.clamp_min(1e-12).log()
        ).sum(dim=-1)
        segment_prior = log_child_prior.unsqueeze(0).expand(
            sequence_count,
            -1,
            -1,
        ).reshape(sequence_count * node_count, -1)
        balance = (
            marginal.clamp_min(1e-12)
            * (
                marginal.clamp_min(1e-12).log()
                - segment_prior
            )
        ).sum(dim=-1)
        observed = row_counts > 0
        marginal_entropy = marginal_entropy.masked_fill(~observed, 0.0)
        conditional = conditional.masked_fill(~observed, 0.0)
        balance = balance.masked_fill(~observed, 0.0)
        mutual_information = marginal_entropy - conditional

        observed_2d = observed.reshape(sequence_count, node_count)
        node_denominator = observed_2d.sum(dim=-1).clamp_min(1)

        def mean_nodes_then_sequences(values: Tensor) -> Tensor:
            return (
                values.reshape(sequence_count, node_count).sum(dim=-1)
                / node_denominator
            ).mean()

        return {
            "distill": distill,
            "mutual_information": mean_nodes_then_sequences(
                mutual_information
            ),
            "balance_kl": mean_nodes_then_sequences(balance),
            "conditional_entropy": mean_nodes_then_sequences(conditional),
            "marginal_entropy": mean_nodes_then_sequences(
                marginal_entropy
            ),
            "observed_gain": (
                reliability_rows * distill_rows
            ).detach().masked_fill(~expanded_mask, 0.0),
            "teacher": branch_target.detach(),
            "energy_teacher": energy_teacher.detach(),
            "student": route_student.detach(),
            "reliability": reliability_rows.detach(),
        }

    def _local_frontier_objective(
        self,
        memory_output: Mapping[str, Any],
        child_energy: Tensor,
    ) -> Dict[str, Tensor]:
        """Fixed-prior child-energy distillation and local regularizers."""
        probability = memory_output["expanded_probability"]
        expanded_mask = memory_output["expanded_mask"]
        expanded_nodes = memory_output["expanded_node_indices"]
        zero = probability.sum() * 0.0
        if not bool(expanded_mask.any()):
            return {
                "distill": zero,
                "mutual_information": zero,
                "balance_kl": zero,
                "conditional_entropy": zero,
                "marginal_entropy": zero,
                "observed_gain": probability.new_zeros(
                    expanded_mask.shape
                ),
                "teacher": probability.new_zeros(probability.shape),
                "energy_teacher": probability.new_zeros(probability.shape),
                "student": probability.new_zeros(probability.shape),
                "reliability": probability.new_zeros(expanded_mask.shape),
            }

        topology = self.tree.frontier_routing._topology_tensors
        safe_expanded = expanded_nodes.clamp_min(0)
        fixed_prior = topology["child_prior"].to(probability).index_select(
            0,
            safe_expanded.reshape(-1),
        ).reshape_as(probability)
        safe_energy = child_energy.detach().masked_fill(
            ~expanded_mask.unsqueeze(-1),
            0.0,
        )
        teacher_logits = (
            fixed_prior.clamp_min(1e-12).log()
            - safe_energy / self.wake_config.route_teacher_temperature
        )
        branch_target = F.softmax(teacher_logits, dim=-1).masked_fill(
            ~expanded_mask.unsqueeze(-1),
            0.0,
        )
        energy_teacher = F.softmax(
            -safe_energy / self.wake_config.route_teacher_temperature,
            dim=-1,
        ).masked_fill(~expanded_mask.unsqueeze(-1), 0.0)
        reliability_rows = (
            1.0
            + (
                energy_teacher.clamp_min(1e-12)
                * energy_teacher.clamp_min(1e-12).log()
            ).sum(dim=-1)
            / math.log(2.0)
        ).clamp(0.0, 1.0).masked_fill(~expanded_mask, 0.0)
        route_student = probability.masked_fill(
            ~expanded_mask.unsqueeze(-1), 0.0
        )
        distill_rows = (
            branch_target.detach()
            * (
                branch_target.detach().clamp_min(1e-12).log()
                - route_student.clamp_min(1e-12).log()
            )
        ).sum(dim=-1)
        selected_reliability = reliability_rows.masked_select(expanded_mask)
        distill = (
            selected_reliability
            * distill_rows.masked_select(expanded_mask)
        ).sum() / selected_reliability.sum().clamp_min(1e-12)

        node_mi = []
        node_balance = []
        node_conditional = []
        node_marginal = []
        child_prior = topology["child_prior"].to(probability)
        for node_index in expanded_nodes[expanded_mask].unique():
            selected = expanded_mask & (expanded_nodes == node_index)
            rows = probability[selected]
            if not rows.numel():
                continue
            marginal = rows.mean(dim=0)
            conditional_entropy = -(
                rows.clamp_min(1e-12)
                * rows.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
            marginal_entropy = -(
                marginal.clamp_min(1e-12)
                * marginal.clamp_min(1e-12).log()
            ).sum()
            prior = child_prior[int(node_index.item())]
            balance = (
                marginal.clamp_min(1e-12)
                * (
                    marginal.clamp_min(1e-12).log()
                    - prior.clamp_min(1e-12).log()
                )
            ).sum()
            node_conditional.append(conditional_entropy)
            node_marginal.append(marginal_entropy)
            node_mi.append(marginal_entropy - conditional_entropy)
            node_balance.append(balance)

        def mean_or_zero(values: Sequence[Tensor]) -> Tensor:
            return torch.stack(list(values)).mean() if values else zero

        return {
            "distill": distill,
            "mutual_information": mean_or_zero(node_mi),
            "balance_kl": mean_or_zero(node_balance),
            "conditional_entropy": mean_or_zero(node_conditional),
            "marginal_entropy": mean_or_zero(node_marginal),
            "observed_gain": (
                reliability_rows * distill_rows
            ).detach().masked_fill(~expanded_mask, 0.0),
            "teacher": branch_target.detach(),
            "energy_teacher": energy_teacher.detach(),
            "student": route_student.detach(),
            "reliability": reliability_rows.detach(),
        }

    def _sequence_route_mean_for_router(
        self,
        sequence: Mapping[str, Tensor],
        *,
        encoder_grad_scale: float = 0.0,
    ) -> Tensor:
        """Compatibility diagnostic: dense mass over actual tree nodes."""
        if not 0.0 <= encoder_grad_scale <= 1.0:
            raise ValueError("encoder_grad_scale must lie in [0, 1]")
        event_routes = []
        for event_index in range(sequence["times"].numel()):
            if encoder_grad_scale == 0.0:
                # During warm-up MI is a Router-only objective.
                with torch.no_grad():
                    z_t = self._encode_memory_event(
                        sequence,
                        event_index,
                    )
                routed_z = z_t.detach()
            else:
                z_t = self._encode_memory_event(
                    sequence,
                    event_index,
                )
                # Forward value is unchanged. Only the gradient entering the
                # Encoder is scaled; Router gradients retain full strength.
                routed_z = (
                    z_t.detach()
                    + encoder_grad_scale * (z_t - z_t.detach())
                )
            route = self.tree.route(routed_z)
            dense = routed_z.new_zeros(len(self.tree.all_node_ids))
            dense.scatter_add_(
                0,
                route.frontier_node_indices[
                    0, route.frontier_mask[0]
                ],
                route.responsibility[0, route.frontier_mask[0]],
            )
            event_routes.append(dense)
        if not event_routes:
            raise ValueError("global batches cannot contain empty sequences")
        return torch.stack(event_routes, dim=0).mean(dim=0)

    def train_global_batch_epoch(
        self,
        dataset: Sequence[Mapping[str, Tensor]],
        generator: torch.Generator,
        *,
        epoch: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Train experts and Router on the actual computed frontier.

        Prediction NLL updates experts with detached routing mass. Router
        updates come from the fixed-prior local child-energy teacher, the
        training-only Regional Probe under unexpanded coarse regions, and weak
        MI/balance regularizers. Frontier posterior and likelihood mixture
        remain detached diagnostics/ownership signals.
        """
        if not dataset:
            raise ValueError("global training requires at least one sequence")
        effective_epoch = (
            self.completed_epochs + 1 if epoch is None else int(epoch)
        )
        if effective_epoch <= 0:
            raise ValueError("global training epoch must be positive")

        order = torch.randperm(len(dataset), generator=generator).tolist()
        batch_size = self.wake_config.route_balance_batch_size
        batches = [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())

        total_loss = 0.0
        total_prediction = 0.0
        total_likelihood_mixture = 0.0
        total_prior_kl = 0.0
        total_posterior_kl = 0.0
        total_distill = 0.0
        total_mi = 0.0
        total_conditional_entropy = 0.0
        total_marginal_entropy = 0.0
        total_probe_loss = 0.0
        total_probe_router_loss = 0.0
        total_probe_expand_loss = 0.0
        total_probe_leaf_loss = 0.0
        total_probe_expand_probability = 0.0
        total_probe_expand_target = 0.0
        total_probe_refinement_gain = 0.0
        total_probe_assignment_confidence = 0.0
        total_probe_regions = 0.0
        total_sequences = 0
        total_events = 0
        optimizer_steps = 0
        max_gradient_norm = 0.0
        total_encoder_grad_scale = 0.0
        reliability_updates = 0
        total_teacher_confidence = 0.0
        total_teacher_student_js = 0.0
        total_teacher_student_alignment = 0.0
        total_controller_loss = 0.0
        min_controller_head_grad_norm = math.inf
        max_controller_grad_norm = 0.0
        max_controller_head_grad_norms = torch.zeros(4, dtype=torch.float64)
        self.tree.train()
        self.encoder.train()

        global_progress = tqdm(
            total=len(dataset),
            desc=f"[Epoch {effective_epoch:03d}] Global",
            unit="seq-pass",
            ascii=True,
            dynamic_ncols=False,
            ncols=110,
            mininterval=2.0,
            maxinterval=10.0,
            smoothing=0.1,
            leave=True,
            disable=not show_progress,
            file=sys.stdout,
        )
        for batch_index, batch_indices in enumerate(batches, start=1):
            # Reliability is updated from the preceding observed batch. This
            # one-step lag keeps the gate causal and prevents its own current
            # gradients from changing the scale used in the same graph.
            encoder_grad_scale = (
                self.wake_config.route_encoder_grad_scale
                * self.encoder_routing_reliability
            )
            self.optimizer.zero_grad(set_to_none=True)
            resident_store = getattr(self, "_resident_sequence_store", None)
            if resident_store is not None:
                # The training dataset was validated and cached once during
                # train() initialisation.  Keep the original mappings for
                # sequence metadata, but skip the hot-path validation and
                # device copies here.
                moved_sequences = [dataset[index] for index in batch_indices]
                self._resident_cache_hits += len(moved_sequences)
            else:
                moved_sequences = [
                    self._move_sequence(dataset[index])
                    for index in batch_indices
                ]
            batch_event_count = sum(
                int(sequence["times"].numel())
                for sequence in moved_sequences
            )
            if batch_event_count <= 0:
                continue
            sequence_count = len(moved_sequences)
            global_progress.set_postfix(
                phase="train",
                batch=f"{batch_index}/{len(batches)}",
                refresh=False,
            )
            z_all, flat = self._encode_global_sequence_batch(
                moved_sequences,
                sequence_indices=batch_indices,
            )
            routed_z = reliability_gated_route_state(
                z_all,
                reliability=self.encoder_routing_reliability,
                alpha_max=self.wake_config.route_encoder_grad_scale,
            )
            projected_routed_z = self.tree.router_compat.project_z(
                routed_z
            )
            memory_query = self.tree.episodic_memory.query_net(z_all)
            memory_output = self.tree(
                z_t=z_all,
                working_delta=torch.zeros(
                    self.tree.param_dim,
                    device=self.device,
                    dtype=z_all.dtype,
                ),
                decays=self.hawkes.decays,
                frontier_projected_z=projected_routed_z,
                frontier_query=memory_query,
                update_memory_state=False,
                update_search_state=False,
                detach_routing=True,
                materialize_diagnostics=False,
            )
            # All three Global parameter variants share the same routing
            # reduction.  Cache the affine semantic/episodic bases once and
            # let controller-effective parameters apply only the requested
            # gate in unconstrained space.
            memory_output["semantic_base"] = (
                memory_output["r"].unsqueeze(-1)
                * memory_output["frontier_semantic_theta"]
            ).sum(dim=1)
            memory_output["episodic_base"] = (
                memory_output["r"].unsqueeze(-1)
                * memory_output["frontier_episodic_delta"]
            ).sum(dim=1)
            sequence_index = flat["sequence_index"]
            # The no-retrieval NLL is the only value needed before the causal
            # controller gate is known.  Keep it detached; the final pass
            # below evaluates full/no-retrieval/gated variants together.
            with torch.no_grad():
                pre_action_terms = self._batched_sequence_event_nll_variants(
                    flat,
                    memory_output["semantic_base"].detach().unsqueeze(1),
                )[:, 0]

            # Posterior/mix remain useful for ownership, memory assignment,
            # prototype credit, and diagnostics, but are outside autograd.
            with torch.no_grad():
                frontier_terms = self._batched_frontier_event_nll(
                    flat, memory_output
                )
                frontier_mask = memory_output["frontier_mask"]
                prior = memory_output["frontier_mass"].detach().masked_fill(
                    ~frontier_mask, 0.0
                )
                component = (
                    prior.clamp_min(1e-12).log()
                    - frontier_terms
                    / self.wake_config.route_energy_temperature
                ).masked_fill(~frontier_mask, -torch.inf)
                mixture_rows = -torch.logsumexp(component, dim=-1)
                mixture_by_sequence, _ = self._segment_mean(
                    mixture_rows, sequence_index, sequence_count
                )
                batch_likelihood_mixture = mixture_by_sequence.mean()
                posterior = F.softmax(component, dim=-1).masked_fill(
                    ~frontier_mask, 0.0
                )
                posterior_kl_rows = (
                    posterior
                    * (
                        posterior.clamp_min(1e-12).log()
                        - prior.clamp_min(1e-12).log()
                    )
                ).masked_fill(~frontier_mask, 0.0).sum(dim=-1)
                posterior_kl_by_sequence, _ = self._segment_mean(
                    posterior_kl_rows, sequence_index, sequence_count
                )
                batch_posterior_kl = posterior_kl_by_sequence.mean()
                child_energy = self._batched_expanded_child_event_nll(
                    flat, memory_output
                )
            owner_indices, _, owner_confidence = (
                self._posterior_owner_indices_batch(
                    memory_output["frontier_node_indices"], posterior
                )
            )
            packed_memory_info = memory_output.get("packed_memory_info")
            visited_indices = memory_output.get("visited_node_indices")
            visited_mask = memory_output.get("visited_node_mask")
            can_reuse_similarity = (
                packed_memory_info is not None
                and visited_indices is not None
                and visited_mask is not None
                and "similarity" in packed_memory_info
                and "valid_mask" in packed_memory_info
            )
            if can_reuse_similarity:
                owner_slot_mask = (
                    visited_indices == owner_indices[:, None]
                ) & visited_mask
                owner_slots = owner_slot_mask.to(torch.long).argmax(dim=1)
                row = torch.arange(
                    owner_indices.size(0),
                    device=owner_indices.device,
                )
                owner_similarity = packed_memory_info["similarity"][
                    row,
                    owner_slots,
                ]
                owner_valid = packed_memory_info["valid_mask"][
                    row,
                    owner_slots,
                ]
                # The current retriever returns zero similarity for a valid
                # row with no context aliases, while the legacy novelty path
                # intentionally produced -inf for that edge case.  Preserve
                # that behavior when the optional context mask is present.
                context_valid = packed_memory_info.get("context_valid")
                if context_valid is not None:
                    owner_context_valid = context_valid[row, owner_slots]
                    owner_similarity = torch.where(
                        owner_valid & ~owner_context_valid.any(dim=-1),
                        torch.full_like(owner_similarity, -torch.inf),
                        owner_similarity,
                    )
                # Every final frontier node is also a visited node in the
                # normal packed route.  Hand-built/legacy outputs can violate
                # that invariant, though, and an async device assertion here
                # would only surface at a later unrelated scalar read (usually
                # ``bool(self.split_enabled)``).  Keep the hot path fully
                # device-side: compute the normal reuse result, then rerun the
                # original packed lookup only for missing owner rows.  The
                # empty-row call is supported and avoids a CUDA-to-host branch
                # when all owners are present.
                owner_present = owner_slot_mask.any(dim=-1)
                safe_owner_indices = owner_indices.clamp(
                    0, len(self.tree.all_node_ids) - 1
                )
                novelty, soft_count, retrieval_similarity = (
                    self.tree.episodic_memory.novelty_from_similarity(
                        owner_similarity,
                        owner_valid,
                        temperature=self.controller.novelty_temperature,
                        count_exponent=self.controller.count_exponent,
                        eps=self.controller.controller_eps,
                        count_similarity_low=(
                            self.controller.count_similarity_low
                        ),
                        count_similarity_high=(
                            self.controller.count_similarity_high
                        ),
                        count_topk=self.controller.count_topk,
                        count_saturation=self.controller.count_saturation,
                    )
                )
                missing_rows = torch.nonzero(
                    ~owner_present,
                    as_tuple=False,
                ).reshape(-1)
                fallback_novelty, fallback_count, fallback_similarity = (
                    self.tree.episodic_memory.novelty_count_packed(
                        memory_query.index_select(0, missing_rows),
                        safe_owner_indices.index_select(0, missing_rows),
                        self.tree.all_node_ids,
                        temperature=self.controller.novelty_temperature,
                        count_exponent=self.controller.count_exponent,
                        eps=self.controller.controller_eps,
                        count_similarity_low=(
                            self.controller.count_similarity_low
                        ),
                        count_similarity_high=(
                            self.controller.count_similarity_high
                        ),
                        count_topk=self.controller.count_topk,
                        count_saturation=self.controller.count_saturation,
                    )
                )
                novelty = novelty.index_copy(
                    0, missing_rows, fallback_novelty
                )
                soft_count = soft_count.index_copy(
                    0, missing_rows, fallback_count
                )
                retrieval_similarity = retrieval_similarity.index_copy(
                    0, missing_rows, fallback_similarity
                )
            else:
                novelty, soft_count, retrieval_similarity = (
                    self.tree.episodic_memory.novelty_count_packed(
                        memory_query,
                        owner_indices,
                        self.tree.all_node_ids,
                        temperature=self.controller.novelty_temperature,
                        count_exponent=self.controller.count_exponent,
                        eps=self.controller.controller_eps,
                        count_similarity_low=(
                            self.controller.count_similarity_low
                        ),
                        count_similarity_high=(
                            self.controller.count_similarity_high
                        ),
                        count_topk=self.controller.count_topk,
                        count_saturation=self.controller.count_saturation,
                    )
                )
            retrieval_norm = memory_output[
                "frontier_episodic_delta"
            ].detach().norm(dim=-1)
            retrieval_norm = (
                retrieval_norm * memory_output["r"].detach()
            ).sum(dim=-1)
            controller_output = self.controller.action_distribution_batch(
                pre_action_terms.detach(),
                novelty.detach(),
                soft_count.detach(),
                update_statistics=False,
                owner_confidence=owner_confidence.detach(),
                retrieval_similarity=retrieval_similarity.detach(),
                retrieval_residual_norm=retrieval_norm.detach(),
                working_memory_norm=z_all.new_zeros(z_all.size(0)),
                pending_write_ratio=z_all.new_zeros(z_all.size(0)),
            )
            if controller_output["logits"].requires_grad:
                controller_output["logits"].retain_grad()
            gated_theta = (
                memory_output["semantic_base"]
                + controller_output["probabilities"][:, 1, None]
                * memory_output["episodic_base"]
            )
            # All three definitions share the same event statistics.  Stack
            # them as [N, 3, P] so softplus/intensity/integral launch as one
            # variant-aware tensor computation.  Only the gated slice keeps
            # the controller gradient; full and pre slices are detached
            # targets/utility measurements.
            theta_variants = torch.stack(
                (
                    memory_output["effective_params"].theta.detach(),
                    memory_output["semantic_base"].detach(),
                    gated_theta,
                ),
                dim=1,
            )
            variant_event_terms = self._batched_sequence_event_nll_variants(
                flat,
                theta_variants,
            )
            full_retrieval_event_terms = variant_event_terms[:, 0].detach()
            pre_action_terms = variant_event_terms[:, 1].detach()
            event_terms = variant_event_terms[:, 2]
            batch_prediction_sum = event_terms.sum()
            batch_prediction = batch_prediction_sum / batch_event_count
            # Epochs 1/2/3 use 0.1, 0.0667, 0.0333; epoch 4+ is utility-only.
            # Real counterfactuals never leak across heads: every label is masked.
            bootstrap_weight = max(0.0, 0.1 * (4 - effective_epoch) / 3.0)
            if self.training_config.controller_only_finetune:
                bootstrap_weight = 0.0
            controller_loss = batch_prediction * 0.0
            if bootstrap_weight > 0.0:
                controller_loss = self.controller.supervision_loss(
                    controller_output,
                    utility_targets=None,
                    bootstrap_weight=bootstrap_weight,
                    write_cost=self.wake_config.lambda_write,
                    split_cost=self.wake_config.controller_split_cost,
                    entropy_weight=self.wake_config.controller_entropy_weight,
                )

            if self.controller.utility_stage_enabled:
                train_retrieve = bool(
                    not self.training_config.controller_only_finetune
                    or "retrieve" in self.training_config.controller_train_heads
                )
                retrieval_utility = (
                    pre_action_terms - full_retrieval_event_terms
                    - self.wake_config.controller_retrieve_cost
                ).detach()
                retrieval_targets = controller_output["probabilities"].new_zeros(
                    controller_output["probabilities"].shape
                )
                retrieval_mask = torch.zeros_like(
                    retrieval_targets, dtype=torch.bool
                )
                retrieval_values = retrieval_targets.clone()
                retrieval_values[:, 1] = retrieval_utility
                retrieval_targets[:, 1] = self.controller.utility_target(
                    retrieval_utility, action_index=1, cost_margin=0.0,
                    update_statistics=train_retrieve,
                )
                retrieval_mask[:, 1] = train_retrieve
                controller_loss = controller_loss + self.controller.masked_utility_loss(
                    controller_output, retrieval_targets, retrieval_mask,
                    retrieval_values,
                    false_positive_weight=self.wake_config.controller_false_positive_weight,
                )

                inputs = torch.stack((
                    pre_action_terms.detach(), novelty.detach(), soft_count.detach(),
                    owner_confidence.detach(), retrieval_similarity.detach(),
                    retrieval_norm.detach(), z_all.new_zeros(z_all.size(0)),
                    z_all.new_zeros(z_all.size(0)),
                ), dim=-1)
                if train_retrieve:
                    sequence_lengths = [
                        int(sequence["times"].numel())
                        for sequence in moved_sequences
                    ]
                    sequence_rows_cpu = sequence_index.detach().cpu().tolist()
                    cluster_ids = [
                        int(torch.as_tensor(
                            sequence.get("cluster_id", -1)
                        ).item())
                        for sequence in moved_sequences
                    ]
                    source_indices = [
                        int(torch.as_tensor(
                            sequence.get("source_index", -1)
                        ).item())
                        for sequence in moved_sequences
                    ]
                    event_indices = [
                        event_index
                        for length in sequence_lengths
                        for event_index in range(length)
                    ]
                    self.controller_utility_replay.add_batch(
                        inputs=inputs,
                        utility=retrieval_values,
                        target=retrieval_targets,
                        label_mask=retrieval_mask,
                        propensity=retrieval_values.new_ones(4).expand_as(
                            retrieval_values
                        ),
                        gate=controller_output["probabilities"].detach(),
                        cluster_ids=[
                            cluster_ids[row] for row in sequence_rows_cpu
                        ],
                        source_indices=[
                            source_indices[row] for row in sequence_rows_cpu
                        ],
                        event_indices=event_indices,
                        owner_indices=owner_indices.detach(),
                        node_ids=self.tree.all_node_ids,
                        action=1,
                    )

                replay_payload = self.controller_utility_replay.sample_packed(
                    self.wake_config.controller_replay_batch_sizes
                )
                if replay_payload is not None:
                    # ``sample_packed`` returns inputs, target, utility,
                    # propensity and mask in one CPU payload.  A single H2D
                    # copy replaces the old per-row/per-field transfers.
                    replay_payload = replay_payload.to(
                        self.device,
                        non_blocking=True,
                    )
                    replay_inputs = replay_payload[:, :8]
                    targets = replay_payload[:, 8:12]
                    utilities = replay_payload[:, 12:16]
                    propensities = replay_payload[:, 16:20]
                    masks = replay_payload[:, 20:24].bool()
                    replay_output = self.controller.action_distribution_batch(
                        replay_inputs[:, 0], replay_inputs[:, 1], replay_inputs[:, 2],
                        update_statistics=False,
                        owner_confidence=replay_inputs[:, 3],
                        retrieval_similarity=replay_inputs[:, 4],
                        retrieval_residual_norm=replay_inputs[:, 5],
                        working_memory_norm=replay_inputs[:, 6],
                        pending_write_ratio=replay_inputs[:, 7],
                    )
                    weights = self.controller.normalized_inverse_propensity(
                        propensities, masks
                    )
                    controller_loss = controller_loss + self.controller.masked_utility_loss(
                        replay_output, targets, masks, utilities,
                        importance_weight=weights,
                        false_positive_weight=self.wake_config.controller_false_positive_weight,
                    )
                if self.training_config.controller_write_ranking:
                    ranking = self.controller_utility_replay.sample_write_ranking(
                        max_rows=96, max_pairs=192
                    )
                    ranking_rows = ranking["rows"]
                    if ranking_rows:
                        ranking_inputs = torch.stack([
                            row["inputs"].to(self.device) for row in ranking_rows
                        ])
                        ranking_output = self.controller.action_distribution_batch(
                            ranking_inputs[:, 0], ranking_inputs[:, 1], ranking_inputs[:, 2],
                            update_statistics=False,
                            owner_confidence=ranking_inputs[:, 3],
                            retrieval_similarity=ranking_inputs[:, 4],
                            retrieval_residual_norm=ranking_inputs[:, 5],
                            working_memory_norm=ranking_inputs[:, 6],
                            pending_write_ratio=ranking_inputs[:, 7],
                        )
                        ranking_loss, ranking_metrics = self.controller.write_ranking_loss(
                            ranking_output, ranking_rows, ranking["pairs"]
                        )
                        controller_loss = controller_loss + ranking_loss
                        self._last_write_ranking_metrics = ranking_metrics
            local = self._batched_local_frontier_objective(
                memory_output,
                child_energy,
                sequence_index,
                sequence_count,
            )
            regional = self._regional_probe_objective(
                memory_output,
                posterior,
                sequence_index,
                sequence_count,
                routed_z,
                flat,
            )
            objective = (
                batch_prediction
                + self.wake_config.lambda_route_distill
                * local["distill"]
                - self.wake_config.lambda_route_mi
                * local["mutual_information"]
                + self.wake_config.lambda_route_balance
                * local["balance_kl"]
                + self.wake_config.lambda_route_probe
                * regional["loss"]
                + controller_loss
            )
            _assert_finite_without_cuda_sync(
                objective,
                "global frontier objective became non-finite",
            )
            objective.backward()
            logit_gradient = controller_output["logits"].grad
            if logit_gradient is None:
                controller_trainable = any(
                    parameter.requires_grad
                    for parameter in self.controller.parameters()
                )
                if controller_trainable:
                    raise RuntimeError("controller logits received no gradient")
                # Heuristic-controller ablations intentionally freeze the
                # learned controller.  Keep the epoch diagnostics defined
                # while allowing the rest of the tree objective to train.
                logit_gradient = torch.zeros_like(controller_output["logits"])
            head_norms = logit_gradient.detach().double().norm(dim=0)
            controller_gradient_values = [
                parameter.grad.detach().double().norm()
                for parameter in self.controller.parameters()
                if parameter.grad is not None
            ]
            controller_gradient = (
                torch.stack(controller_gradient_values).norm()
                if controller_gradient_values
                else controller_output["logits"].new_zeros((), dtype=torch.float64)
            )
            # Keep diagnostics on the device until one compact transfer.  In
            # the old path each head/min/max value called ``.cpu()``
            # separately, serializing the CUDA stream during every Global
            # batch even though these values only feed epoch-level logs.
            controller_status = torch.cat((
                head_norms,
                controller_gradient.reshape(1),
            )).cpu().tolist()
            head_status = torch.tensor(
                controller_status[: len(Action)], dtype=torch.float64
            )
            max_controller_head_grad_norms = torch.maximum(
                max_controller_head_grad_norms, head_status
            )
            min_controller_head_grad_norm = min(
                min_controller_head_grad_norm,
                min(controller_status[: len(Action)]),
            )
            max_controller_grad_norm = max(
                max_controller_grad_norm,
                float(controller_status[-1]),
            )
            global_progress.update(sequence_count)
            gradient_norm = clip_grad_norm_finite(
                self._named_optimized_parameters(),
                self.training_config.grad_clip,
                context="cross-sequence global update",
            )
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            self.optimizer.step()
            reliability = child_teacher_reliability(
                local["energy_teacher"],
                local["student"],
                memory_output["expanded_node_indices"].detach(),
                memory_output["expanded_mask"].detach(),
                node_count=len(self.tree.all_node_ids),
            )
            decay = self.wake_config.route_encoder_reliability_decay
            reliability_status = torch.stack((
                reliability["reliability"],
                reliability["teacher_confidence"],
                reliability["teacher_student_js"],
                reliability["teacher_student_alignment"],
            )).detach().cpu().tolist()
            (
                observed_reliability,
                observed_teacher_confidence,
                observed_teacher_student_js,
                observed_teacher_student_alignment,
            ) = reliability_status
            if not self.training_config.controller_only_finetune:
                self.encoder_routing_reliability = (
                    decay * self.encoder_routing_reliability
                    + (1.0 - decay) * observed_reliability
                )
                self.last_teacher_confidence = observed_teacher_confidence
                self.last_teacher_student_js = observed_teacher_student_js
                self.last_teacher_student_alignment = (
                    observed_teacher_student_alignment
                )
            total_teacher_confidence += observed_teacher_confidence
            total_teacher_student_js += observed_teacher_student_js
            total_teacher_student_alignment += (
                observed_teacher_student_alignment
            )
            reliability_updates += 1
            if not self.training_config.controller_only_finetune:
                self.tree.frontier_routing.prototypes.update_frontier_responsibility(
                    z_all.detach(),
                    memory_output["frontier_node_indices"].detach(),
                    posterior.detach(),
                    frontier_mask.detach(),
                )
                self.tree.frontier_routing.update_expansion_gain(
                    memory_output["expanded_node_indices"].detach(),
                    local["observed_gain"],
                    memory_output["expanded_mask"].detach(),
                )
                if regional["node_indices"].numel():
                    regional_mask = torch.ones_like(
                        regional["node_indices"], dtype=torch.bool
                    )
                    self.tree.frontier_routing.update_expansion_gain(
                        regional["node_indices"].detach(),
                        regional["refinement_gain"].clamp_min(0.0).detach(),
                        regional_mask,
                    )
            optimizer_steps += 1
            total_encoder_grad_scale += encoder_grad_scale

            batch_values = torch.stack(
                [
                    batch_prediction_sum.detach(),
                    batch_likelihood_mixture.detach(),
                    batch_posterior_kl.detach(),
                    local["distill"].detach(),
                    local["balance_kl"].detach(),
                    local["mutual_information"].detach(),
                    local["conditional_entropy"].detach(),
                    local["marginal_entropy"].detach(),
                    regional["loss"].detach(),
                    regional["router_loss"].detach(),
                    regional["expand_loss"].detach(),
                    regional["leaf_loss"].detach(),
                    regional["regions"].detach(),
                    controller_loss.detach(),
                ]
            ).cpu().tolist()
            (
                batch_prediction_sum_value,
                batch_likelihood_mixture_value,
                posterior_kl_value,
                distill_value,
                prior_kl_value,
                mutual_information_value,
                conditional_entropy_value,
                marginal_entropy_value,
                probe_loss_value,
                probe_router_loss_value,
                probe_expand_loss_value,
                probe_leaf_loss_value,
                probe_regions_value,
                controller_loss_value,
            ) = batch_values
            batch_prediction_value = (
                batch_prediction_sum_value / batch_event_count
            )
            batch_loss = (
                batch_prediction_value
                + self.wake_config.lambda_route_distill
                * distill_value
                - self.wake_config.lambda_route_mi
                * mutual_information_value
                + self.wake_config.lambda_route_balance
                * prior_kl_value
                + self.wake_config.lambda_route_probe
                * probe_loss_value
                + controller_loss_value
            )
            total_sequences += sequence_count
            total_events += batch_event_count
            total_loss += batch_loss * sequence_count
            total_prediction += batch_prediction_sum_value
            total_likelihood_mixture += (
                batch_likelihood_mixture_value * sequence_count
            )
            total_posterior_kl += posterior_kl_value * sequence_count
            total_distill += distill_value * sequence_count
            total_prior_kl += prior_kl_value * sequence_count
            total_mi += mutual_information_value * sequence_count
            total_conditional_entropy += (
                conditional_entropy_value * sequence_count
            )
            total_marginal_entropy += (
                marginal_entropy_value * sequence_count
            )
            total_probe_loss += probe_loss_value * sequence_count
            total_probe_router_loss += (
                probe_router_loss_value * sequence_count
            )
            total_probe_expand_loss += (
                probe_expand_loss_value * sequence_count
            )
            total_probe_leaf_loss += probe_leaf_loss_value * sequence_count
            total_probe_regions += probe_regions_value
            total_controller_loss += (
                controller_loss_value * sequence_count
            )
            if regional["expand_probability"].numel():
                probe_status = torch.stack((
                    regional["expand_probability"].mean(),
                    regional["expand_target"].mean(),
                    regional["refinement_gain"].mean(),
                    regional["assignment_confidence"].mean(),
                )).detach().cpu().tolist()
                (
                    probe_expand_probability,
                    probe_expand_target,
                    probe_refinement_gain,
                    probe_assignment_confidence,
                ) = probe_status
                total_probe_expand_probability += (
                    probe_expand_probability * probe_regions_value
                )
                total_probe_expand_target += (
                    probe_expand_target * probe_regions_value
                )
                total_probe_refinement_gain += (
                    probe_refinement_gain * probe_regions_value
                )
                total_probe_assignment_confidence += (
                    probe_assignment_confidence * probe_regions_value
                )
        global_progress.close()

        sequence_denominator = max(total_sequences, 1)
        event_denominator = max(total_events, 1)
        return {
            "loss": total_loss / sequence_denominator,
            "prediction_nll": total_prediction / event_denominator,
            "likelihood_mixture": (
                total_likelihood_mixture / sequence_denominator
            ),
            "balance_kl": total_prior_kl / sequence_denominator,
            "prior_kl": total_prior_kl / sequence_denominator,
            "posterior_kl": total_posterior_kl / sequence_denominator,
            "branch_distill": total_distill / sequence_denominator,
            "mutual_information": total_mi / sequence_denominator,
            "conditional_entropy": (
                total_conditional_entropy / sequence_denominator
            ),
            "marginal_entropy": (
                total_marginal_entropy / sequence_denominator
            ),
            "regional_probe_loss": (
                total_probe_loss / sequence_denominator
            ),
            "regional_probe_router_loss": (
                total_probe_router_loss / sequence_denominator
            ),
            "regional_probe_expand_loss": (
                total_probe_expand_loss / sequence_denominator
            ),
            "regional_probe_leaf_loss": (
                total_probe_leaf_loss / sequence_denominator
            ),
            "regional_probe_regions": total_probe_regions,
            "controller_loss": (
                total_controller_loss / sequence_denominator
            ),
            "controller_grad_norm": max_controller_grad_norm,
            "controller_head_grad_norms": {
                action.value: float(max_controller_head_grad_norms[index])
                for index, action in enumerate(Action)
            },
            "controller_min_head_grad_norm": (
                0.0 if math.isinf(min_controller_head_grad_norm)
                else min_controller_head_grad_norm
            ),
            "controller_utility_stage_enabled": bool(
                self.controller.utility_stage_enabled
            ),
            "regional_probe_expand_probability": (
                total_probe_expand_probability
                / max(total_probe_regions, 1.0)
            ),
            "regional_probe_expand_target": (
                total_probe_expand_target / max(total_probe_regions, 1.0)
            ),
            "regional_probe_refinement_gain": (
                total_probe_refinement_gain
                / max(total_probe_regions, 1.0)
            ),
            "regional_probe_assignment_confidence": (
                total_probe_assignment_confidence
                / max(total_probe_regions, 1.0)
            ),
            "samples": total_sequences,
            "sequences": total_sequences,
            "events": total_events,
            "batches": optimizer_steps,
            "optimizer_steps": optimizer_steps,
            "max_gradient_norm": max_gradient_norm,
            "encoder_route_grad_scale": (
                total_encoder_grad_scale / max(optimizer_steps, 1)
            ),
            # Backward-compatible diagnostics key.
            "mi_encoder_grad_scale": (
                total_encoder_grad_scale / max(optimizer_steps, 1)
            ),
            "encoder_route_gate": self.encoder_routing_reliability,
            "teacher_confidence": (
                total_teacher_confidence / max(reliability_updates, 1)
            ),
            "teacher_student_js": (
                total_teacher_student_js / max(reliability_updates, 1)
            ),
            "teacher_student_alignment": (
                total_teacher_student_alignment
                / max(reliability_updates, 1)
            ),
            "reliability_updates": reliability_updates,
            "steps": optimizer_steps,
        }
