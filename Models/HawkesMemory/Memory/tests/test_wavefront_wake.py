import copy
import unittest

import torch

from HawkesBackbone import (
    HAWKES_HISTORY_STATS_KEY,
    HAWKES_INTERVAL_STATS_KEY,
    HawkesFamily,
)
from LatentHawkesTree import HawkesTree
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory
from MemoryResiduals.WorkingMemory import WorkingMemoryAdapter
from Train.Train import (
    CausalPrefixEncoder,
    MemoryTreeTrainer,
    WakeObjectiveConfig,
)
from Wake.SequentialController import Controller


class MaskedWavefrontWakeTests(unittest.TestCase):
    @staticmethod
    def _trainer(seed: int = 0) -> MemoryTreeTrainer:
        torch.manual_seed(seed)
        hawkes = HawkesFamily(
            2,
            1,
            decays=torch.tensor([1.0]),
        )
        tree = HawkesTree(
            3,
            4,
            2,
            1,
            init_depth=1,
            memory_key_dim=3,
        )
        encoder = CausalPrefixEncoder(
            2,
            3,
            type_dim=4,
            hidden_dim=8,
        )
        return MemoryTreeTrainer(
            tree,
            hawkes,
            encoder,
            wake=WakeObjectiveConfig(
                route_balance_batch_size=2,
                write_horizon=2,
            ),
            device="cpu",
        )

    @staticmethod
    def _cached(
        trainer: MemoryTreeTrainer,
        times,
        types,
    ):
        sequence = {
            "times": torch.tensor(times),
            "types": torch.tensor(types),
        }
        trainer.hawkes.prepare_sequence_cache(sequence, inplace=True)
        return sequence

    @staticmethod
    def _run_prepared_batch(
        trainer: MemoryTreeTrainer,
        sequences,
    ):
        prepared = next(
            trainer._iter_masked_wavefront_batches(
                sequences,
                list(range(len(sequences))),
            )
        )
        return trainer.train_wake_batch(
            sequences=prepared["sequences"],
            sequence_indices=prepared["sequence_indices"],
            z_flat=prepared["z_flat"],
            projected_flat=prepared["projected_flat"],
            query_flat=prepared["query_flat"],
            frontier_static_cache=prepared["frontier_static_cache"],
            frontier_flat=prepared["frontier_flat"],
            frontier_rows=prepared["frontier_rows"],
            flat=prepared["flat"],
        )

    @staticmethod
    def _run_prepared_serial(
        trainer: MemoryTreeTrainer,
        sequences,
    ):
        prepared = next(
            trainer._iter_masked_wavefront_batches(
                sequences,
                list(range(len(sequences))),
            )
        )
        results = []
        offset = 0
        for row, sequence in enumerate(prepared["sequences"]):
            length = int(prepared["lengths"][row])
            end = offset + length
            results.append(trainer.train_wake_sequence(
                sequence,
                sequence_index=prepared["sequence_indices"][row],
                precomputed_z=prepared["z_flat"][offset:end],
                precomputed_projected_z=prepared["projected_flat"][offset:end],
                precomputed_memory_query=prepared["query_flat"][offset:end],
                frontier_static_cache=prepared["frontier_static_cache"],
                precomputed_frontier=prepared["frontier_flat"].slice(
                    offset,
                    end,
                ),
                precomputed_frontier_rows=prepared["frontier_rows"][
                    offset:end
                ],
            ))
            offset = end
        return results

    @staticmethod
    def _force_memorize_policy(trainer: MemoryTreeTrainer) -> None:
        """Make the parity fixture exercise physical write commits."""
        with torch.no_grad():
            trainer.controller.bias_assimilate.fill_(-20.0)
            trainer.controller.bias_retrieve.fill_(10.0)
            trainer.controller.bias_memorize.fill_(20.0)
            trainer.controller.bias_queue_split.fill_(-20.0)
        trainer.controller.exploration_rate = 0.0

    def _assert_memory_state_equal(
        self,
        expected: MemoryTreeTrainer,
        actual: MemoryTreeTrainer,
    ) -> None:
        expected_memory = expected.tree.episodic_memory
        actual_memory = actual.tree.episodic_memory
        self.assertEqual(
            expected_memory._age_clock,
            actual_memory._age_clock,
        )
        self.assertEqual(
            set(expected_memory.banks),
            set(actual_memory.banks),
        )
        tensor_fields = (
            "keys",
            "context_keys",
            "context_valid",
            "context_support",
            "deltas",
            "write_quality",
            "queue_weight",
            "law_keys",
            "support",
            "quality_mass",
            "split_mass",
            "mode_ids",
            "mode_compressed",
            "usage",
            "cycle_usage",
            "stale_cycles",
            "age",
        )
        for node_id in expected_memory.banks:
            expected_bank = expected_memory.banks[node_id]
            actual_bank = actual_memory.banks[node_id]
            self.assertEqual(len(expected_bank), len(actual_bank))
            self.assertEqual(
                expected_bank._next_mode_id,
                actual_bank._next_mode_id,
            )
            self.assertEqual(
                expected_bank._age_reference_clock,
                actual_bank._age_reference_clock,
            )
            for field in tensor_fields:
                expected_value = getattr(expected_bank, field)
                actual_value = getattr(actual_bank, field)
                if expected_value.is_floating_point():
                    self.assertTrue(
                        torch.allclose(
                            expected_value,
                            actual_value,
                            atol=1e-6,
                            rtol=1e-6,
                        ),
                        f"bank field differs: {node_id}.{field}",
                    )
                else:
                    self.assertTrue(
                        torch.equal(expected_value, actual_value),
                        f"bank field differs: {node_id}.{field}",
                    )
            self.assertEqual(
                len(expected_bank.windows),
                len(actual_bank.windows),
            )
            for expected_window, actual_window in zip(
                expected_bank.windows,
                actual_bank.windows,
            ):
                if expected_window is None or actual_window is None:
                    self.assertIs(expected_window, actual_window)
                    continue
                self.assertTrue(torch.equal(
                    expected_window.times,
                    actual_window.times,
                ))
                self.assertTrue(torch.equal(
                    expected_window.types,
                    actual_window.types,
                ))
                self.assertEqual(
                    (expected_window.node_id, expected_window.start_idx,
                     expected_window.end_idx),
                    (actual_window.node_id, actual_window.start_idx,
                     actual_window.end_idx),
                )
        self.assertTrue(torch.allclose(
            expected.tree.working_memory.delta,
            actual.tree.working_memory.delta,
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_batched_working_update_matches_independent_rows(self):
        torch.manual_seed(401)
        adapter = WorkingMemoryAdapter(
            param_dim=7,
            rho=0.73,
            eta=0.04,
            clip_grad_norm=0.6,
        )
        state = torch.randn(4, 7)
        original = state.clone()
        active = torch.tensor([0, 2, 3])
        gradients = torch.randn(3, 7)
        probabilities = torch.tensor([0.2, 0.8, 0.5])

        adapter.update_batch_rows(
            state,
            active,
            gradients,
            adaptation_probability=probabilities,
        )
        expected = original.clone()
        for local_row, state_row in enumerate(active.tolist()):
            scalar = WorkingMemoryAdapter(
                param_dim=7,
                rho=adapter.rho,
                eta=adapter.eta,
                clip_grad_norm=adapter.clip_grad_norm,
            )
            scalar.delta.copy_(original[state_row])
            scalar.update_from_gradient(
                gradients[local_row],
                adaptation_probability=probabilities[local_row],
            )
            expected[state_row] = scalar.delta

        self.assertTrue(torch.allclose(state, expected, atol=1e-7))
        self.assertTrue(torch.equal(state[1], original[1]))

    def test_packed_novelty_matches_scalar_controller(self):
        torch.manual_seed(409)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "left", "right")
        memory.sync_nodes(node_ids)
        for _ in range(3):
            memory.add_memory(
                "left",
                torch.randn(3),
                torch.randn(memory.param_dim),
            )
        memory.add_memory(
            "right",
            torch.randn(3),
            torch.randn(memory.param_dim),
        )
        controller = Controller(
            nll_fn=HawkesFamily(
                2,
                1,
                decays=torch.tensor([1.0]),
            ),
            episodic_memory=memory,
        )
        query = torch.randn(3, 3)
        owner_indices = torch.tensor([0, 1, 2])
        packed = memory.novelty_count_packed(
            query,
            owner_indices,
            node_ids,
            temperature=controller.novelty_temperature,
            count_exponent=controller.count_exponent,
            eps=controller.controller_eps,
            count_similarity_low=controller.count_similarity_low,
            count_similarity_high=controller.count_similarity_high,
            count_topk=controller.count_topk,
            count_saturation=controller.count_saturation,
        )
        reference = [
            torch.stack(values)
            for values in zip(*[
                controller.leaf_novelty_count(query[row], node_ids[node])
                for row, node in enumerate(owner_indices.tolist())
            ])
        ]
        for actual, expected in zip(packed, reference):
            self.assertTrue(torch.allclose(
                actual,
                expected,
                atol=1e-6,
                rtol=1e-6,
            ))

    def test_scalar_wake_reuses_packed_owner_similarity(self):
        trainer = self._trainer(seed=410)
        sequence = self._cached(
            trainer,
            [0.1, 0.4, 0.9],
            [0, 1, 0],
        )

        def unexpected_second_cosine(*_args, **_kwargs):
            raise AssertionError(
                "scalar Wake must reuse similarity from the packed read"
            )

        trainer.controller.leaf_novelty_count = unexpected_second_cosine
        result = trainer.train_wake_sequence(sequence)

        self.assertEqual(result["event_count"], 3)

    def test_raw_theta_likelihood_and_gradient_match_effective_path(self):
        trainer = self._trainer(seed=411)
        sequence = trainer._move_sequence(
            self._cached(trainer, [0.1, 0.4, 0.9], [0, 1, 0])
        )
        times = sequence["times"]
        previous = torch.cat([times.new_zeros(1), times[:-1]])
        flat = {
            "types": sequence["types"],
            "duration": (times - previous).clamp_min(0.0),
            HAWKES_HISTORY_STATS_KEY: sequence[HAWKES_HISTORY_STATS_KEY],
            HAWKES_INTERVAL_STATS_KEY: sequence[HAWKES_INTERVAL_STATS_KEY],
        }
        theta = torch.randn(
            times.numel(),
            trainer.tree.param_dim,
            requires_grad=True,
        )
        effective = trainer._effective_parameters_from_theta(theta)
        expected_nll, expected_grad = (
            trainer._batched_sequence_event_nll_and_grad(flat, effective)
        )
        autodiff_grad = torch.autograd.grad(
            expected_nll.sum(),
            theta,
            retain_graph=True,
        )[0]
        actual_nll, actual_grad = (
            trainer._batched_raw_theta_event_nll_and_grad(flat, theta)
        )

        self.assertTrue(torch.allclose(actual_nll, expected_nll, atol=1e-7))
        self.assertTrue(torch.allclose(actual_grad, expected_grad, atol=1e-7))
        self.assertTrue(torch.allclose(actual_grad, autodiff_grad, atol=1e-6))

    def test_wavefront_batch_size_one_matches_streaming_wake(self):
        base = self._trainer(seed=419)
        streaming = copy.deepcopy(base)
        wavefront = copy.deepcopy(base)
        sequence_stream = self._cached(
            streaming,
            [0.1, 0.4, 0.9],
            [0, 1, 0],
        )
        sequence_wave = self._cached(
            wavefront,
            [0.1, 0.4, 0.9],
            [0, 1, 0],
        )
        expected = streaming.train_wake_sequence(sequence_stream)
        actual = self._run_prepared_batch(
            wavefront,
            [sequence_wave],
        )[0]

        for key in (
            "prediction_nll",
            "wm_penalty",
            "write_penalty",
            "max_gradient_norm",
            "mean_novelty",
            "mean_max_similarity",
            "posterior_entropy",
            "prior_posterior_kl",
        ):
            self.assertAlmostEqual(expected[key], actual[key], places=6)
        self.assertEqual(
            expected["action_counts"],
            actual["action_counts"],
        )
        self.assertEqual(
            expected["memory_assignment_counts"],
            actual["memory_assignment_counts"],
        )

    def test_variable_length_wake_preserves_sequence_event_order(self):
        trainer = self._trainer(seed=421)
        sequences = [
            self._cached(trainer, [0.1, 0.4, 0.9], [0, 1, 0]),
            self._cached(trainer, [0.2, 0.5], [1, 0]),
        ]
        tree_calls = []
        hook = trainer.tree.register_forward_hook(
            lambda *_: tree_calls.append(1)
        )
        results = self._run_prepared_batch(trainer, sequences)
        hook.remove()

        self.assertEqual(
            [result["event_count"] for result in results],
            [3, 2],
        )
        # Prefix/frontier preparation is batched, but the stateful tree call
        # follows strict sequence/event order.
        self.assertEqual(len(tree_calls), 5)
        self.assertEqual(
            [
                sum(result["action_counts"].values())
                for result in results
            ],
            [3, 2],
        )
        for result in results:
            self.assertEqual(
                result["sequence_responsibility"].shape,
                (len(trainer.tree.leaf_ids),),
            )

    def test_legacy_snapshot_retrieval_is_chunked_before_time_loop(self):
        trainer = self._trainer(seed=423)
        trainer.wake_config.retrieval_microbatch = 2
        sequences = [
            self._cached(trainer, [0.1, 0.4, 0.9], [0, 1, 0]),
            self._cached(trainer, [0.2, 0.5], [1, 0]),
        ]
        calls = []
        memory = trainer.tree.episodic_memory
        original_read_packed = memory.read_packed

        def counted_read_packed(*args, **kwargs):
            query = kwargs["query"] if "query" in kwargs else args[0]
            calls.append(query.size(0))
            return original_read_packed(*args, **kwargs)

        memory.read_packed = counted_read_packed
        try:
            prepared = next(
                trainer._iter_masked_wavefront_batches(
                    sequences,
                    list(range(len(sequences))),
                )
            )
            results = trainer._train_wake_batch_snapshot(
                sequences=prepared["sequences"],
                sequence_indices=prepared["sequence_indices"],
                z_flat=prepared["z_flat"],
                projected_flat=prepared["projected_flat"],
                query_flat=prepared["query_flat"],
                frontier_static_cache=prepared["frontier_static_cache"],
                frontier_flat=prepared["frontier_flat"],
                frontier_rows=prepared["frontier_rows"],
                flat=prepared["flat"],
            )
        finally:
            memory.read_packed = original_read_packed

        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(
            [result["event_count"] for result in results],
            [3, 2],
        )

    def test_batched_wake_matches_serial_memory_semantics_for_small_batches(self):
        specs = (
            (
                ([0.1, 0.4, 0.9, 1.4], [0, 1, 0, 1]),
                ([0.2, 0.5, 1.0], [1, 0, 1]),
            ),
            (
                ([0.1, 0.4, 0.9, 1.4], [0, 1, 0, 1]),
                ([0.2, 0.5, 1.0], [1, 0, 1]),
                ([0.15, 0.7, 1.2, 1.8, 2.1], [1, 1, 0, 1, 0]),
                ([0.3, 0.8], [0, 1]),
            ),
        )
        metric_keys = (
            "prediction_nll",
            "wm_penalty",
            "write_penalty",
            "max_gradient_norm",
            "mean_novelty",
            "mean_max_similarity",
            "posterior_entropy",
            "prior_posterior_kl",
            "accepted_write_count",
            "append_count",
            "refresh_count",
            "write_count",
            "write_decision_count",
            "write_probe_count",
            "write_gate_pass_count",
            "write_utility_pass_count",
        )
        for batch_number, batch_specs in zip((2, 4), specs):
            with self.subTest(batch_size=batch_number):
                base = self._trainer(seed=431 + batch_number)
                self._force_memorize_policy(base)
                streaming = copy.deepcopy(base)
                batched = copy.deepcopy(base)
                streaming_sequences = [
                    self._cached(streaming, times, types)
                    for times, types in batch_specs
                ]
                batched_sequences = [
                    self._cached(batched, times, types)
                    for times, types in batch_specs
                ]

                torch.manual_seed(9000 + batch_number)
                expected = self._run_prepared_serial(
                    streaming,
                    streaming_sequences,
                )
                torch.manual_seed(9000 + batch_number)
                actual = self._run_prepared_batch(
                    batched,
                    batched_sequences,
                )

                self.assertEqual(len(expected), batch_number)
                self.assertEqual(len(actual), batch_number)
                self.assertGreater(
                    sum(result["accepted_write_count"] for result in expected),
                    0,
                )
                for expected_result, actual_result in zip(expected, actual):
                    for key in metric_keys:
                        if isinstance(expected_result[key], float):
                            self.assertAlmostEqual(
                                expected_result[key],
                                actual_result[key],
                                places=6,
                            )
                        else:
                            self.assertEqual(
                                expected_result[key],
                                actual_result[key],
                            )
                    self.assertEqual(
                        expected_result["actions"],
                        actual_result["actions"],
                    )
                    self.assertEqual(
                        expected_result["action_counts"],
                        actual_result["action_counts"],
                    )
                    self.assertEqual(
                        expected_result["memory_assignment_counts"],
                        actual_result["memory_assignment_counts"],
                    )
                    self.assertTrue(torch.allclose(
                        expected_result["sequence_responsibility"],
                        actual_result["sequence_responsibility"],
                        atol=1e-6,
                        rtol=1e-6,
                    ))
                self._assert_memory_state_equal(streaming, batched)


if __name__ == "__main__":
    unittest.main()
