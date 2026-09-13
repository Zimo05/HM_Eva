import copy
import unittest

import torch

from MemoryResiduals.MemoryBank import (
    SmoothSparseRetriever,
    entmax15_1d,
    entmax15_masked,
)
from MemoryResiduals.EpisodicMemory import TreeEpisodicMemory


class MaskedEntmaxTests(unittest.TestCase):
    def test_packed_read_with_no_active_memories_stays_zero(self):
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        query = torch.randn(2, 3)
        node_ids = ("root", "root_L")
        indices = torch.tensor([[0, 1], [0, -1]])
        mask = indices >= 0

        retrieved, info = memory.read_packed(
            query,
            indices,
            mask,
            node_ids,
            update_state=False,
        )

        self.assertEqual(retrieved.shape, (2, 2, memory.param_dim))
        self.assertEqual(info["alpha"].shape[:2], (2, 2))
        self.assertEqual(float(retrieved.detach().abs().sum()), 0.0)
        self.assertEqual(float(info["alpha"].detach().abs().sum()), 0.0)

    def test_packed_retrieval_credit_matches_python_path_reference(self):
        torch.manual_seed(83)
        reference = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L", "root_R")
        for node_id in node_ids:
            for _ in range(2):
                reference.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.randn(reference.param_dim),
                )
        packed_memory = copy.deepcopy(reference)
        query = torch.randn(1, 3)
        visited = torch.tensor([[0, 1, 2]])
        visited_mask = torch.ones_like(visited, dtype=torch.bool)
        _, packed_info = packed_memory.read_packed(
            query,
            visited,
            visited_mask,
            node_ids,
            update_state=False,
        )
        _, reference_info = reference.read_packed(
            query,
            visited,
            visited_mask,
            node_ids,
            update_state=False,
        )
        python_info = [{
            node_id: {
                "alpha": reference_info["alpha"][0, node_index]
            }
            for node_index, node_id in enumerate(node_ids)
        }]
        weights = torch.tensor([[0.4, 0.6]])
        retrieve_probability = torch.tensor(0.7)
        reference.credit_retrieval(
            info_by_batch=python_info,
            leaf_paths=[
                ("root", "root_L"),
                ("root", "root_R"),
            ],
            routing_weights=weights,
            retrieval_probability=retrieve_probability,
        )
        packed_memory.credit_retrieval_packed(
            alpha=packed_info["alpha"],
            visited_node_indices=visited,
            visited_node_mask=visited_mask,
            path_incidence=torch.tensor([[
                [True, True, False],
                [True, False, True],
            ]]),
            routing_weights=weights,
            retrieval_probability=retrieve_probability,
            node_ids=node_ids,
        )
        for node_id in node_ids:
            self.assertTrue(torch.allclose(
                reference.banks[node_id].cycle_usage,
                packed_memory.banks[node_id].cycle_usage,
                atol=1e-7,
                rtol=1e-6,
            ))

    def test_packed_bank_mirror_is_reused_and_invalidated_by_write(self):
        torch.manual_seed(89)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L")
        memory.add_memory(
            "root", torch.randn(3), torch.randn(memory.param_dim)
        )
        query = torch.randn(2, 3)
        indices = torch.tensor([[0, 1], [0, -1]])
        mask = indices >= 0

        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 1)
        memory.step_age()
        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 1)

        memory.add_memory(
            "root_L", torch.randn(3), torch.randn(memory.param_dim)
        )
        memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        self.assertEqual(memory._packed_mirror_rebuilds, 2)

    def test_packed_read_snapshot_freezes_age_and_reuses_mirror(self):
        torch.manual_seed(93)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        memory.add_memory(
            "root", torch.randn(3), torch.randn(memory.param_dim)
        )
        query = torch.randn(1, 3)
        indices = torch.tensor([[0]])
        mask = torch.ones_like(indices, dtype=torch.bool)
        snapshot = memory.prepare_packed_read_snapshot(
            ("root",),
            query,
        )

        before, before_info = memory.read_packed(
            query,
            indices,
            mask,
            ("root",),
            update_state=False,
            snapshot=snapshot,
        )
        memory.step_age(5)
        after, after_info = memory.read_packed(
            query,
            indices,
            mask,
            ("root",),
            update_state=False,
            snapshot=snapshot,
        )

        torch.testing.assert_close(after, before)
        for key in before_info:
            torch.testing.assert_close(after_info[key], before_info[key])
        self.assertEqual(snapshot.age_clock, 0)
        self.assertEqual(memory._packed_mirror_rebuilds, 1)

    def test_owner_similarity_uses_visited_node_slots_and_checks_invariant(self):
        torch.manual_seed(94)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L", "root_R")
        for node_id in node_ids[:2]:
            memory.add_memory(
                node_id, torch.randn(3), torch.randn(memory.param_dim)
            )
        query = torch.randn(3, 3)
        visited_indices = torch.tensor([
            [0, 1, -1],
            [1, 0, -1],
            [2, 0, -1],
        ])
        visited_mask = torch.ones_like(visited_indices, dtype=torch.bool)
        visited_mask[:, -1] = False
        _, packed_info = memory.read_packed(
            query,
            visited_indices,
            visited_mask,
            node_ids,
            update_state=False,
        )

        owner_indices = torch.tensor([1, 0, 2])
        owner_similarity, owner_valid = memory.owner_similarity_from_packed(
            packed_info,
            visited_indices,
            visited_mask,
            owner_indices,
        )
        row = torch.arange(query.size(0))
        expected_slots = torch.tensor([1, 1, 0])
        torch.testing.assert_close(
            owner_similarity,
            packed_info["similarity"][row, expected_slots],
        )
        torch.testing.assert_close(
            owner_valid,
            packed_info["valid_mask"][row, expected_slots],
        )

        novelty_kwargs = dict(
            temperature=2.0,
            count_exponent=2.0,
            eps=1e-6,
            count_similarity_low=0.35,
            count_similarity_high=0.65,
            count_topk=None,
            count_saturation=3.0,
        )
        old_result = memory.novelty_count_packed(
            query,
            owner_indices,
            node_ids,
            **novelty_kwargs,
        )
        reused_result = memory.novelty_from_similarity(
            owner_similarity,
            owner_valid,
            **novelty_kwargs,
        )
        for old_value, reused_value in zip(old_result, reused_result):
            torch.testing.assert_close(old_value, reused_value)

        with self.assertRaisesRegex(
            RuntimeError,
            "posterior owner must belong to visited path union",
        ):
            memory.owner_similarity_from_packed(
                packed_info,
                visited_indices,
                visited_mask,
                torch.tensor([3, 0, 2]),
            )

    def test_packed_visited_node_read_matches_reference_calls(self):
        torch.manual_seed(97)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L", "root_R")
        for node_index, node_id in enumerate(node_ids):
            for _ in range(node_index + 1):
                memory.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.randn(memory.param_dim),
                )
        query = torch.randn(2, 3)
        indices = torch.tensor([[0, 1, -1], [0, 2, 1]])
        mask = indices >= 0
        packed, _ = memory.read_packed(
            query, indices, mask, node_ids, update_state=False
        )
        for row in range(query.size(0)):
            active = indices[row, mask[row]].tolist()
            delta_by_node, _ = memory.read_nodes(
                query[row],
                [node_ids[index] for index in active],
                update_state=False,
            )
            reference = torch.stack([
                delta_by_node[node_ids[index]] for index in active
            ])
            self.assertTrue(torch.allclose(
                packed[row, mask[row]], reference, atol=1e-6
            ))
            self.assertEqual(
                float(
                    packed[row, ~mask[row]].detach().abs().sum()
                ),
                0.0,
            )

    def test_packed_read_chunks_large_active_frontier(self):
        torch.manual_seed(99)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        for _ in range(3):
            memory.add_memory(
                "root",
                torch.randn(3),
                torch.randn(memory.param_dim),
            )

        query = torch.randn(129, 3)
        indices = torch.zeros(129, 1, dtype=torch.long)
        mask = torch.ones_like(indices, dtype=torch.bool)
        packed_by_chunk = {}
        info_by_chunk = {}
        for chunk_size in (64, 128, 256, 512):
            packed, info = memory.read_packed(
                query,
                indices,
                mask,
                ("root",),
                update_state=False,
                visit_chunk_size=chunk_size,
            )
            packed_by_chunk[chunk_size] = packed
            info_by_chunk[chunk_size] = info

        packed = packed_by_chunk[64]
        for chunk_size in (128, 256, 512):
            torch.testing.assert_close(
                packed,
                packed_by_chunk[chunk_size],
                atol=1e-6,
                rtol=1e-5,
            )
            for key in info_by_chunk[64]:
                torch.testing.assert_close(
                    info_by_chunk[64][key],
                    info_by_chunk[chunk_size][key],
                    atol=1e-6,
                    rtol=1e-5,
                )
        reference = torch.stack([
            memory.read_nodes(
                query[row],
                ["root"],
                update_state=False,
            )[0]["root"]
            for row in range(query.size(0))
        ])

        torch.testing.assert_close(
            packed[:, 0],
            reference,
            atol=1e-6,
            rtol=1e-5,
        )

    def test_packed_read_materializes_only_requested_info_fields(self):
        torch.manual_seed(100)
        memory = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        memory.add_memory(
            "root", torch.randn(3), torch.randn(memory.param_dim)
        )
        query = torch.randn(3, 3)
        indices = torch.zeros(3, 1, dtype=torch.long)
        mask = torch.ones_like(indices, dtype=torch.bool)

        complete, complete_info = memory.read_packed(
            query, indices, mask, ("root",), update_state=False
        )
        alpha_only, alpha_info = memory.read_packed(
            query,
            indices,
            mask,
            ("root",),
            update_state=False,
            info_fields=("alpha",),
        )
        no_info, empty_info = memory.read_packed(
            query,
            indices,
            mask,
            ("root",),
            update_state=False,
            info_fields=(),
        )

        self.assertEqual(
            set(complete_info),
            {
                "alpha",
                "similarity",
                "effective_k",
                "null_alpha",
                "valid_mask",
                "context_valid",
            },
        )
        self.assertEqual(set(alpha_info), {"alpha"})
        self.assertEqual(empty_info, {})
        torch.testing.assert_close(alpha_only, complete)
        torch.testing.assert_close(no_info, complete)
        torch.testing.assert_close(alpha_info["alpha"], complete_info["alpha"])

    def test_single_scatter_chunking_preserves_retrieval_gradients(self):
        torch.manual_seed(102)
        chunked = TreeEpisodicMemory(
            key_dim=3,
            num_event_types=2,
            num_basis=1,
            capacity_per_node=5,
            device="cpu",
        )
        node_ids = ("root", "root_L")
        for node_id in node_ids:
            for _ in range(3):
                chunked.add_memory(
                    node_id,
                    torch.randn(3),
                    torch.randn(chunked.param_dim),
                )
        unbounded = copy.deepcopy(chunked)
        chunked_query = torch.randn(5, 3, requires_grad=True)
        unbounded_query = chunked_query.detach().clone().requires_grad_(True)
        indices = torch.tensor([
            [0, 1], [1, 0], [0, 1], [1, 0], [0, 1],
        ])
        mask = torch.ones_like(indices, dtype=torch.bool)
        chunked_delta, _ = chunked.read_packed(
            chunked_query,
            indices,
            mask,
            node_ids,
            update_state=False,
            visit_chunk_size=2,
            info_fields=(),
        )
        unbounded_delta, _ = unbounded.read_packed(
            unbounded_query,
            indices,
            mask,
            node_ids,
            update_state=False,
            visit_chunk_size=64,
            info_fields=(),
        )
        torch.testing.assert_close(chunked_delta, unbounded_delta)
        objective_weight = torch.randn_like(chunked_delta)
        (chunked_delta * objective_weight).sum().backward()
        (unbounded_delta * objective_weight).sum().backward()
        torch.testing.assert_close(chunked_query.grad, unbounded_query.grad)
        for actual, expected in zip(
            chunked.retriever.parameters(), unbounded.retriever.parameters()
        ):
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_padded_rows_match_independent_entmax_forward_and_backward(self):
        torch.manual_seed(101)
        lengths = (1, 3, 6, 4)
        logits_batch = torch.randn(4, 6, requires_grad=True)
        valid_mask = (
            torch.arange(6).unsqueeze(0)
            < torch.tensor(lengths).unsqueeze(1)
        )
        batched = entmax15_masked(logits_batch, valid_mask)

        logits_reference = logits_batch.detach().clone().requires_grad_(True)
        reference_rows = []
        for row, length in zip(logits_reference, lengths):
            probabilities = entmax15_1d(row[:length])
            reference_rows.append(torch.nn.functional.pad(
                probabilities,
                (0, 6 - length),
            ))
        reference = torch.stack(reference_rows)
        self.assertTrue(torch.allclose(batched, reference, atol=1e-7, rtol=1e-6))

        weights = torch.randn_like(batched)
        (batched * weights).sum().backward()
        (reference * weights).sum().backward()
        self.assertTrue(torch.allclose(
            logits_batch.grad,
            logits_reference.grad,
            atol=1e-7,
            rtol=1e-6,
        ))

    def test_batched_retriever_matches_independent_bank_calls_and_gradients(self):
        torch.manual_seed(103)
        row_count, width, key_dim, param_dim = 5, 7, 4, 9
        lengths = (1, 4, 7, 2, 5)
        valid_mask = (
            torch.arange(width).unsqueeze(0)
            < torch.tensor(lengths).unsqueeze(1)
        )
        keys = torch.randn(row_count, width, key_dim)
        deltas = torch.randn(row_count, width, param_dim)
        usage = torch.rand(row_count, width)
        age = torch.rand(row_count, width) * 10.0

        batched_retriever = SmoothSparseRetriever()
        reference_retriever = copy.deepcopy(batched_retriever)
        batched_query = torch.randn(
            row_count, key_dim, requires_grad=True
        )
        reference_query = batched_query.detach().clone().requires_grad_(True)

        batched_delta, batched_info = batched_retriever.forward_batched(
            query=batched_query,
            keys=keys,
            deltas=deltas,
            usage=usage,
            age=age,
            valid_mask=valid_mask,
        )
        reference_delta = []
        reference_alpha = []
        for row_index, length in enumerate(lengths):
            delta, info = reference_retriever(
                query=reference_query[row_index],
                keys=keys[row_index, :length],
                deltas=deltas[row_index, :length],
                usage=usage[row_index, :length],
                age=age[row_index, :length],
            )
            reference_delta.append(delta)
            reference_alpha.append(torch.nn.functional.pad(
                info["alpha"],
                (0, width - length),
            ))
        reference_delta = torch.stack(reference_delta)
        reference_alpha = torch.stack(reference_alpha)
        self.assertTrue(torch.allclose(
            batched_delta, reference_delta, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            batched_info["alpha"],
            reference_alpha,
            atol=1e-6,
            rtol=1e-5,
        ))

        objective_weight = torch.randn_like(batched_delta)
        (batched_delta * objective_weight).sum().backward()
        (reference_delta * objective_weight).sum().backward()
        self.assertTrue(torch.allclose(
            batched_query.grad,
            reference_query.grad,
            atol=2e-6,
            rtol=2e-5,
        ))
        for batched_parameter, reference_parameter in zip(
            batched_retriever.parameters(),
            reference_retriever.parameters(),
        ):
            self.assertTrue(torch.allclose(
                batched_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
            ))


if __name__ == "__main__":
    unittest.main()
