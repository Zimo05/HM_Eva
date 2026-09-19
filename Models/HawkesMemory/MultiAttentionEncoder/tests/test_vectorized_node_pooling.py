import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AttentionEncoder.AttenEncoderMain_v1 import MultiAttentionEncoderPipeline
from AttentionEncoder.Model.Node_Embedding import NodeEmbedding


class VectorizedNodePoolingTests(unittest.TestCase):
    def test_matches_ragged_loop_outputs_and_gradients(self):
        torch.manual_seed(7)
        d_model = 6
        padded = torch.randn(4, 5, d_model)
        mask = torch.tensor([
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, False, False],
            [False, False, False, False, False],
        ])

        loop_model = NodeEmbedding(d_model=d_model, hidden_dim=9)
        batch_model = copy.deepcopy(loop_model)
        loop_input = padded.clone().requires_grad_(True)
        batch_input = padded.clone().requires_grad_(True)

        expected_rows = []
        for node_index in range(mask.shape[0]):
            if not mask[node_index].any():
                expected_rows.append(torch.zeros(d_model))
                continue
            values = loop_input[node_index, mask[node_index]]
            expected_rows.append(
                loop_model.out_proj(loop_model.attention_pool(values))
            )
        expected = torch.stack(expected_rows)
        member_rows = mask.nonzero(as_tuple=False)
        member_embeddings = batch_input[member_rows[:, 0], member_rows[:, 1]]
        actual = batch_model.attention_pool_indexed(
            member_embeddings,
            node_index=member_rows[:, 0],
            num_nodes=mask.shape[0],
        )

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

        weights = torch.randn_like(expected)
        (expected * weights).sum().backward()
        (actual * weights).sum().backward()
        torch.testing.assert_close(
            batch_input.grad, loop_input.grad, rtol=1e-5, atol=1e-7
        )
        for loop_param, batch_param in zip(
            loop_model.parameters(), batch_model.parameters()
        ):
            torch.testing.assert_close(
                batch_param.grad, loop_param.grad, rtol=1e-5, atol=1e-7
            )

    def test_pipeline_membership_filter_matches_legacy_pooling(self):
        torch.manual_seed(11)
        pipeline = MultiAttentionEncoderPipeline.__new__(
            MultiAttentionEncoderPipeline
        )
        pipeline.device = torch.device("cpu")
        pipeline.d_model = 4
        pipeline.node_ids = ["root", "l", "r", "empty"]
        pipeline.node_sequences = {
            "root": [0, 1, 2, 3],
            "l": [0, 2],
            "r": [1, 3],
            "empty": [99],
        }
        pipeline.key_to_global_id = {f"s{i}": i for i in range(4)}
        pipeline.seq_embeddings = {
            f"s{i}": torch.randn(pipeline.d_model) for i in range(4)
        }
        pipeline.Z_matrix = None
        pipeline._node_sequence_indices = None
        pipeline._node_membership_nodes = None
        pipeline._node_sequence_counts = None
        pipeline._global_id_to_z_row = {}
        pipeline._allowed_node_mask_cache = {}
        pipeline.node_embedder = NodeEmbedding(d_model=pipeline.d_model)

        allowed = frozenset({0, 1, 3})
        actual = pipeline._pool_node_embeddings(allowed)

        expected_rows = []
        for node_id in pipeline.node_ids:
            embeddings = [
                pipeline.seq_embeddings[f"s{global_id}"]
                for global_id in pipeline.node_sequences[node_id]
                if (
                    global_id in allowed
                    and f"s{global_id}" in pipeline.seq_embeddings
                )
            ]
            if embeddings:
                values = torch.stack(embeddings)
                expected_rows.append(
                    pipeline.node_embedder.out_proj(
                        pipeline.node_embedder.attention_pool(values)
                    )
                )
            else:
                expected_rows.append(torch.zeros(pipeline.d_model))

        torch.testing.assert_close(
            actual, torch.stack(expected_rows), rtol=1e-6, atol=1e-7
        )
        self.assertEqual(
            pipeline._node_sequence_indices.numel(),
            8,
        )
        self.assertIs(
            pipeline._node_pool_mask(allowed),
            pipeline._node_pool_mask(allowed),
        )


if __name__ == "__main__":
    unittest.main()
