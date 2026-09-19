import os
import unittest
from unittest.mock import patch

import torch

from Train.DistributedRuntime import DistributedRuntime


class DistributedRuntimeTests(unittest.TestCase):
    def test_local_runtime_is_noop_and_keeps_order(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False):
            runtime = DistributedRuntime.from_environment(device="cpu")
        self.assertFalse(runtime.enabled)
        self.assertTrue(runtime.is_rank0)
        self.assertEqual(runtime.contiguous_shard(list(range(7))), list(range(7)))
        self.assertEqual(runtime.gather_transactions(["a", "b"]), ["a", "b"])
        self.assertEqual(runtime.broadcast_object({"value": 3}), {"value": 3})

    def test_contiguous_shards_are_balanced_and_stable(self):
        values = list(range(10))
        shards = [
            DistributedRuntime(rank=rank, local_rank=rank, world_size=4)
            .contiguous_shard(values)
            for rank in range(4)
        ]
        self.assertEqual(shards, [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]])
        self.assertEqual([item for shard in shards for item in shard], values)

    def test_local_event_weighted_reduction_preserves_gradient(self):
        runtime = DistributedRuntime(device=torch.device("cpu"))
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        (parameter.square().sum()).backward()
        before = parameter.grad.clone()
        total = runtime.event_weighted_all_reduce_gradients([parameter], 5)
        self.assertEqual(total, 5)
        torch.testing.assert_close(parameter.grad, before)


if __name__ == "__main__":
    unittest.main()
