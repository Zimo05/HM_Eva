"""Regional structural evidence is separate from Wake leaf routing."""

import unittest

import torch

from HawkesBackbone import HawkesFamily
from LatentHawkesTree import HawkesTree
from Train.RegionalProbe import counterfactual_energy_probe
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer, WakeObjectiveConfig


class RegionalProbeTests(unittest.TestCase):
    def test_counterfactual_energy_helper_remains_available_for_sleep(self):
        probe = counterfactual_energy_probe(
            coarse_energy=torch.tensor([2.0, 1.0]),
            leaf_energy=torch.tensor([[0.2, 3.0], [2.0, 3.0]]),
            coarse_responsibility=torch.ones(2),
            leaf_prior=torch.ones(2),
            teacher_temperature=0.2,
            gain_temperature=0.2,
            leaf_smoothing=0.05,
        )
        self.assertGreater(float(probe.expand_target[0]), 0.99)
        self.assertLess(float(probe.expand_target[1]), 0.01)
        self.assertFalse(probe.teacher.requires_grad)

    def test_wake_probe_does_not_train_or_change_flat_router(self):
        tree = HawkesTree(3, 5, 2, 1, init_depth=2, memory_key_dim=3)
        trainer = MemoryTreeTrainer(
            tree,
            HawkesFamily(2, 1, decays=torch.tensor([1.0])),
            CausalPrefixEncoder(2, 3, type_dim=4, hidden_dim=6),
            wake=WakeObjectiveConfig(lambda_route_probe=1.0),
            device="cpu",
        )
        z = torch.randn(2, 3)
        output = tree(
            z, update_memory_state=False, update_search_state=False,
            materialize_diagnostics=False,
        )
        selected_before = output["frontier_node_indices"].clone()
        probe = trainer._regional_probe_objective(
            output,
            output["frontier_mass"].detach(),
            torch.tensor([0, 0]),
            1,
            z,
            {},
        )
        self.assertEqual(float(probe["loss"]), 0.0)
        self.assertEqual(float(probe["regions"]), 0.0)
        self.assertFalse(hasattr(tree, "expansion_predictor"))
        self.assertEqual(tree.frontier_routing.probe_leaf_visits, {})
        self.assertTrue(torch.equal(
            output["frontier_node_indices"], selected_before
        ))

    def test_legacy_probe_coverage_state_survives_checkpoint(self):
        tree = HawkesTree(3, 5, 2, 1, init_depth=2, memory_key_dim=3)
        tree.frontier_routing.probe_leaf_visits = {tree.leaf_ids[0]: 3}
        state = tree.state_dict()
        restored = HawkesTree(3, 5, 2, 1, init_depth=0, memory_key_dim=3)
        restored.load_state_dict(state)
        self.assertEqual(
            restored.frontier_routing.probe_leaf_visits,
            tree.frontier_routing.probe_leaf_visits,
        )


if __name__ == "__main__":
    unittest.main()
