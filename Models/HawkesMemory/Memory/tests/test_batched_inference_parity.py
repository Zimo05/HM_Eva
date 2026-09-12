"""Parity and scheduling tests for the compact HM evaluation path.

The numerical tests intentionally compare the packed implementation with the
ordinary causal ``run_sequence`` reference.  The optional CL test compares
the same checkpoint matrix at batch sizes one and 64, ignoring only runtime
metadata, so it can be enabled on a machine that has real CL checkpoints.
"""

from __future__ import annotations

import csv
import copy
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - allows collection without HM deps
    torch = None

if torch is not None:
    from HawkesBackbone import (
        HAWKES_HISTORY_STATS_KEY,
        HAWKES_INTERVAL_STATS_KEY,
        HawkesFamily,
    )
    from LatentHawkesTree import HawkesTree
    from Train.Inference import (
        EvaluationProtocol,
        MemoryTreeInference,
        inference_config_for_protocol,
    )
    from Train.Train import CausalPrefixEncoder
    from EvaluateCL import (
        EVENT_PREDICTION_SCOPES,
        EvaluationSet,
        _event_prediction_set_names,
    )
    from Wake.HawkesParams import HawkesParams


def _sequence(length: int, source_index: int) -> dict:
    times = torch.arange(1, length + 1, dtype=torch.float32) * 0.1
    types = (torch.arange(length, dtype=torch.long) + source_index) % 2
    return {
        "times": times,
        "types": types,
        "source_index": source_index,
    }


def _tied_sequence(source_index: int) -> dict:
    """Return a non-decreasing sequence with two tied-time groups."""
    return {
        "times": torch.tensor(
            [0.1, 0.2, 0.2, 0.45, 0.45, 0.8],
            dtype=torch.float32,
        ),
        "types": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
        "source_index": source_index,
    }


def _make_inference(protocol: EvaluationProtocol) -> MemoryTreeInference:
    """Create two-identical-node synthetic HM inference for fast CPU tests."""

    torch.manual_seed(1729)
    hawkes = HawkesFamily(
        2,
        1,
        decays=torch.tensor([1.0]),
    )
    tree = HawkesTree(
        z_dim=3,
        node_dim=4,
        num_event_types=2,
        num_basis=1,
        init_depth=1,
        memory_key_dim=3,
        memory_capacity_per_node=16,
    )
    encoder = CausalPrefixEncoder(
        num_event_types=2,
        z_dim=3,
        type_dim=4,
        hidden_dim=8,
    )
    inference = MemoryTreeInference(
        tree,
        hawkes,
        encoder,
        inference_config=inference_config_for_protocol(protocol),
        device="cpu",
    )
    # Keep a resident row in both leaves so routing/retrieval parity exercises
    # the packed bank mirror instead of only the empty-bank branch.
    for node_index, node_id in enumerate(inference.tree.leaf_ids):
        key = torch.zeros(3)
        key[node_index % 3] = 1.0
        delta = torch.linspace(
            0.01 * (node_index + 1),
            0.02 * (node_index + 1),
            inference.tree.param_dim,
        )
        inference.tree.episodic_memory.add_memory(node_id, key, delta)
    return inference


def _batch_run(
    inference: MemoryTreeInference,
    sequences: list[dict],
    batch_size: int,
    *,
    capture_event_predictions: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    static_cache = inference.tree.frontier_routing.build_static_cache(
        detach=True
    )
    totals = {
        "events": 0,
        "nll_sum": 0.0,
        "correct": 0,
        "time_abs_sum": 0.0,
    }
    results: list[dict] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        prepared, static_cache = inference.prepare_sequence_batch(
            batch,
            frontier_static_cache=static_cache,
        )
        batch_results = inference.run_sequence_batch_compact(
            prepared,
            frontier_static_cache=static_cache,
            capture_event_predictions=capture_event_predictions,
        )
        results.extend(batch_results)
        for result in batch_results:
            scalar = result["scalar_metrics"]
            totals["events"] += int(scalar["events"])
            totals["nll_sum"] += float(scalar["nll_sum"])
            totals["correct"] += int(scalar["correct"])
            totals["time_abs_sum"] += float(scalar["time_abs_sum"])
    return totals, results


def _scalar_run(
    inference: MemoryTreeInference,
    sequence: dict,
) -> dict:
    static_cache = inference.tree.frontier_routing.build_static_cache(
        detach=True
    )
    return inference.run_sequence(
        sequence,
        frontier_static_cache=static_cache,
    )


def _tree_snapshot(inference: MemoryTreeInference) -> dict:
    state = {}
    for name, value in inference.tree.state_dict().items():
        if name.startswith("working_memory."):
            continue
        if isinstance(value, torch.Tensor):
            state[name] = value.detach().cpu().clone()
        else:
            state[name] = copy.deepcopy(value)
    return {
        "state": state,
        "leaf_ids": tuple(inference.tree.leaf_ids),
        "bank_lengths": {
            node_id: len(bank)
            for node_id, bank in inference.tree.episodic_memory.banks.items()
        },
    }


def _assert_nested_equal(
    test: unittest.TestCase,
    left,
    right,
    name: str,
) -> None:
    """Compare snapshots containing ordinary values and nested tensors."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        test.assertIsInstance(left, torch.Tensor, name)
        test.assertIsInstance(right, torch.Tensor, name)
        test.assertTrue(torch.equal(left, right), name)
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        test.assertIsInstance(left, Mapping, name)
        test.assertIsInstance(right, Mapping, name)
        test.assertEqual(set(left), set(right), name)
        for key in left:
            _assert_nested_equal(test, left[key], right[key], f"{name}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        test.assertIsInstance(right, type(left), name)
        test.assertEqual(len(left), len(right), name)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(test, left_item, right_item, f"{name}[{index}]")
        return
    test.assertEqual(left, right, name)


def _assert_event_parity(test: unittest.TestCase, scalar, packed, source_index):
    test.assertEqual(len(scalar["events"]), len(packed["events"]))
    for scalar_event, packed_event in zip(scalar["events"], packed["events"]):
        test.assertEqual(scalar_event["event_index"], packed_event["event_index"])
        test.assertEqual(scalar_event["predicted_type"], packed_event["predicted_type"])
        test.assertEqual(
            tuple(scalar_event["frontier_node_ids"]),
            tuple(packed_event["frontier_node_ids"]),
        )
        test.assertEqual(scalar_event["owner_id"], packed_event["owner_id"])
        test.assertEqual(
            scalar_event["write_token"],
            f"{source_index}:{scalar_event['event_index']}",
        )
        for field in (
            "nll",
            "predicted_time",
            "retrieve_gate",
            "retrieval_alpha_mass",
        ):
            test.assertLess(
                abs(float(scalar_event[field]) - float(packed_event[field])),
                1e-5,
                msg=field,
            )


@unittest.skipUnless(torch is not None, "requires the HM PyTorch dependencies")
class BatchedInferenceParityTests(unittest.TestCase):
    def test_batched_event_nll_reduces_all_hawkes_axes(self):
        inference = _make_inference(EvaluationProtocol.FROZEN)
        sequence = _sequence(5, 7)
        cached = inference.hawkes.prepare_sequence_cache(
            dict(sequence),
            inplace=False,
        )
        D = inference.hawkes.num_types
        M = inference.hawkes.num_basis
        parameter_dim = D + D * D * M
        torch.manual_seed(2718)
        theta = torch.randn(5, parameter_dim)
        history = cached[HAWKES_HISTORY_STATS_KEY]
        interval = cached[HAWKES_INTERVAL_STATS_KEY]
        times = sequence["times"]
        durations = torch.cat([
            times[:1],
            times[1:] - times[:-1],
        ])

        actual = inference._batched_event_nll(
            theta,
            history,
            interval,
            sequence["types"],
            durations,
        )
        expected = torch.stack([
            inference.hawkes.event_NLL(
                cached,
                HawkesParams(
                    theta[index, :D],
                    theta[index, D:].reshape(D, D, M),
                ),
                index,
            )
            for index in range(times.numel())
        ])
        self.assertEqual(tuple(actual.shape), (times.numel(),))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

        # Also cover the [N, K, P] frontier-candidate form used by routing.
        candidate_count = 3
        actual_candidates = inference._batched_event_nll(
            theta.unsqueeze(1).expand(-1, candidate_count, -1),
            history,
            interval,
            sequence["types"],
            durations,
        )
        self.assertEqual(
            tuple(actual_candidates.shape),
            (times.numel(), candidate_count),
        )
        self.assertTrue(
            torch.allclose(
                actual_candidates,
                expected.unsqueeze(1).expand(-1, candidate_count),
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_batched_owner_indices_match_scalar_lca_resolution(self):
        inference = _make_inference(EvaluationProtocol.FROZEN)
        node_ids = tuple(inference.tree.all_node_ids)
        width = len(node_ids)
        if width < 2:
            self.skipTest("synthetic tree needs at least two nodes")
        frontier_node_indices = torch.tensor([
            list(range(width)),
            list(reversed(range(width))),
        ], dtype=torch.long)
        frontier_mask = torch.zeros(2, width, dtype=torch.bool)
        frontier_mask[0] = True
        frontier_mask[1, :2] = True
        posterior = torch.zeros(2, width)
        posterior[0] = torch.arange(1, width + 1, dtype=torch.float32)
        posterior[1, :2] = torch.tensor([0.35, 0.65])
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)

        actual = inference._posterior_owner_indices_batch(
            frontier_node_indices,
            frontier_mask,
            posterior,
        )
        expected = []
        for row in range(2):
            active_ids = tuple(
                node_ids[int(index)]
                for index in frontier_node_indices[row][frontier_mask[row]]
            )
            scalar_owner = inference._posterior_owner(
                active_ids,
                posterior[row][frontier_mask[row]],
            )
            expected.append(node_ids.index(scalar_owner))
        self.assertTrue(torch.equal(actual.cpu(), torch.tensor(expected)))

    def test_scalar_and_batch_frozen_parity_and_no_state_transition(self):
        sequence = _sequence(7, 42)
        scalar_inference = _make_inference(EvaluationProtocol.FROZEN)
        batch_inference = _make_inference(EvaluationProtocol.FROZEN)
        scalar = _scalar_run(scalar_inference, dict(sequence))
        before = _tree_snapshot(batch_inference)
        _, batch_results = _batch_run(
            batch_inference,
            [dict(sequence)],
            batch_size=1,
            capture_event_predictions=True,
        )
        packed = batch_results[0]

        _assert_event_parity(self, scalar, packed, sequence["source_index"])
        self.assertLess(abs(scalar["total_nll"] - packed["total_nll"]), 1e-5)
        self.assertEqual(len(packed["events"]), sequence["times"].numel())

        after = _tree_snapshot(batch_inference)
        self.assertEqual(before["leaf_ids"], after["leaf_ids"])
        self.assertEqual(before["bank_lengths"], after["bank_lengths"])
        self.assertEqual(before["state"].keys(), after["state"].keys())
        for name in before["state"]:
            left = before["state"][name]
            right = after["state"][name]
            _assert_nested_equal(self, left, right, name)

    def test_fast_adapt_has_one_independent_working_state_per_sequence(self):
        sequences = [_sequence(3, 10), _sequence(6, 11), _sequence(4, 12)]
        scalar_inference = _make_inference(EvaluationProtocol.FAST_ADAPT)
        scalar_results = []
        scalar_trajectories = []
        for sequence in sequences:
            trajectory = []
            adapter = scalar_inference.tree.working_memory
            original_update = adapter.update_from_gradient

            def record_scalar_update(
                grad,
                adaptation_probability=None,
                *,
                _original=original_update,
                _adapter=adapter,
                _trajectory=trajectory,
            ):
                result = _original(
                    grad,
                    adaptation_probability=adaptation_probability,
                )
                _trajectory.append(_adapter.delta.detach().clone())
                return result

            adapter.update_from_gradient = record_scalar_update
            try:
                scalar_results.append(_scalar_run(scalar_inference, dict(sequence)))
            finally:
                adapter.update_from_gradient = original_update
            scalar_trajectories.append(trajectory)

        batch_inference = _make_inference(EvaluationProtocol.FAST_ADAPT)
        batch_trajectories = [[] for _ in sequences]
        adapter = batch_inference.tree.working_memory
        original_batch_update = adapter.update_batch_rows

        def record_batch_update(
            state,
            row_indices,
            grad,
            adaptation_probability=None,
            *,
            _original=original_batch_update,
            _trajectories=batch_trajectories,
        ):
            result = _original(
                state,
                row_indices,
                grad,
                adaptation_probability=adaptation_probability,
            )
            for local_index, row_index in enumerate(
                row_indices.detach().cpu().tolist()
            ):
                _trajectories[int(row_index)].append(
                    result[local_index].detach().clone()
                )
            return result

        adapter.update_batch_rows = record_batch_update
        _, batch_results = _batch_run(
            batch_inference,
            [dict(sequence) for sequence in sequences],
            batch_size=len(sequences),
            capture_event_predictions=True,
        )
        adapter.update_batch_rows = original_batch_update

        for sequence, scalar, packed in zip(
            sequences, scalar_results, batch_results
        ):
            _assert_event_parity(self, scalar, packed, sequence["source_index"])
            self.assertLess(abs(scalar["total_nll"] - packed["total_nll"]), 1e-5)
            self.assertEqual(
                len(packed["events"]),
                int(sequence["times"].numel()),
            )
        for row, (scalar_trace, batch_trace) in enumerate(
            zip(scalar_trajectories, batch_trajectories)
        ):
            self.assertEqual(len(scalar_trace), len(batch_trace), row)
            for step, (scalar_state, batch_state) in enumerate(
                zip(scalar_trace, batch_trace)
            ):
                self.assertTrue(
                    torch.allclose(
                        scalar_state,
                        batch_state,
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                    f"working-memory trajectory row={row} step={step}",
                )

    def test_tied_timestamps_have_scalar_batch_parity(self):
        sequences = [_tied_sequence(70), _sequence(4, 71)]
        scalar_inference = _make_inference(EvaluationProtocol.FAST_ADAPT)
        scalar_results = [
            _scalar_run(scalar_inference, dict(sequence))
            for sequence in sequences
        ]
        batch_inference = _make_inference(EvaluationProtocol.FAST_ADAPT)
        _, batch_results = _batch_run(
            batch_inference,
            [dict(sequence) for sequence in sequences],
            batch_size=2,
            capture_event_predictions=True,
        )
        for sequence, scalar, packed in zip(
            sequences, scalar_results, batch_results
        ):
            _assert_event_parity(self, scalar, packed, sequence["source_index"])
            self.assertLess(abs(scalar["total_nll"] - packed["total_nll"]), 1e-5)

    def test_online_write_keeps_scalar_semantics(self):
        sequence = _sequence(4, 80)
        inference = _make_inference(EvaluationProtocol.ONLINE_WRITE)
        self.assertTrue(inference.config.allow_memory_writes)
        self.assertTrue(inference.config.update_memory_usage)
        scalar = _scalar_run(inference, dict(sequence))
        self.assertEqual(len(scalar["events"]), 4)

        read_only_prepare, static_cache = inference.prepare_sequence_batch(
            [dict(sequence)]
        )
        with self.assertRaisesRegex(
            ValueError,
            "packed inference is restricted to read-only protocols",
        ):
            inference.run_sequence_batch_compact(
                read_only_prepare,
                frontier_static_cache=static_cache,
            )

    def test_variable_lengths_do_not_score_padded_positions(self):
        lengths = [20, 80, 5, 200]
        sequences = [
            _sequence(length, 100 + row)
            for row, length in enumerate(lengths)
        ]
        inference = _make_inference(EvaluationProtocol.FAST_ADAPT)
        static_cache = inference.tree.frontier_routing.build_static_cache(
            detach=True
        )
        prepared, static_cache = inference.prepare_sequence_batch(
            [dict(sequence) for sequence in sequences],
            frontier_static_cache=static_cache,
        )
        self.assertEqual(
            [int(item["z"].size(0)) for item in prepared],
            lengths,
        )
        results = inference.run_sequence_batch_compact(
            prepared,
            frontier_static_cache=static_cache,
            capture_event_predictions=True,
        )
        for length, result in zip(lengths, results):
            self.assertEqual(len(result["events"]), length)
            self.assertTrue(
                all(0 <= int(event["event_index"]) < length for event in result["events"])
            )
            self.assertEqual(result["scalar_metrics"]["events"], length)

        # The short sequence must have exactly the same causal trajectory when
        # the other, longer rows are present in the packed wavefront.
        single = _make_inference(EvaluationProtocol.FAST_ADAPT)
        reference = _scalar_run(single, dict(sequences[2]))
        self.assertLess(
            abs(reference["total_nll"] - results[2]["total_nll"]),
            1e-5,
        )
        _assert_event_parity(self, reference, results[2], sequences[2]["source_index"])

    def test_metrics_are_invariant_to_sequence_batch_size(self):
        lengths = [2 + (index % 8) for index in range(64)]
        metrics_by_batch_size = {}
        for batch_size in (1, 2, 8, 32, 64):
            inference = _make_inference(EvaluationProtocol.FROZEN)
            sequences = [
                _sequence(length, 200 + row)
                for row, length in enumerate(lengths)
            ]
            totals, _ = _batch_run(
                inference,
                sequences,
                batch_size=batch_size,
            )
            denominator = max(int(totals["events"]), 1)
            metrics_by_batch_size[batch_size] = {
                "nll_per_event": totals["nll_sum"] / denominator,
                "accuracy": totals["correct"] / denominator,
                "time_mae": totals["time_abs_sum"] / denominator,
            }
        reference = metrics_by_batch_size[1]
        for batch_size, metrics in metrics_by_batch_size.items():
            for field in reference:
                self.assertLess(
                    abs(metrics[field] - reference[field]),
                    1e-5,
                    msg=f"batch_size={batch_size} field={field}",
                )

    def test_event_prediction_scope_selection(self):
        self.assertEqual(
            EVENT_PREDICTION_SCOPES,
            ("none", "current", "final", "all"),
        )
        evaluation_sets = [
            EvaluationSet("task_02_test", "task_test", Path("task_02.csv"), 2),
            EvaluationSet("task_02_control", "matched_control", Path("c.csv"), 2),
            EvaluationSet("task_03_test_pre", "task_test_pre", Path("task_03.csv"), 3),
            EvaluationSet("anchor_A", "anchor", Path("a.csv"), None),
        ]
        for scope, expected in {
            "none": set(),
            "current": {"task_02_test", "task_02_control"},
            "final": {
                "task_02_test",
                "task_02_control",
                "task_03_test_pre",
                "anchor_A",
            },
            "all": {
                "task_02_test",
                "task_02_control",
                "task_03_test_pre",
                "anchor_A",
            },
        }.items():
            args = SimpleNamespace(
                event_prediction_scope=scope,
                _event_prediction_final_task=2,
            )
            self.assertEqual(
                _event_prediction_set_names(
                    args=args,
                    checkpoint_task=2,
                    evaluation_sets=evaluation_sets,
                ),
                expected,
            )
        args = SimpleNamespace(
            event_prediction_scope="final",
            _event_prediction_final_task=2,
        )
        self.assertEqual(
            _event_prediction_set_names(
                args=args,
                checkpoint_task=1,
                evaluation_sets=evaluation_sets,
            ),
            set(),
        )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _stable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dynamic = {"elapsed_seconds", "events_per_second", "from_cache"}
    stable = [
        {key: value for key, value in row.items() if key not in dynamic}
        for row in rows
    ]
    return sorted(
        stable,
        key=lambda row: tuple(
            row.get(field, "")
            for field in (
                "checkpoint_task",
                "eval_name",
                "eval_task",
                "variant",
                "regime_id",
                "task_id",
            )
        ),
    )


def _assert_csv_rows_close(
    test: unittest.TestCase,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
    name: str,
) -> None:
    first = _stable_rows(first)
    second = _stable_rows(second)
    test.assertEqual(len(first), len(second), msg=name)
    for row_index, (left, right) in enumerate(zip(first, second)):
        test.assertEqual(set(left), set(right), msg=f"{name} row {row_index}")
        for key in left:
            left_value = left[key]
            right_value = right[key]
            try:
                left_number = float(left_value)
                right_number = float(right_value)
            except (TypeError, ValueError):
                test.assertEqual(
                    left_value,
                    right_value,
                    msg=f"{name} row {row_index} field {key}",
                )
            else:
                if math.isnan(left_number) or math.isnan(right_number):
                    test.assertTrue(
                        math.isnan(left_number) and math.isnan(right_number),
                        msg=f"{name} row {row_index} field {key}",
                    )
                else:
                    test.assertLess(
                        abs(left_number - right_number),
                        1e-5,
                        msg=f"{name} row {row_index} field {key}",
                    )


@unittest.skipUnless(
    torch is not None
    and os.environ.get("HM_CL_REGRESSION_CHECKPOINT_DIR")
    and os.environ.get("HM_CL_REGRESSION_DATA_ROOT"),
    "set HM_CL_REGRESSION_CHECKPOINT_DIR and HM_CL_REGRESSION_DATA_ROOT",
)
class CLLevelBatchRegressionTests(unittest.TestCase):
    def test_task_0_to_2_csv_contract_is_batch_schedule_invariant(self):
        checkpoint_dir = Path(
            os.environ["HM_CL_REGRESSION_CHECKPOINT_DIR"]
        ).expanduser()
        data_root = Path(os.environ["HM_CL_REGRESSION_DATA_ROOT"]).expanduser()
        repo_root = Path(__file__).resolve().parents[4]
        memory_root = repo_root / "Models" / "HawkesMemory" / "Memory"
        hawkes_root = repo_root / "Models" / "HawkesMemory"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(repo_root),
                str(memory_root),
                str(hawkes_root),
                environment.get("PYTHONPATH", ""),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {}
            for batch_size in (1, 64):
                output_dir = root / f"batch_{batch_size}"
                command = [
                    sys.executable,
                    "-m",
                    "EvaluateCL",
                    "--data-root",
                    str(data_root),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--output-dir",
                    str(output_dir),
                    "--task-start",
                    "0",
                    "--task-end",
                    "2",
                    "--variants",
                    "frozen/full",
                    "--eval-batch-size",
                    str(batch_size),
                    "--event-prediction-scope",
                    "none",
                    "--bootstrap-samples",
                    "8",
                    "--max-sequences",
                    "2",
                    "--no-hawkes-law-evaluation",
                    "--no-summary-plots",
                    "--no-adaptation-evaluation",
                ]
                subprocess.run(
                    command,
                    cwd=memory_root,
                    env=environment,
                    check=True,
                    timeout=1800,
                )
                outputs[batch_size] = output_dir

            # EvaluateCL's native name is ``anchor_nll_matrix.csv``; the
            # outer Evaluation runner publishes the same matrix under the
            # benchmark-wide ``frozen_anchor_matrix.csv`` contract.
            artifacts = {
                "task_metrics.csv": "task_metrics.csv",
                "anchor_metrics.csv": "anchor_metrics.csv",
                "continual_summary.csv": "continual_summary.csv",
                "frozen_anchor_matrix.csv": "anchor_nll_matrix.csv",
                "law_metrics.csv": "law_metrics.csv",
                "fwt_metrics.csv": "fwt_metrics.csv",
                "rrr_metrics.csv": "rrr_metrics.csv",
            }
            for contract_name, native_name in artifacts.items():
                first = outputs[1] / native_name
                second = outputs[64] / native_name
                self.assertTrue(first.is_file(), first)
                self.assertTrue(second.is_file(), second)
                _assert_csv_rows_close(
                    self,
                    _read_csv_rows(first),
                    _read_csv_rows(second),
                    contract_name,
                )


if __name__ == "__main__":
    unittest.main()
