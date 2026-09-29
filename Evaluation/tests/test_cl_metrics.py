from core.cl_metrics import (
    AdaptationRecord,
    CLMetricEngine,
    FrozenAnchorRecord,
    TaskBoundaryRecord,
)
from core.cl_protocol import CLTaskSpec


class FakeProtocol:
    first_seen = {"A": 0, "B": 1, "C": 3}
    persistent_regimes = frozenset(first_seen)
    task_ids = (0, 1, 2, 3)
    tasks = {
        0: CLTaskSpec(0, {"A": 1.0}, "initial"),
        1: CLTaskSpec(1, {"B": 1.0}, "novel"),
        2: CLTaskSpec(2, {"A": 1.0}, "exact_recurrence", "A"),
        3: CLTaskSpec(3, {"C": 1.0}, "specialization", "A"),
    }

    def task(self, task_id):
        return self.tasks[task_id]


def test_engine_excludes_transient_anchor_and_uses_law_macro_metrics():
    engine = CLMetricEngine(FakeProtocol())
    report = engine.evaluate(frozen_anchor_records=[
        FrozenAnchorRecord(0, "A", 2.0, 10),
        FrozenAnchorRecord(1, "A", 3.0, 10),
        FrozenAnchorRecord(1, "B", 4.0, 100),
        FrozenAnchorRecord(1, "X_transient", 100.0, 1, "diagnostic_only"),
    ])

    assert report["frozen_anchor_matrix"] == [
        {"checkpoint_task": 0, "A": 2.0, "B": None, "C": None},
        {"checkpoint_task": 1, "A": 3.0, "B": 4.0, "C": None},
    ]
    assert report["continual_summary"][-1]["clnll"] == 3.5
    assert report["continual_summary"][-1]["average_forgetting"] == 0.5
    assert report["continual_summary"][0]["average_bwt"] is None


def test_engine_fwt_and_rrr_are_protocol_driven():
    engine = CLMetricEngine(FakeProtocol())
    boundaries = [
        TaskBoundaryRecord(0, 5.0, 3.0, None),
        TaskBoundaryRecord(1, 6.0, 4.0, 7.0),
        TaskBoundaryRecord(2, 4.0, 3.5, 8.0),
        TaskBoundaryRecord(3, 7.0, 5.0, 9.0),
    ]

    fwt = engine.fwt(boundaries)
    assert fwt["eligible_task_count"] == 2
    assert [row["task_id"] for row in fwt["rows"] if row["fwt_eligible"]] == [1, 3]
    assert fwt["average_fwt"] == 1.5

    rrr = engine.rrr(boundaries)
    assert len(rrr["rows"]) == 1
    assert rrr["rows"][0]["return_task"] == 2
    assert rrr["rows"][0]["rrr"] == 0.5


def test_rrr_recovers_task_zero_c_init_from_fwt_scratch():
    engine = CLMetricEngine(FakeProtocol())
    boundaries = [
        TaskBoundaryRecord(0, None, 3.0, 5.0),
        TaskBoundaryRecord(2, 4.0, 3.5, 8.0),
    ]

    rrr = engine.rrr(boundaries)

    assert rrr["status"] == "available"
    assert rrr["rows"][0]["first_pre_nll"] == 5.0
    assert rrr["rows"][0]["rrr"] == 0.5


def test_adaptation_auc_uses_k_span_and_requires_zero():
    engine = CLMetricEngine(FakeProtocol())
    report = engine.adaptation([
        AdaptationRecord(3, 0, 5.0, 5.0),
        AdaptationRecord(3, 2, 5.0, 3.0),
        AdaptationRecord(3, 8, 5.0, 1.0),
        AdaptationRecord(1, 2, 5.0, 3.0),
    ])
    task_three = next(row for row in report["summary"] if row["task_id"] == 3)
    assert task_three["adaptation_auc"] == 2.5
    task_one = next(row for row in report["summary"] if row["task_id"] == 1)
    assert task_one["status"] == "not_available_missing_K0"


def test_adaptation_burden_auc_uses_nonuniform_k_trapezoids():
    k_values = (0, 1, 2, 4, 8, 16, 32)
    nll_values = (10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)
    report = CLMetricEngine(FakeProtocol()).adaptation([
        AdaptationRecord(
            task_id=4,
            K=k,
            pre_nll=10.0,
            adapted_nll=nll,
            protocol="fast_adapt",
            law_id="A_1",
            shift_type="exact_recurrence",
            recurrence_of="A_1",
            query_frozen=True,
        )
        for k, nll in zip(k_values, nll_values)
    ])

    summary = report["summary"][0]
    assert summary["reference_nll"] == 3.0
    assert summary["total_adaptation_gain"] == 7.0
    assert summary["adaptation_burden_auc"] == 47.5 / 32.0
    assert summary["law_id"] == "A_1"
    point_zero = next(row for row in report["points"] if row["K"] == 0)
    assert point_zero["nll"] == 10.0
    assert point_zero["gain_from_K0"] == 0.0
    assert point_zero["burden_nll"] == 7.0


def test_adaptation_burden_is_not_reported_when_query_keeps_mutating():
    report = CLMetricEngine(FakeProtocol()).adaptation([
        AdaptationRecord(4, 0, 10.0, 10.0, protocol="online_write", query_frozen=False),
        AdaptationRecord(4, 2, 10.0, 6.0, protocol="online_write", query_frozen=False),
    ])
    assert report["summary"][0]["adaptation_burden_auc"] is None
    assert report["summary"][0]["burden_status"] == "not_applicable_query_not_frozen"
