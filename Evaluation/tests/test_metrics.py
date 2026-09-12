from core.metrics import adaptation_auc, continual_metrics, prediction_metrics


def test_prediction_metrics_hand_calculation():
    rows = [
        {"true_type": 0, "predicted_type": 0, "true_delta_time": 1.0, "predicted_delta_time": 2.0, "event_nll": 1.0},
        {"true_type": 1, "predicted_type": 0, "true_delta_time": 3.0, "predicted_delta_time": 2.0, "event_nll": 3.0},
    ]
    result = prediction_metrics(rows)
    assert result["nll_per_event"] == 2.0
    assert result["accuracy"] == 0.5
    assert result["time_mae"] == 1.0
    assert result["time_rmse"] == 1.0
    assert result["macro_f1"] == 1.0 / 3.0


def test_prediction_metrics_keeps_unavailable_event_nll_null():
    rows = [
        {
            "true_type": 0,
            "predicted_type": 0,
            "true_delta_time": 1.0,
            "predicted_delta_time": 1.5,
            "event_nll": None,
        }
    ]
    assert prediction_metrics(rows)["nll_per_event"] is None


def test_adaptation_auc_trapezoid():
    assert adaptation_auc({0: 4.0, 4: 2.0}) == 3.0


def test_continual_loss_metrics():
    matrix = {0: {"A": 2.0}, 1: {"A": 3.0, "B": 4.0}}
    rows = continual_metrics(matrix, {"A": 0, "B": 1})
    assert rows[-1]["clnll"] == 3.5
    assert rows[-1]["average_forgetting"] == 0.5
    assert rows[-1]["average_bwt"] == -1.0
