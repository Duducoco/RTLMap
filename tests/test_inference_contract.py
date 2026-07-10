import pytest
import torch

from models.inference import build_prediction_records


def test_prediction_records_preserve_coverage_order_and_rectangle():
    records = build_prediction_records(
        sample_ids=("a", "b"),
        coverage_keys=("branch", "line"),
        graph_pred=torch.tensor([[0.4, 0.5], [0.6, 0.7]]),
        hyper_min=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        hyper_max=torch.tensor([[0.5, 0.6], [0.6, 0.7]]),
    )

    assert records[0]["coverage"] == pytest.approx({"branch": 0.4, "line": 0.5})
    assert records[1]["sample_id"] == "b"
    assert records[1]["hyper_min"] == pytest.approx([0.2, 0.3])


@pytest.mark.parametrize(
    "graph_pred,hyper_min,hyper_max,match",
    [
        (torch.zeros(1, 1), None, None, "requires hyper"),
        (torch.tensor([[float("nan")]]), torch.zeros(1, 1), torch.ones(1, 1), "NaN"),
        (torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1), "exceeds"),
    ],
)
def test_prediction_records_reject_invalid_model_outputs(
    graph_pred, hyper_min, hyper_max, match
):
    with pytest.raises(ValueError, match=match):
        build_prediction_records(
            sample_ids=("a",),
            coverage_keys=("branch",),
            graph_pred=graph_pred,
            hyper_min=hyper_min,
            hyper_max=hyper_max,
        )
