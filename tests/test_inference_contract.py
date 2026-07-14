import pytest
import torch

from models.inference import build_prediction_records


def test_prediction_records_preserve_coverage_order_and_rectangle():
    hyper_min = torch.full((2, 5, 10), 0.1)
    hyper_max = torch.full((2, 5, 10), 0.6)
    records = build_prediction_records(
        sample_ids=("a", "b"),
        coverage_keys=("branch", "line"),
        graph_pred=torch.tensor([[0.4, 0.5], [0.6, 0.7]]),
        hyper_min=hyper_min,
        hyper_max=hyper_max,
    )

    assert records[0]["coverage"] == pytest.approx({"branch": 0.4, "line": 0.5})
    assert records[1]["sample_id"] == "b"
    assert records[0]["schema_version"] == "rtlmap_prediction.v2"
    assert records[1]["hyperrectangles"]["line"]["min"] == pytest.approx(
        [0.1] * 10
    )
    assert records[1]["hyperrectangles"]["line"]["volume"] == pytest.approx(
        0.5**10
    )
    assert records[1]["coverage_space_size"] == pytest.approx(0.5**10)


@pytest.mark.parametrize(
    "graph_pred,hyper_min,hyper_max,match",
    [
        (torch.zeros(1, 1), None, None, "requires hyper"),
        (
            torch.tensor([[float("nan")]]),
            torch.zeros(1, 5, 10),
            torch.ones(1, 5, 10),
            "NaN",
        ),
        (
            torch.zeros(1, 1),
            torch.ones(1, 5, 10),
            torch.zeros(1, 5, 10),
            "exceeds",
        ),
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
