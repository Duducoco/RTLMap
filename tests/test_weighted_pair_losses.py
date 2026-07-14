"""Tests for sample-balanced endpoint losses in Geometry Pair batches."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.losses import compute_weighted_graph_loss
from models.contrastive_loss import compute_coverage_geometry_losses


def test_weighted_graph_loss_preserves_equal_coverage_type_weighting() -> None:
    graph_pred = torch.zeros((4, 2))
    target = torch.tensor(
        [
            [1.0, float("nan"), 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [float("nan"), 1.0, 0.0, 0.0, 0.0],
        ]
    )
    endpoint_weight = torch.tensor([1.0, 3.0, 1.0, 1.0])

    loss = compute_weighted_graph_loss(
        graph_pred,
        target,
        endpoint_weight,
        coverage_target_keys=("branch", "line"),
    )

    # branch: (0.5*1 + 0*3 + 0.5*1) / 5 = 0.2
    # line:   (0.5*3 + 0*1 + 0.5*1) / 5 = 0.4
    assert torch.isclose(loss, torch.tensor(0.3))


def test_volume_is_endpoint_weighted_but_iou_is_pair_uniform() -> None:
    output_a = SimpleNamespace(
        hyper_min=torch.zeros((2, 1, 1)),
        hyper_max=torch.full((2, 1, 1), 0.5),
    )
    output_b = SimpleNamespace(
        hyper_min=torch.zeros((2, 1, 1)),
        hyper_max=torch.full((2, 1, 1), 0.5),
    )
    common = dict(
        density_a=torch.tensor([[0.5], [1.0]]),
        density_b=torch.tensor([[1.0], [0.5]]),
        size_mask=torch.ones((2, 1), dtype=torch.bool),
        jaccard=torch.tensor([[1.0], [0.5]]),
        iou_mask=torch.ones((2, 1), dtype=torch.bool),
        min_width=0.01,
        smooth_temperature=None,
    )

    weighted = compute_coverage_geometry_losses(
        output_a,
        output_b,
        endpoint_weight_a=torch.tensor([1.0, 3.0]),
        endpoint_weight_b=torch.tensor([1.0, 1.0]),
        **common,
    )
    reweighted = compute_coverage_geometry_losses(
        output_a,
        output_b,
        endpoint_weight_a=torch.tensor([20.0, 1.0]),
        endpoint_weight_b=torch.tensor([1.0, 20.0]),
        **common,
    )

    assert torch.isclose(weighted.volume_loss, torch.tensor(0.160151), atol=1e-6)
    assert torch.equal(weighted.iou_loss, reweighted.iou_loss)
    assert torch.isclose(weighted.iou_loss, torch.tensor(0.120113), atol=1e-6)
