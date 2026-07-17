from types import SimpleNamespace

import pytest
import torch

from models.contrastive_loss import compute_coverage_geometry_losses
from models.hyperrectangle import HyperrectangleHead, hyperrectangle_geometry


def test_typed_center_radius_head_preserves_bounds_and_minimum_width() -> None:
    head = HyperrectangleHead(hidden_dim=16, num_types=5, dim_per_type=10, margin=0.01)

    hyper_min, hyper_max = head(torch.randn(4, 16))

    assert hyper_min.shape == (4, 5, 10)
    assert hyper_max.shape == (4, 5, 10)
    assert torch.all(hyper_min >= 0.0)
    assert torch.all(hyper_max <= 1.0)
    assert torch.all(hyper_max - hyper_min >= 0.01 - 1e-6)


def test_hyperrectangle_head_starts_with_broad_non_saturated_boxes() -> None:
    head = HyperrectangleHead(
        hidden_dim=16,
        num_types=5,
        dim_per_type=5,
        margin=0.01,
        dropout=0.0,
    )

    hyper_min, hyper_max = head(torch.zeros(2, 16))
    volume = (hyper_max - hyper_min).prod(dim=-1)

    assert hasattr(head, "feature_proj")
    assert torch.all(volume > 0.4)
    assert torch.all(hyper_min > 0.01)
    assert torch.all(hyper_max < 0.99)


def test_true_volume_iou_matches_known_partial_overlap() -> None:
    min_a = torch.zeros(1, 1, 10)
    max_a = torch.ones(1, 1, 10)
    min_b = torch.zeros(1, 1, 10)
    max_b = torch.ones(1, 1, 10)
    max_a[..., 0] = 0.5
    min_b[..., 0] = 0.25
    max_b[..., 0] = 0.75

    geometry = hyperrectangle_geometry(min_a, max_a, min_b, max_b)

    assert geometry.volume_a.item() == pytest.approx(0.5)
    assert geometry.volume_b.item() == pytest.approx(0.5)
    assert geometry.intersection.item() == pytest.approx(0.25)
    assert geometry.iou.item() == pytest.approx(1 / 3)


def test_smooth_intersection_recovers_gradient_for_disjoint_boxes() -> None:
    min_a = torch.zeros(1, 1, 10, requires_grad=True)
    max_a = torch.full((1, 1, 10), 0.2, requires_grad=True)
    min_b = torch.full((1, 1, 10), 0.8, requires_grad=True)
    max_b = torch.ones(1, 1, 10, requires_grad=True)

    hard = hyperrectangle_geometry(min_a, max_a, min_b, max_b)
    losses = compute_coverage_geometry_losses(
        SimpleNamespace(hyper_min=min_a, hyper_max=max_a),
        SimpleNamespace(hyper_min=min_b, hyper_max=max_b),
        density_a=torch.full((1, 1), 0.2**10),
        density_b=torch.full((1, 1), 0.2**10),
        size_mask=torch.ones(1, 1, dtype=torch.bool),
        jaccard=torch.full((1, 1), 0.5),
        iou_mask=torch.ones(1, 1, dtype=torch.bool),
        min_width=0.01,
        smooth_temperature=0.01,
    )
    losses.iou_loss.backward()

    assert hard.intersection.item() == 0.0
    assert torch.isfinite(losses.geometry.log_iou).all()
    assert min_b.grad is not None
    assert torch.isfinite(min_b.grad).all()
    assert min_b.grad.abs().sum() > 0.0


def test_geometry_losses_are_zero_for_exact_volume_and_iou_targets() -> None:
    hyper_min = torch.zeros(1, 1, 10)
    hyper_max = torch.full((1, 1, 10), 0.5)
    output = SimpleNamespace(hyper_min=hyper_min, hyper_max=hyper_max)
    density = torch.full((1, 1), 0.5**10)

    losses = compute_coverage_geometry_losses(
        output,
        output,
        density_a=density,
        density_b=density,
        size_mask=torch.ones(1, 1, dtype=torch.bool),
        jaccard=torch.ones(1, 1),
        iou_mask=torch.ones(1, 1, dtype=torch.bool),
        min_width=0.01,
        smooth_temperature=None,
    )

    assert losses.volume_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert losses.iou_loss.item() == pytest.approx(0.0, abs=1e-7)


def test_volume_loss_pushes_identical_full_boxes_inward() -> None:
    hyper_min = torch.zeros(1, 1, 10, requires_grad=True)
    hyper_max = torch.ones(1, 1, 10, requires_grad=True)
    output = SimpleNamespace(hyper_min=hyper_min, hyper_max=hyper_max)

    losses = compute_coverage_geometry_losses(
        output,
        output,
        density_a=torch.full((1, 1), 0.25),
        density_b=torch.full((1, 1), 0.25),
        size_mask=torch.ones(1, 1, dtype=torch.bool),
        jaccard=torch.ones(1, 1),
        iou_mask=torch.ones(1, 1, dtype=torch.bool),
        min_width=0.01,
        smooth_temperature=None,
    )
    losses.volume_loss.backward()

    assert losses.iou_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert losses.volume_loss.item() > 0.0
    assert torch.all(hyper_min.grad < 0.0)
    assert torch.all(hyper_max.grad > 0.0)


def test_hard_validation_loss_is_finite_for_disjoint_positive_pair() -> None:
    output_a = SimpleNamespace(
        hyper_min=torch.zeros(1, 1, 10),
        hyper_max=torch.full((1, 1, 10), 0.2),
    )
    output_b = SimpleNamespace(
        hyper_min=torch.full((1, 1, 10), 0.8),
        hyper_max=torch.ones(1, 1, 10),
    )
    density = torch.full((1, 1), 0.2**10)

    losses = compute_coverage_geometry_losses(
        output_a,
        output_b,
        density_a=density,
        density_b=density,
        size_mask=torch.ones(1, 1, dtype=torch.bool),
        jaccard=torch.full((1, 1), 0.5),
        iou_mask=torch.ones(1, 1, dtype=torch.bool),
        min_width=0.01,
        smooth_temperature=None,
    )

    assert torch.isfinite(losses.iou_loss)
    assert losses.iou_loss > 0.0
