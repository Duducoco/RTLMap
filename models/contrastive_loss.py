#!/usr/bin/env python3
"""Losses for typed true-volume coverage-space geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from models.hyperrectangle import HyperrectangleGeometry, hyperrectangle_geometry


@dataclass(frozen=True)
class CoverageGeometryLosses:
    volume_loss: torch.Tensor
    iou_loss: torch.Tensor
    geometry: HyperrectangleGeometry


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(
            f"masked values and mask shapes differ: {values.shape} != {mask.shape}"
        )
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_coverage_geometry_losses(
    output_a,
    output_b,
    *,
    density_a: torch.Tensor,
    density_b: torch.Tensor,
    size_mask: torch.Tensor,
    jaccard: torch.Tensor,
    iou_mask: torch.Tensor,
    min_width: float,
    smooth_temperature: float | None,
) -> CoverageGeometryLosses:
    """Supervise typed true volumes and pairwise true-volume IoU."""
    geometry = hyperrectangle_geometry(
        output_a.hyper_min,
        output_a.hyper_max,
        output_b.hyper_min,
        output_b.hyper_max,
        smooth_temperature=smooth_temperature,
    )
    expected_shape = geometry.iou.shape
    targets = (density_a, density_b, size_mask, jaccard, iou_mask)
    if any(target.shape != expected_shape for target in targets):
        raise ValueError(
            "coverage geometry targets must match [B, T] box geometry; "
            f"expected {expected_shape}, got {[target.shape for target in targets]}"
        )

    box_dim = output_a.hyper_min.shape[-1]
    minimum_volume = float(min_width) ** box_dim
    target_log_scale_a = torch.log(
        density_a.float().clamp(min=minimum_volume, max=1.0)
    ) / box_dim
    target_log_scale_b = torch.log(
        density_b.float().clamp(min=minimum_volume, max=1.0)
    ) / box_dim
    volume_error_a = F.smooth_l1_loss(
        geometry.mean_log_width_a, target_log_scale_a, reduction="none"
    )
    volume_error_b = F.smooth_l1_loss(
        geometry.mean_log_width_b, target_log_scale_b, reduction="none"
    )
    volume_loss = 0.5 * (
        _masked_mean(volume_error_a, size_mask)
        + _masked_mean(volume_error_b, size_mask)
    )
    jaccard_float = jaccard.float()
    positive_target = jaccard_float > 0.0
    log_iou_for_loss = (
        geometry.log_iou
        if smooth_temperature is not None
        else geometry.log_iou.clamp_min(math.log(1e-12))
    )
    positive_error = F.smooth_l1_loss(
        log_iou_for_loss,
        torch.log(jaccard_float.clamp_min(1e-12)),
        reduction="none",
    )
    zero_error = geometry.iou.square()
    iou_error = torch.where(positive_target, positive_error, zero_error)
    iou_loss = _masked_mean(iou_error, iou_mask)
    return CoverageGeometryLosses(
        volume_loss=volume_loss,
        iou_loss=iou_loss,
        geometry=geometry,
    )
