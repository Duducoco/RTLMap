#!/usr/bin/env python3
"""Losses for typed true-volume coverage-space geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from models.hyperrectangle import HyperrectangleGeometry, hyperrectangle_geometry

DEFAULT_IOU_RANK_LOSS_WEIGHT = 0.5
DEFAULT_IOU_RANK_MARGIN = 0.05
DEFAULT_IOU_RANK_MIN_TARGET_GAP = 0.05


@dataclass(frozen=True)
class IoURankingConfig:
    weight: float = DEFAULT_IOU_RANK_LOSS_WEIGHT
    margin: float = DEFAULT_IOU_RANK_MARGIN
    min_target_gap: float = DEFAULT_IOU_RANK_MIN_TARGET_GAP

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("IoU ranking weight must be non-negative")
        if self.margin < 0.0:
            raise ValueError("IoU ranking margin must be non-negative")
        if self.min_target_gap < 0.0:
            raise ValueError("IoU ranking minimum target gap must be non-negative")


DEFAULT_IOU_RANKING_CONFIG = IoURankingConfig()


@dataclass(frozen=True)
class CoverageGeometryLosses:
    volume_loss: torch.Tensor
    iou_loss: torch.Tensor
    iou_calibration_loss: torch.Tensor
    iou_rank_loss: torch.Tensor
    geometry: HyperrectangleGeometry


def _weighted_type_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Weight samples within each type, then give each defined type equal weight."""
    if values.shape != mask.shape:
        raise ValueError("weighted values and mask must have the same shape")
    if sample_weight.shape != (values.shape[0],):
        raise ValueError("sample_weight must have shape [B]")
    if torch.any(sample_weight < 0):
        raise ValueError("sample_weight must be non-negative")
    weights = mask.to(values.dtype) * sample_weight.to(values.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=0)
    defined = denominator > 0
    if not defined.any():
        return values.sum() * 0.0
    per_type = (values * weights).sum(dim=0) / denominator.clamp_min(1e-12)
    return per_type[defined].mean()


def _equal_type_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_type_mean(
        values,
        mask,
        torch.ones(values.shape[0], device=values.device, dtype=values.dtype),
    )


def _pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
    min_target_gap: float,
) -> torch.Tensor:
    type_losses: list[torch.Tensor] = []
    for column in range(prediction.shape[1]):
        valid = mask[:, column]
        pred = prediction[valid, column]
        truth = target[valid, column]
        if pred.numel() < 2:
            continue
        row, col = torch.triu_indices(
            pred.numel(), pred.numel(), offset=1, device=prediction.device
        )
        target_delta = truth[row] - truth[col]
        target_gap = target_delta.abs()
        informative = (target_gap > 0.0) & (target_gap >= min_target_gap)
        if not informative.any():
            continue
        signed_prediction_delta = target_delta[informative].sign() * (
            pred[row[informative]] - pred[col[informative]]
        )
        type_losses.append(
            F.relu(margin - signed_prediction_delta).mean()
        )
    if not type_losses:
        return prediction.sum() * 0.0
    return torch.stack(type_losses).mean()


def compute_coverage_geometry_losses(
    output_a,
    output_b,
    *,
    density_a: torch.Tensor,
    density_b: torch.Tensor,
    size_mask: torch.Tensor,
    jaccard: torch.Tensor,
    iou_mask: torch.Tensor,
    endpoint_weight_a: torch.Tensor | None = None,
    endpoint_weight_b: torch.Tensor | None = None,
    min_width: float,
    smooth_temperature: float | None,
    iou_ranking: IoURankingConfig = DEFAULT_IOU_RANKING_CONFIG,
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
    batch_size = expected_shape[0]
    if endpoint_weight_a is None:
        endpoint_weight_a = torch.ones(batch_size, device=density_a.device)
    if endpoint_weight_b is None:
        endpoint_weight_b = torch.ones(batch_size, device=density_b.device)
    if endpoint_weight_a.shape != (batch_size,) or endpoint_weight_b.shape != (
        batch_size,
    ):
        raise ValueError("endpoint weights must have shape [B]")

    box_dim = output_a.hyper_min.shape[-1]
    minimum_volume = float(min_width) ** box_dim
    target_log_volume_a = torch.log(
        density_a.float().clamp(min=minimum_volume, max=1.0)
    )
    target_log_volume_b = torch.log(
        density_b.float().clamp(min=minimum_volume, max=1.0)
    )
    volume_error_a = F.smooth_l1_loss(
        geometry.log_volume_a, target_log_volume_a, reduction="none"
    )
    volume_error_b = F.smooth_l1_loss(
        geometry.log_volume_b, target_log_volume_b, reduction="none"
    )
    volume_loss = _weighted_type_mean(
        torch.cat((volume_error_a, volume_error_b), dim=0),
        torch.cat((size_mask, size_mask), dim=0),
        torch.cat((endpoint_weight_a, endpoint_weight_b), dim=0),
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
    iou_calibration_loss = _equal_type_mean(iou_error, iou_mask)
    iou_rank_loss = _pairwise_rank_loss(
        geometry.iou,
        jaccard_float,
        iou_mask,
        margin=iou_ranking.margin,
        min_target_gap=iou_ranking.min_target_gap,
    )
    iou_loss = iou_calibration_loss + iou_ranking.weight * iou_rank_loss
    return CoverageGeometryLosses(
        volume_loss=volume_loss,
        iou_loss=iou_loss,
        iou_calibration_loss=iou_calibration_loss,
        iou_rank_loss=iou_rank_loss,
        geometry=geometry,
    )
