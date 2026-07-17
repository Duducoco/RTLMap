#!/usr/bin/env python3
"""Typed true-volume hyperrectangles for learned RTL coverage spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HyperrectangleGeometry:
    volume_a: torch.Tensor
    volume_b: torch.Tensor
    intersection: torch.Tensor
    iou: torch.Tensor
    log_iou: torch.Tensor
    mean_log_width_a: torch.Tensor
    mean_log_width_b: torch.Tensor
    has_hard_intersection: torch.Tensor


class HyperrectangleHead(nn.Module):
    """Map graph embeddings to one center-radius subrectangle per coverage type."""

    def __init__(
        self,
        hidden_dim: int,
        num_types: int = 5,
        dim_per_type: int = 10,
        margin: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < margin < 1.0:
            raise ValueError("margin must be between zero and one")
        if num_types <= 0 or dim_per_type <= 0:
            raise ValueError("num_types and dim_per_type must be positive")
        output_dim = num_types * dim_per_type
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.center_proj = nn.Linear(hidden_dim, output_dim)
        self.radius_proj = nn.Linear(hidden_dim, output_dim)
        self.num_types = num_types
        self.dim_per_type = dim_per_type
        self.margin = margin
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        feature_linear = self.feature_proj[1]
        nn.init.xavier_uniform_(feature_linear.weight)
        nn.init.zeros_(feature_linear.bias)
        nn.init.xavier_uniform_(self.center_proj.weight, gain=0.1)
        nn.init.zeros_(self.center_proj.bias)
        nn.init.xavier_uniform_(self.radius_proj.weight, gain=0.1)
        nn.init.constant_(self.radius_proj.bias, 2.0)

    def forward(self, rtl_graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (-1, self.num_types, self.dim_per_type)
        features = rtl_graph + self.feature_proj(rtl_graph)
        center_ratio = torch.sigmoid(self.center_proj(features)).view(shape)
        radius_ratio = torch.sigmoid(self.radius_proj(features)).view(shape)

        half_margin = self.margin / 2.0
        center = half_margin + (1.0 - self.margin) * center_ratio
        max_half_width = torch.minimum(center, 1.0 - center)
        half_width = half_margin + radius_ratio * (max_half_width - half_margin)
        return center - half_width, center + half_width


def _overlap_width(
    v_min_1: torch.Tensor,
    v_max_1: torch.Tensor,
    v_min_2: torch.Tensor,
    v_max_2: torch.Tensor,
    *,
    smooth_temperature: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.minimum(v_max_1, v_max_2) - torch.maximum(v_min_1, v_min_2)
    hard_overlap = delta.clamp_min(0.0)
    if smooth_temperature is None:
        return hard_overlap, hard_overlap
    if smooth_temperature <= 0.0:
        raise ValueError("smooth_temperature must be positive")
    soft_overlap = smooth_temperature * F.softplus(delta / smooth_temperature)
    max_overlap = torch.minimum(v_max_1 - v_min_1, v_max_2 - v_min_2)
    return torch.minimum(soft_overlap, max_overlap), hard_overlap


def hyperrectangle_geometry(
    v_min_1: torch.Tensor,
    v_max_1: torch.Tensor,
    v_min_2: torch.Tensor,
    v_max_2: torch.Tensor,
    *,
    smooth_temperature: float | None = None,
    eps: float = 1e-12,
) -> HyperrectangleGeometry:
    """Compute per-type true volumes and IoU in float32 log space."""
    tensors = (v_min_1, v_max_1, v_min_2, v_max_2)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("typed hyperrectangles must have shape [B, T, D_box]")
    if any(tensor.shape != v_min_1.shape for tensor in tensors[1:]):
        raise ValueError("all hyperrectangle tensors must have the same shape")

    min_1, max_1, min_2, max_2 = (tensor.float() for tensor in tensors)
    width_1 = (max_1 - min_1).clamp_min(eps)
    width_2 = (max_2 - min_2).clamp_min(eps)
    overlap, hard_overlap = _overlap_width(
        min_1,
        max_1,
        min_2,
        max_2,
        smooth_temperature=smooth_temperature,
    )

    log_volume_1 = torch.log(width_1).sum(dim=-1)
    log_volume_2 = torch.log(width_2).sum(dim=-1)
    log_intersection = torch.log(overlap.clamp_min(eps)).sum(dim=-1)
    volume_1 = torch.exp(log_volume_1)
    volume_2 = torch.exp(log_volume_2)
    intersection = torch.exp(log_intersection)
    has_hard_intersection = (hard_overlap > 0.0).all(dim=-1)
    if smooth_temperature is None:
        intersection = torch.where(
            has_hard_intersection, intersection, torch.zeros_like(intersection)
        )
    max_log_volume = torch.maximum(log_volume_1, log_volume_2)
    scaled_union = (
        torch.exp(log_volume_1 - max_log_volume)
        + torch.exp(log_volume_2 - max_log_volume)
        - torch.exp(log_intersection - max_log_volume)
    ).clamp_min(eps)
    log_union = max_log_volume + torch.log(scaled_union)
    log_iou = log_intersection - log_union
    if smooth_temperature is None:
        log_iou = torch.where(
            has_hard_intersection,
            log_iou,
            torch.full_like(log_iou, -torch.inf),
        )

    return HyperrectangleGeometry(
        volume_a=volume_1,
        volume_b=volume_2,
        intersection=intersection,
        iou=torch.exp(log_iou).clamp(0.0, 1.0),
        log_iou=log_iou,
        mean_log_width_a=log_volume_1 / width_1.shape[-1],
        mean_log_width_b=log_volume_2 / width_2.shape[-1],
        has_hard_intersection=has_hard_intersection,
    )
