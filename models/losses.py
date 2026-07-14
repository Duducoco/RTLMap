#!/usr/bin/env python3
"""模型基础监督损失。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from datasets import DualGraphData
from datasets.data_types import coverage_key_index
from .data_types import ModelOutput


def compute_weighted_graph_loss(
    graph_pred: torch.Tensor,
    target: torch.Tensor,
    endpoint_weight: torch.Tensor,
    *,
    coverage_target_keys: tuple = ("branch",),
) -> torch.Tensor:
    """Compute a type-equal graph loss with sample weights inside each type."""
    if graph_pred.ndim != 2 or graph_pred.shape[1] != len(coverage_target_keys):
        raise ValueError("graph_pred must have shape [B, len(coverage_target_keys)]")
    if target.ndim != 2 or target.shape[0] != graph_pred.shape[0]:
        raise ValueError("target must have shape [B, num_coverage_types]")
    if endpoint_weight.shape != (graph_pred.shape[0],):
        raise ValueError("endpoint_weight must have shape [B]")
    if torch.any(endpoint_weight < 0):
        raise ValueError("endpoint_weight must be non-negative")

    col_idx = [coverage_key_index(key) for key in coverage_target_keys]
    selected_target = target[:, col_idx]
    type_losses: list[torch.Tensor] = []
    for column in range(selected_target.shape[1]):
        valid = ~torch.isnan(selected_target[:, column])
        if not valid.any():
            continue
        error = F.smooth_l1_loss(
            graph_pred[valid, column],
            selected_target[valid, column],
            reduction="none",
        )
        weights = endpoint_weight[valid].to(error.dtype)
        type_losses.append((error * weights).sum() / valid.sum())
    if not type_losses:
        return graph_pred.sum() * 0.0
    return torch.stack(type_losses).mean()


def compute_supervised_losses(
    output: ModelOutput,
    data: DualGraphData,
    coverage_target_keys: tuple = ("branch",),
    graph_loss_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """计算图级覆盖率回归监督损失。"""
    if output.graph_pred is not None:
        device = output.graph_pred.device
    else:
        device = data.y.device if data.y is not None else torch.device("cpu")
    losses = {}

    if data.y is not None and output.graph_pred is not None:
        col_idx = [coverage_key_index(k) for k in coverage_target_keys]
        target = data.y[:, col_idx]
        valid = ~torch.isnan(target)
        if valid.any():
            graph_loss = compute_weighted_graph_loss(
                output.graph_pred,
                data.y,
                torch.ones(
                    output.graph_pred.shape[0],
                    device=output.graph_pred.device,
                    dtype=output.graph_pred.dtype,
                ),
                coverage_target_keys=coverage_target_keys,
            )
            losses["num_valid_graph_targets"] = int(valid.sum().item())
        else:
            graph_loss = output.graph_pred.sum() * 0.0
            losses["num_valid_graph_targets"] = 0
        losses["graph_loss"] = graph_loss * graph_loss_weight
    else:
        losses["graph_loss"] = torch.tensor(0.0, device=device)
        losses["num_valid_graph_targets"] = 0

    losses["total_loss"] = losses["graph_loss"]
    return losses
