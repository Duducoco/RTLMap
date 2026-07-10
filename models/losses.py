#!/usr/bin/env python3
"""模型基础监督损失。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from datasets import DualGraphData
from datasets.data_types import coverage_key_index
from .data_types import ModelOutput


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
            per_target_losses = [
                F.smooth_l1_loss(
                    output.graph_pred[column_valid, column],
                    target[column_valid, column],
                )
                for column in range(target.size(1))
                if (column_valid := valid[:, column]).any()
            ]
            graph_loss = torch.stack(per_target_losses).mean()
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
