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
    edge_loss_weight: float = 1.0,
    graph_loss_weight: float = 1.0,
    label_smoothing: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """计算边分类和图级回归监督损失。"""
    device = output.edge_logits.device
    losses = {}

    edge_labels = getattr(data, "edge_labels", None)
    if edge_labels is not None:
        valid_mask = edge_labels != -1
        num_valid = valid_mask.sum().item()

        if num_valid > 0:
            valid_logits = output.edge_logits[valid_mask]
            valid_labels = edge_labels[valid_mask]
            edge_loss = F.cross_entropy(
                valid_logits, valid_labels, label_smoothing=label_smoothing
            )
            losses["edge_loss"] = edge_loss * edge_loss_weight
        else:
            losses["edge_loss"] = torch.tensor(0.0, device=device)

        losses["num_valid_edges"] = num_valid
    else:
        losses["edge_loss"] = torch.tensor(0.0, device=device)
        losses["num_valid_edges"] = 0

    if data.y is not None:
        col_idx = [coverage_key_index(k) for k in coverage_target_keys]
        target = data.y[:, col_idx]
        valid = ~torch.isnan(target)
        if valid.any():
            graph_loss = F.smooth_l1_loss(output.graph_pred[valid], target[valid])
            losses["num_valid_graph_targets"] = int(valid.sum().item())
        else:
            graph_loss = output.graph_pred.sum() * 0.0
            losses["num_valid_graph_targets"] = 0
        losses["graph_loss"] = graph_loss * graph_loss_weight
    else:
        losses["graph_loss"] = torch.tensor(0.0, device=device)
        losses["num_valid_graph_targets"] = 0

    losses["total_loss"] = losses["edge_loss"] + losses["graph_loss"]
    return losses
