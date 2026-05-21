#!/usr/bin/env python3
"""训练策略级损失函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.data_types import ModelOutput


def compute_or_consistency_loss(
    out_a: ModelOutput,
    out_b: ModelOutput,
    out_m: ModelOutput,
    merged_labels: torch.Tensor | None,
) -> torch.Tensor:
    """计算 merged 覆盖率的逻辑 OR 一致性约束。"""
    device = out_a.edge_logits.device
    loss = torch.tensor(0.0, device=device)
    if merged_labels is None:
        return loss
    if out_a.edge_logits.size(0) != out_b.edge_logits.size(0):
        return loss
    if out_a.edge_logits.size(0) != out_m.edge_logits.size(0):
        return loss

    prob_a = torch.softmax(out_a.edge_logits, dim=-1)[:, 1]
    prob_b = torch.softmax(out_b.edge_logits, dim=-1)[:, 1]
    prob_or = 1.0 - (1.0 - prob_a) * (1.0 - prob_b)
    valid = merged_labels != -1
    if valid.any():
        loss = F.binary_cross_entropy(
            prob_or[valid].clamp(1e-6, 1 - 1e-6),
            merged_labels[valid].float(),
        )
    return loss
