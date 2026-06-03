#!/usr/bin/env python3
"""对比损失 — 对齐超矩形交集度与覆盖率 Jaccard"""

import torch
import torch.nn.functional as F

from models.hyperrectangle import hyperrectangle_intersection


def compute_contrastive_loss(
    output_a,
    output_b,
    similarity: torch.Tensor,
    loss_type: str = "mse",
    margin: float = 0.2,
) -> torch.Tensor:
    """对齐超矩形交集度与覆盖率 Jaccard 相似度

    Args:
        output_a: ModelOutput（含 hyper_min, hyper_max）
        output_b: ModelOutput（含 hyper_min, hyper_max）
        similarity: [B] 两侧样本的覆盖率 Jaccard ∈ [0, 1]
        loss_type: "mse" | "bce" | "margin"
        margin: margin 模式下不相似对的交集度上限

    Returns:
        对比损失 scalar
    """
    intersection = hyperrectangle_intersection(
        output_a.hyper_min,
        output_a.hyper_max,
        output_b.hyper_min,
        output_b.hyper_max,
    )  # [B]

    if loss_type == "mse":
        return F.mse_loss(intersection, similarity)

    elif loss_type == "bce":
        return F.binary_cross_entropy(intersection.clamp(1e-7, 1 - 1e-7), similarity)

    elif loss_type == "margin":
        sim_mask = similarity > 0.5
        loss_sim = (
            F.mse_loss(intersection[sim_mask], similarity[sim_mask])
            if sim_mask.any()
            else torch.tensor(0.0, device=intersection.device)
        )
        loss_dissim = (
            F.relu(intersection[~sim_mask] - margin).mean()
            if (~sim_mask).any()
            else torch.tensor(0.0, device=intersection.device)
        )
        return loss_sim + loss_dissim

    else:
        raise ValueError(f"未知对比损失类型: {loss_type}")
