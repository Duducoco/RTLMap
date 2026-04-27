#!/usr/bin/env python3
"""覆盖率相似度计算 — 用于对比学习的对间 Jaccard 相似度"""

import torch


def compute_edge_jaccard(
    edge_labels_1: torch.Tensor, edge_labels_2: torch.Tensor
) -> float:
    """计算两个刺激在共同有效边上的覆盖率 Jaccard 相似度

    仅在两边都标注的边上比较（排除 -1 未标注边）。
    Jaccard = |intersection| / |union| ∈ [0, 1]。
    两个刺激均未覆盖任何共同有效边时返回 0.0。

    Args:
        edge_labels_1: [E] 第一个刺激的边标签（-1/0/1）
        edge_labels_2: [E] 第二个刺激的边标签（-1/0/1）

    Returns:
        Jaccard 相似度 ∈ [0.0, 1.0]
    """
    valid_both = (edge_labels_1 != -1) & (edge_labels_2 != -1)
    covered_1 = (edge_labels_1 == 1) & valid_both
    covered_2 = (edge_labels_2 == 1) & valid_both
    intersection = (covered_1 & covered_2).sum().float()
    union = (covered_1 | covered_2).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()