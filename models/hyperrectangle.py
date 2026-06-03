#!/usr/bin/env python3
"""超矩形表示头 — 将图级嵌入映射为单位超立方体中的超矩形"""

import torch
import torch.nn as nn


class HyperrectangleHead(nn.Module):
    """将 rtl_graph [B, D] 映射为单位超立方体中的超矩形 (v_min, v_max)

    参数化方式：v_min = sigmoid(min_proj(x))
                extent_raw = sigmoid(extent_proj(x))
                v_max = v_min + margin + extent_raw * (1 - v_min - margin)

    该组合公式保证：
    - v_min ∈ [0, 1]
    - v_max >= v_min + margin（每个维度最小宽度，防止矩形坍缩为点）
    - v_max <= 1.0
    - 全程可微，无需 clamp
    """

    def __init__(self, hidden_dim: int, margin: float = 0.01):
        super().__init__()
        self.min_proj = nn.Linear(hidden_dim, hidden_dim)
        self.extent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.margin = margin

    def forward(self, rtl_graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_min = torch.sigmoid(self.min_proj(rtl_graph))
        extent_raw = torch.sigmoid(self.extent_proj(rtl_graph))
        v_max = v_min + self.margin + extent_raw * (1.0 - v_min - self.margin)
        return v_min, v_max


def hyperrectangle_intersection(
    v_min_1: torch.Tensor,
    v_max_1: torch.Tensor,
    v_min_2: torch.Tensor,
    v_max_2: torch.Tensor,
) -> torch.Tensor:
    """计算两个超矩形的交集度（维度平均）

    每个维度上的交集长度 = max(0, min(v_max_1_i, v_max_2_i) - max(v_min_1_i, v_min_2_i))
    交集度 = 各维度交集长度的均值 ∈ [0, 1]

    使用维度平均而非乘积（体积），因为 D=256 维乘积梯度近零。

    Args:
        v_min_1, v_max_1: [B, D] 或 [D] 第一个超矩形
        v_min_2, v_max_2: [B, D] 或 [D] 第二个超矩形

    Returns:
        [B] 或 scalar 交集度 ∈ [0, 1]
    """
    overlap = torch.clamp(
        torch.min(v_max_1, v_max_2) - torch.max(v_min_1, v_min_2), min=0.0
    )
    return overlap.mean(dim=-1)
