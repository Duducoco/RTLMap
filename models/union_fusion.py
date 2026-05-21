#!/usr/bin/env python3
"""联合覆盖率训练使用的对称融合模块。"""

import torch
import torch.nn as nn


class SymmetricUnionFusion(nn.Module):
    """Order-invariant gated union fusion for two aligned feature tensors."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        union_input = torch.cat(
            [
                feat_a + feat_b,
                torch.maximum(feat_a, feat_b),
                torch.abs(feat_a - feat_b),
                feat_a * feat_b,
            ],
            dim=-1,
        )
        gate = self.gate(union_input)
        h_base = 0.5 * (feat_a + feat_b)
        h_max = torch.maximum(feat_a, feat_b)
        return self.norm(gate * h_max + (1.0 - gate) * h_base)
