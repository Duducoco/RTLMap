#!/usr/bin/env python3
"""模型任务头。"""

import torch
import torch.nn as nn


class CoverageHead(nn.Module):
    """单个图级覆盖率预测头。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rtl_graph_emb: torch.Tensor) -> torch.Tensor:
        return self.net(rtl_graph_emb)


class GraphRegressor(nn.Module):
    """图级覆盖率回归头集合。"""

    def __init__(
        self,
        hidden_dim: int,
        coverage_target_keys: tuple[str, ...] = ("branch",),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.coverage_target_keys = tuple(coverage_target_keys)
        self.heads = nn.ModuleDict(
            {
                key: CoverageHead(hidden_dim, dropout)
                for key in self.coverage_target_keys
            }
        )

    def forward(self, rtl_graph_emb: torch.Tensor) -> torch.Tensor:
        preds = [self.heads[key](rtl_graph_emb) for key in self.coverage_target_keys]
        return torch.cat(preds, dim=-1)
