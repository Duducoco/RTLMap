#!/usr/bin/env python3
"""模型任务头。"""

from typing import Optional

import torch
import torch.nn as nn

from .encoder import RTLEdgeFeatureEncoder


class EdgeClassifier(nn.Module):
    """RTL 边覆盖分类头。"""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        num_edge_types: int = 5,
        max_ports: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_encoder = RTLEdgeFeatureEncoder(num_edge_types, hidden_dim, max_ports)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_width: Optional[torch.Tensor] = None,
        source_port_idx: Optional[torch.Tensor] = None,
        target_port_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, tgt = edge_index
        src_feat = node_features[src]
        tgt_feat = node_features[tgt]
        edge_feat = self.edge_encoder(
            edge_type, edge_width, source_port_idx, target_port_idx
        )
        combined = torch.cat([src_feat, tgt_feat, edge_feat], dim=-1)
        return self.net(combined)


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
