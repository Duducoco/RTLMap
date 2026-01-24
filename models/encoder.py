#!/usr/bin/env python3
"""
图编码器模块

支持边类型嵌入的 GNN 编码器。
CDFG 中边类型语义重要：DATA（数据流）、CONTROL（控制流）、CLOCK、RESET、ENABLE

设计选择：
- 使用 GINEConv（GIN with Edge features）支持边特征
- RTL 和 ASM 使用独立的 GNN 层和边类型嵌入
- DropPath 替代 Dropout（更稳定）
- 跨图注意力共享参数（连接两图的桥梁）
"""

import warnings
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple

from .interaction import CrossGraphInteraction


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic Depth (论文: Deep Networks with Stochastic Depth)"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class GNNLayer(nn.Module):
    """
    单层 GNN（GINEConv + LayerNorm + DropPath）

    使用 GINEConv 支持边特征，边类型嵌入在 forward 时传入
    """

    def __init__(self, hidden_dim: int, drop_path_rate: float = 0.1):
        super().__init__()
        # GINEConv 需要边特征维度与节点特征维度相同
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(self.mlp, edge_dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop_path_rate = drop_path_rate

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, D] 边特征（边类型嵌入）

        Returns:
            [N, D] 更新后的节点特征
        """
        h = self.conv(x, edge_index, edge_attr)
        h = drop_path(h, self.drop_path_rate, self.training)
        return self.norm(x + h)


class InteractiveDualEncoder(nn.Module):
    """
    交互式双图编码器

    架构：每层 GNN 后进行跨图交互
    特性：
    - RTL 和 ASM 使用独立的 GNN 层（不共享参数）
    - RTL 和 ASM 使用独立的边类型嵌入（不共享参数）
    - DropPath 逐层递增
    - 跨图注意力共享参数（连接两图的桥梁）
    """

    def __init__(
        self,
        rtl_node_dim: int,
        asm_node_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_edge_types: int = 5,  # DATA, CONTROL, CLOCK, RESET, ENABLE
        **kwargs,  # 兼容旧参数
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 输入投影（独立）
        self.rtl_input = nn.Sequential(
            nn.Linear(rtl_node_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.asm_input = nn.Sequential(
            nn.Linear(asm_node_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        # 边类型嵌入（独立）
        self.rtl_edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.asm_edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # GNN 层（独立，逐层增加 drop_path 概率）
        drop_rates = [
            dropout * i / (num_layers - 1) if num_layers > 1 else 0
            for i in range(num_layers)
        ]

        self.rtl_layers = nn.ModuleList(
            [GNNLayer(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )
        self.asm_layers = nn.ModuleList(
            [GNNLayer(hidden_dim, drop_rates[i]) for i in range(num_layers)]
        )

        # 跨图交互（共享参数，作为连接两图的桥梁）
        self.cross_attn = CrossGraphInteraction(hidden_dim, num_heads, dropout)

        # 输出投影（独立）
        self.rtl_output = nn.Linear(hidden_dim, output_dim)
        self.asm_output = nn.Linear(hidden_dim, output_dim)

    def _get_edge_attr(
        self,
        edge_type: Optional[torch.Tensor],
        embedding: nn.Embedding,
        num_edges: int,
        device,
    ) -> torch.Tensor:
        """获取边特征：如果有边类型则嵌入，否则使用零向量"""
        if edge_type is not None:
            return embedding(edge_type)
        else:
            warnings.warn(
                "Edge types are required for GINEConv. Using zero vectors instead."
            )
            # 无边类型时使用零向量（不影响 GINEConv 聚合）
            return torch.zeros(num_edges, self.hidden_dim, device=device)

    def forward(
        self,
        rtl_x: torch.Tensor,
        rtl_edge_index: torch.Tensor,
        asm_x: torch.Tensor,
        asm_edge_index: torch.Tensor,
        rtl_batch: Optional[torch.Tensor] = None,
        asm_batch: Optional[torch.Tensor] = None,
        rtl_edge_type: Optional[torch.Tensor] = None,
        asm_edge_type: Optional[torch.Tensor] = None,
        **kwargs,  # 忽略其他参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        前向传播

        Args:
            rtl_x: [N1, D1] RTL 节点特征
            rtl_edge_index: [2, E1] RTL 边索引
            asm_x: [N2, D2] ASM 节点特征
            asm_edge_index: [2, E2] ASM 边索引
            rtl_batch: [N1] RTL batch 索引
            asm_batch: [N2] ASM batch 索引
            rtl_edge_type: [E1] RTL 边类型（可选）
            asm_edge_type: [E2] ASM 边类型（可选）

        Returns:
            rtl_node: [N1, D] RTL 节点嵌入
            rtl_graph: [B, D] RTL 图级嵌入
            asm_node: [N2, D] ASM 节点嵌入
            asm_graph: [B, D] ASM 图级嵌入
            cross_attentions: 跨图注意力权重列表
        """
        device = rtl_x.device

        # 输入投影
        rtl_h = self.rtl_input(rtl_x)
        asm_h = self.asm_input(asm_x)

        # 边类型嵌入（使用各自独立的嵌入层）
        rtl_edge_attr = self._get_edge_attr(
            rtl_edge_type, self.rtl_edge_type_embedding, rtl_edge_index.size(1), device
        )
        asm_edge_attr = self._get_edge_attr(
            asm_edge_type, self.asm_edge_type_embedding, asm_edge_index.size(1), device
        )

        cross_attentions = []

        for i in range(self.num_layers):
            # GNN（带边特征，独立参数）
            rtl_h = self.rtl_layers[i](rtl_h, rtl_edge_index, rtl_edge_attr)
            asm_h = self.asm_layers[i](asm_h, asm_edge_index, asm_edge_attr)

            # 跨图交互
            if rtl_batch is not None:
                rtl_dense, rtl_mask = to_dense_batch(rtl_h, rtl_batch)
                asm_dense, asm_mask = to_dense_batch(asm_h, asm_batch)
            else:
                rtl_dense = rtl_h.unsqueeze(0)
                asm_dense = asm_h.unsqueeze(0)
                rtl_mask = asm_mask = None

            rtl_inter, asm_inter = self.cross_attn(
                rtl_dense, asm_dense, rtl_mask, asm_mask
            )
            cross_attentions.append(None)  # 简化：不返回注意力权重

            # 转回 sparse
            if rtl_batch is not None:
                rtl_h = rtl_inter[rtl_mask]
                asm_h = asm_inter[asm_mask]
            else:
                rtl_h = rtl_inter.squeeze(0)
                asm_h = asm_inter.squeeze(0)

        # 输出
        rtl_node = self.rtl_output(rtl_h)
        asm_node = self.asm_output(asm_h)

        if rtl_batch is not None:
            rtl_graph = global_mean_pool(rtl_node, rtl_batch)
            asm_graph = global_mean_pool(asm_node, asm_batch)
        else:
            rtl_graph = rtl_node.mean(0, keepdim=True)
            asm_graph = asm_node.mean(0, keepdim=True)

        return rtl_node, rtl_graph, asm_node, asm_graph, cross_attentions
