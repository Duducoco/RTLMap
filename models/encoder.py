#!/usr/bin/env python3
"""
图编码器模块

RTL 节点编码：node_cell_type + node_width
RTL 边编码：edge_type + edge_width

ASM 节点编码：node_type + instruction_encoding（外部模型生成）
ASM 边编码：edge_type（仅类型嵌入）

设计选择：
- 使用 MessagePassing 实现支持边特征更新的 GNN 层
- RTL 和 ASM 使用独立的 GNN 层和边类型嵌入
- DropPath 替代 Dropout（更稳定）
- 跨图注意力共享参数（连接两图的桥梁）
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
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


class RTLNodeFeatureEncoder(nn.Module):
    """
    RTL 节点特征编码器

    将 cell_type 索引和 width 编码为节点嵌入：
    - cell_type: 通过 Embedding 层嵌入
    - width: 使用 log2(width + 1) 编码后通过线性层
    - 融合方式: 加法融合
    """

    def __init__(self, num_types: int, hidden_dim: int):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.width_encoder = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_type: torch.Tensor, node_width: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_type: [N] 节点类型索引
            node_width: [N] 节点 width

        Returns:
            [N, hidden_dim] 节点嵌入
        """
        type_emb = self.type_embedding(node_type)  # [N, D]
        width_log = torch.log2(node_width.float() + 1).unsqueeze(-1)  # [N, 1]
        width_emb = self.width_encoder(width_log)  # [N, D]
        return self.norm(type_emb + width_emb)


class AsmNodeFeatureEncoder(nn.Module):
    """
    ASM 节点特征编码器

    将 node_type 和 instruction_encoding 编码为节点嵌入：
    - node_type: 通过 Embedding 层嵌入
    - instruction_encoding: 通过线性层投影（由外部模型生成）
    - 融合方式: 加法融合
    """

    def __init__(self, num_node_types: int, instruction_dim: int, hidden_dim: int):
        """
        Args:
            num_node_types: ASM 节点类型数量 (AsmNodeType 枚举)
            instruction_dim: 指令编码维度（由外部模型生成）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.instruction_proj = nn.Linear(instruction_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_type: torch.Tensor, instruction_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_type: [M] ASM 节点类型索引
            instruction_encoding: [M, D_instr] 指令编码（由外部模型生成）

        Returns:
            [M, hidden_dim] 节点嵌入
        """
        type_emb = self.node_type_embedding(node_type)  # [M, D]
        instr_emb = self.instruction_proj(instruction_encoding)  # [M, D]
        return self.norm(type_emb + instr_emb)


class RTLEdgeFeatureEncoder(nn.Module):
    """
    RTL 边特征编码器

    将 edge_type 和 width 编码为边嵌入：
    - edge_type: 通过 Embedding 层嵌入
    - width: 使用 log2(width + 1) 编码后通过线性层
    - 融合方式: 加法融合
    """

    def __init__(self, num_edge_types: int, hidden_dim: int):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.width_encoder = nn.Linear(1, hidden_dim)

    def forward(
        self,
        edge_type: torch.Tensor,
        edge_width: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            edge_type: [E] 边类型索引
            edge_width: [E] 边 width（可选）

        Returns:
            [E, hidden_dim] 边嵌入
        """
        type_emb = self.edge_type_embedding(edge_type)  # [E, D]

        if edge_width is not None:
            width_log = torch.log2(edge_width.float() + 1).unsqueeze(-1)  # [E, 1]
            width_emb = self.width_encoder(width_log)  # [E, D]
            return type_emb + width_emb
        else:
            return type_emb


class AsmEdgeFeatureEncoder(nn.Module):
    """
    ASM 边特征编码器

    仅使用 edge_type 嵌入（ASM 边不需要 width）
    """

    def __init__(self, num_edge_types: int, hidden_dim: int):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_type: [F] 边类型索引

        Returns:
            [F, hidden_dim] 边嵌入
        """
        return self.edge_type_embedding(edge_type)


class GNNLayer(MessagePassing):
    """
    支持边特征更新的 GNN 层

    每次前向传播同时更新：
    1. 节点特征：通过消息聚合（邻居特征 + 边特征）
    2. 边特征：通过源/目标节点特征和当前边特征

    使用残差连接和 LayerNorm 确保训练稳定性
    """

    def __init__(self, hidden_dim: int, drop_path_rate: float = 0.1):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path_rate

        # 节点更新 MLP: [neighbor, edge] -> node_update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 边更新 MLP: [src, tgt, edge] -> edge_update
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, D] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, D] 边特征

        Returns:
            x: [N, D] 更新后的节点特征
            edge_attr: [E, D] 更新后的边特征
        """
        # 1. 更新节点特征
        node_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_out = drop_path(node_out, self.drop_path_rate, self.training)
        x = self.node_norm(x + node_out)

        # 2. 更新边特征
        edge_out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        edge_out = drop_path(edge_out, self.drop_path_rate, self.training)
        edge_attr = self.edge_norm(edge_attr + edge_out)

        return x, edge_attr

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """消息函数：邻居特征 + 边特征"""
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def edge_update(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """边更新函数：源节点 + 目标节点 + 当前边"""
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class InteractiveDualEncoder(nn.Module):
    """
    交互式双图编码器

    架构：每层 GNN 后进行跨图交互
    特性：
    - RTL 和 ASM 使用独立的 GNN 层（不共享参数）
    - RTL 和 ASM 使用独立的特征编码器（不共享参数）
    - DropPath 逐层递增
    - 跨图注意力共享参数（连接两图的桥梁）
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        # RTL 特征配置
        num_cell_types: int = 74,
        num_edge_types: int = 5,
        # ASM 特征配置
        num_asm_node_types: int = 22,
        num_asm_edge_types: int = 10,
        asm_instruction_dim: int = 256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # RTL 节点编码器
        self.rtl_node_encoder = RTLNodeFeatureEncoder(num_cell_types, hidden_dim)

        # ASM 节点编码器
        self.asm_node_encoder = AsmNodeFeatureEncoder(
            num_asm_node_types, asm_instruction_dim, hidden_dim
        )

        # RTL 边特征编码器
        self.rtl_edge_encoder = RTLEdgeFeatureEncoder(num_edge_types, hidden_dim)

        # ASM 边特征编码器（仅类型嵌入）
        self.asm_edge_encoder = AsmEdgeFeatureEncoder(num_asm_edge_types, hidden_dim)

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

        # 跨图交互（共享参数）
        self.cross_attn = CrossGraphInteraction(hidden_dim, num_heads, dropout)

        # 输出投影（独立）
        self.rtl_output = nn.Linear(hidden_dim, output_dim)
        self.asm_output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        # RTL 图
        rtl_edge_index: torch.Tensor,
        rtl_node_cell_type: torch.Tensor,
        rtl_node_width: torch.Tensor,
        rtl_edge_type: torch.Tensor,
        rtl_batch: Optional[torch.Tensor] = None,
        rtl_edge_width: Optional[torch.Tensor] = None,
        # ASM 图
        asm_edge_index: torch.Tensor = None,
        asm_node_type: torch.Tensor = None,
        asm_instruction_encoding: torch.Tensor = None,
        asm_edge_type: torch.Tensor = None,
        asm_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            rtl_edge_index: [2, E1] RTL 边索引
            rtl_node_cell_type: [N1] RTL 节点 cell_type 索引
            rtl_node_width: [N1] RTL 节点 width
            rtl_edge_type: [E1] RTL 边类型
            rtl_batch: [N1] RTL batch 索引
            rtl_edge_width: [E1] RTL 边 width
            asm_edge_index: [2, E2] ASM 边索引
            asm_node_type: [N2] ASM 节点类型索引
            asm_instruction_encoding: [N2, D_instr] ASM 指令编码
            asm_edge_type: [E2] ASM 边类型
            asm_batch: [N2] ASM batch 索引

        Returns:
            rtl_node: [N1, D] RTL 节点嵌入
            rtl_graph: [B, D] RTL 图级嵌入
            asm_node: [N2, D] ASM 节点嵌入
            asm_graph: [B, D] ASM 图级嵌入
        """
        # 编码节点特征
        rtl_h = self.rtl_node_encoder(rtl_node_cell_type, rtl_node_width)
        asm_h = self.asm_node_encoder(asm_node_type, asm_instruction_encoding)

        # 边特征编码
        rtl_edge_attr = self.rtl_edge_encoder(rtl_edge_type, rtl_edge_width)
        asm_edge_attr = self.asm_edge_encoder(asm_edge_type)

        for i in range(self.num_layers):
            # GNN（同时更新节点和边特征）
            rtl_h, rtl_edge_attr = self.rtl_layers[i](
                rtl_h, rtl_edge_index, rtl_edge_attr
            )
            asm_h, asm_edge_attr = self.asm_layers[i](
                asm_h, asm_edge_index, asm_edge_attr
            )

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

        return rtl_node, rtl_graph, asm_node, asm_graph
