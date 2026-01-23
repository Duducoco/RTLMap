#!/usr/bin/env python3
"""双图神经网络数据类型"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DualGraphData:
    """双图数据"""
    # RTL 图
    rtl_x: torch.Tensor                           # [N1, D] 节点特征
    rtl_edge_index: torch.Tensor                  # [2, E1] 边索引
    rtl_edge_type: Optional[torch.Tensor] = None  # [E1] 边类型 {0: DATA, 1: CONTROL, 2: CLOCK, 3: RESET, 4: ENABLE}
    rtl_batch: Optional[torch.Tensor] = None      # [N1] batch 索引

    # ASM 图
    asm_x: torch.Tensor = None
    asm_edge_index: torch.Tensor = None
    asm_edge_type: Optional[torch.Tensor] = None  # [E2] 边类型
    asm_batch: Optional[torch.Tensor] = None

    # 标签
    rtl_edge_labels: Optional[torch.Tensor] = None  # [E1] 边标签 {-1, 0, 1}
    graph_label: Optional[torch.Tensor] = None      # [B, 1] 图级标签

    # 可选扩展字段
    rtl_edge_attr: Optional[torch.Tensor] = None    # [E1, D_e] 边特征（可选）
    rtl_node_type: Optional[torch.Tensor] = None    # [N1] 节点类型（可选）
    asm_edge_attr: Optional[torch.Tensor] = None    # [E2, D_e] 边特征（可选）
    asm_node_type: Optional[torch.Tensor] = None    # [N2] 节点类型（可选）
    node_mapping: Optional[torch.Tensor] = None     # 跨图节点映射（可选）

    def to(self, device) -> "DualGraphData":
        def mv(t):
            return t.to(device) if t is not None else None
        return DualGraphData(
            rtl_x=mv(self.rtl_x), rtl_edge_index=mv(self.rtl_edge_index),
            rtl_edge_type=mv(self.rtl_edge_type), rtl_batch=mv(self.rtl_batch),
            asm_x=mv(self.asm_x), asm_edge_index=mv(self.asm_edge_index),
            asm_edge_type=mv(self.asm_edge_type), asm_batch=mv(self.asm_batch),
            rtl_edge_labels=mv(self.rtl_edge_labels), graph_label=mv(self.graph_label),
            rtl_edge_attr=mv(self.rtl_edge_attr), rtl_node_type=mv(self.rtl_node_type),
            asm_edge_attr=mv(self.asm_edge_attr), asm_node_type=mv(self.asm_node_type),
            node_mapping=mv(self.node_mapping)
        )

    @property
    def num_rtl_nodes(self) -> int:
        return self.rtl_x.size(0)

    @property
    def num_asm_nodes(self) -> int:
        return self.asm_x.size(0) if self.asm_x is not None else 0

    @property
    def num_rtl_edges(self) -> int:
        return self.rtl_edge_index.size(1)

    @property
    def num_asm_edges(self) -> int:
        return self.asm_edge_index.size(1) if self.asm_edge_index is not None else 0


@dataclass
class ModelConfig:
    """模型配置"""
    rtl_node_dim: int = 64
    asm_node_dim: int = 64
    hidden_dim: int = 256
    num_gnn_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    num_edge_classes: int = 2  # {0: 未覆盖, 1: 已覆盖}，-1 被 mask 掉
    num_graph_targets: int = 1

    # 兼容旧代码
    rtl_edge_dim: Optional[int] = None
    asm_edge_dim: Optional[int] = None
    num_edge_types: int = 5
    encoder_type: str = "gin"

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0


@dataclass
class ModelOutput:
    """模型输出"""
    edge_logits: Optional[torch.Tensor] = None
    graph_pred: Optional[torch.Tensor] = None
    rtl_final: Optional[torch.Tensor] = None
    asm_final: Optional[torch.Tensor] = None
    matching_matrix: Optional[torch.Tensor] = None
    cross_attentions: Optional[List] = None
