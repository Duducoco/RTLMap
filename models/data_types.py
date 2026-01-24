#!/usr/bin/env python3
"""模型配置和输出数据类型

注意：DualGraphData 已迁移至 datasets.data_types 模块
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


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
