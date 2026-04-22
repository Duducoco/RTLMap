#!/usr/bin/env python3
"""模型配置和输出数据类型"""

import torch
from dataclasses import dataclass
from typing import Optional

# 从 cdfg_rtl 导入 cell 类型数量
from cdfg_rtl import NUM_CELL_TYPES


@dataclass
class ModelConfig:
    """模型配置"""

    # 隐藏层和输出维度
    hidden_dim: int = 256
    num_gnn_layers: int = 4
    dropout: float = 0.1

    # 任务配置
    num_edge_classes: int = 2  # {0: 未覆盖, 1: 已覆盖}，-1 被 mask 掉
    num_graph_targets: int = 1

    # RTL 特征配置
    num_cell_types: int = NUM_CELL_TYPES  # cell_type 词汇表大小 (74)
    num_edge_types: int = (
        7  # DATA, DATA_TRUE, DATA_FALSE, CONTROL, CLOCK, RESET, ENABLE
    )
    max_ports: int = 8  # 端口位置索引最大值

    # ASM 特征配置
    num_asm_node_types: int = 22  # AsmNodeType 枚举数量
    num_asm_edge_types: int = 10  # AsmEdgeType 枚举数量
    asm_instruction_dim: int = 256  # ASM 指令编码维度（由外部模型生成）

    # 融合配置（PerceiverCrossFusion）
    perceiver_num_latents: int = 16   # latent token 数量 K
    perceiver_num_heads: int = 4      # Multi-head Attention 头数

    # ASM 节点编码器适配器
    use_asm_adapter: bool = True      # 在 instruction_encoding 投影前插入可训练 Adapter


@dataclass
class ModelOutput:
    """模型输出"""

    edge_logits: Optional[torch.Tensor] = None
    graph_pred: Optional[torch.Tensor] = None
    rtl_final: Optional[torch.Tensor] = None
    asm_final: Optional[torch.Tensor] = None
    matching_matrix: Optional[torch.Tensor] = None
