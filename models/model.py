#!/usr/bin/env python3
"""
双图融合模型

设计理念：
- ASM 和 RTL 通过双向 FiLM 注入相互增强（ASM ↔ RTL）
- 图级覆盖率回归只使用 RTL CDFG 的特征
- RTL 节点特征：node_cell_type + node_width
- RTL 边特征：edge_type + edge_width
- ASM 节点特征：node_type + instruction_encoding（外部模型生成）
- ASM 边特征：edge_type（仅类型嵌入）
- Label Smoothing 防止过拟合
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from datasets import DualGraphData
from .data_types import ModelConfig, ModelOutput
from .encoder import PerceiverDualEncoder
from .heads import GraphRegressor
from .hyperrectangle import HyperrectangleHead


class DualGraphFusionModel(nn.Module):
    """
    双图融合模型

    架构设计：
    - 编码阶段：ASM 和 RTL 通过双向融合相互增强（ASM ↔ RTL）
    - 预测阶段：仅使用 RTL CDFG 进行图级覆盖率回归
    - ASM CDFG 作为上下文，模拟"测试激励驱动硬件"

    支持两种融合模式：
    - film: 标准 FiLM 注入（全局池化上下文）
    - ssm_film: SSM 增强的 FiLM（状态空间动态上下文）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 双图编码器（PerceiverCrossFusion 双向融合）
        self.encoder = PerceiverDualEncoder(
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_gnn_layers,
            dropout=config.dropout,
            num_edge_types=config.num_edge_types,
            num_cell_types=config.num_cell_types,
            max_ports=config.max_ports,
            num_asm_node_types=config.num_asm_node_types,
            num_asm_edge_types=config.num_asm_edge_types,
            asm_instruction_dim=config.asm_instruction_dim,
            use_asm_adapter=config.use_asm_adapter,
            perceiver_num_latents=config.perceiver_num_latents,
            perceiver_num_heads=config.perceiver_num_heads,
        )

        # 图回归器（仅 RTL）
        self.graph_regressor = GraphRegressor(
            config.hidden_dim, config.coverage_target_keys, config.dropout
        )

        # 超矩形头（条件创建）
        if config.use_hyperrectangle:
            self.hyperrectangle_head = HyperrectangleHead(
                hidden_dim=config.hidden_dim,
                num_types=len(config.hyperrectangle_type_names),
                dim_per_type=config.hyperrectangle_dim_per_type,
                margin=config.hyper_min_margin,
                dropout=config.dropout,
            )

    def forward(self, data: DualGraphData) -> ModelOutput:
        """
        前向传播

        Args:
            data: 双图数据，包含 RTL 和 ASM 图
                RTL: node_cell_type, node_width, edge_index, edge_type, edge_width,
                     edge_source_port_idx, edge_target_port_idx
                ASM: asm_node_type, asm_instruction_encoding, asm_edge_index, asm_edge_type

        Returns:
            ModelOutput: 图级覆盖率预测（基于 RTL）
        """
        # 编码阶段（双向 FiLM 注入：ASM ↔ RTL）
        rtl_node, rtl_edge_attr, rtl_graph, asm_node, _ = self.encoder(
            # RTL 图
            rtl_edge_index=data.edge_index,
            rtl_node_cell_type=data.node_cell_type,
            rtl_node_width=data.node_width,
            rtl_edge_type=data.edge_type,
            rtl_node_type=data.node_type,
            rtl_batch=getattr(data, "batch", None),
            rtl_edge_width=getattr(data, "edge_width", None),
            rtl_edge_source_port_idx=getattr(data, "edge_source_port_idx", None),
            rtl_edge_target_port_idx=getattr(data, "edge_target_port_idx", None),
            # ASM 图
            asm_edge_index=data.asm_edge_index,
            asm_node_type=data.asm_node_type,
            asm_instruction_encoding=data.asm_instruction_encoding,
            asm_edge_type=data.asm_edge_type,
            asm_batch=getattr(data, "asm_node_type_batch", None),
        )

        # 预测阶段：仅使用 RTL
        graph_pred = self.graph_regressor(rtl_graph)

        # 超矩形输出（条件计算）
        hyper_min, hyper_max = None, None
        if self.config.use_hyperrectangle and hasattr(self, "hyperrectangle_head"):
            hyper_min, hyper_max = self.hyperrectangle_head(rtl_graph)

        return ModelOutput(
            graph_pred=graph_pred,
            rtl_final=rtl_node,
            asm_final=asm_node,
            hyper_min=hyper_min,
            hyper_max=hyper_max,
            rtl_graph_emb=rtl_graph,
        )

    def compute_loss(
        self,
        output: ModelOutput,
        data: DualGraphData,
        graph_loss_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """计算模型基础监督损失。"""
        from .losses import compute_supervised_losses

        return compute_supervised_losses(
            output,
            data,
            coverage_target_keys=self.config.coverage_target_keys,
            graph_loss_weight=graph_loss_weight,
        )


def create_model(
    config: Optional[ModelConfig] = None, **kwargs
) -> DualGraphFusionModel:
    """创建模型"""
    if config is None:
        config = ModelConfig(**kwargs)
    return DualGraphFusionModel(config)


def create_small_model(**kwargs) -> DualGraphFusionModel:
    """创建小型模型（调试用）"""
    defaults = {"hidden_dim": 128, "num_gnn_layers": 2}
    defaults.update(kwargs)
    return create_model(**defaults)


def create_base_model(**kwargs) -> DualGraphFusionModel:
    """创建基础模型"""
    defaults = {"hidden_dim": 256, "num_gnn_layers": 4}
    defaults.update(kwargs)
    return create_model(**defaults)
