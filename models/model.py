#!/usr/bin/env python3
"""
双图融合模型

设计理念：
- ASM 和 RTL 通过双向 Perceiver 注入相互增强（ASM ↔ RTL）
- 图级覆盖率回归直接使用 RTL 与 ASM 的图级特征
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
from .data_types import (
    MODEL_ARCHITECTURE_PERCEIVER_FUSION,
    MODEL_ARCHITECTURE_POOLED_ADD,
    ModelConfig,
    ModelOutput,
)
from .encoder import DualGraphEncoder
from .heads import GraphRegressor
from .hyperrectangle import HyperrectangleHead
from .losses import (
    DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR,
    DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT,
    compute_supervised_losses,
)


class _DualGraphModelBase(nn.Module):
    """Shared dual-graph prediction implementation behind architecture adapters."""

    enable_cross_fusion: bool
    graph_embedding_multiplier: int

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        graph_embedding_dim = config.hidden_dim * self.graph_embedding_multiplier

        self.encoder = DualGraphEncoder(
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
            enable_cross_fusion=self.enable_cross_fusion,
        )

        self.graph_regressor = GraphRegressor(
            graph_embedding_dim,
            config.hidden_dim,
            config.coverage_target_keys,
            config.dropout,
        )

        if config.use_hyperrectangle:
            self.hyperrectangle_head = HyperrectangleHead(
                input_dim=graph_embedding_dim,
                hidden_dim=config.hidden_dim,
                num_types=len(config.hyperrectangle_type_names),
                dim_per_type=config.hyperrectangle_dim_per_type,
                margin=config.hyper_min_margin,
                dropout=config.dropout,
            )

    def _combine_graph_embeddings(
        self, rtl_graph: torch.Tensor, asm_graph: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data: DualGraphData) -> ModelOutput:
        rtl_node, rtl_edge_attr, rtl_graph, asm_node, asm_graph = self.encoder(
            rtl_edge_index=data.edge_index,
            rtl_node_cell_type=data.node_cell_type,
            rtl_node_width=data.node_width,
            rtl_edge_type=data.edge_type,
            rtl_node_type=data.node_type,
            rtl_batch=getattr(data, "batch", None),
            rtl_edge_width=getattr(data, "edge_width", None),
            rtl_edge_source_port_idx=getattr(data, "edge_source_port_idx", None),
            rtl_edge_target_port_idx=getattr(data, "edge_target_port_idx", None),
            asm_edge_index=data.asm_edge_index,
            asm_node_type=data.asm_node_type,
            asm_instruction_encoding=data.asm_instruction_encoding,
            asm_edge_type=data.asm_edge_type,
            asm_batch=getattr(data, "asm_node_type_batch", None),
        )

        joint_graph = self._combine_graph_embeddings(rtl_graph, asm_graph)
        graph_pred = self.graph_regressor(joint_graph)

        hyper_min, hyper_max = None, None
        if self.config.use_hyperrectangle and hasattr(self, "hyperrectangle_head"):
            hyper_min, hyper_max = self.hyperrectangle_head(joint_graph)

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
        relative_loss_weight: float = DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT,
        relative_loss_floor: float = DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR,
    ) -> Dict[str, torch.Tensor]:
        """计算模型基础监督损失。"""
        return compute_supervised_losses(
            output,
            data,
            coverage_target_keys=self.config.coverage_target_keys,
            graph_loss_weight=graph_loss_weight,
            relative_loss_weight=relative_loss_weight,
            relative_loss_floor=relative_loss_floor,
        )


class DualGraphFusionModel(_DualGraphModelBase):
    """
    双图融合模型

    架构设计：
    - 编码阶段：ASM 和 RTL 通过双向融合相互增强（ASM ↔ RTL）
    - 预测阶段：拼接 RTL 与 ASM 图级嵌入进行覆盖率回归
    - ASM CDFG 作为上下文，模拟"测试激励驱动硬件"

    ASM 与 RTL 在每层 GNN 后通过双向 Perceiver cross-attention 交互。
    """

    enable_cross_fusion = True
    graph_embedding_multiplier = 2

    def _combine_graph_embeddings(
        self, rtl_graph: torch.Tensor, asm_graph: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((rtl_graph, asm_graph), dim=-1)


class PooledAddBaselineModel(_DualGraphModelBase):
    """Independent RTL/ASM GNN branches combined only after mean pooling."""

    enable_cross_fusion = False
    graph_embedding_multiplier = 1

    def _combine_graph_embeddings(
        self, rtl_graph: torch.Tensor, asm_graph: torch.Tensor
    ) -> torch.Tensor:
        if rtl_graph.shape != asm_graph.shape:
            raise ValueError(
                "pooled_add requires matching RTL and ASM graph embedding shapes, "
                f"got {tuple(rtl_graph.shape)} and {tuple(asm_graph.shape)}"
            )
        return rtl_graph + asm_graph


def create_model(
    config: Optional[ModelConfig] = None, **kwargs
) -> DualGraphFusionModel | PooledAddBaselineModel:
    """创建模型"""
    if config is None:
        config = ModelConfig(**kwargs)
    if config.model_architecture == MODEL_ARCHITECTURE_PERCEIVER_FUSION:
        return DualGraphFusionModel(config)
    if config.model_architecture == MODEL_ARCHITECTURE_POOLED_ADD:
        return PooledAddBaselineModel(config)
    raise ValueError(f"unsupported model_architecture: {config.model_architecture!r}")


def create_small_model(**kwargs) -> DualGraphFusionModel | PooledAddBaselineModel:
    """创建小型模型（调试用）"""
    defaults = {"hidden_dim": 128, "num_gnn_layers": 2}
    defaults.update(kwargs)
    return create_model(**defaults)


def create_base_model(**kwargs) -> DualGraphFusionModel | PooledAddBaselineModel:
    """创建基础模型"""
    defaults = {"hidden_dim": 256, "num_gnn_layers": 4}
    defaults.update(kwargs)
    return create_model(**defaults)
