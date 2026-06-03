#!/usr/bin/env python3
"""
双图融合模型

设计理念：
- ASM 和 RTL 通过双向 FiLM 注入相互增强（ASM ↔ RTL）
- 边分类和图回归均只使用 RTL CDFG 的特征
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
from .heads import EdgeClassifier, GraphRegressor
from .hyperrectangle import HyperrectangleHead
from .union_fusion import SymmetricUnionFusion


class DualGraphFusionModel(nn.Module):
    """
    双图融合模型

    架构设计：
    - 编码阶段：ASM 和 RTL 通过双向融合相互增强（ASM ↔ RTL）
    - 预测阶段：仅使用 RTL CDFG 进行边分类和图回归
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

        # 边分类器（仅 RTL，支持端口位置索引）
        self.edge_classifier = EdgeClassifier(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_edge_classes,
            num_edge_types=config.num_edge_types,
            max_ports=config.max_ports,
            dropout=config.dropout,
        )

        # 图回归器（仅 RTL）
        self.graph_regressor = GraphRegressor(
            config.hidden_dim, config.coverage_target_keys, config.dropout
        )

        # 超矩形头（条件创建）
        if config.use_hyperrectangle:
            self.hyperrectangle_head = HyperrectangleHead(
                hidden_dim=config.hidden_dim, margin=config.hyper_min_margin
            )

        self.union_node_fusion = SymmetricUnionFusion(config.hidden_dim, config.dropout)
        self.union_graph_fusion = SymmetricUnionFusion(
            config.hidden_dim, config.dropout
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
            ModelOutput: 边分类 logits 和图级预测（均基于 RTL）
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
        edge_logits = self.edge_classifier(
            rtl_node,
            data.edge_index,
            data.edge_type,
            getattr(data, "edge_width", None),
            getattr(data, "edge_source_port_idx", None),
            getattr(data, "edge_target_port_idx", None),
        )
        graph_pred = self.graph_regressor(rtl_graph)

        # 超矩形输出（条件计算）
        hyper_min, hyper_max = None, None
        if self.config.use_hyperrectangle and hasattr(self, "hyperrectangle_head"):
            hyper_min, hyper_max = self.hyperrectangle_head(rtl_graph)

        return ModelOutput(
            edge_logits=edge_logits,
            graph_pred=graph_pred,
            rtl_final=rtl_node,
            asm_final=asm_node,
            hyper_min=hyper_min,
            hyper_max=hyper_max,
            rtl_graph_emb=rtl_graph,
        )

    def merge_outputs(
        self,
        output_a: ModelOutput,
        output_b: ModelOutput,
        reference: DualGraphData,
    ) -> ModelOutput:
        """Fuse two aligned test-conditioned outputs into merged coverage output."""
        if output_a.rtl_final is None or output_b.rtl_final is None:
            raise ValueError("merge_outputs requires rtl_final in both outputs")
        if output_a.rtl_graph_emb is None or output_b.rtl_graph_emb is None:
            raise ValueError("merge_outputs requires rtl_graph_emb in both outputs")
        if output_a.rtl_final.shape != output_b.rtl_final.shape:
            raise ValueError(
                "merge_outputs requires aligned node features: "
                f"{tuple(output_a.rtl_final.shape)} != {tuple(output_b.rtl_final.shape)}"
            )
        if output_a.rtl_graph_emb.shape != output_b.rtl_graph_emb.shape:
            raise ValueError(
                "merge_outputs requires aligned graph features: "
                f"{tuple(output_a.rtl_graph_emb.shape)} != "
                f"{tuple(output_b.rtl_graph_emb.shape)}"
            )

        rtl_union = self.union_node_fusion(output_a.rtl_final, output_b.rtl_final)
        graph_union = self.union_graph_fusion(
            output_a.rtl_graph_emb, output_b.rtl_graph_emb
        )

        edge_logits = self.edge_classifier(
            rtl_union,
            reference.edge_index,
            reference.edge_type,
            getattr(reference, "edge_width", None),
            getattr(reference, "edge_source_port_idx", None),
            getattr(reference, "edge_target_port_idx", None),
        )
        graph_pred = self.graph_regressor(graph_union)

        hyper_min, hyper_max = None, None
        if self.config.use_hyperrectangle and hasattr(self, "hyperrectangle_head"):
            hyper_min, hyper_max = self.hyperrectangle_head(graph_union)

        return ModelOutput(
            edge_logits=edge_logits,
            graph_pred=graph_pred,
            rtl_final=rtl_union,
            asm_final=None,
            hyper_min=hyper_min,
            hyper_max=hyper_max,
            rtl_graph_emb=graph_union,
        )

    def compute_loss(
        self,
        output: ModelOutput,
        data: DualGraphData,
        edge_loss_weight: float = 1.0,
        graph_loss_weight: float = 1.0,
        label_smoothing: float = 0.0,
        edge_loss_type: str = "focal",
        focal_gamma: float = 2.0,
        edge_class_weight: tuple[float, float] = (8.0, 1.0),
    ) -> Dict[str, torch.Tensor]:
        """计算模型基础监督损失。"""
        from .losses import compute_supervised_losses

        return compute_supervised_losses(
            output,
            data,
            coverage_target_keys=self.config.coverage_target_keys,
            edge_loss_weight=edge_loss_weight,
            graph_loss_weight=graph_loss_weight,
            label_smoothing=label_smoothing,
            edge_loss_type=edge_loss_type,
            focal_gamma=focal_gamma,
            edge_class_weight=edge_class_weight,
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
