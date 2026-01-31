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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from datasets import DualGraphData
from .data_types import ModelConfig, ModelOutput
from .encoder import FiLMDualEncoder, RTLEdgeFeatureEncoder


class EdgeClassifier(nn.Module):
    """
    边分类器

    仅使用 RTL CDFG 的节点特征和边特征进行分类
    边特征 = edge_type 嵌入 + edge_width 编码（log2 + 线性层，加法融合）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        num_edge_types: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 边特征编码器
        self.edge_encoder = RTLEdgeFeatureEncoder(num_edge_types, hidden_dim)

        # 输入维度：src + tgt + edge_feat = 3 * hidden_dim
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
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, D] RTL 节点特征
            edge_index: [2, E] RTL 边索引
            edge_type: [E] RTL 边类型
            edge_width: [E] RTL 边 width（可选）

        Returns:
            [E, num_classes] 边分类 logits
        """
        src, tgt = edge_index
        src_feat = node_features[src]  # [E, D]
        tgt_feat = node_features[tgt]  # [E, D]

        # 边特征编码
        edge_feat = self.edge_encoder(edge_type, edge_width)  # [E, D]

        combined = torch.cat([src_feat, tgt_feat, edge_feat], dim=-1)
        return self.net(combined)


class GraphRegressor(nn.Module):
    """
    图级回归器

    仅使用 RTL CDFG 的图级嵌入进行回归
    """

    def __init__(self, hidden_dim: int, num_targets: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, rtl_graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rtl_graph_emb: [B, D] RTL 图级嵌入

        Returns:
            [B, num_targets] 图级预测
        """
        return self.net(rtl_graph_emb)


class DualGraphFusionModel(nn.Module):
    """
    双图融合模型

    架构设计：
    - 编码阶段：ASM 和 RTL 通过双向 FiLM 注入相互增强（ASM ↔ RTL）
    - 预测阶段：仅使用 RTL CDFG 进行边分类和图回归
    - ASM CDFG 作为上下文，模拟"测试激励驱动硬件"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 双图编码器（FiLM 注入）
        self.encoder = FiLMDualEncoder(
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_gnn_layers,
            dropout=config.dropout,
            num_edge_types=config.num_edge_types,
            num_cell_types=config.num_cell_types,
            num_asm_node_types=config.num_asm_node_types,
            num_asm_edge_types=config.num_asm_edge_types,
            asm_instruction_dim=config.asm_instruction_dim,
        )

        # 边分类器（仅 RTL）
        self.edge_classifier = EdgeClassifier(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_edge_classes,
            num_edge_types=config.num_edge_types,
            dropout=config.dropout,
        )

        # 图回归器（仅 RTL）
        self.graph_regressor = GraphRegressor(
            config.hidden_dim, config.num_graph_targets, config.dropout
        )

    def forward(self, data: DualGraphData) -> ModelOutput:
        """
        前向传播

        Args:
            data: 双图数据，包含 RTL 和 ASM 图
                RTL: node_cell_type, node_width, edge_index, edge_type, edge_width
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
            rtl_batch=getattr(data, "batch", None),
            rtl_edge_width=getattr(data, "edge_width", None),
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
        )
        graph_pred = self.graph_regressor(rtl_graph)

        return ModelOutput(
            edge_logits=edge_logits,
            graph_pred=graph_pred,
            rtl_final=rtl_node,
            asm_final=asm_node,
        )

    def compute_loss(
        self,
        output: ModelOutput,
        data: DualGraphData,
        edge_loss_weight: float = 1.0,
        graph_loss_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失

        Args:
            output: 模型输出
            data: 输入数据（含标签）
            edge_loss_weight: 边分类损失权重
            graph_loss_weight: 图回归损失权重
            label_smoothing: Label Smoothing 系数

        Returns:
            损失字典 {'edge_loss', 'graph_loss', 'total_loss', 'num_valid_edges'}
        """
        device = output.edge_logits.device
        losses = {}

        # 边分类损失（mask 掉 label=-1 的边）
        edge_labels = getattr(data, "edge_labels", None)
        if edge_labels is not None:
            valid_mask = edge_labels != -1
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                valid_logits = output.edge_logits[valid_mask]
                valid_labels = edge_labels[valid_mask]
                edge_loss = F.cross_entropy(
                    valid_logits, valid_labels, label_smoothing=label_smoothing
                )
                losses["edge_loss"] = edge_loss * edge_loss_weight
            else:
                losses["edge_loss"] = torch.tensor(0.0, device=device)

            losses["num_valid_edges"] = num_valid
        else:
            losses["edge_loss"] = torch.tensor(0.0, device=device)
            losses["num_valid_edges"] = 0

        # 图回归损失
        if data.y is not None:
            graph_loss = F.smooth_l1_loss(output.graph_pred, data.y)
            losses["graph_loss"] = graph_loss * graph_loss_weight
        else:
            losses["graph_loss"] = torch.tensor(0.0, device=device)

        losses["total_loss"] = losses["edge_loss"] + losses["graph_loss"]
        return losses


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
