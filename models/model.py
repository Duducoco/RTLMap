#!/usr/bin/env python3
"""
双图融合模型

设计理念：
- ASM CDFG 作为上下文信息，在编码阶段通过跨图交互增强 RTL 表示
- 边分类和图回归均只使用 RTL CDFG 的特征
- 支持边类型嵌入（DATA, CONTROL, CLOCK, RESET, ENABLE）
- Label Smoothing 防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .data_types import DualGraphData, ModelConfig, ModelOutput
from .encoder import InteractiveDualEncoder


class EdgeClassifier(nn.Module):
    """
    边分类器

    仅使用 RTL CDFG 的节点特征和边类型进行分类
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,
        num_edge_types: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 边类型嵌入
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)

        # 输入维度：src + tgt + edge_type = 3 * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, D] RTL 节点特征
            edge_index: [2, E] RTL 边索引
            edge_type: [E] RTL 边类型（可选）

        Returns:
            [E, num_classes] 边分类 logits
        """
        src, tgt = edge_index
        src_feat = node_features[src]  # [E, D]
        tgt_feat = node_features[tgt]  # [E, D]

        # 边类型嵌入
        if edge_type is not None:
            edge_type_feat = self.edge_type_embedding(edge_type)  # [E, D]
        else:
            edge_type_feat = torch.zeros(
                edge_index.size(1), self.hidden_dim,
                device=node_features.device
            )

        edge_feat = torch.cat([src_feat, tgt_feat, edge_type_feat], dim=-1)
        return self.net(edge_feat)


class GraphRegressor(nn.Module):
    """
    图级回归器

    仅使用 RTL CDFG 的图级嵌入进行回归
    """

    def __init__(self, hidden_dim: int, num_targets: int = 1, dropout: float = 0.1):
        super().__init__()
        # 输入维度：仅 RTL 图嵌入 = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets)
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
    - 编码阶段：RTL 和 ASM 通过跨图交互相互增强
    - 预测阶段：仅使用 RTL CDFG 进行边分类和图回归
    - ASM CDFG 作为上下文，帮助模型理解"测试激励如何影响硬件"
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 双图编码器（RTL 和 ASM 交互）
        self.encoder = InteractiveDualEncoder(
            rtl_node_dim=config.rtl_node_dim,
            asm_node_dim=config.asm_node_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_gnn_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_edge_types=config.num_edge_types
        )

        # 边分类器（仅 RTL）
        self.edge_classifier = EdgeClassifier(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_edge_classes,
            num_edge_types=config.num_edge_types,
            dropout=config.dropout
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

        Returns:
            ModelOutput: 边分类 logits 和图级预测（均基于 RTL）
        """
        # 编码阶段：双图交互
        rtl_node, rtl_graph, asm_node, asm_graph, cross_attn = self.encoder(
            rtl_x=data.rtl_x,
            rtl_edge_index=data.rtl_edge_index,
            asm_x=data.asm_x,
            asm_edge_index=data.asm_edge_index,
            rtl_batch=data.rtl_batch,
            asm_batch=data.asm_batch,
            rtl_edge_type=data.rtl_edge_type,
            asm_edge_type=data.asm_edge_type
        )

        # 预测阶段：仅使用 RTL
        edge_logits = self.edge_classifier(
            rtl_node, data.rtl_edge_index, data.rtl_edge_type
        )
        graph_pred = self.graph_regressor(rtl_graph)  # 仅 RTL 图嵌入

        return ModelOutput(
            edge_logits=edge_logits,
            graph_pred=graph_pred,
            rtl_final=rtl_node,
            asm_final=asm_node,
            cross_attentions=cross_attn
        )

    def compute_loss(
        self,
        output: ModelOutput,
        data: DualGraphData,
        edge_loss_weight: float = 1.0,
        graph_loss_weight: float = 1.0,
        label_smoothing: float = 0.1
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

        # 边分类损失（mask 掉 label=-1 的边，仅对 label∈{0,1} 计算损失）
        if data.rtl_edge_labels is not None:
            # mask: True 表示有效边（label != -1）
            valid_mask = data.rtl_edge_labels != -1
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                # 筛选有效边
                valid_logits = output.edge_logits[valid_mask]  # [num_valid, 2]
                valid_labels = data.rtl_edge_labels[valid_mask]  # [num_valid], 值为 {0, 1}

                # 二分类：{0: 未覆盖, 1: 已覆盖}
                edge_loss = F.cross_entropy(
                    valid_logits,
                    valid_labels,
                    label_smoothing=label_smoothing
                )
                losses['edge_loss'] = edge_loss * edge_loss_weight
            else:
                losses['edge_loss'] = torch.tensor(0., device=device)

            losses['num_valid_edges'] = num_valid
        else:
            losses['edge_loss'] = torch.tensor(0., device=device)
            losses['num_valid_edges'] = 0

        # 图回归损失
        if data.graph_label is not None:
            graph_loss = F.smooth_l1_loss(output.graph_pred, data.graph_label)
            losses['graph_loss'] = graph_loss * graph_loss_weight
        else:
            losses['graph_loss'] = torch.tensor(0., device=device)

        losses['total_loss'] = losses['edge_loss'] + losses['graph_loss']
        return losses


def create_model(config: Optional[ModelConfig] = None, **kwargs) -> DualGraphFusionModel:
    """创建模型"""
    if config is None:
        config = ModelConfig(**kwargs)
    return DualGraphFusionModel(config)


def create_small_model(**kwargs) -> DualGraphFusionModel:
    """创建小型模型（调试用）"""
    defaults = {'hidden_dim': 128, 'num_gnn_layers': 2, 'num_heads': 4}
    defaults.update(kwargs)
    return create_model(**defaults)


def create_base_model(**kwargs) -> DualGraphFusionModel:
    """创建基础模型"""
    defaults = {'hidden_dim': 256, 'num_gnn_layers': 4, 'num_heads': 4}
    defaults.update(kwargs)
    return create_model(**defaults)
