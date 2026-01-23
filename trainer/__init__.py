#!/usr/bin/env python3
"""
PyTorch Lightning 训练模块

提供双图融合模型的训练封装：
- DualGraphLightningModule: 模型训练封装
- DualGraphDataModule: 数据加载封装
- TrainerConfig: 训练配置
- train_model: 便捷训练函数
"""

from .config import TrainerConfig
from .datamodule import DualGraphDataModule, DualGraphDataset, collate_dual_graph
from .module import DualGraphLightningModule
from .utils import train_model

__all__ = [
    # 配置
    "TrainerConfig",
    # 数据
    "DualGraphDataModule",
    "DualGraphDataset",
    "collate_dual_graph",
    # 模型
    "DualGraphLightningModule",
    # 工具
    "train_model",
]
