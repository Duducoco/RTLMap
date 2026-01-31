#!/usr/bin/env python3
"""
PyTorch Lightning 训练模块

提供双图融合模型的训练封装：
- DualGraphLightningModule: 模型训练封装
- TrainerConfig: 训练配置
- CallbackFactory: Callback 工厂类
- train_model: 便捷训练函数
- LightningTrainer: 训练器包装类
"""

from .config import TrainerConfig
from .lightning_module import DualGraphLightningModule
from .callbacks import CallbackFactory
from .utils import train_model
from .lightning_trainer import LightningTrainer

__all__ = [
    # 配置
    "TrainerConfig",
    # 模型
    "DualGraphLightningModule",
    # Callbacks
    "CallbackFactory",
    # 工具
    "train_model",
    # 训练器
    "LightningTrainer",
]
