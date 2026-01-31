#!/usr/bin/env python3
"""便捷训练函数"""

from typing import List, Optional, Tuple
import lightning as L
from lightning.pytorch.callbacks import Callback

from datasets import DualGraphData, DualGraphDataModule
from models.data_types import ModelConfig
from .config import TrainerConfig
from .lightning_module import DualGraphLightningModule
from .lightning_trainer import LightningTrainer


def train_model(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    data_root: str,
    train_data: Optional[List[DualGraphData]] = None,
    val_data: Optional[List[DualGraphData]] = None,
    test_data: Optional[List[DualGraphData]] = None,
    logger_type: str = "tensorboard",
    experiment_name: str = "dual_graph_gnn",
    extra_callbacks: Optional[List[Callback]] = None,
) -> Tuple[DualGraphLightningModule, L.Trainer]:
    """
    便捷训练函数

    Args:
        model_config: 模型配置
        trainer_config: 训练配置
        data_root: 数据集根目录
        train_data: 训练数据（首次运行时提供，之后从磁盘加载）
        val_data: 验证数据（可选）
        test_data: 测试数据（可选）
        logger_type: 日志类型（tensorboard / csv）
        experiment_name: 实验名称
        extra_callbacks: 额外的自定义 callbacks（可选）

    Returns:
        (module, trainer): 训练后的模块和 Trainer
    """
    # 创建 LightningModule
    module = DualGraphLightningModule(
        model_config=model_config,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        edge_loss_weight=trainer_config.edge_loss_weight,
        graph_loss_weight=trainer_config.graph_loss_weight,
        label_smoothing=trainer_config.label_smoothing,
        warmup_steps=trainer_config.warmup_steps,
        scheduler_type=trainer_config.scheduler_type,
    )

    # 创建 DataModule
    datamodule = DualGraphDataModule(
        root=data_root,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )

    # 使用 LightningTrainer
    lightning_trainer = LightningTrainer(
        config=trainer_config,
        experiment_name=experiment_name,
        logger_type=logger_type,
        extra_callbacks=extra_callbacks,
        has_validation=val_data is not None,
    )

    # 训练
    lightning_trainer.fit(module, datamodule)

    # 测试（如果有测试数据）
    if test_data:
        lightning_trainer.test(module, datamodule)

    return module, lightning_trainer.trainer
