#!/usr/bin/env python3
"""便捷训练函数"""

from typing import List, Optional, Tuple
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from models.data_types import DualGraphData, ModelConfig
from .config import TrainerConfig
from datasets import DualGraphDataModule
from .module import DualGraphLightningModule
from .callbacks import CallbackFactory


def train_model(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    train_data: List[DualGraphData],
    val_data: Optional[List[DualGraphData]] = None,
    test_data: Optional[List[DualGraphData]] = None,
    logger_type: str = "tensorboard",
    experiment_name: str = "dual_graph_gnn",
    extra_callbacks: Optional[List[Callback]] = None
) -> Tuple[DualGraphLightningModule, L.Trainer]:
    """
    便捷训练函数

    Args:
        model_config: 模型配置
        trainer_config: 训练配置
        train_data: 训练数据
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
        scheduler_type=trainer_config.scheduler_type
    )

    # 创建 DataModule
    datamodule = DualGraphDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers
    )

    # 创建 Logger
    if logger_type == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=trainer_config.checkpoint_dir,
            name=experiment_name
        )
    else:
        logger = CSVLogger(
            save_dir=trainer_config.checkpoint_dir,
            name=experiment_name
        )

    # 创建 Callbacks
    callbacks = CallbackFactory.create_default_callbacks(
        config=trainer_config,
        experiment_name=experiment_name,
        has_validation=val_data is not None,
        extra_callbacks=extra_callbacks
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        gradient_clip_val=trainer_config.gradient_clip_val,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # 训练
    trainer.fit(module, datamodule)

    # 测试（如果有测试数据）
    if test_data:
        trainer.test(module, datamodule)

    return module, trainer
