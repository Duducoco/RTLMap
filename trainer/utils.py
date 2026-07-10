#!/usr/bin/env python3
"""便捷训练函数"""

from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
from lightning.pytorch.callbacks import Callback

from .app_config import AppConfig
from .datamodule_factory import build_training_datamodule as _build_training_datamodule
from .lightning_module import DualGraphLightningModule
from .lightning_trainer import LightningTrainer
from models.config_artifact import (
    build_model_config_artifact,
    save_model_config_artifact,
)


def persist_model_config_artifact(app_config: AppConfig) -> Path:
    """Write the resolved inference contract beside experiment checkpoints."""

    output = (
        Path(app_config.trainer.checkpoint_dir)
        / app_config.runtime.experiment_name
        / "model_config.yaml"
    )
    artifact = build_model_config_artifact(
        app_config.model,
        app_config.text_encoder,
    )
    return save_model_config_artifact(output, artifact)


def train_model(
    app_config: AppConfig,
    extra_callbacks: Optional[List[Callback]] = None,
) -> Tuple[DualGraphLightningModule, L.Trainer]:
    """便捷训练函数（AppConfig 入口）。

    Args:
        app_config: 应用级统一配置
        extra_callbacks: 额外的自定义 callbacks（可选）

    Returns:
        (module, trainer): 训练后的模块和 Trainer
    """
    model_config = app_config.model
    trainer_config = app_config.trainer
    dataset_dir = app_config.data.dataset_dir
    model_config_artifact = build_model_config_artifact(
        model_config,
        app_config.text_encoder,
    )
    persist_model_config_artifact(app_config)

    module = DualGraphLightningModule(
        model_config=model_config,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        graph_loss_weight=trainer_config.graph_loss_weight,
        warmup_steps=trainer_config.warmup_steps,
        scheduler_type=trainer_config.scheduler_type,
        contrastive_loss_weight=trainer_config.contrastive_loss_weight
        if trainer_config.use_hyperrectangle
        else 0.0,
        contrastive_loss_type=trainer_config.contrastive_loss_type,
        contrastive_margin=trainer_config.contrastive_margin,
        coverage_target_keys=trainer_config.coverage_target_keys,
        joint_contrastive=trainer_config.joint_contrastive,
        lambda_ce=trainer_config.lambda_ce,
        lambda_cl=trainer_config.lambda_cl,
        model_config_artifact=model_config_artifact,
    )

    effective_datamodule, inferred_has_validation = _build_training_datamodule(
        data_root=app_config.data.data_root,
        dataset_dir=dataset_dir,
        text_encoder_config=app_config.text_encoder,
        trainer_config=trainer_config,
    )
    has_validation = app_config.runtime.has_validation and inferred_has_validation

    lightning_trainer = LightningTrainer(
        config=trainer_config,
        experiment_name=app_config.runtime.experiment_name,
        logger_type=app_config.runtime.logger_type,
        extra_callbacks=extra_callbacks,
        has_validation=has_validation,
    )

    lightning_trainer.fit(
        module, effective_datamodule, ckpt_path=app_config.runtime.ckpt_path
    )

    if app_config.runtime.has_test:
        lightning_trainer.test(module, effective_datamodule)

    return module, lightning_trainer.trainer
