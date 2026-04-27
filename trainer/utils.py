#!/usr/bin/env python3
"""便捷训练函数"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import lightning as L
from lightning.pytorch.callbacks import Callback

from datasets import DualGraphDataModule
from datasets.pair_datamodule import ContrastivePairDataModule
from models.data_types import ModelConfig
from .config import TrainerConfig
from .lightning_module import DualGraphLightningModule
from .lightning_trainer import LightningTrainer

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig


def train_model(
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    data_root: str,
    dataset_dir: Optional[Union[str, Path]] = None,
    text_encoder_config: Optional["TextEncoderConfig"] = None,
    has_validation: bool = True,
    has_test: bool = False,
    logger_type: str = "tensorboard",
    experiment_name: str = "dual_graph_gnn",
    extra_callbacks: Optional[List[Callback]] = None,
    ckpt_path: Optional[str] = None,
) -> Tuple[DualGraphLightningModule, L.Trainer]:
    """
    便捷训练函数

    Args:
        model_config: 模型配置
        trainer_config: 训练配置
        data_root: 数据集缓存根目录
        dataset_dir: coverage-report-extractor 输出目录（含 dataset_index.jsonlines）
        text_encoder_config: 文本编码器配置，None 时回退到零向量
        has_validation: 是否有验证集
        has_test: 是否有测试集
        logger_type: 日志类型（tensorboard / csv）
        experiment_name: 实验名称
        extra_callbacks: 额外的自定义 callbacks（可选）
        ckpt_path: 检查点路径（用于恢复训练）

    Returns:
        (module, trainer): 训练后的模块和 Trainer
    """
    module = DualGraphLightningModule(
        model_config=model_config,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        edge_loss_weight=trainer_config.edge_loss_weight,
        graph_loss_weight=trainer_config.graph_loss_weight,
        label_smoothing=trainer_config.label_smoothing,
        warmup_steps=trainer_config.warmup_steps,
        scheduler_type=trainer_config.scheduler_type,
        contrastive_loss_weight=trainer_config.contrastive_loss_weight if trainer_config.use_hyperrectangle else 0.0,
        contrastive_loss_type=trainer_config.contrastive_loss_type,
        contrastive_margin=trainer_config.contrastive_margin,
    )

    datamodule = DualGraphDataModule(
        root=data_root,
        dataset_dir=dataset_dir,
        text_encoder_config=text_encoder_config,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )

    # 对比学习 DataLoader（条件创建）
    if trainer_config.use_hyperrectangle and model_config.use_hyperrectangle:
        datamodule.setup("fit")
        contrastive_dm = ContrastivePairDataModule(
            base_dataset=datamodule.train_dataset,
            pairs_per_epoch=trainer_config.contrastive_pairs_per_epoch,
            batch_size=trainer_config.contrastive_batch_size,
            num_workers=min(trainer_config.num_workers, 2),
        )
        contrastive_dm.setup("fit")
        module.set_contrastive_dataloader(contrastive_dm.contrastive_train_dataloader())

    lightning_trainer = LightningTrainer(
        config=trainer_config,
        experiment_name=experiment_name,
        logger_type=logger_type,
        extra_callbacks=extra_callbacks,
        has_validation=has_validation,
    )

    lightning_trainer.fit(module, datamodule, ckpt_path=ckpt_path)

    if has_test:
        lightning_trainer.test(module, datamodule)

    return module, lightning_trainer.trainer
