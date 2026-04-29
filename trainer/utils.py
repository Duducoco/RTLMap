#!/usr/bin/env python3
"""便捷训练函数"""

import lightning as L
from typing import List, Optional, Tuple
from lightning.pytorch.callbacks import Callback

from datasets import DualGraphDataModule
from datasets.triple_datamodule import ContrastiveTripleDataModule
from .app_config import AppConfig
from .lightning_module import DualGraphLightningModule
from .lightning_trainer import LightningTrainer


class _JointTrainDataModule(L.LightningDataModule):
    """包装 DataModule：训练 dataloader 来自 triple loader，验证/测试来自原 DataModule。

    这样 Lightning 的 fit/val/test 接口保持不变，joint 模式对 Trainer 透明。
    """

    def __init__(
        self,
        base: DualGraphDataModule,
        triple_dm: ContrastiveTripleDataModule,
    ):
        super().__init__()
        self._base = base
        self._triple_dm = triple_dm

    def prepare_data(self):
        self._base.prepare_data()

    def setup(self, stage=None):
        self._base.setup(stage)
        self._triple_dm.setup(stage)

    def train_dataloader(self):
        return self._triple_dm.train_dataloader()

    def val_dataloader(self):
        return self._base.val_dataloader()

    def test_dataloader(self):
        return self._base.test_dataloader()


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

    module = DualGraphLightningModule(
        model_config=model_config,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        edge_loss_weight=trainer_config.edge_loss_weight,
        graph_loss_weight=trainer_config.graph_loss_weight,
        label_smoothing=trainer_config.label_smoothing,
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
        or_consistency_weight=trainer_config.or_consistency_weight,
    )

    datamodule = DualGraphDataModule(
        root=app_config.data.data_root,
        dataset_dir=dataset_dir,
        text_encoder_config=app_config.text_encoder,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
        use_bucketing=trainer_config.use_bucketing,
        token_budget=trainer_config.token_budget,
    )

    effective_datamodule = datamodule
    if trainer_config.joint_contrastive and trainer_config.contrastive_triple_index:
        triple_dm = ContrastiveTripleDataModule(
            dataset_dir=dataset_dir,
            index_file=trainer_config.contrastive_triple_index,
            batch_size=trainer_config.contrastive_batch_size,
            num_workers=min(trainer_config.num_workers, 4),
        )
        effective_datamodule = _JointTrainDataModule(datamodule, triple_dm)

    lightning_trainer = LightningTrainer(
        config=trainer_config,
        experiment_name=app_config.runtime.experiment_name,
        logger_type=app_config.runtime.logger_type,
        extra_callbacks=extra_callbacks,
        has_validation=app_config.runtime.has_validation,
    )

    lightning_trainer.fit(
        module, effective_datamodule, ckpt_path=app_config.runtime.ckpt_path
    )

    if app_config.runtime.has_test:
        lightning_trainer.test(module, effective_datamodule)

    return module, lightning_trainer.trainer
