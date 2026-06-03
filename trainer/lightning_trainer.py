#!/usr/bin/env python3
"""Lightning 训练器包装类"""

from typing import Optional, List, Any
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from .config import TrainerConfig
from .callbacks import CallbackFactory


class LightningTrainer:
    """Lightning 训练器包装类

    提供面向对象的训练 API，支持分离的 fit/test/validate/predict 操作。

    Example:
        >>> config = TrainerConfig(max_epochs=10, learning_rate=1e-3)
        >>> trainer = LightningTrainer(config, experiment_name="my_exp")
        >>> trainer.fit(model, datamodule)
        >>> trainer.test(model, datamodule)
    """

    def __init__(
        self,
        config: TrainerConfig,
        experiment_name: str = "dual_graph_gnn",
        logger_type: str = "tensorboard",
        extra_callbacks: Optional[List[Callback]] = None,
        has_validation: bool = True,
    ):
        """
        Args:
            config: 训练配置
            experiment_name: 实验名称
            logger_type: 日志类型 (tensorboard / csv)
            extra_callbacks: 额外的自定义 callbacks
            has_validation: 是否有验证数据（影响 checkpoint 和 early stopping）
        """
        self.config = config
        self.experiment_name = experiment_name
        self._has_validation = has_validation

        # 创建 Logger
        self.logger = self._create_logger(logger_type)

        # 创建 Callbacks（仅在启用 checkpointing 时创建 checkpoint 相关 callbacks）
        effective_has_validation = has_validation and config.enable_checkpointing
        self.callbacks = CallbackFactory.create_default_callbacks(
            config=config,
            experiment_name=experiment_name,
            has_validation=effective_has_validation,
            extra_callbacks=extra_callbacks,
        )

        # 创建 Lightning Trainer
        self.trainer = L.Trainer(
            # 训练轮数
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs,
            # 设备
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            strategy=config.strategy,
            # 回调和日志
            callbacks=self.callbacks,
            logger=self.logger,
            # 检查点和进度
            enable_checkpointing=config.enable_checkpointing,
            enable_progress_bar=config.enable_progress_bar,
            enable_model_summary=config.enable_model_summary,
            # 调试选项
            fast_dev_run=config.fast_dev_run,
            overfit_batches=config.overfit_batches,
            # 梯度
            gradient_clip_val=config.gradient_clip_val,
            gradient_clip_algorithm=config.gradient_clip_algorithm,
            accumulate_grad_batches=config.accumulate_grad_batches,
            # 验证
            val_check_interval=config.val_check_interval,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            # 性能
            deterministic=config.deterministic,
            benchmark=config.benchmark,
            use_distributed_sampler=not config.use_bucketing,
            # 日志
            log_every_n_steps=config.log_every_n_steps,
            # sanity check
            num_sanity_val_steps=config.num_sanity_val_steps,
        )

    def _create_logger(self, logger_type: str):
        """创建日志记录器"""
        if logger_type == "tensorboard":
            return TensorBoardLogger(
                save_dir=self.config.checkpoint_dir,
                name=self.experiment_name,
            )
        else:
            return CSVLogger(
                save_dir=self.config.checkpoint_dir,
                name=self.experiment_name,
            )

    def fit(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """训练模型

        Args:
            model: Lightning 模型
            datamodule: 数据模块
            ckpt_path: 检查点路径（用于恢复训练）
        """
        self.trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    def test(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> List[dict]:
        """测试模型

        Args:
            model: Lightning 模型
            datamodule: 数据模块
            ckpt_path: 检查点路径（默认使用最佳模型）

        Returns:
            测试结果列表
        """
        return self.trainer.test(model, datamodule, ckpt_path=ckpt_path)

    def validate(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> List[dict]:
        """验证模型

        Args:
            model: Lightning 模型
            datamodule: 数据模块
            ckpt_path: 检查点路径

        Returns:
            验证结果列表
        """
        return self.trainer.validate(model, datamodule, ckpt_path=ckpt_path)

    def predict(
        self,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        ckpt_path: Optional[str] = None,
    ) -> List[Any]:
        """预测

        Args:
            model: Lightning 模型
            datamodule: 数据模块
            ckpt_path: 检查点路径

        Returns:
            预测结果列表
        """
        return self.trainer.predict(model, datamodule, ckpt_path=ckpt_path)

    @property
    def checkpoint_callback(self) -> Optional[Callback]:
        """获取 ModelCheckpoint callback"""
        return self.trainer.checkpoint_callback

    @property
    def early_stopping_callback(self) -> Optional[Callback]:
        """获取 EarlyStopping callback"""
        for callback in self.callbacks:
            if callback.__class__.__name__ == "EarlyStopping":
                return callback
        return None
