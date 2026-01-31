#!/usr/bin/env python3
"""训练回调函数"""

from typing import List, Optional, Union
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ProgressBar,
)

from .config import TrainerConfig

# RichProgressBar 是可选依赖
try:
    from lightning.pytorch.callbacks import RichProgressBar

    _RICH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _RICH_AVAILABLE = False

try:
    # 检查 rich 库是否可用
    import rich

    _RICH_AVAILABLE = _RICH_AVAILABLE and True
except ImportError:
    _RICH_AVAILABLE = False


class CallbackFactory:
    """Callback 工厂类 - 根据配置创建 Lightning callbacks"""

    @staticmethod
    def create_lr_monitor() -> LearningRateMonitor:
        """创建学习率监控 callback"""
        return LearningRateMonitor(logging_interval="step")

    @staticmethod
    def create_progress_bar(use_rich: bool = True) -> Union[ProgressBar, Callback]:
        """
        创建进度条 callback

        Args:
            use_rich: 是否使用 RichProgressBar（需要 rich 库）

        Returns:
            进度条 callback，如果 rich 不可用则返回 None（使用默认进度条）
        """
        if use_rich and _RICH_AVAILABLE:
            return RichProgressBar()
        # 返回 None 表示使用 Lightning 默认进度条
        return None

    @staticmethod
    def create_checkpoint(
        config: TrainerConfig, experiment_name: str
    ) -> ModelCheckpoint:
        """创建模型检查点 callback"""
        return ModelCheckpoint(
            dirpath=f"{config.checkpoint_dir}/{experiment_name}",
            filename="{epoch:02d}-{val/total_loss:.4f}",
            monitor=config.checkpoint_monitor,
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True,
        )

    @staticmethod
    def create_early_stopping(config: TrainerConfig) -> EarlyStopping:
        """创建早停 callback"""
        return EarlyStopping(
            monitor=config.early_stopping_monitor,
            mode=config.early_stopping_mode,
            patience=config.early_stopping_patience,
            verbose=True,
        )

    @classmethod
    def create_default_callbacks(
        cls,
        config: TrainerConfig,
        experiment_name: str,
        has_validation: bool = True,
        extra_callbacks: Optional[List[Callback]] = None,
        use_rich_progress: bool = True,
        enable_progress_bar: Optional[bool] = None,
    ) -> List[Callback]:
        """
        创建默认 callback 列表

        Args:
            config: 训练配置
            experiment_name: 实验名称
            has_validation: 是否有验证数据
            extra_callbacks: 额外的自定义 callbacks
            use_rich_progress: 是否使用 RichProgressBar
            enable_progress_bar: 是否启用进度条（None 时从 config 读取）

        Returns:
            callback 列表
        """
        callbacks = [cls.create_lr_monitor()]

        # 确定是否启用进度条
        if enable_progress_bar is None:
            enable_progress_bar = config.enable_progress_bar

        # 添加进度条（如果启用且可用）
        if enable_progress_bar:
            progress_bar = cls.create_progress_bar(use_rich_progress)
            if progress_bar is not None:
                callbacks.append(progress_bar)

        if has_validation:
            callbacks.append(cls.create_checkpoint(config, experiment_name))
            callbacks.append(cls.create_early_stopping(config))

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        return callbacks
