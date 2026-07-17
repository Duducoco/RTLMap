#!/usr/bin/env python3
"""训练回调函数"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any, List, Optional, Union

import lightning as L
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

if importlib.util.find_spec("rich") is None:
    _RICH_AVAILABLE = False


if _RICH_AVAILABLE:

    class _SafeRichProgressBar(RichProgressBar):
        """DDP 安全的 RichProgressBar

        DDP 多进程下 rich.Console._live_stack 可能为空，
        导致 _init_progress → clear_live() 时 pop from empty list，
        且后续回调断言 self.progress is not None 失败。
        检测到异常时禁用自身，避免级联错误。
        """

        _disabled: bool = False

        def _init_progress(self, trainer: L.Trainer) -> None:
            try:
                super()._init_progress(trainer)
            except IndexError:
                self._disabled = True

        def on_validation_batch_start(self, *args, **kwargs) -> None:
            if self._disabled:
                return
            super().on_validation_batch_start(*args, **kwargs)

        def on_validation_batch_end(self, *args, **kwargs) -> None:
            if self._disabled:
                return
            super().on_validation_batch_end(*args, **kwargs)

        def on_train_batch_start(self, *args, **kwargs) -> None:
            if self._disabled:
                return
            super().on_train_batch_start(*args, **kwargs)

        def on_train_batch_end(self, *args, **kwargs) -> None:
            if self._disabled:
                return
            super().on_train_batch_end(*args, **kwargs)


class RuntimeConfiguredEarlyStopping(EarlyStopping):
    """Restore progress from checkpoints while keeping runtime stop policy."""

    @property
    def state_key(self) -> str:
        return f"{EarlyStopping.__qualname__}{repr({'monitor': self.monitor, 'mode': self.mode})}"

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        configured_patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = configured_patience


class EvalMetricJsonLogger(Callback):
    """在 logger 的 version 目录下导出每个验证 epoch 的指标 JSON。"""

    _FILENAME = "eval_metric.json"

    @staticmethod
    def _resolve_log_dir(trainer: L.Trainer) -> Optional[Path]:
        logger = getattr(trainer, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if not log_dir:
            return None
        return Path(log_dir)

    @staticmethod
    def _to_scalar(value: Any) -> Optional[float]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numel"):
            if value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @classmethod
    def _load_payload(cls, output_path: Path) -> list[dict[str, Any]]:
        if not output_path.exists():
            return []
        try:
            payload = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    @classmethod
    def _collect_metrics(cls, trainer: L.Trainer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            key_str = str(key)
            if not (key_str.startswith("val/") or key_str.startswith("hp/val_")):
                continue
            scalar = cls._to_scalar(value)
            if scalar is None:
                continue
            metrics[key_str] = scalar
        return metrics

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False) or not getattr(
            trainer, "is_global_zero", True
        ):
            return

        log_dir = self._resolve_log_dir(trainer)
        if log_dir is None:
            return

        metrics = self._collect_metrics(trainer)
        if not metrics:
            return

        output_path = log_dir / self._FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_payload(output_path)
        payload.append(
            {
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
                "metrics": metrics,
            }
        )
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        temporary_path.replace(output_path)


class CallbackFactory:
    """Callback 工厂类 - 根据配置创建 Lightning callbacks"""

    @staticmethod
    def create_lr_monitor() -> LearningRateMonitor:
        """创建学习率监控 callback"""
        return LearningRateMonitor(logging_interval="step")

    @staticmethod
    def create_progress_bar(
        use_rich: bool = True,
    ) -> Union[ProgressBar, Callback, None]:
        """
        创建进度条 callback

        Args:
            use_rich: 是否使用 RichProgressBar（需要 rich 库）

        Returns:
            进度条 callback，如果 rich 不可用则返回 None（使用默认进度条）
        """
        if use_rich and _RICH_AVAILABLE:
            # DDP 下 RichProgressBar 的 live display 与多进程 stdout 冲突，
            # 回退到 Lightning 默认 TQDM 进度条
            return _SafeRichProgressBar()
        # 返回 None 表示使用 Lightning 默认进度条
        return None

    @staticmethod
    def create_checkpoint(
        config: TrainerConfig, experiment_name: str
    ) -> ModelCheckpoint:
        """创建模型检查点 callback"""
        return ModelCheckpoint(
            dirpath=f"{config.checkpoint_dir}/{experiment_name}",
            filename=(
                "{epoch:02d}-pair_loss={val/pair_total_loss:.4f}"
                if config.checkpoint_monitor == "val/pair_total_loss"
                else "{epoch:02d}-graph_loss={val/total_loss:.4f}"
            ),
            auto_insert_metric_name=False,
            monitor=config.checkpoint_monitor,
            mode="min",
            save_top_k=config.save_top_k,
            save_last=True,
        )

    @staticmethod
    def create_early_stopping(config: TrainerConfig) -> EarlyStopping:
        """创建早停 callback"""
        return RuntimeConfiguredEarlyStopping(
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
        callbacks = [cls.create_lr_monitor(), EvalMetricJsonLogger()]

        # 确定是否启用进度条
        if enable_progress_bar is None:
            enable_progress_bar = config.enable_progress_bar

        # 添加进度条（如果启用且可用）
        # DDP 下 RichProgressBar 的 live display 与多进程 stdout 冲突，回退到 TQDM
        is_ddp = "ddp" in getattr(config, "strategy", "")
        if enable_progress_bar:
            progress_bar = cls.create_progress_bar(
                use_rich=use_rich_progress and not is_ddp,
            )
            if progress_bar is not None:
                callbacks.append(progress_bar)

        if has_validation:
            callbacks.append(cls.create_checkpoint(config, experiment_name))
            callbacks.append(cls.create_early_stopping(config))

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        return callbacks
