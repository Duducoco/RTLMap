#!/usr/bin/env python3
"""应用级配置聚合对象。"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Union
from pathlib import Path

from models.data_types import ModelConfig
from .config import TrainerConfig

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig


@dataclass
class DataConfig:
    """数据路径配置。"""

    data_root: str
    dataset_dir: Optional[Union[str, Path, list[str], list[Path]]] = None


@dataclass
class RuntimeConfig:
    """运行时控制配置。"""

    has_validation: bool = True
    has_test: bool = False
    logger_type: str = "tensorboard"
    experiment_name: str = "dual_graph_gnn"
    ckpt_path: Optional[str] = None


@dataclass
class AppConfig:
    """训练入口统一配置。"""

    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    text_encoder: Optional["TextEncoderConfig"]
    runtime: RuntimeConfig
