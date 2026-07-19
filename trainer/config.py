#!/usr/bin/env python3
"""训练配置"""

from dataclasses import dataclass
from typing import Union

from contrastive_defaults import DEFAULT_PAIR_CANDIDATE_POOL_SIZE
from models.contrastive_loss import (
    DEFAULT_IOU_RANKING_CONFIG,
    IoURankingConfig,
)
from models.losses import (
    DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR,
    DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT,
)


@dataclass
class TrainerConfig:
    """训练配置"""

    # 优化器
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    scheduler_type: str = "cosine"  # cosine / linear / plateau / none
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_learning_rate: float = 1e-6

    # 损失权重
    graph_loss_weight: float = 1.0
    graph_relative_loss_weight: float = DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT
    graph_relative_loss_floor: float = DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR

    # 训练参数
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    use_bucketing: bool = False  # 按 RTL 图大小分桶的动态 batch（减少 padding 浪费）
    token_budget: int = 8192  # bucketing 模式下每 batch 最大 RTL 节点总数

    # 早停
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val/total_loss"
    early_stopping_mode: str = "min"

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    checkpoint_monitor: str = "val/total_loss"

    # 设备
    accelerator: str = "auto"  # auto / gpu / cpu
    devices: Union[int, str] = "auto"
    precision: str = "32-true"  # 32-true / 16-mixed / bf16-mixed
    strategy: str = "ddp_find_unused_parameters_true"  # auto / ddp / ddp_find_unused_parameters_true

    # 训练控制
    min_epochs: int = 1
    fast_dev_run: bool = False
    overfit_batches: float = 0.0

    # 验证
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1

    # 性能
    deterministic: bool = False
    benchmark: bool = True
    gradient_clip_algorithm: str = "norm"

    num_sanity_val_steps: int = 0  # DDP 小数据集下跳过 sanity check 避免挂起

    # 日志
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True

    # 对比学习超矩形配置
    contrastive_batch_size: int = 16  # 对比 DataLoader batch 大小
    pair_candidate_pool_size: int = DEFAULT_PAIR_CANDIDATE_POOL_SIZE
    pair_relative_low_quota: int = 2
    pair_relative_mid_quota: int = 1
    pair_relative_high_quota: int = 1
    pair_sampling_seed: int = 42

    # coverage-vector pair 对比学习配置（joint mode）
    joint_contrastive: bool = False  # 启用 pair 对比训练
    lambda_ce: float = 1.0  # a/b 两路监督损失的合并权重
    lambda_iou: float = 1.0  # 逐 coverage type 真实体积 IoU 损失权重
    iou_rank_loss_weight: float = DEFAULT_IOU_RANKING_CONFIG.weight
    iou_rank_margin: float = DEFAULT_IOU_RANKING_CONFIG.margin
    iou_rank_min_target_gap: float = DEFAULT_IOU_RANKING_CONFIG.min_target_gap
    lambda_volume: float = 1.0  # 单样本真实体积校准损失权重
    volume_warmup_epochs: int = 5
    smooth_intersection_temperature: float = 0.01

    # 图回归目标选择
    coverage_target_keys: tuple = ("branch",)  # 实际用于 loss 的覆盖率列子集

    def __post_init__(self) -> None:
        if not 0.0 < self.plateau_factor < 1.0:
            raise ValueError("plateau_factor must be between zero and one")
        if self.plateau_patience < 0:
            raise ValueError("plateau_patience must be non-negative")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.graph_relative_loss_weight < 0.0:
            raise ValueError("graph_relative_loss_weight must be non-negative")
        if self.graph_relative_loss_floor <= 0.0:
            raise ValueError("graph_relative_loss_floor must be positive")
        IoURankingConfig(
            weight=self.iou_rank_loss_weight,
            margin=self.iou_rank_margin,
            min_target_gap=self.iou_rank_min_target_gap,
        )
        if self.pair_candidate_pool_size <= 0:
            raise ValueError("pair_candidate_pool_size must be positive")
        quotas = (
            self.pair_relative_low_quota,
            self.pair_relative_mid_quota,
            self.pair_relative_high_quota,
        )
        if any(quota < 0 for quota in quotas):
            raise ValueError("relative pair quotas must be non-negative")
        if sum(quotas) > self.pair_candidate_pool_size:
            raise ValueError("relative pair quotas exceed pair_candidate_pool_size")
