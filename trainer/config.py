#!/usr/bin/env python3
"""训练配置"""

from dataclasses import dataclass
from typing import Union


@dataclass
class TrainerConfig:
    """训练配置"""

    # 优化器
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    scheduler_type: str = "cosine"  # cosine / linear / none

    # 损失权重
    edge_loss_weight: float = 1.0
    graph_loss_weight: float = 1.0
    label_smoothing: float = 0.1

    # 训练参数
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    use_bucketing: bool = False    # 按 RTL 图大小分桶的动态 batch（减少 padding 浪费）
    token_budget: int = 8192       # bucketing 模式下每 batch 最大 RTL 节点总数

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
    use_hyperrectangle: bool = False              # 启用超矩形对比学习
    contrastive_loss_weight: float = 0.5          # 对比损失权重
    contrastive_pairs_per_epoch: int = 512        # 每 epoch 采样的对比对数量
    contrastive_batch_size: int = 16              # 对比 DataLoader batch 大小
    contrastive_margin: float = 0.2               # margin 模式下不相似对的交集上限
    contrastive_loss_type: str = "mse"            # mse / bce / margin
    hyper_min_margin: float = 0.01                # 超矩形每维度最小宽度

    # 图回归目标选择
    coverage_target_keys: tuple = ("branch",)     # 实际用于 loss 的覆盖率列子集
