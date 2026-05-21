#!/usr/bin/env python3
"""优化器和学习率调度器构建。"""

import math

import torch


def configure_adamw_with_scheduler(
    module,
    learning_rate: float,
    weight_decay: float,
    scheduler_type: str,
    warmup_steps: int,
):
    """按现有 LightningModule 语义创建 AdamW 和可选 step scheduler。"""
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if scheduler_type == "none":
        return optimizer

    trainer = module.trainer
    if trainer.max_steps > 0:
        total_steps = trainer.max_steps
    elif hasattr(trainer, "estimated_stepping_batches"):
        total_steps = trainer.estimated_stepping_batches
    else:
        total_steps = 10000

    effective_warmup = min(warmup_steps, int(total_steps * 0.1))

    def lr_lambda(current_step: int) -> float:
        if current_step < effective_warmup:
            return float(current_step) / float(max(1, effective_warmup))
        progress = float(current_step - effective_warmup) / float(
            max(1, total_steps - effective_warmup)
        )
        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
