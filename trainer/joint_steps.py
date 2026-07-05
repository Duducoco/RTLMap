#!/usr/bin/env python3
"""Joint contrastive 训练步骤。"""

import torch

from datasets.data_types import ContrastivePairBatch
from models.contrastive_loss import compute_contrastive_loss


def run_pair_training_step(module, batch: ContrastivePairBatch) -> torch.Tensor:
    """执行覆盖向量 pair 对比训练步骤。"""
    batch.batch_a = batch.batch_a.to(module.device)
    batch.batch_b = batch.batch_b.to(module.device)
    batch.similarity = batch.similarity.to(module.device)

    out_a = module(batch.batch_a)
    out_b = module(batch.batch_b)

    losses_a = module.model.compute_loss(
        out_a,
        batch.batch_a,
        graph_loss_weight=module.graph_loss_weight,
    )
    losses_b = module.model.compute_loss(
        out_b,
        batch.batch_b,
        graph_loss_weight=module.graph_loss_weight,
    )

    l_sup = losses_a["graph_loss"] + losses_b["graph_loss"]
    l_cl = compute_contrastive_loss(
        out_a,
        out_b,
        batch.similarity,
        loss_type=module.contrastive_loss_type,
        margin=module.contrastive_margin,
    )
    total = module.lambda_ce * l_sup + module.lambda_cl * l_cl

    module.log_dict(
        {
            "train/graph_loss_a": losses_a["graph_loss"].detach(),
            "train/graph_loss_b": losses_b["graph_loss"].detach(),
            "train/supervised_loss": l_sup.detach(),
            "train/contrastive_loss": l_cl.detach(),
            "train/total_joint_loss": total.detach(),
        },
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        batch_size=1,
    )

    c_step = module._compute_contrastive_step_metrics(out_a, out_b, batch.similarity)
    module.log_dict(c_step, on_step=True, on_epoch=True, prog_bar=False, batch_size=1)

    return total
