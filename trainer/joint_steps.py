#!/usr/bin/env python3
"""Joint contrastive 训练步骤。"""

import torch

from datasets.data_types import ContrastiveTripleBatch
from models.contrastive_loss import compute_contrastive_loss
from .losses import compute_or_consistency_loss


def run_joint_training_step(module, batch: ContrastiveTripleBatch) -> torch.Tensor:
    """执行三元组联合训练步骤，保持原日志 key 和 loss 语义不变。"""
    batch.batch_a = batch.batch_a.to(module.device)
    batch.batch_b = batch.batch_b.to(module.device)
    batch.batch_merged = batch.batch_merged.to(module.device)
    batch.similarity = batch.similarity.to(module.device)

    out_a = module(batch.batch_a)
    out_b = module(batch.batch_b)
    out_m = module.model.merge_outputs(out_a, out_b, batch.batch_merged)

    common_loss_kwargs = {
        "edge_loss_weight": module.edge_loss_weight,
        "label_smoothing": module.label_smoothing,
        "edge_loss_type": module.edge_loss_type,
        "focal_gamma": module.focal_gamma,
        "edge_class_weight": (
            module.edge_class_weight_neg,
            module.edge_class_weight_pos,
        ),
    }

    losses_a = module.model.compute_loss(
        out_a,
        batch.batch_a,
        graph_loss_weight=module.graph_loss_weight,
        **common_loss_kwargs,
    )
    losses_b = module.model.compute_loss(
        out_b,
        batch.batch_b,
        graph_loss_weight=module.graph_loss_weight,
        **common_loss_kwargs,
    )
    losses_m = module.model.compute_loss(
        out_m,
        batch.batch_merged,
        graph_loss_weight=module.graph_loss_weight,
        **common_loss_kwargs,
    )

    l_ce = losses_a["total_loss"] + losses_b["total_loss"] + losses_m["total_loss"]
    l_cl = compute_contrastive_loss(
        out_a,
        out_b,
        batch.similarity,
        loss_type=module.contrastive_loss_type,
        margin=module.contrastive_margin,
    )

    total = module.lambda_ce * l_ce + module.lambda_cl * l_cl

    l_or = torch.tensor(0.0, device=module.device)
    if module.or_consistency_weight > 0:
        merged_labels = getattr(batch.batch_merged, "edge_labels", None)
        l_or = compute_or_consistency_loss(out_a, out_b, out_m, merged_labels)
        total = total + module.or_consistency_weight * l_or

    module.log_dict(
        {
            "train/edge_loss_a": losses_a["edge_loss"].detach(),
            "train/edge_loss_b": losses_b["edge_loss"].detach(),
            "train/edge_loss_merged": losses_m["edge_loss"].detach(),
            "train/graph_loss_merged": losses_m["graph_loss"].detach(),
            "train/merged_total_loss": losses_m["total_loss"].detach(),
            "train/ce_loss": l_ce.detach(),
            "train/contrastive_loss": l_cl.detach(),
            "train/or_consistency_loss": l_or.detach(),
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
