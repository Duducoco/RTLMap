#!/usr/bin/env python3
"""Joint contrastive 训练步骤。"""

import torch

from datasets.data_types import ContrastivePairBatch
from models.contrastive_loss import compute_coverage_geometry_losses
from models.hyperrectangle import hyperrectangle_geometry


def _move_geometry_targets(batch: ContrastivePairBatch, device) -> None:
    batch.density_a = batch.density_a.to(device)
    batch.density_b = batch.density_b.to(device)
    batch.size_mask = batch.size_mask.to(device)
    batch.jaccard = batch.jaccard.to(device)
    batch.iou_mask = batch.iou_mask.to(device)


def _geometry_losses(module, out_a, out_b, batch, *, training: bool):
    return compute_coverage_geometry_losses(
        out_a,
        out_b,
        density_a=batch.density_a,
        density_b=batch.density_b,
        size_mask=batch.size_mask,
        jaccard=batch.jaccard,
        iou_mask=batch.iou_mask,
        min_width=module.model.config.hyper_min_margin,
        smooth_temperature=(
            module.smooth_intersection_temperature if training else None
        ),
    )


def _hard_geometry_metrics(module, out_a, out_b, batch):
    geometry = hyperrectangle_geometry(
        out_a.hyper_min,
        out_a.hyper_max,
        out_b.hyper_min,
        out_b.hyper_max,
    )
    metrics = module.contrastive_metrics.compute(
        geometry=geometry,
        density_a=batch.density_a,
        density_b=batch.density_b,
        size_mask=batch.size_mask,
        jaccard=batch.jaccard,
        iou_mask=batch.iou_mask,
        hyper_min_a=out_a.hyper_min,
        hyper_max_a=out_a.hyper_max,
        hyper_min_b=out_b.hyper_min,
        hyper_max_b=out_b.hyper_max,
    )
    return geometry, metrics


def run_pair_training_step(module, batch: ContrastivePairBatch) -> torch.Tensor:
    """执行覆盖向量 pair 对比训练步骤。"""
    batch.batch_a = batch.batch_a.to(module.device)
    batch.batch_b = batch.batch_b.to(module.device)
    _move_geometry_targets(batch, module.device)

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
    geometry_losses = _geometry_losses(module, out_a, out_b, batch, training=True)
    warmup = (
        1.0
        if module.volume_warmup_epochs <= 0
        else min(1.0, module.current_epoch / module.volume_warmup_epochs)
    )
    effective_volume_weight = module.lambda_volume * warmup
    total = (
        module.lambda_ce * l_sup
        + module.lambda_iou * geometry_losses.iou_loss
        + effective_volume_weight * geometry_losses.volume_loss
    )

    module.log_dict(
        {
            "train/graph_loss_a": losses_a["graph_loss"].detach(),
            "train/graph_loss_b": losses_b["graph_loss"].detach(),
            "train/supervised_loss": l_sup.detach(),
            "train/iou_loss": geometry_losses.iou_loss.detach(),
            "train/volume_loss": geometry_losses.volume_loss.detach(),
            "train/effective_volume_weight": torch.tensor(
                effective_volume_weight, device=module.device
            ),
            "train/total_joint_loss": total.detach(),
        },
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        batch_size=batch.jaccard.shape[0],
        sync_dist=True,
    )

    _, c_step = _hard_geometry_metrics(module, out_a, out_b, batch)
    module.log_dict(
        {f"train/geometry_{key}": value for key, value in c_step.items()},
        on_step=True,
        on_epoch=True,
        prog_bar=False,
        batch_size=batch.jaccard.shape[0],
        sync_dist=True,
    )

    return total


def run_pair_validation_step(module, batch: ContrastivePairBatch) -> torch.Tensor:
    batch.batch_a = batch.batch_a.to(module.device)
    batch.batch_b = batch.batch_b.to(module.device)
    _move_geometry_targets(batch, module.device)
    out_a = module(batch.batch_a)
    out_b = module(batch.batch_b)
    losses = _geometry_losses(module, out_a, out_b, batch, training=False)
    geometry, metrics = _hard_geometry_metrics(module, out_a, out_b, batch)
    module.contrastive_metrics.collect(
        geometry=geometry,
        density_a=batch.density_a,
        density_b=batch.density_b,
        size_mask=batch.size_mask,
        jaccard=batch.jaccard,
        iou_mask=batch.iou_mask,
    )
    total = module.lambda_iou * losses.iou_loss + module.lambda_volume * losses.volume_loss
    values = {
        "val/pair_total_loss": total.detach(),
        "val/pair_iou_loss": losses.iou_loss.detach(),
        "val/pair_volume_loss": losses.volume_loss.detach(),
    }
    values.update({f"val/pair_{key}": value for key, value in metrics.items()})
    module.log_dict(
        values,
        on_step=False,
        on_epoch=True,
        batch_size=batch.jaccard.shape[0],
        sync_dist=True,
        add_dataloader_idx=False,
    )
    return total
