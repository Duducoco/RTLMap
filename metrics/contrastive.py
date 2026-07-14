#!/usr/bin/env python3
"""Metrics for typed true-volume coverage-space geometry."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _local_metric_pairs(
    predictions: list[torch.Tensor], targets: list[torch.Tensor]
) -> torch.Tensor:
    if not predictions:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.stack(
        (torch.cat(predictions).float(), torch.cat(targets).float()), dim=-1
    )


def _gather_metric_pairs(
    pairs: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    pairs = pairs.to(device=device, dtype=torch.float32)
    if not dist.is_available() or not dist.is_initialized():
        return pairs.cpu()

    world_size = dist.get_world_size()
    local_size = torch.tensor([pairs.shape[0]], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    max_size = max(1, max(int(size.item()) for size in sizes))
    padded = torch.zeros((max_size, 2), dtype=pairs.dtype, device=device)
    padded[: pairs.shape[0]] = pairs
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    return torch.cat(
        [
            rank_pairs[: int(rank_size.item())]
            for rank_pairs, rank_size in zip(gathered, sizes, strict=True)
        ],
        dim=0,
    ).cpu()


class ContrastiveMetrics:
    """Compute and accumulate hard-geometry calibration metrics."""

    def __init__(self):
        self._volume_predictions: list[torch.Tensor] = []
        self._volume_targets: list[torch.Tensor] = []
        self._iou_predictions: list[torch.Tensor] = []
        self._iou_targets: list[torch.Tensor] = []

    def compute(
        self,
        *,
        geometry,
        density_a: torch.Tensor,
        density_b: torch.Tensor,
        size_mask: torch.Tensor,
        jaccard: torch.Tensor,
        iou_mask: torch.Tensor,
        hyper_min_a: torch.Tensor,
        hyper_max_a: torch.Tensor,
        hyper_min_b: torch.Tensor,
        hyper_max_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        volume_predictions = torch.cat((geometry.volume_a, geometry.volume_b), dim=0)
        volume_targets = torch.cat((density_a.float(), density_b.float()), dim=0)
        volume_mask = torch.cat((size_mask, size_mask), dim=0)
        positive_pairs = iou_mask & (jaccard > 0.0)
        zero_intersection = ~geometry.has_hard_intersection

        all_min = torch.cat((hyper_min_a, hyper_min_b), dim=0)
        all_max = torch.cat((hyper_max_a, hyper_max_b), dim=0)
        return {
            "volume_mae": _masked_mean(
                (volume_predictions - volume_targets).abs(), volume_mask
            ),
            "iou_mse": _masked_mean(
                F.mse_loss(geometry.iou, jaccard.float(), reduction="none"), iou_mask
            ),
            "iou_mae": _masked_mean(
                (geometry.iou - jaccard.float()).abs(), iou_mask
            ),
            "predicted_volume_mean": _masked_mean(volume_predictions, volume_mask),
            "target_density_mean": _masked_mean(volume_targets, volume_mask),
            "predicted_iou_mean": _masked_mean(geometry.iou, iou_mask),
            "target_jaccard_mean": _masked_mean(jaccard.float(), iou_mask),
            "positive_pair_zero_intersection_ratio": _masked_mean(
                zero_intersection.float(), positive_pairs
            ),
            "min_boundary_saturation": (all_min < 0.01).float().mean(),
            "max_boundary_saturation": (all_max > 0.99).float().mean(),
        }

    def collect(
        self,
        *,
        geometry,
        density_a: torch.Tensor,
        density_b: torch.Tensor,
        size_mask: torch.Tensor,
        jaccard: torch.Tensor,
        iou_mask: torch.Tensor,
    ) -> None:
        volume_predictions = torch.cat((geometry.volume_a, geometry.volume_b), dim=0)
        volume_targets = torch.cat((density_a.float(), density_b.float()), dim=0)
        volume_mask = torch.cat((size_mask, size_mask), dim=0)
        self._volume_predictions.append(volume_predictions[volume_mask].detach().cpu())
        self._volume_targets.append(volume_targets[volume_mask].detach().cpu())
        self._iou_predictions.append(geometry.iou[iou_mask].detach().cpu())
        self._iou_targets.append(jaccard[iou_mask].detach().float().cpu())

    @staticmethod
    def _spearman(pairs: torch.Tensor) -> float:
        if pairs.shape[0] < 2:
            return 0.0
        prediction = pairs[:, 0]
        target = pairs[:, 1]
        if (
            prediction.unique().numel() < 2
            or target.unique().numel() < 2
        ):
            return 0.0
        from scipy.stats import spearmanr

        value, _ = spearmanr(prediction.numpy(), target.numpy())
        return float(value) if value == value else 0.0

    def compute_epoch(self, *, device: torch.device) -> dict[str, float]:
        volume_pairs = _gather_metric_pairs(
            _local_metric_pairs(self._volume_predictions, self._volume_targets),
            device=device,
        )
        iou_pairs = _gather_metric_pairs(
            _local_metric_pairs(self._iou_predictions, self._iou_targets),
            device=device,
        )
        result = {
            "volume_spearman": self._spearman(volume_pairs),
            "iou_spearman": self._spearman(iou_pairs),
        }
        self.reset()
        return result

    def reset(self) -> None:
        self._volume_predictions.clear()
        self._volume_targets.clear()
        self._iou_predictions.clear()
        self._iou_targets.clear()
