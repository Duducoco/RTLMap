#!/usr/bin/env python3
"""对比学习指标 — 超矩形交集度与 Jaccard 对齐质量评估"""

import torch
import torch.nn.functional as F
from typing import Dict


class ContrastiveMetrics:
    """对比学习指标计算器

    评估超矩形交集度与真实覆盖率 Jaccard 的对齐质量。

    Step 级指标（每 batch 计算）: alignment_mse, alignment_mae, intersection_stats, jaccard_stats
    Epoch 级指标（聚合后计算）: alignment_pearson_r, alignment_spearman_r

    Pearson R 和 Spearman R 在 step 级计算不稳定（batch 太小时方差为 0），
    通过 collect_step → compute_epoch 两步完成。
    """

    def __init__(self):
        self._step_intersections: list[torch.Tensor] = []
        self._step_jaccards: list[torch.Tensor] = []

    def compute(
        self,
        intersection: torch.Tensor,
        jaccard: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算对比学习 step 级指标

        Args:
            intersection: [B] 超矩形交集度
            jaccard: [B] 真实覆盖率 Jaccard 相似度
        """
        alignment_mse = F.mse_loss(intersection, jaccard)
        alignment_mae = F.l1_loss(intersection, jaccard)

        result = {
            "alignment_mse": alignment_mse,
            "alignment_mae": alignment_mae,
            "intersection_mean": intersection.mean(),
            "intersection_std": intersection.std() if intersection.numel() > 1 else torch.tensor(0.0, device=intersection.device),
            "intersection_min": intersection.min(),
            "intersection_max": intersection.max(),
            "jaccard_mean": jaccard.mean(),
            "jaccard_std": jaccard.std() if jaccard.numel() > 1 else torch.tensor(0.0, device=jaccard.device),
            "jaccard_min": jaccard.min(),
            "jaccard_max": jaccard.max(),
        }

        return result

    def collect_step(self, intersection: torch.Tensor, jaccard: torch.Tensor):
        """收集 step 级数据用于 epoch 级 Pearson/Spearman 计算"""
        self._step_intersections.append(intersection.detach().cpu())
        self._step_jaccards.append(jaccard.detach().cpu())

    def compute_epoch(self) -> Dict[str, float]:
        """epoch 级聚合计算 Pearson R 和 Spearman R

        Returns:
            dict with alignment_pearson_r and alignment_spearman_r (float)
        """
        result = {"alignment_pearson_r": 0.0, "alignment_spearman_r": 0.0}

        if not self._step_intersections:
            return result

        all_inter = torch.cat(self._step_intersections, dim=0)
        all_jacc = torch.cat(self._step_jaccards, dim=0)

        if all_inter.numel() < 2:
            return result

        # Pearson R (tensor 计算)
        inter_centered = all_inter - all_inter.mean()
        jacc_centered = all_jacc - all_jacc.mean()
        cov = (inter_centered * jacc_centered).sum()
        inter_std = inter_centered.pow(2).sum().sqrt()
        jacc_std = jacc_centered.pow(2).sum().sqrt()
        if inter_std > 1e-8 and jacc_std > 1e-8:
            result["alignment_pearson_r"] = (cov / (inter_std * jacc_std)).item()

        # Spearman R (需要 scipy)
        try:
            from scipy.stats import spearmanr
            sr, _ = spearmanr(all_inter.numpy(), all_jacc.numpy())
            result["alignment_spearman_r"] = float(sr)
        except (ImportError, ValueError):
            pass

        self.reset()
        return result

    def reset(self):
        self._step_intersections.clear()
        self._step_jaccards.clear()