#!/usr/bin/env python3
"""对比学习指标 - 超矩形交集度与覆盖向量相似度对齐质量评估。"""

import torch
import torch.nn.functional as F
from typing import Dict


class ContrastiveMetrics:
    """对比学习指标计算器

    评估超矩形交集度与真实覆盖向量相似度的对齐质量。

    Step 级指标（每 batch 计算）: alignment_mse, alignment_mae, intersection_stats, similarity_stats
    Epoch 级指标（聚合后计算）: alignment_pearson_r, alignment_spearman_r

    Pearson R 和 Spearman R 在 step 级计算不稳定（batch 太小时方差为 0），
    通过 collect_step → compute_epoch 两步完成。
    """

    def __init__(self):
        self._step_intersections: list[torch.Tensor] = []
        self._step_similarities: list[torch.Tensor] = []

    def compute(
        self,
        intersection: torch.Tensor,
        similarity: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算对比学习 step 级指标

        Args:
            intersection: [B] 超矩形交集度
            similarity: [B] 真实覆盖向量相似度
        """
        alignment_mse = F.mse_loss(intersection, similarity)
        alignment_mae = F.l1_loss(intersection, similarity)

        result = {
            "alignment_mse": alignment_mse,
            "alignment_mae": alignment_mae,
            "intersection_mean": intersection.mean(),
            "intersection_std": intersection.std()
            if intersection.numel() > 1
            else torch.tensor(0.0, device=intersection.device),
            "intersection_min": intersection.min(),
            "intersection_max": intersection.max(),
            "similarity_mean": similarity.mean(),
            "similarity_std": similarity.std()
            if similarity.numel() > 1
            else torch.tensor(0.0, device=similarity.device),
            "similarity_min": similarity.min(),
            "similarity_max": similarity.max(),
        }

        return result

    def collect_step(self, intersection: torch.Tensor, similarity: torch.Tensor):
        """收集 step 级数据用于 epoch 级 Pearson/Spearman 计算"""
        self._step_intersections.append(intersection.detach().cpu())
        self._step_similarities.append(similarity.detach().cpu())

    def compute_epoch(self) -> Dict[str, float]:
        """epoch 级聚合计算 Pearson R 和 Spearman R

        Returns:
            dict with alignment_pearson_r and alignment_spearman_r (float)
        """
        result = {"alignment_pearson_r": 0.0, "alignment_spearman_r": 0.0}

        if not self._step_intersections:
            return result

        all_inter = torch.cat(self._step_intersections, dim=0)
        all_sim = torch.cat(self._step_similarities, dim=0)

        if all_inter.numel() < 2:
            return result

        # Pearson R (tensor 计算)
        inter_centered = all_inter - all_inter.mean()
        sim_centered = all_sim - all_sim.mean()
        cov = (inter_centered * sim_centered).sum()
        inter_std = inter_centered.pow(2).sum().sqrt()
        sim_std = sim_centered.pow(2).sum().sqrt()
        if inter_std > 1e-8 and sim_std > 1e-8:
            result["alignment_pearson_r"] = (cov / (inter_std * sim_std)).item()

        # Spearman R (需要 scipy)
        try:
            from scipy.stats import spearmanr

            sr, _ = spearmanr(all_inter.numpy(), all_sim.numpy())
            result["alignment_spearman_r"] = float(sr)
        except (ImportError, ValueError):
            pass

        self.reset()
        return result

    def reset(self):
        self._step_intersections.clear()
        self._step_similarities.clear()
