#!/usr/bin/env python3
"""覆盖率回归指标 — 图级覆盖率百分比预测评估"""

import torch
import torch.nn.functional as F
from typing import Dict


class CoverageRegressionMetrics:
    """覆盖率回归指标计算器

    支持 lite 模式（训练 step 级）和 full 模式（验证/测试 step 级）。

    MASE（Mean Absolute Scaled Error）需要训练集的 y 均值作为朴素基准。
    调用 update_naive_baseline() 在训练集扫描后设置基准。
    """

    def __init__(self):
        self._naive_mae: Optional[float] = None

    def update_naive_baseline(self, targets: torch.Tensor) -> None:
        """用训练集的 y 均值设置 MASE 朴素基准

        MASE = MAE / MAE_naive，其中 MAE_naive = mean(|y_i - y_mean|)
        """
        y_mean = targets.mean()
        mae_naive = (targets - y_mean).abs().mean().item()
        self._naive_mae = mae_naive if mae_naive > 1e-8 else None

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lite: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """计算覆盖率回归指标

        Args:
            pred: [B, 1] 预测覆盖率
            target: [B, 1] 真实覆盖率
            lite: True 时仅计算核心指标
        """
        device = pred.device
        _zero = torch.tensor(0.0, device=device)

        if target is None or pred is None:
            result = {
                "graph_mse": _zero,
                "graph_mae": _zero,
            }
            if not lite:
                result.update({
                    "graph_rmse": _zero,
                    "graph_mape": _zero,
                    "graph_smape": _zero,
                    "graph_medae": _zero,
                    "graph_r2": _zero,
                    "graph_r": _zero,
                    "graph_mase": _zero,
                })
            return result

        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        result = {
            "graph_mse": mse,
            "graph_mae": mae,
        }

        if lite:
            return result

        # full 模式指标
        rmse = mse.sqrt()

        # SMAPE: symmetric MAPE
        smape = 2.0 * (pred - target).abs() / (pred.abs() + target.abs() + 1e-8)
        smape = smape.mean() * 100

        # MedAE: median absolute error
        errors = (pred - target).abs().view(-1)
        medae = errors.median()

        # MAPE: 仅对 target ≠ 0 计算
        non_zero_mask = target.abs() > 1e-8
        if non_zero_mask.any():
            mape = (
                (pred[non_zero_mask] - target[non_zero_mask]).abs()
                / target[non_zero_mask].abs()
            ).mean() * 100
        else:
            mape = _zero

        # R² (决定系数)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        ss_res = (pred_flat - target_flat).pow(2).sum()
        ss_tot = (target_flat - target_flat.mean()).pow(2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        # Pearson R
        if pred.numel() > 1:
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            cov = (pred_centered * target_centered).sum()
            pred_std = pred_centered.pow(2).sum().sqrt()
            target_std = target_centered.pow(2).sum().sqrt()
            r = cov / (pred_std * target_std + 1e-8)
        else:
            r = _zero

        # MASE
        if self._naive_mae is not None and self._naive_mae > 1e-8:
            mase = mae / self._naive_mae
        else:
            mase = mae  # 无基准时回退为 MAE

        result.update({
            "graph_rmse": rmse,
            "graph_mape": mape,
            "graph_smape": smape,
            "graph_medae": medae,
            "graph_r2": r2,
            "graph_r": r,
            "graph_mase": mase,
        })

        return result