#!/usr/bin/env python3
"""覆盖率回归指标 — 图级覆盖率百分比预测评估"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


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
        pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        lite: bool = False,
        coverage_keys: Tuple[str, ...] = ("branch",),
    ) -> Dict[str, torch.Tensor]:
        """计算覆盖率回归指标

        Args:
            pred: [B, K] 预测覆盖率（None 时返回零指标）
            target: [B, K] 真实覆盖率（None 时返回零指标，NaN 行应在调用前过滤）
            lite: True 时仅计算核心指标
            coverage_keys: 与 pred/target 列对应的覆盖率类型名（用于多列时的指标后缀）
        """
        use_suffix = len(coverage_keys) > 1

        def _suffix(key: str) -> str:
            return f"_{key}" if use_suffix else ""

        # 单列时保持原接口不变，多列时按列分别计算后合并
        if pred is None or target is None or pred.numel() == 0:
            result: Dict[str, torch.Tensor] = {}
            device = pred.device if pred is not None else torch.device("cpu")
            _zero = torch.tensor(0.0, device=device)
            for key in coverage_keys:
                sfx = _suffix(key)
                result[f"graph_mse{sfx}"] = _zero
                result[f"graph_mae{sfx}"] = _zero
                if not lite:
                    result.update({
                        f"graph_rmse{sfx}": _zero,
                        f"graph_mape{sfx}": _zero,
                        f"graph_smape{sfx}": _zero,
                        f"graph_medae{sfx}": _zero,
                        f"graph_r2{sfx}": _zero,
                        f"graph_r{sfx}": _zero,
                        f"graph_mase{sfx}": _zero,
                    })
            return result

        result = {}
        for col, key in enumerate(coverage_keys):
            sfx = _suffix(key)
            p = pred[:, col] if pred.dim() == 2 else pred.view(-1)
            t = target[:, col] if target.dim() == 2 else target.view(-1)
            result.update(self._compute_single(p, t, lite=lite, suffix=sfx))
        return result

    def _compute_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lite: bool,
        suffix: str,
    ) -> Dict[str, torch.Tensor]:
        device = pred.device
        _zero = torch.tensor(0.0, device=device)

        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        result = {
            f"graph_mse{suffix}": mse,
            f"graph_mae{suffix}": mae,
        }

        if lite:
            return result

        rmse = mse.sqrt()

        smape = 2.0 * (pred - target).abs() / (pred.abs() + target.abs() + 1e-8)
        smape = smape.mean() * 100

        errors = (pred - target).abs().view(-1)
        medae = errors.median()

        non_zero_mask = target.abs() > 1e-8
        if non_zero_mask.any():
            mape = (
                (pred[non_zero_mask] - target[non_zero_mask]).abs()
                / target[non_zero_mask].abs()
            ).mean() * 100
        else:
            mape = _zero

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        ss_res = (pred_flat - target_flat).pow(2).sum()
        ss_tot = (target_flat - target_flat.mean()).pow(2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        if pred.numel() > 1:
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            cov = (pred_centered * target_centered).sum()
            pred_std = pred_centered.pow(2).sum().sqrt()
            target_std = target_centered.pow(2).sum().sqrt()
            r = cov / (pred_std * target_std + 1e-8)
        else:
            r = _zero

        if self._naive_mae is not None and self._naive_mae > 1e-8:
            mase = mae / self._naive_mae
        else:
            mase = mae

        result.update({
            f"graph_rmse{suffix}": rmse,
            f"graph_mape{suffix}": mape,
            f"graph_smape{suffix}": smape,
            f"graph_medae{suffix}": medae,
            f"graph_r2{suffix}": r2,
            f"graph_r{suffix}": r,
            f"graph_mase{suffix}": mase,
        })

        return result