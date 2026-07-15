#!/usr/bin/env python3
"""覆盖率回归指标 — 图级覆盖率百分比预测评估"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _gather_rows(rows: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Gather variable-length metric rows from every DDP rank."""
    rows = rows.to(device=device, dtype=torch.float32)
    if not dist.is_available() or not dist.is_initialized():
        return rows.cpu()

    world_size = dist.get_world_size()
    local_size = torch.tensor([rows.shape[0]], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    max_size = max(1, max(int(size.item()) for size in sizes))
    padded = torch.zeros((max_size, rows.shape[1]), dtype=rows.dtype, device=device)
    padded[: rows.shape[0]] = rows
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [
            rank_rows[: int(rank_size.item())]
            for rank_rows, rank_size in zip(gathered, sizes, strict=True)
        ],
        dim=0,
    ).cpu()


class CoverageRegressionMetrics:
    """覆盖率回归指标计算器

    支持 lite 模式（训练 step 级）和 full 模式（验证/测试 step 级）。

    """

    def __init__(self) -> None:
        self._epoch_predictions: list[torch.Tensor] = []
        self._epoch_targets: list[torch.Tensor] = []

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
            target: [B, K] 真实覆盖率（None 时返回零指标，NaN 值会按列过滤）
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
                    result.update(
                        {
                            f"graph_rmse{sfx}": _zero,
                            f"graph_mape{sfx}": _zero,
                            f"graph_smape{sfx}": _zero,
                            f"graph_medae{sfx}": _zero,
                            f"graph_r2{sfx}": _zero,
                            f"graph_r{sfx}": _zero,
                        }
                    )
            return result

        result = {}
        for col, key in enumerate(coverage_keys):
            sfx = _suffix(key)
            p = pred[:, col] if pred.dim() == 2 else pred.view(-1)
            t = target[:, col] if target.dim() == 2 else target.view(-1)
            valid = ~torch.isnan(t)
            if valid.any():
                result.update(
                    self._compute_single(p[valid], t[valid], lite=lite, suffix=sfx)
                )
            else:
                result.update(
                    self._zero_single(device=pred.device, lite=lite, suffix=sfx)
                )
        return result

    def collect_epoch(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Buffer validation predictions for global epoch-level metrics."""
        self._epoch_predictions.append(pred.detach().float().cpu())
        self._epoch_targets.append(target.detach().float().cpu())

    def compute_epoch(
        self, *, device: torch.device, coverage_keys: Tuple[str, ...]
    ) -> Dict[str, torch.Tensor]:
        """Compute full-dataset validation metrics, including global R²."""
        if (
            not self._epoch_predictions
            and (not dist.is_available() or not dist.is_initialized())
        ):
            return {}

        if self._epoch_predictions:
            predictions = torch.cat(self._epoch_predictions, dim=0)
            targets = torch.cat(self._epoch_targets, dim=0)
        else:
            predictions = torch.empty((0, len(coverage_keys)), dtype=torch.float32)
            targets = torch.empty((0, len(coverage_keys)), dtype=torch.float32)
        try:
            predictions = _gather_rows(predictions, device=device)
            targets = _gather_rows(targets, device=device)
            return self.compute(
                predictions,
                targets,
                lite=False,
                coverage_keys=coverage_keys,
            )
        finally:
            self.reset_epoch()

    def reset_epoch(self) -> None:
        self._epoch_predictions.clear()
        self._epoch_targets.clear()

    def _zero_single(
        self,
        device: torch.device,
        lite: bool,
        suffix: str,
    ) -> Dict[str, torch.Tensor]:
        _zero = torch.tensor(0.0, device=device)
        result = {
            f"graph_mse{suffix}": _zero,
            f"graph_mae{suffix}": _zero,
        }
        if not lite:
            result.update(
                {
                    f"graph_rmse{suffix}": _zero,
                    f"graph_mape{suffix}": _zero,
                    f"graph_smape{suffix}": _zero,
                    f"graph_medae{suffix}": _zero,
                    f"graph_r2{suffix}": _zero,
                    f"graph_r{suffix}": _zero,
                }
            )
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
        if ss_tot <= 1e-8:
            r2 = torch.ones((), device=device) if ss_res <= 1e-8 else _zero
        else:
            r2 = 1.0 - ss_res / ss_tot

        if pred.numel() > 1:
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            cov = (pred_centered * target_centered).sum()
            pred_std = pred_centered.pow(2).sum().sqrt()
            target_std = target_centered.pow(2).sum().sqrt()
            r = cov / (pred_std * target_std + 1e-8)
        else:
            r = _zero

        result.update(
            {
                f"graph_rmse{suffix}": rmse,
                f"graph_mape{suffix}": mape,
                f"graph_smape{suffix}": smape,
                f"graph_medae{suffix}": medae,
                f"graph_r2{suffix}": r2,
                f"graph_r{suffix}": r,
            }
        )

        return result
