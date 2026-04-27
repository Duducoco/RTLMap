#!/usr/bin/env python3
"""边分类指标 — 二分类（覆盖/未覆盖）评估"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


class EdgeClassificationMetrics:
    """边分类指标计算器

    支持 lite 模式（训练 step 级，保留 tensor 无 .item() 同步）
    和 full 模式（验证/测试 step 级，含完整指标）。

    AUC-ROC 和 AUPRC 需要在 epoch 级聚合后计算（batch 太小时数值不稳定），
    通过 collect_step → compute_epoch 两步完成。
    """

    def __init__(self):
        self._step_logits: list[torch.Tensor] = []
        self._step_labels: list[torch.Tensor] = []

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        lite: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """计算边分类指标

        Args:
            logits: [E, 2] 边分类 logits
            labels: [E] 边标签（-1/0/1）
            valid_mask: [E] 有效边掩码，None 时从 labels 推导
            lite: True 时仅计算核心指标（训练用）
        """
        if valid_mask is None:
            valid_mask = labels != -1 if labels is not None else None

        device = logits.device
        _zero = torch.tensor(0.0, device=device)

        if valid_mask is None or not valid_mask.any():
            result = {
                "edge_accuracy": _zero,
                "precision": _zero,
                "recall": _zero,
                "f1": _zero,
                "specificity": _zero,
                "balanced_accuracy": _zero,
                "tp": torch.tensor(0, device=device),
                "fp": torch.tensor(0, device=device),
                "fn": torch.tensor(0, device=device),
                "tn": torch.tensor(0, device=device),
            }
            if not lite:
                result["num_valid_edges"] = torch.tensor(0, device=device)
                result["num_pred_covered"] = torch.tensor(0, device=device)
                result["num_pred_uncovered"] = torch.tensor(0, device=device)
            return result

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        valid_preds = valid_logits.argmax(dim=-1)

        accuracy = (valid_preds == valid_labels).float().mean()

        tp = ((valid_preds == 1) & (valid_labels == 1)).sum().float()
        fp = ((valid_preds == 1) & (valid_labels == 0)).sum().float()
        fn = ((valid_preds == 0) & (valid_labels == 1)).sum().float()
        tn = ((valid_preds == 0) & (valid_labels == 0)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        balanced_accuracy = (recall + specificity) / 2.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        result = {
            "edge_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if not lite:
            result["num_valid_edges"] = valid_mask.sum()
            result["num_pred_covered"] = (valid_preds == 1).sum()
            result["num_pred_uncovered"] = (valid_preds == 0).sum()

        return result

    def collect_step(self, logits: torch.Tensor, labels: torch.Tensor):
        """收集 step 级 logits 和 labels，用于 epoch 级 AUC 计算"""
        valid_mask = labels != -1
        if valid_mask.any():
            self._step_logits.append(logits[valid_mask].detach().cpu())
            self._step_labels.append(labels[valid_mask].detach().cpu())

    def compute_epoch(self) -> Dict[str, float]:
        """epoch 级聚合计算 AUC-ROC 和 AUPRC

        Returns:
            dict with auc_roc and auprc (float)
        """
        result = {"auc_roc": 0.0, "auprc": 0.0}

        if not self._step_logits:
            return result

        all_logits = torch.cat(self._step_logits, dim=0)
        all_labels = torch.cat(self._step_labels, dim=0)

        # 至少需要两个类别才能计算 AUC
        unique_labels = all_labels.unique()
        if len(unique_labels) < 2:
            return result

        # 提取正类（covered=1）的概率
        probs = F.softmax(all_logits, dim=-1)[:, 1]

        probs_np = probs.numpy()
        labels_np = all_labels.numpy()

        from sklearn.metrics import roc_auc_score, average_precision_score

        try:
            result["auc_roc"] = roc_auc_score(labels_np, probs_np)
        except ValueError:
            pass

        try:
            result["auprc"] = average_precision_score(labels_np, probs_np)
        except ValueError:
            pass

        self.reset()
        return result

    def reset(self):
        self._step_logits.clear()
        self._step_labels.clear()