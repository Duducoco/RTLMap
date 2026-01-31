#!/usr/bin/env python3
"""Lightning 模型封装"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import lightning as L

from datasets import DualGraphData
from models.data_types import ModelConfig, ModelOutput
from models.model import create_model


class DualGraphLightningModule(L.LightningModule):
    """双图融合模型的 Lightning 封装"""

    def __init__(
        self,
        model_config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        edge_loss_weight: float = 1.0,
        graph_loss_weight: float = 1.0,
        label_smoothing: float = 0.1,
        warmup_steps: int = 100,
        scheduler_type: str = "cosine",
    ):
        super().__init__()
        # 保存超参数（包括 ModelConfig 的所有字段）
        self.save_hyperparameters(ignore=["model_config"])
        # 手动保存 ModelConfig 的字段到 hparams
        self.hparams.update(
            {
                "hidden_dim": model_config.hidden_dim,
                "num_gnn_layers": model_config.num_gnn_layers,
                "dropout": model_config.dropout,
                "num_edge_classes": model_config.num_edge_classes,
                "num_graph_targets": model_config.num_graph_targets,
                "num_cell_types": model_config.num_cell_types,
                "num_edge_types": model_config.num_edge_types,
                "num_asm_node_types": model_config.num_asm_node_types,
                "num_asm_edge_types": model_config.num_asm_edge_types,
                "asm_instruction_dim": model_config.asm_instruction_dim,
            }
        )

        # 模型
        self.model = create_model(model_config)

        # 超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.edge_loss_weight = edge_loss_weight
        self.graph_loss_weight = graph_loss_weight
        self.label_smoothing = label_smoothing
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type

        # 验证/测试指标累积
        self._val_outputs: List[Dict] = []
        self._test_outputs: List[Dict] = []

    def forward(self, data: DualGraphData) -> ModelOutput:
        """前向传播"""
        return self.model(data)

    def _shared_step(
        self, batch: DualGraphData, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """共享的训练/验证/测试步骤"""
        # 前向传播
        output = self(batch)

        # 计算损失
        losses = self.model.compute_loss(
            output,
            batch,
            edge_loss_weight=self.edge_loss_weight,
            graph_loss_weight=self.graph_loss_weight,
            label_smoothing=self.label_smoothing,
        )

        # 计算指标
        metrics = self._compute_metrics(output, batch, losses, stage)

        return losses["total_loss"], metrics

    def _compute_metrics(
        self,
        output: ModelOutput,
        data: DualGraphData,
        losses: Dict[str, torch.Tensor],
        stage: str,
    ) -> Dict[str, Any]:
        """计算并记录指标"""
        metrics = {
            f"{stage}/total_loss": losses["total_loss"].detach(),
            f"{stage}/edge_loss": losses["edge_loss"].detach(),
            f"{stage}/graph_loss": losses["graph_loss"].detach(),
        }

        # 边分类指标（仅对有效边）
        edge_labels = getattr(data, "edge_labels", None)
        if edge_labels is not None:
            valid_mask = edge_labels != -1
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                valid_logits = output.edge_logits[valid_mask]
                valid_labels = edge_labels[valid_mask]
                valid_preds = valid_logits.argmax(dim=-1)

                # 准确率
                accuracy = (valid_preds == valid_labels).float().mean()
                metrics[f"{stage}/edge_accuracy"] = accuracy

                # 统计预测分布
                num_pred_pos = (valid_preds == 1).sum().item()
                num_pred_neg = (valid_preds == 0).sum().item()
                metrics[f"{stage}/num_valid_edges"] = num_valid
                metrics[f"{stage}/num_pred_covered"] = num_pred_pos
                metrics[f"{stage}/num_pred_uncovered"] = num_pred_neg

                # Precision/Recall/F1（以 covered=1 为正类）
                tp = ((valid_preds == 1) & (valid_labels == 1)).sum().float()
                fp = ((valid_preds == 1) & (valid_labels == 0)).sum().float()
                fn = ((valid_preds == 0) & (valid_labels == 1)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                metrics[f"{stage}/precision"] = precision
                metrics[f"{stage}/recall"] = recall
                metrics[f"{stage}/f1"] = f1

        # 图回归指标
        if data.y is not None and output.graph_pred is not None:
            pred = output.graph_pred
            target = data.y

            # MSE、RMSE 和 MAE
            mse = F.mse_loss(pred, target)
            rmse = mse.sqrt()
            mae = F.l1_loss(pred, target)

            metrics[f"{stage}/graph_mse"] = mse
            metrics[f"{stage}/graph_rmse"] = rmse
            metrics[f"{stage}/graph_mae"] = mae

            # MAPE (Mean Absolute Percentage Error)
            # 避免除零：仅对 target != 0 的样本计算
            non_zero_mask = target.abs() > 1e-8
            if non_zero_mask.any():
                mape = (
                    (pred[non_zero_mask] - target[non_zero_mask]).abs()
                    / target[non_zero_mask].abs()
                ).mean() * 100  # 百分比形式
                metrics[f"{stage}/graph_mape"] = mape

            # Pearson 相关系数 R
            if pred.numel() > 1:
                pred_flat = pred.view(-1)
                target_flat = target.view(-1)
                pred_mean = pred_flat.mean()
                target_mean = target_flat.mean()
                pred_centered = pred_flat - pred_mean
                target_centered = target_flat - target_mean
                cov = (pred_centered * target_centered).sum()
                pred_std = pred_centered.pow(2).sum().sqrt()
                target_std = target_centered.pow(2).sum().sqrt()
                r = cov / (pred_std * target_std + 1e-8)
                metrics[f"{stage}/graph_r"] = r

        return metrics

    def on_fit_start(self):
        """训练开始时记录超参数（用于 TensorBoard HPARAMS 面板）"""
        if self.logger is not None:
            # 定义要跟踪的指标（用于超参数对比）
            metrics_to_track = {
                "hp/val_loss": 0.0,
                "hp/val_accuracy": 0.0,
                "hp/val_f1": 0.0,
            }
            # 记录超参数与指标关联（仅 TensorBoard 支持）
            try:
                self.logger.log_hyperparams(self.hparams, metrics_to_track)
            except TypeError:
                # CSVLogger 等不支持 metrics 参数
                self.logger.log_hyperparams(self.hparams)

    def on_validation_epoch_end(self):
        """验证 epoch 结束时更新超参数指标"""
        # 获取当前 epoch 的验证指标
        val_loss = self.trainer.callback_metrics.get("val/total_loss")
        val_acc = self.trainer.callback_metrics.get("val/edge_accuracy")
        val_f1 = self.trainer.callback_metrics.get("val/f1")

        # 记录到 hp/ 命名空间（用于 HPARAMS 面板对比）
        if val_loss is not None:
            self.log("hp/val_loss", val_loss, sync_dist=True)
        if val_acc is not None:
            self.log("hp/val_accuracy", val_acc, sync_dist=True)
        if val_f1 is not None:
            self.log("hp/val_f1", val_f1, sync_dist=True)

        self._val_outputs.clear()

    def training_step(self, batch: DualGraphData, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        loss, metrics = self._shared_step(batch, "train")

        # 记录指标
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        return loss

    def validation_step(self, batch: DualGraphData, batch_idx: int):
        """验证步骤"""
        loss, metrics = self._shared_step(batch, "val")
        self._val_outputs.append(metrics)

        # 记录指标
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=1
        )

    def test_step(self, batch: DualGraphData, batch_idx: int):
        """测试步骤"""
        loss, metrics = self._shared_step(batch, "test")
        self._test_outputs.append(metrics)

        # 记录指标
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=1)

    def on_test_epoch_end(self):
        """测试 epoch 结束"""
        self._test_outputs.clear()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # AdamW 优化器
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.scheduler_type == "none":
            return optimizer

        # 计算总步数
        if self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            # 估算总步数
            if hasattr(self.trainer, "estimated_stepping_batches"):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000  # 默认值

        # Warmup + 主调度器
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # 线性 warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine 或 Linear 衰减
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, total_steps - self.warmup_steps)
                )
                if self.scheduler_type == "cosine":
                    import math

                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                elif self.scheduler_type == "linear":
                    return max(0.0, 1.0 - progress)
                else:
                    return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
