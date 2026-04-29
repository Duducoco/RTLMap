#!/usr/bin/env python3
"""Lightning 模型封装"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import lightning as L

from datasets import DualGraphData
from datasets.data_types import ContrastiveTripleBatch
from models.data_types import ModelConfig, ModelOutput
from models.model import create_model
from models.contrastive_loss import compute_contrastive_loss
from models.hyperrectangle import hyperrectangle_intersection
from metrics import (
    EdgeClassificationMetrics,
    CoverageRegressionMetrics,
    ContrastiveMetrics,
)


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
        contrastive_loss_weight: float = 0.0,
        contrastive_loss_type: str = "mse",
        contrastive_margin: float = 0.2,
        coverage_target_keys: tuple = ("branch",),
        joint_contrastive: bool = False,
        lambda_ce: float = 1.0,
        lambda_cl: float = 0.5,
        or_consistency_weight: float = 0.0,
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
                "coverage_target_keys": coverage_target_keys,
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
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_margin = contrastive_margin
        self.use_hyperrectangle = model_config.use_hyperrectangle
        self.coverage_target_keys = coverage_target_keys
        from datasets.data_types import coverage_key_index

        self._coverage_key_index = coverage_key_index
        self.joint_contrastive = joint_contrastive
        self.lambda_ce = lambda_ce
        self.lambda_cl = lambda_cl
        self.or_consistency_weight = or_consistency_weight

        # 指标计算器
        self.edge_metrics = EdgeClassificationMetrics()
        self.regression_metrics = CoverageRegressionMetrics()
        self.contrastive_metrics = ContrastiveMetrics()

        # epoch 级指标收集器（用于 AUC-ROC/AUPRC/Spearman R）
        self._val_edge_metrics = EdgeClassificationMetrics()
        self._test_edge_metrics = EdgeClassificationMetrics()

        # 验证/测试指标累积
        self._val_outputs: List[Dict] = []
        self._test_outputs: List[Dict] = []

    def forward(self, data: DualGraphData) -> ModelOutput:
        """前向传播"""
        return self.model(data)

    def set_regression_baseline(self, targets: torch.Tensor):
        """设置回归指标的 MASE 朴素基准（用训练集 y 均值）"""
        self.regression_metrics.update_naive_baseline(targets)

    def _shared_step(
        self, batch: DualGraphData, stage: str, lite: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """共享的训练/验证/测试步骤"""
        output = self(batch)
        losses = self.model.compute_loss(
            output,
            batch,
            edge_loss_weight=self.edge_loss_weight,
            graph_loss_weight=self.graph_loss_weight,
            label_smoothing=self.label_smoothing,
        )

        # 损失指标
        metrics = {
            f"{stage}/total_loss": losses["total_loss"].detach(),
            f"{stage}/edge_loss": losses["edge_loss"].detach(),
            f"{stage}/graph_loss": losses["graph_loss"].detach(),
        }

        # 边分类指标
        edge_labels = getattr(batch, "edge_labels", None)
        valid_mask = edge_labels != -1 if edge_labels is not None else None
        edge_m = self.edge_metrics.compute(
            output.edge_logits, edge_labels, valid_mask, lite=lite
        )
        for k, v in edge_m.items():
            metrics[f"{stage}/{k}"] = v

        # 图回归指标（选列 + NaN 行过滤）
        col_idx = [self._coverage_key_index(k) for k in self.coverage_target_keys]
        y_target = batch.y[:, col_idx] if batch.y is not None else None
        valid_rows = (
            ~torch.isnan(y_target).any(dim=-1) if y_target is not None else None
        )
        pred_valid = (
            output.graph_pred[valid_rows]
            if valid_rows is not None and valid_rows.any()
            else None
        )
        target_valid = (
            y_target[valid_rows]
            if valid_rows is not None and valid_rows.any()
            else None
        )
        graph_m = self.regression_metrics.compute(
            pred_valid,
            target_valid,
            lite=lite,
            coverage_keys=self.coverage_target_keys,
        )
        for k, v in graph_m.items():
            metrics[f"{stage}/{k}"] = v

        # epoch 级指标收集（验证/测试）
        if not lite and valid_mask is not None and valid_mask.any():
            collector = (
                self._val_edge_metrics if stage == "val" else self._test_edge_metrics
            )
            collector.collect_step(output.edge_logits, edge_labels)

        return losses["total_loss"], metrics

    def _compute_contrastive_step_metrics(
        self, output_a: ModelOutput, output_b: ModelOutput, similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算对比学习 step 级指标"""
        intersection = hyperrectangle_intersection(
            output_a.hyper_min,
            output_a.hyper_max,
            output_b.hyper_min,
            output_b.hyper_max,
        )
        c_metrics = self.contrastive_metrics.compute(intersection, similarity)
        return {f"train/contrastive_{k}": v for k, v in c_metrics.items()}

    def on_fit_start(self):
        """训练开始时记录超参数"""
        if self.logger is not None:
            metrics_to_track = {
                "hp/val_loss": 0.0,
                "hp/val_accuracy": 0.0,
                "hp/val_f1": 0.0,
            }
            try:
                self.logger.log_hyperparams(self.hparams, metrics_to_track)
            except TypeError:
                self.logger.log_hyperparams(self.hparams)

    def on_validation_epoch_end(self):
        """验证 epoch 结束：计算 epoch 级 AUC/Spearman 等指标"""
        val_loss = self.trainer.callback_metrics.get("val/total_loss")
        val_acc = self.trainer.callback_metrics.get("val/edge_accuracy")
        val_f1 = self.trainer.callback_metrics.get("val/f1")

        if val_loss is not None:
            self.log("hp/val_loss", val_loss, sync_dist=True)
        if val_acc is not None:
            self.log("hp/val_accuracy", val_acc, sync_dist=True)
        if val_f1 is not None:
            self.log("hp/val_f1", val_f1, sync_dist=True)

        # epoch 级 AUC-ROC / AUPRC
        auc_metrics = self._val_edge_metrics.compute_epoch()
        self.log("val/auc_roc", auc_metrics["auc_roc"], sync_dist=False)
        self.log("val/auprc", auc_metrics["auprc"], sync_dist=False)

        self._val_outputs.clear()

    def on_test_epoch_end(self):
        """测试 epoch 结束：计算 epoch 级指标"""
        auc_metrics = self._test_edge_metrics.compute_epoch()
        self.log("test/auc_roc", auc_metrics["auc_roc"], sync_dist=False)
        self.log("test/auprc", auc_metrics["auprc"], sync_dist=False)

        self._test_outputs.clear()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.joint_contrastive and isinstance(batch, ContrastiveTripleBatch):
            return self._joint_training_step(batch, batch_idx)
        loss, metrics = self._shared_step(batch, "train", lite=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def _joint_training_step(
        self, batch: ContrastiveTripleBatch, batch_idx: int
    ) -> torch.Tensor:
        """联合三元组训练步骤：单次 training_step 内完成三路覆盖率预测 + 对比学习。

        L_total = lambda_ce * (L_ce_a + L_ce_b + L_ce_merged) + lambda_cl * L_contrastive
                  [+ or_consistency_weight * L_or]
        """
        batch.batch_a = batch.batch_a.to(self.device)
        batch.batch_b = batch.batch_b.to(self.device)
        batch.batch_merged = batch.batch_merged.to(self.device)
        batch.similarity = batch.similarity.to(self.device)

        # 三次 forward（共享编码器参数）
        out_a = self(batch.batch_a)
        out_b = self(batch.batch_b)
        out_m = self(batch.batch_merged)

        # 三路边级 CE 损失
        losses_a = self.model.compute_loss(
            out_a,
            batch.batch_a,
            edge_loss_weight=self.edge_loss_weight,
            graph_loss_weight=self.graph_loss_weight,
            label_smoothing=self.label_smoothing,
        )
        losses_b = self.model.compute_loss(
            out_b,
            batch.batch_b,
            edge_loss_weight=self.edge_loss_weight,
            graph_loss_weight=self.graph_loss_weight,
            label_smoothing=self.label_smoothing,
        )
        losses_m = self.model.compute_loss(
            out_m,
            batch.batch_merged,
            edge_loss_weight=self.edge_loss_weight,
            graph_loss_weight=0.0,  # merged 无图回归标签，只算边级
            label_smoothing=self.label_smoothing,
        )

        l_ce = losses_a["total_loss"] + losses_b["total_loss"] + losses_m["total_loss"]

        # 对比损失（超矩形交集对齐 Jaccard 相似度）
        l_cl = compute_contrastive_loss(
            out_a,
            out_b,
            batch.similarity,
            loss_type=self.contrastive_loss_type,
            margin=self.contrastive_margin,
        )

        total = self.lambda_ce * l_ce + self.lambda_cl * l_cl

        # 可选：merged 逻辑 OR 一致性约束
        l_or = torch.tensor(0.0, device=self.device)
        if self.or_consistency_weight > 0:
            # 仅在三路边数完全对齐时启用 OR 一致性约束
            if out_a.edge_logits.size(0) == out_b.edge_logits.size(
                0
            ) and out_a.edge_logits.size(0) == out_m.edge_logits.size(0):
                prob_a = torch.softmax(out_a.edge_logits, dim=-1)[:, 1]
                prob_b = torch.softmax(out_b.edge_logits, dim=-1)[:, 1]
                prob_or = 1.0 - (1.0 - prob_a) * (1.0 - prob_b)
                merged_labels = getattr(batch.batch_merged, "edge_labels", None)
                if merged_labels is not None:
                    valid = merged_labels != -1
                    if valid.any():
                        l_or = F.binary_cross_entropy(
                            prob_or[valid].clamp(1e-6, 1 - 1e-6),
                            merged_labels[valid].float(),
                        )
                        total = total + self.or_consistency_weight * l_or

        # 日志
        self.log_dict(
            {
                "train/edge_loss_a": losses_a["edge_loss"].detach(),
                "train/edge_loss_b": losses_b["edge_loss"].detach(),
                "train/edge_loss_merged": losses_m["edge_loss"].detach(),
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

        # 对比 step 指标
        c_step = self._compute_contrastive_step_metrics(out_a, out_b, batch.similarity)
        self.log_dict(c_step, on_step=True, on_epoch=True, prog_bar=False, batch_size=1)

        return total

    def validation_step(self, batch: DualGraphData, batch_idx: int):
        """验证步骤"""
        loss, metrics = self._shared_step(batch, "val")
        self._val_outputs.append(metrics)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, batch: DualGraphData, batch_idx: int):
        """测试步骤"""
        loss, metrics = self._shared_step(batch, "test")
        self._test_outputs.append(metrics)
        self.log_dict(
            metrics, on_step=False, on_epoch=True, batch_size=1, sync_dist=True
        )

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.scheduler_type == "none":
            return optimizer

        if self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            if hasattr(self.trainer, "estimated_stepping_batches"):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000

        effective_warmup = min(self.warmup_steps, int(total_steps * 0.1))

        def lr_lambda(current_step: int) -> float:
            if current_step < effective_warmup:
                return float(current_step) / float(max(1, effective_warmup))
            else:
                progress = float(current_step - effective_warmup) / float(
                    max(1, total_steps - effective_warmup)
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
