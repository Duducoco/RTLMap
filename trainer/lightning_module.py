#!/usr/bin/env python3
"""Lightning 模型封装"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import lightning as L

from datasets import DualGraphData
from datasets.data_types import ContrastivePairBatch
from models.data_types import ModelConfig, ModelOutput
from models.model import create_model
from models.contrastive_loss import compute_contrastive_loss
from models.hyperrectangle import hyperrectangle_intersection
from metrics import EdgeClassificationMetrics, CoverageRegressionMetrics, ContrastiveMetrics


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
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_margin = contrastive_margin
        self.use_hyperrectangle = model_config.use_hyperrectangle

        # 指标计算器
        self.edge_metrics = EdgeClassificationMetrics()
        self.regression_metrics = CoverageRegressionMetrics()
        self.contrastive_metrics = ContrastiveMetrics()

        # epoch 级指标收集器（用于 AUC-ROC/AUPRC/Spearman R）
        self._val_edge_metrics = EdgeClassificationMetrics()
        self._test_edge_metrics = EdgeClassificationMetrics()
        self._val_contrastive_metrics = ContrastiveMetrics()
        self._test_contrastive_metrics = ContrastiveMetrics()

        # 对比 DataLoader（外部注入）
        self._contrastive_dataloader = None
        self._contrastive_iter = None

        # 验证/测试指标累积
        self._val_outputs: List[Dict] = []
        self._test_outputs: List[Dict] = []

    def forward(self, data: DualGraphData) -> ModelOutput:
        """前向传播"""
        return self.model(data)

    def set_contrastive_dataloader(self, dataloader):
        """注入对比学习 DataLoader"""
        self._contrastive_dataloader = dataloader
        self._contrastive_iter = None

    def _get_next_contrastive_batch(self) -> ContrastivePairBatch:
        """从对比 DataLoader 取下一批"""
        if self._contrastive_iter is None:
            self._contrastive_iter = iter(self._contrastive_dataloader)
        try:
            return next(self._contrastive_iter)
        except StopIteration:
            self._contrastive_iter = iter(self._contrastive_dataloader)
            return next(self._contrastive_iter)

    def set_regression_baseline(self, targets: torch.Tensor):
        """设置回归指标的 MASE 朴素基准（用训练集 y 均值）"""
        self.regression_metrics.update_naive_baseline(targets)

    def _shared_step(
        self, batch: DualGraphData, stage: str, lite: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """共享的训练/验证/测试步骤"""
        output = self(batch)
        losses = self.model.compute_loss(
            output, batch,
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

        # 图回归指标
        graph_m = self.regression_metrics.compute(
            output.graph_pred, batch.y, lite=lite
        )
        for k, v in graph_m.items():
            metrics[f"{stage}/{k}"] = v

        # epoch 级指标收集（验证/测试）
        if not lite and valid_mask is not None and valid_mask.any():
            collector = self._val_edge_metrics if stage == "val" else self._test_edge_metrics
            collector.collect_step(output.edge_logits, edge_labels)

        return losses["total_loss"], metrics

    def _compute_contrastive_step_metrics(
        self, output_a: ModelOutput, output_b: ModelOutput, similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算对比学习 step 级指标"""
        intersection = hyperrectangle_intersection(
            output_a.hyper_min, output_a.hyper_max,
            output_b.hyper_min, output_b.hyper_max,
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

        # epoch 级对比学习指标
        if self.contrastive_loss_weight > 0:
            cl_metrics = self._val_contrastive_metrics.compute_epoch()
            self.log("val/alignment_pearson_r", cl_metrics["alignment_pearson_r"], sync_dist=False)
            self.log("val/alignment_spearman_r", cl_metrics["alignment_spearman_r"], sync_dist=False)

        self._val_outputs.clear()

    def on_test_epoch_end(self):
        """测试 epoch 结束：计算 epoch 级指标"""
        auc_metrics = self._test_edge_metrics.compute_epoch()
        self.log("test/auc_roc", auc_metrics["auc_roc"], sync_dist=False)
        self.log("test/auprc", auc_metrics["auprc"], sync_dist=False)

        if self.contrastive_loss_weight > 0:
            cl_metrics = self._test_contrastive_metrics.compute_epoch()
            self.log("test/alignment_pearson_r", cl_metrics["alignment_pearson_r"], sync_dist=False)
            self.log("test/alignment_spearman_r", cl_metrics["alignment_spearman_r"], sync_dist=False)

        self._test_outputs.clear()

    def training_step(self, batch: DualGraphData, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        loss, metrics = self._shared_step(batch, "train", lite=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        # 对比损失
        if self.contrastive_loss_weight > 0 and self._contrastive_dataloader is not None:
            pair_batch = self._get_next_contrastive_batch()
            pair_batch.batch_a = pair_batch.batch_a.to(self.device)
            pair_batch.batch_b = pair_batch.batch_b.to(self.device)
            pair_batch.similarity = pair_batch.similarity.to(self.device)
            output_a = self(pair_batch.batch_a)
            output_b = self(pair_batch.batch_b)
            cl = compute_contrastive_loss(
                output_a, output_b, pair_batch.similarity,
                loss_type=self.contrastive_loss_type,
                margin=self.contrastive_margin,
            )

            # 对比 step 级指标
            c_step = self._compute_contrastive_step_metrics(output_a, output_b, pair_batch.similarity)
            self.log_dict(c_step, on_step=True, on_epoch=True, prog_bar=False, batch_size=1)

            total = loss + self.contrastive_loss_weight * cl
            self.log("train/contrastive_loss", cl, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
            return total

        return loss

    def validation_step(self, batch: DualGraphData, batch_idx: int):
        """验证步骤"""
        loss, metrics = self._shared_step(batch, "val")
        self._val_outputs.append(metrics)
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, sync_dist=True
        )

    def test_step(self, batch: DualGraphData, batch_idx: int):
        """测试步骤"""
        loss, metrics = self._shared_step(batch, "test")
        self._test_outputs.append(metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

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