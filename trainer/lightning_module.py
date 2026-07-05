#!/usr/bin/env python3
"""Lightning 模型封装"""

import torch
from typing import Dict, List, Tuple, Any
import lightning as L

from datasets import DualGraphData
from datasets.data_types import ContrastivePairBatch
from models.data_types import ModelConfig, ModelOutput
from models.model import create_model
from models.hyperrectangle import hyperrectangle_intersection
from metrics import (
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
        graph_loss_weight: float = 1.0,
        warmup_steps: int = 100,
        scheduler_type: str = "cosine",
        contrastive_loss_weight: float = 0.0,
        contrastive_loss_type: str = "mse",
        contrastive_margin: float = 0.2,
        coverage_target_keys: tuple = ("branch",),
        joint_contrastive: bool = False,
        lambda_ce: float = 1.0,
        lambda_cl: float = 0.5,
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
        self.graph_loss_weight = graph_loss_weight
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

        # 指标计算器
        self.regression_metrics = CoverageRegressionMetrics()
        self.contrastive_metrics = ContrastiveMetrics()

        # 验证/测试指标累积
        self._val_outputs: List[Dict] = []
        self._test_outputs: List[Dict] = []

    def forward(self, data: DualGraphData) -> ModelOutput:
        """前向传播"""
        return self.model(data)

    def _shared_step(
        self, batch: DualGraphData, stage: str, lite: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """共享的训练/验证/测试步骤"""
        output = self(batch)
        losses = self.model.compute_loss(
            output,
            batch,
            graph_loss_weight=self.graph_loss_weight,
        )
        supervised_loss = losses["graph_loss"]

        # 损失指标
        metrics = {
            f"{stage}/total_loss": supervised_loss.detach(),
            f"{stage}/graph_loss": losses["graph_loss"].detach(),
        }

        # 图回归指标（选列，指标内部按覆盖率列独立过滤 NaN）
        col_idx = [self._coverage_key_index(k) for k in self.coverage_target_keys]
        y_target = batch.y[:, col_idx] if batch.y is not None else None
        graph_m = self.regression_metrics.compute(
            output.graph_pred if y_target is not None else None,
            y_target,
            lite=lite,
            coverage_keys=self.coverage_target_keys,
        )
        for k, v in graph_m.items():
            metrics[f"{stage}/{k}"] = v

        return supervised_loss, metrics

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
                "hp/val_graph_mse": 0.0,
            }
            try:
                self.logger.log_hyperparams(self.hparams, metrics_to_track)
            except TypeError:
                self.logger.log_hyperparams(self.hparams)

    def on_validation_epoch_end(self):
        """验证 epoch 结束：计算 epoch 级 AUC/Spearman 等指标"""
        val_loss = self.trainer.callback_metrics.get("val/total_loss")
        val_graph_mse = self.trainer.callback_metrics.get("val/graph_mse")

        if val_loss is not None:
            self.log("hp/val_loss", val_loss, sync_dist=True)
        if val_graph_mse is not None:
            self.log("hp/val_graph_mse", val_graph_mse, sync_dist=True)

        self._val_outputs.clear()

    def on_test_epoch_end(self):
        """测试 epoch 结束：计算 epoch 级指标"""
        self._test_outputs.clear()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.joint_contrastive and isinstance(batch, ContrastivePairBatch):
            return self._pair_training_step(batch, batch_idx)
        loss, metrics = self._shared_step(batch, "train", lite=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

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

    def _pair_training_step(
        self, batch: ContrastivePairBatch, batch_idx: int
    ) -> torch.Tensor:
        """覆盖向量 pair 对比训练步骤。"""
        from .joint_steps import run_pair_training_step

        return run_pair_training_step(self, batch)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        from .optim import configure_adamw_with_scheduler

        return configure_adamw_with_scheduler(
            self,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            scheduler_type=self.scheduler_type,
            warmup_steps=self.warmup_steps,
        )
