#!/usr/bin/env python3
"""Lightning 模型封装"""

import torch
from typing import Dict, List, Tuple, Any
import lightning as L

from datasets import DualGraphData
from datasets.data_types import ContrastivePairBatch
from models.data_types import ModelConfig, ModelOutput
from models.config_artifact import build_model_config_artifact
from models.model import create_model
from models.losses import (
    DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR,
    DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT,
)
from models.contrastive_loss import (
    DEFAULT_IOU_RANKING_CONFIG,
    IoURankingConfig,
)
from metrics import (
    CoverageRegressionMetrics,
    ContrastiveMetrics,
)


class DualGraphLightningModule(L.LightningModule):
    """双图融合模型的 Lightning 封装"""

    _GLOBAL_VALIDATION_METRIC_PREFIXES = (
        "val/graph_rmse",
        "val/graph_mape",
        "val/graph_smape",
        "val/graph_medae",
        "val/graph_r2",
        "val/graph_r",
    )

    def __init__(
        self,
        model_config: ModelConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        graph_loss_weight: float = 1.0,
        graph_relative_loss_weight: float = DEFAULT_GRAPH_RELATIVE_LOSS_WEIGHT,
        graph_relative_loss_floor: float = DEFAULT_GRAPH_RELATIVE_LOSS_FLOOR,
        warmup_steps: int = 100,
        scheduler_type: str = "cosine",
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        min_learning_rate: float = 1e-6,
        coverage_target_keys: tuple = ("branch",),
        joint_contrastive: bool = False,
        lambda_ce: float = 1.0,
        lambda_iou: float = 1.0,
        iou_rank_loss_weight: float = DEFAULT_IOU_RANKING_CONFIG.weight,
        iou_rank_margin: float = DEFAULT_IOU_RANKING_CONFIG.margin,
        iou_rank_min_target_gap: float = DEFAULT_IOU_RANKING_CONFIG.min_target_gap,
        lambda_volume: float = 1.0,
        volume_warmup_epochs: int = 5,
        smooth_intersection_temperature: float = 0.01,
        model_config_artifact: dict | None = None,
    ):
        super().__init__()
        model_config_artifact = model_config_artifact or build_model_config_artifact(
            model_config, None
        )
        # 保存超参数（包括 ModelConfig 的所有字段）
        self.save_hyperparameters(ignore=["model_config", "model_config_artifact"])
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
                "model_config_artifact": model_config_artifact,
            }
        )

        # 模型
        self.model = create_model(model_config)

        # 超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.graph_loss_weight = graph_loss_weight
        self.graph_relative_loss_weight = graph_relative_loss_weight
        self.graph_relative_loss_floor = graph_relative_loss_floor
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.min_learning_rate = min_learning_rate
        self.use_hyperrectangle = model_config.use_hyperrectangle
        self.coverage_target_keys = coverage_target_keys
        from datasets.data_types import coverage_key_index

        self._coverage_key_index = coverage_key_index
        self.joint_contrastive = joint_contrastive
        self.lambda_ce = lambda_ce
        self.lambda_iou = lambda_iou
        self.iou_ranking = IoURankingConfig(
            weight=iou_rank_loss_weight,
            margin=iou_rank_margin,
            min_target_gap=iou_rank_min_target_gap,
        )
        self.lambda_volume = lambda_volume
        self.volume_warmup_epochs = max(0, int(volume_warmup_epochs))
        self.smooth_intersection_temperature = smooth_intersection_temperature
        if self.smooth_intersection_temperature <= 0.0:
            raise ValueError("smooth_intersection_temperature must be positive")

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
            relative_loss_weight=self.graph_relative_loss_weight,
            relative_loss_floor=self.graph_relative_loss_floor,
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
        if stage == "val" and y_target is not None:
            self.regression_metrics.collect_epoch(output.graph_pred, y_target)
        for k, v in graph_m.items():
            metrics[f"{stage}/{k}"] = v

        return supervised_loss, metrics

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
        """验证 epoch 结束：计算整个验证集上的回归指标。"""
        val_loss = self.trainer.callback_metrics.get("val/total_loss")
        val_graph_mse = self.trainer.callback_metrics.get("val/graph_mse")

        if val_loss is not None:
            self.log("hp/val_loss", val_loss, sync_dist=True)
        if val_graph_mse is not None:
            self.log("hp/val_graph_mse", val_graph_mse, sync_dist=True)

        for key, value in self.regression_metrics.compute_epoch(
            device=self.device,
            coverage_keys=self.coverage_target_keys,
        ).items():
            metric_name = f"val/{key}"
            if not metric_name.startswith(self._GLOBAL_VALIDATION_METRIC_PREFIXES):
                continue
            self.log(
                metric_name,
                value,
                sync_dist=False,
                add_dataloader_idx=False,
            )

        if self.joint_contrastive:
            for key, value in self.contrastive_metrics.compute_epoch(
                device=self.device
            ).items():
                self.log(
                    f"val/pair_{key}",
                    value,
                    sync_dist=False,
                    add_dataloader_idx=False,
                )

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

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """验证步骤"""
        if self.joint_contrastive and isinstance(batch, ContrastivePairBatch):
            from .joint_steps import run_pair_validation_step

            return run_pair_validation_step(self, batch)
        loss, metrics = self._shared_step(batch, "val", lite=True)
        self._val_outputs.append(metrics)
        self.log_dict(
            {
                key: value
                for key, value in metrics.items()
                if not key.startswith(self._GLOBAL_VALIDATION_METRIC_PREFIXES)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
            add_dataloader_idx=False,
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
            plateau_monitor=(
                "val/pair_total_loss" if self.joint_contrastive else "val/total_loss"
            ),
            plateau_factor=self.plateau_factor,
            plateau_patience=self.plateau_patience,
            min_learning_rate=self.min_learning_rate,
        )
