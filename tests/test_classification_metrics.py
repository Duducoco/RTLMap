#!/usr/bin/env python3
"""Tests for edge classification metrics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from metrics.classification import EdgeClassificationMetrics
from trainer.callbacks import EvalMetricJsonLogger


def test_epoch_metrics_accept_bfloat16_logits() -> None:
    metrics = EdgeClassificationMetrics()
    logits = torch.tensor(
        [[-1.0, 1.0], [1.0, -1.0], [-0.5, 0.5]],
        dtype=torch.bfloat16,
    )
    labels = torch.tensor([1, 0, 1])

    metrics.collect_step(logits, labels)
    result = metrics.compute_epoch()

    assert set(result) == {"auc_roc", "auprc"}
    assert result["auc_roc"] >= 0.0
    assert result["auprc"] >= 0.0


def test_eval_metric_json_logger_writes_validation_metrics(tmp_path: Path) -> None:
    callback = EvalMetricJsonLogger()
    logger = SimpleNamespace(log_dir=str(tmp_path / "version_0"))
    trainer = SimpleNamespace(
        logger=logger,
        current_epoch=0,
        global_step=12,
        callback_metrics={
            "val/total_loss": torch.tensor(0.5),
            "val/f1": torch.tensor(0.8),
            "train/total_loss": torch.tensor(0.2),
        },
        sanity_checking=False,
    )

    callback.on_validation_epoch_end(trainer, pl_module=None)

    payload = json.loads((tmp_path / "version_0" / "eval_metric.json").read_text())
    assert payload == [
        {
            "epoch": 0,
            "step": 12,
            "metrics": {
                "val/total_loss": 0.5,
                "val/f1": 0.800000011920929,
            },
        }
    ]
