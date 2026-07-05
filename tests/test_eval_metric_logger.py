#!/usr/bin/env python3
"""Tests for validation metric JSON logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from trainer.callbacks import EvalMetricJsonLogger


def test_eval_metric_json_logger_writes_validation_metrics(tmp_path: Path) -> None:
    callback = EvalMetricJsonLogger()
    logger = SimpleNamespace(log_dir=str(tmp_path / "version_0"))
    trainer = SimpleNamespace(
        logger=logger,
        current_epoch=0,
        global_step=12,
        callback_metrics={
            "val/total_loss": torch.tensor(0.5),
            "val/graph_mse": torch.tensor(0.25),
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
                "val/graph_mse": 0.25,
            },
        }
    ]
