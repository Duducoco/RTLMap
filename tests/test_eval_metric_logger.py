#!/usr/bin/env python3
"""Tests for validation metric JSON logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from trainer.callbacks import CallbackFactory, EvalMetricJsonLogger
from trainer.config import TrainerConfig


def test_eval_metric_json_logger_writes_validation_metrics(tmp_path: Path) -> None:
    callback = EvalMetricJsonLogger()
    logger = SimpleNamespace(log_dir=str(tmp_path / "version_0"))
    trainer = SimpleNamespace(
        logger=logger,
        is_global_zero=True,
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


def test_eval_metric_json_logger_skips_nonzero_rank(tmp_path: Path) -> None:
    callback = EvalMetricJsonLogger()
    trainer = SimpleNamespace(
        logger=SimpleNamespace(log_dir=str(tmp_path / "version_0")),
        is_global_zero=False,
        current_epoch=0,
        global_step=12,
        callback_metrics={"val/total_loss": torch.tensor(0.5)},
        sanity_checking=False,
    )

    callback.on_validation_epoch_end(trainer, pl_module=None)

    assert not (tmp_path / "version_0" / "eval_metric.json").exists()


def test_early_stopping_resume_preserves_runtime_patience() -> None:
    callback = CallbackFactory.create_early_stopping(
        TrainerConfig(
            early_stopping_monitor="val/pair_total_loss",
            early_stopping_patience=30,
        )
    )

    assert callback.state_key == (
        "EarlyStopping{'monitor': 'val/pair_total_loss', 'mode': 'min'}"
    )

    callback.load_state_dict(
        {
            "wait_count": 10,
            "stopped_epoch": 16,
            "best_score": torch.tensor(0.0732),
            "patience": 10,
        }
    )

    assert callback.patience == 30
    assert callback.wait_count == 10
    assert callback.best_score == torch.tensor(0.0732)
