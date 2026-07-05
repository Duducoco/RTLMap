#!/usr/bin/env python3
"""Tests for graph-only supervised training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DualGraphData
from models import ModelConfig
from models.data_types import ModelOutput
from trainer import DualGraphLightningModule


def _minimal_batch() -> DualGraphData:
    return DualGraphData(
        node_cell_type=torch.randint(0, 74, (4,)),
        node_type=torch.randint(0, 12, (4,)),
        node_width=torch.ones(4, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_type=torch.randint(0, 7, (3,)),
        edge_width=torch.ones(3, dtype=torch.long),
        edge_source_port_idx=torch.zeros(3, dtype=torch.long),
        edge_target_port_idx=torch.zeros(3, dtype=torch.long),
        asm_node_type=torch.zeros(1, dtype=torch.long),
        asm_instruction_encoding=torch.zeros(1, 32),
        asm_edge_index=torch.empty((2, 0), dtype=torch.long),
        asm_edge_type=torch.empty(0, dtype=torch.long),
        edge_labels=torch.tensor([0, 1, 1], dtype=torch.long),
        y=torch.tensor([[0.5]], dtype=torch.float),
    )


def test_standard_training_step_uses_graph_loss_only(monkeypatch) -> None:
    module = DualGraphLightningModule(
        model_config=ModelConfig(
            hidden_dim=32,
            num_gnn_layers=1,
            asm_instruction_dim=32,
        ),
    )
    batch = _minimal_batch()

    monkeypatch.setattr(
        module,
        "forward",
        lambda data: ModelOutput(
            graph_pred=torch.zeros(1, 1),
        ),
    )

    def fake_compute_loss(output, data, *args, **kwargs):
        del output, data, args
        assert kwargs == {"graph_loss_weight": module.graph_loss_weight}
        unrelated_loss = torch.tensor(100.0)
        graph_loss = torch.tensor(5.0)
        return {
            "graph_loss": graph_loss,
            "total_loss": unrelated_loss + graph_loss,
            "num_valid_graph_targets": 1,
        }

    monkeypatch.setattr(module.model, "compute_loss", fake_compute_loss)

    loss, metrics = module._shared_step(batch, "train", lite=True)

    assert torch.equal(loss, torch.tensor(5.0))
    assert set(metrics) == {
        "train/total_loss",
        "train/graph_loss",
        "train/graph_mse",
        "train/graph_mae",
    }


if __name__ == "__main__":
    test_standard_training_step_uses_graph_loss_only()
