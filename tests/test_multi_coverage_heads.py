#!/usr/bin/env python3
"""Tests for per-coverage graph prediction heads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DualGraphData
from models import DualGraphFusionModel, ModelConfig


def _batch_with_missing_targets() -> DualGraphData:
    return DualGraphData(
        node_cell_type=torch.randint(0, 74, (6,)),
        node_type=torch.randint(0, 12, (6,)),
        node_width=torch.ones(6, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_type=torch.randint(0, 7, (4,)),
        edge_width=torch.ones(4, dtype=torch.long),
        edge_source_port_idx=torch.zeros(4, dtype=torch.long),
        edge_target_port_idx=torch.zeros(4, dtype=torch.long),
        asm_node_type=torch.randint(0, 19, (3,)),
        asm_instruction_encoding=torch.randn(3, 256),
        asm_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        asm_edge_type=torch.randint(0, 10, (2,)),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        asm_node_type_batch=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_labels=torch.tensor([1, 0, -1, 1], dtype=torch.long),
        y=torch.tensor(
            [
                [0.5, float("nan"), float("nan"), 0.8, float("nan")],
                [0.7, 0.2, float("nan"), float("nan"), 0.1],
            ],
            dtype=torch.float,
        ),
    )


def test_graph_regressor_uses_one_head_per_coverage_target() -> None:
    config = ModelConfig(
        hidden_dim=32,
        num_gnn_layers=1,
        coverage_target_keys=("branch", "line", "toggle", "condition"),
    )
    model = DualGraphFusionModel(config)

    assert set(model.graph_regressor.heads.keys()) == {
        "branch",
        "line",
        "toggle",
        "condition",
    }
    assert (
        model.graph_regressor.heads["branch"] is not model.graph_regressor.heads["line"]
    )

    batch = _batch_with_missing_targets()
    output = model(batch)
    losses = model.compute_loss(output, batch)

    assert output.graph_pred.shape == (2, 4)
    assert torch.isfinite(output.graph_pred).all()
    assert torch.isfinite(losses["graph_loss"])
    assert losses["num_valid_graph_targets"] == 5


if __name__ == "__main__":
    test_graph_regressor_uses_one_head_per_coverage_target()
