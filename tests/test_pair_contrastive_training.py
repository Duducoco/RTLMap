#!/usr/bin/env python3
"""Tests for coverage-vector pair contrastive training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import ContrastivePairBatch, DualGraphData
from models import ModelConfig
from trainer import DualGraphLightningModule
import trainer.joint_steps as joint_steps


def make_contrastive_pair_batch(
    batch_size: int = 2, asm_instruction_dim: int = 32
) -> ContrastivePairBatch:
    items_a = []
    items_b = []
    for _ in range(batch_size):
        num_nodes = 6
        num_edges = 8
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        common = dict(
            node_cell_type=torch.randint(0, 74, (num_nodes,)),
            node_type=torch.randint(0, 12, (num_nodes,)),
            node_width=torch.ones(num_nodes, dtype=torch.long),
            edge_index=edge_index,
            edge_type=torch.randint(0, 7, (num_edges,)),
            edge_width=torch.ones(num_edges, dtype=torch.long),
            edge_source_port_idx=torch.zeros(num_edges, dtype=torch.long),
            edge_target_port_idx=torch.zeros(num_edges, dtype=torch.long),
        )
        asm_common = dict(
            asm_node_type=torch.randint(0, 22, (3,)),
            asm_instruction_encoding=torch.randn(3, asm_instruction_dim),
            asm_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            asm_edge_type=torch.randint(0, 10, (2,)),
        )
        items_a.append(
            DualGraphData(
                **common,
                **asm_common,
                edge_labels=torch.randint(-1, 2, (num_edges,)),
                y=torch.rand(1, 1),
            )
        )
        items_b.append(
            DualGraphData(
                **common,
                **asm_common,
                edge_labels=torch.randint(-1, 2, (num_edges,)),
                y=torch.rand(1, 1),
            )
        )

    return ContrastivePairBatch(
        batch_a=Batch.from_data_list(items_a, follow_batch=["asm_node_type"]),
        batch_b=Batch.from_data_list(items_b, follow_batch=["asm_node_type"]),
        density_a=torch.rand(batch_size, 5),
        density_b=torch.rand(batch_size, 5),
        size_mask=torch.ones(batch_size, 5, dtype=torch.bool),
        jaccard=torch.rand(batch_size, 5),
        iou_mask=torch.ones(batch_size, 5, dtype=torch.bool),
        endpoint_weight_a=torch.arange(1, batch_size + 1, dtype=torch.float32),
        endpoint_weight_b=torch.arange(
            batch_size + 1, 2 * batch_size + 1, dtype=torch.float32
        ),
    )


def test_pair_training_step_uses_only_a_and_b() -> None:
    torch.manual_seed(0)
    batch = make_contrastive_pair_batch(batch_size=2, asm_instruction_dim=32)
    config = ModelConfig(
        hidden_dim=32,
        num_gnn_layers=1,
        asm_instruction_dim=32,
        use_hyperrectangle=True,
    )
    module = DualGraphLightningModule(
        model_config=config,
        joint_contrastive=True,
        lambda_ce=1.0,
        lambda_iou=0.5,
    )
    module.train()

    original_forward = module.forward
    seen_ids: list[int] = []

    def counting_forward(data):
        seen_ids.append(id(data))
        return original_forward(data)

    module.forward = counting_forward
    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert seen_ids == [id(batch.batch_a), id(batch.batch_b)]


def test_pair_training_step_uses_one_weighted_endpoint_graph_loss(monkeypatch) -> None:
    torch.manual_seed(0)
    batch = make_contrastive_pair_batch(batch_size=2, asm_instruction_dim=32)
    config = ModelConfig(
        hidden_dim=32,
        num_gnn_layers=1,
        asm_instruction_dim=32,
        use_hyperrectangle=True,
    )
    module = DualGraphLightningModule(
        model_config=config,
        joint_contrastive=True,
        lambda_ce=1.0,
        lambda_iou=0.5,
    )
    module.train()

    monkeypatch.setattr(
        module.model,
        "compute_loss",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pair training must use the weighted endpoint reduction")
        ),
    )
    seen_weights: list[torch.Tensor] = []

    def fake_weighted_graph_loss(graph_pred, target, endpoint_weight, **kwargs):
        del graph_pred, target, kwargs
        seen_weights.append(endpoint_weight.detach().cpu())
        return torch.tensor(5.0, device=module.device)

    monkeypatch.setattr(
        joint_steps,
        "compute_weighted_graph_loss",
        fake_weighted_graph_loss,
        raising=False,
    )
    monkeypatch.setattr(
        joint_steps,
        "compute_coverage_geometry_losses",
        lambda *args, **kwargs: SimpleNamespace(
            iou_loss=torch.tensor(0.0, device=module.device),
            volume_loss=torch.tensor(0.0, device=module.device),
        ),
    )

    loss = module.training_step(batch, 0)

    assert torch.equal(loss, torch.tensor(5.0, device=module.device))
    assert len(seen_weights) == 1
    assert torch.equal(seen_weights[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


if __name__ == "__main__":
    test_pair_training_step_uses_only_a_and_b()
    test_pair_training_step_uses_one_weighted_endpoint_graph_loss()
