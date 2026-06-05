#!/usr/bin/env python3
"""Tests for feature-level union fusion used by joint contrastive training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import ContrastiveTripleBatch, DualGraphData
from models import DualGraphFusionModel, ModelConfig
from models.data_types import ModelOutput
from trainer import DualGraphLightningModule


def _reference_graph(num_nodes: int = 6, num_edges: int = 8) -> DualGraphData:
    return DualGraphData(
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_type=torch.randint(0, 7, (num_edges,)),
        edge_width=torch.ones(num_edges, dtype=torch.long),
        edge_source_port_idx=torch.zeros(num_edges, dtype=torch.long),
        edge_target_port_idx=torch.zeros(num_edges, dtype=torch.long),
        edge_labels=torch.randint(-1, 2, (num_edges,)),
        y=torch.rand(1, 1),
    )


def make_contrastive_triple_batch(
    batch_size: int = 2, asm_instruction_dim: int = 32
) -> ContrastiveTripleBatch:
    items_a = []
    items_b = []
    items_m = []
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
        empty_asm = dict(
            asm_node_type=torch.zeros(1, dtype=torch.long),
            asm_instruction_encoding=torch.zeros(1, asm_instruction_dim),
            asm_edge_index=torch.empty((2, 0), dtype=torch.long),
            asm_edge_type=torch.empty(0, dtype=torch.long),
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
        items_m.append(
            DualGraphData(
                **common,
                **empty_asm,
                edge_labels=torch.randint(-1, 2, (num_edges,)),
                y=torch.rand(1, 1),
            )
        )

    from torch_geometric.data import Batch

    return ContrastiveTripleBatch(
        batch_a=Batch.from_data_list(items_a, follow_batch=["asm_node_type"]),
        batch_b=Batch.from_data_list(items_b, follow_batch=["asm_node_type"]),
        batch_merged=Batch.from_data_list(items_m, follow_batch=["asm_node_type"]),
        similarity=torch.rand(batch_size),
    )


def test_merge_outputs_is_symmetric_and_decodes_reference_edges() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        hidden_dim=16,
        num_gnn_layers=1,
        use_hyperrectangle=True,
    )
    model = DualGraphFusionModel(config)
    model.eval()

    num_nodes = 6
    hidden_dim = config.hidden_dim
    out_a = ModelOutput(
        rtl_final=torch.randn(num_nodes, hidden_dim),
        rtl_graph_emb=torch.randn(1, hidden_dim),
    )
    out_b = ModelOutput(
        rtl_final=torch.randn(num_nodes, hidden_dim),
        rtl_graph_emb=torch.randn(1, hidden_dim),
    )
    reference = _reference_graph(num_nodes=num_nodes, num_edges=8)

    merged_ab = model.merge_outputs(out_a, out_b, reference)
    merged_ba = model.merge_outputs(out_b, out_a, reference)

    assert merged_ab.edge_logits.shape == (8, 2)
    assert merged_ab.graph_pred.shape == (1, 1)
    assert merged_ab.hyper_min.shape == (1, hidden_dim)
    assert merged_ab.hyper_max.shape == (1, hidden_dim)
    assert torch.allclose(merged_ab.edge_logits, merged_ba.edge_logits)
    assert torch.allclose(merged_ab.graph_pred, merged_ba.graph_pred)


def test_joint_training_step_does_not_forward_merged_batch() -> None:
    torch.manual_seed(0)
    batch = make_contrastive_triple_batch(batch_size=2, asm_instruction_dim=32)
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
        lambda_cl=0.5,
    )
    module.train()

    original_forward = module.forward
    seen_ids: list[int] = []

    def counting_forward(data):
        seen_ids.append(id(data))
        if data is batch.batch_merged:
            raise AssertionError("batch_merged must not be forwarded through encoder")
        return original_forward(data)

    module.forward = counting_forward
    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert seen_ids == [id(batch.batch_a), id(batch.batch_b)]


def test_joint_training_step_uses_graph_loss_weight_for_merged_branch(monkeypatch) -> None:
    torch.manual_seed(0)
    batch = make_contrastive_triple_batch(batch_size=2, asm_instruction_dim=32)
    config = ModelConfig(
        hidden_dim=32,
        num_gnn_layers=1,
        asm_instruction_dim=32,
        use_hyperrectangle=True,
    )
    module = DualGraphLightningModule(
        model_config=config,
        graph_loss_weight=0.75,
        joint_contrastive=True,
        lambda_ce=1.0,
        lambda_cl=0.5,
    )
    module.train()

    original_compute_loss = module.model.compute_loss
    merged_graph_weights: list[float] = []

    def recording_compute_loss(output, data, *args, **kwargs):
        if data is batch.batch_merged:
            merged_graph_weights.append(kwargs["graph_loss_weight"])
        return original_compute_loss(output, data, *args, **kwargs)

    monkeypatch.setattr(module.model, "compute_loss", recording_compute_loss)

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert merged_graph_weights == [module.graph_loss_weight]


if __name__ == "__main__":
    test_merge_outputs_is_symmetric_and_decodes_reference_edges()
    test_joint_training_step_does_not_forward_merged_batch()
