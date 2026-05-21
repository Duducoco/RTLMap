#!/usr/bin/env python3
"""Tests for feature-level union fusion used by joint contrastive training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DualGraphData
from models import DualGraphFusionModel, ModelConfig
from models.data_types import ModelOutput
from trainer import DualGraphLightningModule
from trainer.test_trainer import make_contrastive_triple_batch


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


if __name__ == "__main__":
    test_merge_outputs_is_symmetric_and_decodes_reference_edges()
    test_joint_training_step_does_not_forward_merged_batch()
