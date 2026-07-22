from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from datasets import DualGraphData
from models import (
    MODEL_ARCHITECTURE_POOLED_ADD,
    PerceiverCrossFusion,
    PooledAddBaselineModel,
    ModelConfig,
    create_model,
)


class _StubGraphEncoder(torch.nn.Module):
    def forward(self, **kwargs):
        del kwargs
        return (
            torch.zeros((2, 4)),
            torch.zeros((0, 4)),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]),
            torch.zeros((2, 4)),
            torch.tensor([[5.0, 6.0, 7.0, 8.0], [1.0, 1.0, 1.0, 1.0]]),
        )


class _CapturingHead(torch.nn.Module):
    def __init__(self, *, hyperrectangle: bool = False) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None
        self.hyperrectangle = hyperrectangle

    def forward(self, graph_embedding: torch.Tensor):
        self.input = graph_embedding
        if self.hyperrectangle:
            shape = (graph_embedding.shape[0], 5, 7)
            return torch.zeros(shape), torch.ones(shape)
        return graph_embedding[:, :1]


def _batch() -> SimpleNamespace:
    value = torch.empty(0)
    return SimpleNamespace(
        edge_index=value,
        node_cell_type=value,
        node_width=value,
        edge_type=value,
        node_type=value,
        asm_edge_index=value,
        asm_node_type=value,
        asm_instruction_encoding=value,
        asm_edge_type=value,
    )


def test_factory_builds_baseline_without_cross_fusion_modules() -> None:
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_POOLED_ADD,
        hidden_dim=4,
        num_gnn_layers=1,
    )

    model = create_model(config)

    assert isinstance(model, PooledAddBaselineModel)
    assert not any(
        isinstance(module, PerceiverCrossFusion) for module in model.modules()
    )
    assert not any("fusion_" in name for name, _ in model.named_parameters())


def test_baseline_heads_read_the_sum_of_pooled_graph_embeddings() -> None:
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_POOLED_ADD,
        hidden_dim=4,
        num_gnn_layers=1,
        use_hyperrectangle=True,
        hyperrectangle_dim_per_type=7,
    )
    model = create_model(config)
    model.encoder = _StubGraphEncoder()
    graph_head = _CapturingHead()
    geometry_head = _CapturingHead(hyperrectangle=True)
    model.graph_regressor = graph_head
    model.hyperrectangle_head = geometry_head

    output = model(_batch())

    expected = torch.tensor([[6.0, 8.0, 10.0, 12.0], [3.0, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(graph_head.input, expected)
    torch.testing.assert_close(geometry_head.input, expected)
    assert output.graph_pred.shape == (2, 1)
    assert output.hyper_min.shape == (2, 5, 7)
    assert output.hyper_max.shape == (2, 5, 7)


def test_baseline_hyperrectangle_head_respects_configured_dimension() -> None:
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_POOLED_ADD,
        hidden_dim=4,
        num_gnn_layers=1,
        use_hyperrectangle=True,
        hyperrectangle_dim_per_type=7,
        hyper_min_margin=0.02,
    )
    model = create_model(config)
    model.encoder = _StubGraphEncoder()

    output = model(_batch())

    assert output.hyper_min.shape == (2, 5, 7)
    assert output.hyper_max.shape == (2, 5, 7)
    assert torch.all(output.hyper_max - output.hyper_min >= 0.02 - 1e-6)


def test_baseline_real_encoder_supports_hyperrectangle_backward() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_POOLED_ADD,
        hidden_dim=16,
        num_gnn_layers=1,
        asm_instruction_dim=16,
        use_hyperrectangle=True,
        hyperrectangle_dim_per_type=3,
    )
    model = create_model(config)
    data = DualGraphData(
        node_cell_type=torch.randint(0, 74, (4,)),
        node_type=torch.randint(0, 12, (4,)),
        node_width=torch.ones(4, dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_type=torch.randint(0, 7, (3,)),
        edge_width=torch.ones(3, dtype=torch.long),
        edge_source_port_idx=torch.zeros(3, dtype=torch.long),
        edge_target_port_idx=torch.zeros(3, dtype=torch.long),
        asm_node_type=torch.randint(0, 22, (2,)),
        asm_instruction_encoding=torch.randn(2, 16),
        asm_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        asm_edge_type=torch.zeros(1, dtype=torch.long),
        y=torch.tensor([[0.5]], dtype=torch.float),
    )

    output = model(data)
    loss = model.compute_loss(output, data)["graph_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert output.graph_pred.shape == (1, 1)
    assert output.hyper_min.shape == (1, 5, 3)
    assert output.hyper_max.shape == (1, 5, 3)


def test_model_config_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported model_architecture"):
        ModelConfig(model_architecture="unknown")
