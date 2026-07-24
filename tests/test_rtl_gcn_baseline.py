from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
from torch_geometric.nn import GCNConv

from datasets import DualGraphData
from models import (
    GNNLayer,
    MODEL_ARCHITECTURE_RTL_GCN,
    GCNFusionBaselineModel,
    ModelConfig,
    PerceiverCrossFusion,
    RTLGCNLayer,
    create_model,
)
from models.config_artifact import (
    build_model_config_artifact,
    load_model_config_artifact,
    save_model_config_artifact,
)


class _StubGraphEncoder(torch.nn.Module):
    def forward(self, **kwargs):
        del kwargs
        return (
            torch.zeros((1, 4)),
            torch.zeros((0, 4)),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.zeros((1, 4)),
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
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


def test_factory_replaces_only_rtl_message_passing_with_standard_gcn() -> None:
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_RTL_GCN,
        hidden_dim=8,
        num_gnn_layers=2,
    )

    model = create_model(config)

    assert isinstance(model, GCNFusionBaselineModel)
    assert all(isinstance(layer, RTLGCNLayer) for layer in model.encoder.rtl_layers)
    assert all(isinstance(layer.conv, GCNConv) for layer in model.encoder.rtl_layers)
    assert all(isinstance(layer, GNNLayer) for layer in model.encoder.asm_layers)
    assert any(isinstance(module, PerceiverCrossFusion) for module in model.modules())
    assert model.encoder.rtl_edge_encoder is None
    assert not any(
        name.startswith("encoder.rtl_edge_encoder")
        for name, _ in model.named_parameters()
    )


def test_rtl_gcn_layer_uses_only_nodes_and_adjacency() -> None:
    torch.manual_seed(0)
    layer = RTLGCNLayer(hidden_dim=4, drop_path_rate=0.0).eval()
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    output_a = layer(x, edge_index)
    output_b = layer(x, edge_index)

    torch.testing.assert_close(output_a, output_b)


def test_rtl_gcn_heads_keep_concat_and_configurable_hyperrectangle() -> None:
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_RTL_GCN,
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

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    torch.testing.assert_close(graph_head.input, expected)
    torch.testing.assert_close(geometry_head.input, expected)
    assert output.hyper_min.shape == (1, 5, 7)
    assert output.hyper_max.shape == (1, 5, 7)


def test_rtl_gcn_real_encoder_supports_hyperrectangle_backward() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        model_architecture=MODEL_ARCHITECTURE_RTL_GCN,
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


def test_rtl_gcn_architecture_round_trips_in_model_artifact(tmp_path: Path) -> None:
    path = tmp_path / "model_config.yaml"
    config = ModelConfig(model_architecture=MODEL_ARCHITECTURE_RTL_GCN)
    save_model_config_artifact(path, build_model_config_artifact(config, None))

    _, loaded_config, _ = load_model_config_artifact(path)

    assert loaded_config == config
