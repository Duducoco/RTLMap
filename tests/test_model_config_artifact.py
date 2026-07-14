from pathlib import Path

import pytest
import yaml

from models.config_artifact import (
    ModelConfigMismatch,
    build_model_config_artifact,
    load_model_config_artifact,
    save_model_config_artifact,
    validate_checkpoint_artifact,
)
from models.data_types import ModelConfig
from text_encoder import TextEncoderConfig
from trainer.app_config import AppConfig, DataConfig, RuntimeConfig
from trainer.config import TrainerConfig
from trainer.lightning_module import DualGraphLightningModule
from trainer.utils import persist_model_config_artifact


def _model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=128,
        num_gnn_layers=3,
        dropout=0.2,
        coverage_target_keys=("branch", "fsm", "line", "condition", "toggle"),
        num_cell_types=75,
        num_edge_types=8,
        max_ports=12,
        num_asm_node_types=23,
        num_asm_edge_types=11,
        asm_instruction_dim=192,
        perceiver_num_latents=8,
        perceiver_num_heads=2,
        use_asm_adapter=False,
        use_hyperrectangle=True,
        hyper_min_margin=0.02,
    )


def _text_config() -> TextEncoderConfig:
    return TextEncoderConfig(
        model_name="example/code-model",
        output_dim=192,
        max_length=256,
        batch_size=64,
        device="cuda",
        pooling="cls",
    )


def test_model_config_yaml_round_trip_preserves_inference_fields(tmp_path: Path):
    path = tmp_path / "model_config.yaml"
    artifact = build_model_config_artifact(_model_config(), _text_config())

    save_model_config_artifact(path, artifact)
    loaded, model_config, text_config = load_model_config_artifact(path)

    assert loaded == artifact
    assert model_config == _model_config()
    assert text_config == _text_config()
    assert yaml.safe_load(path.read_text())["schema_version"] == "rtlmap_model_config.v2"
    assert loaded["model"]["coverage_target_keys"] == [
        "branch",
        "fsm",
        "line",
        "condition",
        "toggle",
    ]
    assert loaded["model"]["hyperrectangle_type_names"] == [
        "line",
        "condition",
        "toggle",
        "fsm",
        "branch",
    ]
    assert loaded["model"]["hyperrectangle_dim_per_type"] == 10
    assert "num_graph_targets" not in loaded["model"]


def test_checkpoint_artifact_mismatch_is_rejected():
    sidecar = build_model_config_artifact(_model_config(), _text_config())
    checkpoint = build_model_config_artifact(_model_config(), _text_config())
    checkpoint["model"]["hidden_dim"] = 256

    with pytest.raises(ModelConfigMismatch, match="sidecar.*checkpoint"):
        validate_checkpoint_artifact(sidecar, {"model_config_artifact": checkpoint})


def test_checkpoint_without_full_artifact_is_rejected():
    sidecar = build_model_config_artifact(_model_config(), _text_config())

    with pytest.raises(ModelConfigMismatch, match="model_config_artifact"):
        validate_checkpoint_artifact(sidecar, {"hidden_dim": 128})


def test_lightning_hparams_embed_complete_model_config_artifact():
    artifact = build_model_config_artifact(_model_config(), _text_config())

    module = DualGraphLightningModule(
        model_config=_model_config(),
        coverage_target_keys=_model_config().coverage_target_keys,
        model_config_artifact=artifact,
    )

    assert module.hparams["model_config_artifact"] == artifact


def test_training_config_is_written_next_to_experiment_checkpoints(tmp_path: Path):
    app_config = AppConfig(
        data=DataConfig(data_root=str(tmp_path / "data")),
        model=_model_config(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        text_encoder=_text_config(),
        runtime=RuntimeConfig(experiment_name="experiment_a"),
    )

    output = persist_model_config_artifact(app_config)

    assert output == tmp_path / "checkpoints" / "experiment_a" / "model_config.yaml"
    _, model_config, text_config = load_model_config_artifact(output)
    assert model_config == _model_config()
    assert text_config == _text_config()
