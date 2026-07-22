"""Versioned model/preprocessing configuration artifacts for inference."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from text_encoder import TextEncoderConfig

from .data_types import MODEL_ARCHITECTURE_PERCEIVER_FUSION, ModelConfig


SCHEMA_VERSION = "rtlmap_model_config.v6"
LEGACY_SCHEMA_VERSION = "rtlmap_model_config.v5"


class ModelConfigMismatch(ValueError):
    """Raised when sidecar and checkpoint inference contracts disagree."""


def _model_mapping(config: ModelConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_info in fields(ModelConfig):
        if not field_info.init:
            continue
        value = getattr(config, field_info.name)
        result[field_info.name] = list(value) if isinstance(value, tuple) else value
    return result


def build_model_config_artifact(
    model_config: ModelConfig,
    text_encoder_config: TextEncoderConfig | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": _model_mapping(model_config),
        "text_encoder": (
            asdict(text_encoder_config) if text_encoder_config is not None else None
        ),
    }


def save_model_config_artifact(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(artifact), sort_keys=False),
        encoding="utf-8",
    )
    return output


def _require_exact_fields(
    raw: Mapping[str, Any], expected: set[str], section: str
) -> None:
    actual = set(raw)
    if actual != expected:
        raise ValueError(
            f"{section} fields mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _normalize_artifact(
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], ModelConfig, TextEncoderConfig | None]:
    if not isinstance(artifact, Mapping):
        raise ValueError("model config artifact must be a mapping")
    schema_version = artifact.get("schema_version")
    if schema_version not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
        raise ValueError(f"unsupported model config schema: {schema_version!r}")

    model_raw = artifact.get("model")
    if not isinstance(model_raw, Mapping):
        raise ValueError("model config artifact missing model mapping")
    model_fields = {item.name for item in fields(ModelConfig) if item.init}
    model_values = dict(model_raw)
    if schema_version == LEGACY_SCHEMA_VERSION:
        _require_exact_fields(
            model_values, model_fields - {"model_architecture"}, "model"
        )
        model_values["model_architecture"] = MODEL_ARCHITECTURE_PERCEIVER_FUSION
    else:
        _require_exact_fields(model_values, model_fields, "model")
    model_values["coverage_target_keys"] = tuple(model_values["coverage_target_keys"])
    model_values["hyperrectangle_type_names"] = tuple(
        model_values["hyperrectangle_type_names"]
    )
    model_config = ModelConfig(**model_values)

    text_raw = artifact.get("text_encoder")
    text_config = None
    if text_raw is not None:
        if not isinstance(text_raw, Mapping):
            raise ValueError("text_encoder must be a mapping or null")
        text_fields = {item.name for item in fields(TextEncoderConfig) if item.init}
        _require_exact_fields(text_raw, text_fields, "text_encoder")
        text_config = TextEncoderConfig(**text_raw)

    normalized = build_model_config_artifact(model_config, text_config)
    return normalized, model_config, text_config


def load_model_config_artifact(
    path: str | Path,
) -> tuple[dict[str, Any], ModelConfig, TextEncoderConfig | None]:
    artifact = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _normalize_artifact(artifact)


def validate_checkpoint_artifact(
    sidecar: Mapping[str, Any], checkpoint_hparams: Mapping[str, Any]
) -> None:
    checkpoint_artifact = checkpoint_hparams.get("model_config_artifact")
    if not isinstance(checkpoint_artifact, Mapping):
        raise ModelConfigMismatch(
            "checkpoint hyperparameters missing model_config_artifact"
        )
    try:
        normalized_sidecar, _, _ = _normalize_artifact(sidecar)
        normalized_checkpoint, _, _ = _normalize_artifact(checkpoint_artifact)
    except ValueError as exc:
        raise ModelConfigMismatch(str(exc)) from exc
    if normalized_sidecar != normalized_checkpoint:
        raise ModelConfigMismatch(
            "sidecar model config does not match checkpoint model config artifact"
        )
