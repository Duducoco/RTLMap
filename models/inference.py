"""Strict batch inference for unlabeled candidate datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from torch_geometric.loader import DataLoader

from datasets import DualGraphDataset
from datasets.manifest import read_manifest_samples
from models.config_artifact import (
    load_model_config_artifact,
    validate_checkpoint_artifact,
)
from trainer.lightning_module import DualGraphLightningModule


def build_prediction_records(
    *,
    sample_ids: Sequence[str],
    coverage_keys: Sequence[str],
    graph_pred: torch.Tensor,
    hyper_min: torch.Tensor | None,
    hyper_max: torch.Tensor | None,
) -> list[dict]:
    """Validate a model batch and convert it to JSON-serializable records."""
    batch_size = len(sample_ids)
    if graph_pred.ndim != 2 or tuple(graph_pred.shape) != (
        batch_size,
        len(coverage_keys),
    ):
        raise ValueError(
            "graph_pred shape mismatch: "
            f"got {tuple(graph_pred.shape)}, "
            f"expected {(batch_size, len(coverage_keys))}"
        )
    if hyper_min is None or hyper_max is None:
        raise ValueError("inference requires hyper_min and hyper_max")
    if hyper_min.ndim != 2 or hyper_min.shape != hyper_max.shape:
        raise ValueError("hyper_min and hyper_max must have the same 2D shape")
    if hyper_min.shape[0] != batch_size:
        raise ValueError("hyperrectangle batch dimension mismatch")
    if not torch.isfinite(graph_pred).all():
        raise ValueError("graph_pred contains NaN or Inf")
    if not torch.isfinite(hyper_min).all() or not torch.isfinite(hyper_max).all():
        raise ValueError("hyperrectangle contains NaN or Inf")
    if torch.any(hyper_min > hyper_max):
        raise ValueError("hyper_min exceeds hyper_max")

    predictions = graph_pred.detach().cpu().tolist()
    lower = hyper_min.detach().cpu().tolist()
    upper = hyper_max.detach().cpu().tolist()
    return [
        {
            "schema_version": "rtlmap_prediction.v1",
            "sample_id": sample_id,
            "coverage": dict(zip(coverage_keys, predictions[index], strict=True)),
            "hyper_min": lower[index],
            "hyper_max": upper[index],
        }
        for index, sample_id in enumerate(sample_ids)
    ]


def run_candidate_inference(
    *,
    dataset_dir: Path,
    checkpoint_path: Path,
    model_config_path: Path,
    cache_root: Path,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cpu",
) -> Path:
    """Run strict inference and persist one record per manifest sample."""
    sidecar, model_config, text_config = load_model_config_artifact(model_config_path)
    if not model_config.use_hyperrectangle:
        raise ValueError("candidate inference requires use_hyperrectangle=true")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    validate_checkpoint_artifact(sidecar, hparams)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint missing state_dict")

    module = DualGraphLightningModule(
        model_config=model_config,
        coverage_target_keys=model_config.coverage_target_keys,
        model_config_artifact=sidecar,
    )
    module.load_state_dict(state_dict, strict=True)
    module.eval().to(device)

    samples = read_manifest_samples(Path(dataset_dir))
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("candidate dataset contains duplicate sample_id values")
    dataset = DualGraphDataset(
        root=str(cache_root),
        dataset_dir=dataset_dir,
        text_encoder_config=text_config,
        num_workers=num_workers,
    )
    if len(dataset) != len(sample_ids):
        raise ValueError(
            f"processed dataset size {len(dataset)} != manifest size {len(sample_ids)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        follow_batch=["asm_node_type"],
    )

    records: list[dict] = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            output = module(batch)
            current_ids = sample_ids[cursor : cursor + output.graph_pred.shape[0]]
            records.extend(
                build_prediction_records(
                    sample_ids=current_ids,
                    coverage_keys=model_config.coverage_target_keys,
                    graph_pred=output.graph_pred,
                    hyper_min=output.hyper_min,
                    hyper_max=output.hyper_max,
                )
            )
            cursor += len(current_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.inference")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_candidate_inference(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        cache_root=args.cache_root,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
