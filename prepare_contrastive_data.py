#!/usr/bin/env python3
"""Prepare CodeBERT and graph caches for joint contrastive training."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _normalize_encoder_devices(devices: str) -> str:
    devices = devices.strip()
    if devices.lower() in {"cpu", "none"}:
        return ""
    return devices


def _read_encoder_devices_arg(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "--encoder-devices" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--encoder-devices="):
            return arg.split("=", 1)[1]
    return None


def _configure_visible_devices() -> str:
    # Must run before importing modules that import torch. The encoder itself
    # then sees only this device list and uses all visible GPUs in threads.
    requested = _read_encoder_devices_arg(sys.argv[1:])
    if requested is None:
        requested = os.environ.get("PREPARE_ENCODER_DEVICES")
    if requested is None:
        requested = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

    visible_devices = _normalize_encoder_devices(requested)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return visible_devices


_ENCODER_DEVICES = _configure_visible_devices()

from datasets import DualGraphDataModule  # noqa: E402
from datasets.pair_datamodule import ContrastivePairDataModule  # noqa: E402
from text_encoder import TextEncoderConfig  # noqa: E402


DEFAULT_DATASET_ROOT = "/home/u1/projects/coverage-report-extractor/out"
DEFAULT_DATASET_NAMES = (
    "archgen_single",
    "ibex",
    "picorv32",
    "riscv_simple_multicycle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets and ASM CodeBERT caches before training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="./data_contrastive")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset-dir",
        nargs="+",
        default=None,
        help="Explicit dataset.v1 directories. Overrides --dataset-root defaults.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--contrastive-batch-size", type=int, default=16)
    parser.add_argument("--contrastive-pairs-per-sample", type=int, default=4)
    parser.add_argument("--text-model-name", default="microsoft/codebert-base")
    parser.add_argument("--text-output-dim", type=int, default=256)
    parser.add_argument("--text-max-length", type=int, default=512)
    parser.add_argument("--text-batch-size", type=int, default=1024)
    parser.add_argument("--text-pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--encoder-devices",
        default=_ENCODER_DEVICES,
        help=(
            "CUDA device list for CodeBERT preparation, e.g. '0,1'. "
            "Use 'cpu' to disable CUDA."
        ),
    )
    parser.add_argument(
        "--asm-chunk-files",
        type=int,
        default=64,
        help="Number of ASM JSON files loaded before each CodeBERT encode pass.",
    )
    return parser.parse_args()


def default_dataset_dirs(dataset_root: str) -> list[str]:
    root = Path(dataset_root)
    return [str(root / name) for name in DEFAULT_DATASET_NAMES]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_dirs = args.dataset_dir or default_dataset_dirs(args.dataset_root)
    text_config = TextEncoderConfig(
        model_name=args.text_model_name,
        output_dim=args.text_output_dim,
        max_length=args.text_max_length,
        batch_size=args.text_batch_size,
        pooling=args.text_pooling,
    )

    logging.info("Preparing %d dataset directories", len(dataset_dirs))
    logging.info(
        "CodeBERT encoder devices: %s",
        args.encoder_devices if args.encoder_devices else "cpu",
    )
    for dataset_dir in dataset_dirs:
        logging.info("dataset: %s", dataset_dir)

    logging.info("Step 1/2: preparing DualGraphDataModule with CodeBERT")
    base_dm = DualGraphDataModule(
        root=args.data_root,
        dataset_dir=dataset_dirs,
        text_encoder_config=text_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        asm_chunk_files=args.asm_chunk_files,
    )
    base_dm.prepare_data()
    base_dm.setup("fit")
    train_len = len(base_dm.train_dataset) if base_dm.train_dataset is not None else 0
    val_len = len(base_dm.val_dataset) if base_dm.val_dataset is not None else 0
    logging.info("DualGraphDataModule ready: train=%d, val=%d", train_len, val_len)

    logging.info("Step 2/2: preparing contrastive pair dataset with CodeBERT")
    pair_dm = ContrastivePairDataModule(
        dataset_dir=dataset_dirs,
        batch_size=args.contrastive_batch_size,
        num_workers=0,
        pairs_per_sample=args.contrastive_pairs_per_sample,
        text_encoder_config=text_config,
        encoding_cache_root=str(Path(args.data_root) / "train"),
        asm_chunk_files=args.asm_chunk_files,
    )
    pair_dm.setup("fit")
    logging.info("ContrastivePairDataModule ready: pairs=%d", len(pair_dm))
    logging.info("Preparation complete. Use the same --data-root for training.")


if __name__ == "__main__":
    main()
