#!/usr/bin/env python3
"""Pair contrastive data pipeline based on coverage vectors."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from .coverage_vector_similarity import (
    CoverageGeometryTargets,
    compute_coverage_geometry_targets,
    total_coverage_vector_length,
)
from .data_types import ContrastivePairBatch, DualGraphData
from .datamodule import DualGraphDataset
from .manifest import (
    DatasetDirInput,
    normalize_dataset_dirs as _normalize_dataset_dirs,
    read_manifest_samples_from_dirs as _read_manifest_samples_from_dirs,
)

if TYPE_CHECKING:
    from text_encoder import TextEncoderConfig

logger = logging.getLogger(__name__)


class ContrastivePairDataset(Dataset):
    """Pair dataset using coverage vectors and processed graph cache."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        pairs_per_sample: int = 4,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        encoding_cache_root: str | Path = "dataset_root",
        processed_root: str | Path | None = None,
        sample_indices: Optional[Sequence[int]] = None,
        asm_chunk_files: int = 64,
    ):
        self.dataset_dirs = _normalize_dataset_dirs(dataset_dir)
        self.pairs_per_sample = max(1, int(pairs_per_sample))
        self._samples = _read_manifest_samples_from_dirs(self.dataset_dirs)
        self._sample_indices = (
            set(int(idx) for idx in sample_indices)
            if sample_indices is not None
            else None
        )
        self._graph_dataset = DualGraphDataset(
            root=str(processed_root or encoding_cache_root),
            dataset_dir=self.dataset_dirs,
            text_encoder_config=text_encoder_config,
            num_workers=1,
            asm_chunk_files=asm_chunk_files,
        )
        if len(self._graph_dataset) < len(self._samples):
            raise RuntimeError(
                "processed graph cache has fewer samples than the manifest; "
                "run prepare_contrastive_data.py before joint contrastive training"
            )
        self._pairs: list[tuple[int, int, CoverageGeometryTargets]] = []
        self._build_pairs()

    def _build_pairs(self) -> None:
        by_source_module: dict[tuple[str, str], list[int]] = {}
        for idx, sample in enumerate(self._samples):
            if self._sample_indices is not None and idx not in self._sample_indices:
                continue
            vectors = sample.get("targets", {}).get("coverage_vectors")
            if vectors is None:
                raise ValueError(
                    "joint contrastive training requires targets.coverage_vectors "
                    f"for sample {sample.get('sample_id')}"
                )
            if total_coverage_vector_length(vectors) <= 0:
                continue
            group_key = (sample["_dataset_dir"], sample["module_name"])
            by_source_module.setdefault(group_key, []).append(idx)

        for indices in by_source_module.values():
            if len(indices) < 2:
                continue
            for pos, idx_a in enumerate(indices):
                upper = min(len(indices), pos + 1 + self.pairs_per_sample)
                for idx_b in indices[pos + 1 : upper]:
                    sample_a = self._samples[idx_a]
                    sample_b = self._samples[idx_b]
                    targets = compute_coverage_geometry_targets(
                        sample_a["targets"]["coverage_vectors"],
                        sample_b["targets"]["coverage_vectors"],
                    )
                    self._pairs.append((idx_a, idx_b, targets))

        logger.info("ContrastivePairDataset: built %d pairs", len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[DualGraphData, DualGraphData, CoverageGeometryTargets]:
        idx_a, idx_b, targets = self._pairs[idx]
        return (
            self._build_data(idx_a),
            self._build_data(idx_b),
            targets,
        )

    def _build_data(self, sample_idx: int) -> DualGraphData:
        return self._graph_dataset[sample_idx]


def contrastive_pair_collate(
    pair_list: list[tuple[DualGraphData, DualGraphData, CoverageGeometryTargets]],
) -> ContrastivePairBatch:
    items_a, items_b, targets = zip(*pair_list)
    follow = ["asm_node_type"]
    return ContrastivePairBatch(
        batch_a=Batch.from_data_list(list(items_a), follow_batch=follow),
        batch_b=Batch.from_data_list(list(items_b), follow_batch=follow),
        density_a=torch.tensor([target.density_a for target in targets]),
        density_b=torch.tensor([target.density_b for target in targets]),
        size_mask=torch.tensor([target.size_mask for target in targets]),
        jaccard=torch.tensor([target.jaccard for target in targets]),
        iou_mask=torch.tensor([target.iou_mask for target in targets]),
    )


class ContrastivePairDataModule:
    """DataModule for typed coverage-geometry pair batches."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = True,
        pairs_per_sample: int = 4,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        encoding_cache_root: str | Path = "dataset_root",
        processed_root: str | Path | None = None,
        sample_indices: Optional[Sequence[int]] = None,
        asm_chunk_files: int = 64,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pairs_per_sample = pairs_per_sample
        self.text_encoder_config = text_encoder_config
        self.encoding_cache_root = encoding_cache_root
        self.processed_root = processed_root
        self.sample_indices = (
            list(sample_indices) if sample_indices is not None else None
        )
        self.asm_chunk_files = max(1, int(asm_chunk_files))
        self._dataset: Optional[ContrastivePairDataset] = None

    def set_sample_indices(self, sample_indices: Optional[Sequence[int]]) -> None:
        self.sample_indices = (
            list(sample_indices) if sample_indices is not None else None
        )
        self._dataset = None

    def setup(self, stage: Optional[str] = None):
        if self._dataset is None:
            self._dataset = ContrastivePairDataset(
                dataset_dir=self.dataset_dir,
                pairs_per_sample=self.pairs_per_sample,
                text_encoder_config=self.text_encoder_config,
                encoding_cache_root=self.encoding_cache_root,
                processed_root=self.processed_root,
                sample_indices=self.sample_indices,
                asm_chunk_files=self.asm_chunk_files,
            )

    def train_dataloader(self) -> DataLoader:
        if self._dataset is None:
            self.setup()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=contrastive_pair_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._dataset is None:
            self.setup()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=contrastive_pair_collate,
            pin_memory=True,
        )

    def __len__(self) -> int:
        if self._dataset is None:
            self.setup()
        return len(self._dataset)
