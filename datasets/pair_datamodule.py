#!/usr/bin/env python3
"""Pair contrastive data pipeline based on coverage vectors."""

from __future__ import annotations

import json
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from .coverage_vector_similarity import (
    compute_coverage_vector_agreement,
    total_coverage_vector_length,
)
from .data_types import ContrastivePairBatch, DualGraphData
from .graph_builders import (
    build_asm_graph as _build_asm_graph,
    build_rtl_structure as _build_rtl_structure,
    empty_asm_graph as _empty_asm_graph,
)
from .manifest import (
    DatasetDirInput,
    normalize_dataset_dirs as _normalize_dataset_dirs,
    read_manifest_samples_from_dirs as _read_manifest_samples_from_dirs,
)
from .targets import extract_targets_from_sample as _extract_targets_from_sample

logger = logging.getLogger(__name__)


class ContrastivePairDataset(Dataset):
    """Pair dataset using ordinary dataset.v1 samples and coverage_vectors."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        pairs_per_sample: int = 4,
    ):
        self.dataset_dirs = _normalize_dataset_dirs(dataset_dir)
        self.pairs_per_sample = max(1, int(pairs_per_sample))
        self._samples = _read_manifest_samples_from_dirs(self.dataset_dirs)
        self._pairs: list[tuple[int, int, float]] = []
        self._build_pairs()

    def _build_pairs(self) -> None:
        by_source_module: dict[tuple[str, str], list[int]] = {}
        for idx, sample in enumerate(self._samples):
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
                    similarity = compute_coverage_vector_agreement(
                        sample_a["targets"]["coverage_vectors"],
                        sample_b["targets"]["coverage_vectors"],
                    )
                    self._pairs.append((idx_a, idx_b, similarity))

        logger.info("ContrastivePairDataset: built %d pairs", len(self._pairs))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[DualGraphData, DualGraphData, float]:
        idx_a, idx_b, similarity = self._pairs[idx]
        return (
            self._build_data(self._samples[idx_a]),
            self._build_data(self._samples[idx_b]),
            similarity,
        )

    def _build_data(self, sample: dict) -> DualGraphData:
        with open(sample["_rtl_path"], encoding="utf-8") as f:
            rtl_json = json.load(f)
        rtl_struct = _build_rtl_structure(rtl_json)
        valid_edge_mask = rtl_struct["_valid_edge_mask"]
        rtl_struct.pop("_valid_edge_mask", None)

        edge_labels, y = _extract_targets_from_sample(sample, valid_edge_mask)
        asm_path = sample.get("_asm_path")
        asm = _load_asm_graph(asm_path) if asm_path else _empty_asm()

        return DualGraphData(
            **rtl_struct,
            **asm,
            edge_labels=edge_labels,
            y=y,
        )


def contrastive_pair_collate(
    pair_list: list[tuple[DualGraphData, DualGraphData, float]],
) -> ContrastivePairBatch:
    items_a, items_b, sims = zip(*pair_list)
    follow = ["asm_node_type"]
    return ContrastivePairBatch(
        batch_a=Batch.from_data_list(list(items_a), follow_batch=follow),
        batch_b=Batch.from_data_list(list(items_b), follow_batch=follow),
        similarity=torch.tensor(sims, dtype=torch.float32),
    )


class ContrastivePairDataModule:
    """Training-only DataModule for pair contrastive batches."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = True,
        pairs_per_sample: int = 4,
    ):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pairs_per_sample = pairs_per_sample
        self._dataset: Optional[ContrastivePairDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self._dataset is None:
            self._dataset = ContrastivePairDataset(
                dataset_dir=self.dataset_dir,
                pairs_per_sample=self.pairs_per_sample,
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

    def __len__(self) -> int:
        if self._dataset is None:
            self.setup()
        return len(self._dataset)


def _load_asm_graph(asm_json_path: str) -> dict:
    return _build_asm_graph(asm_json_path, asm_encoding=None)


def _empty_asm() -> dict:
    return _empty_asm_graph()
