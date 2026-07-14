#!/usr/bin/env python3
"""Pair contrastive data pipeline based on coverage vectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import logging
import math
import random
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


@dataclass(frozen=True)
class GeometryPairRecord:
    """One sampled, oriented Geometry Pair selection."""

    idx_a: int
    idx_b: int
    targets: CoverageGeometryTargets
    stratum: str
    degree_a: int = 0
    degree_b: int = 0
    endpoint_weight_a: float = 0.0
    endpoint_weight_b: float = 0.0


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _coverage_similarity(targets: CoverageGeometryTargets) -> float:
    scores = [
        score
        for score, active in zip(targets.jaccard, targets.iou_mask, strict=True)
        if active
    ]
    return sum(scores) / len(scores) if scores else 0.0


class ContrastivePairDataset(Dataset):
    """Pair dataset using coverage vectors and processed graph cache."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        candidate_pool_size: int = 128,
        relative_low_quota: int = 2,
        relative_mid_quota: int = 1,
        relative_high_quota: int = 1,
        sampling_seed: int = 42,
        text_encoder_config: Optional["TextEncoderConfig"] = None,
        encoding_cache_root: str | Path = "dataset_root",
        processed_root: str | Path | None = None,
        sample_indices: Optional[Sequence[int]] = None,
        asm_chunk_files: int = 64,
    ):
        self.dataset_dirs = _normalize_dataset_dirs(dataset_dir)
        self.candidate_pool_size = int(candidate_pool_size)
        self.relative_quotas = {
            "relative-low": int(relative_low_quota),
            "relative-mid": int(relative_mid_quota),
            "relative-high": int(relative_high_quota),
        }
        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if any(quota < 0 for quota in self.relative_quotas.values()):
            raise ValueError("relative pair quotas must be non-negative")
        if sum(self.relative_quotas.values()) > self.candidate_pool_size:
            raise ValueError("relative pair quotas exceed candidate_pool_size")
        self.sampling_seed = int(sampling_seed)
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
        self._pair_records: list[GeometryPairRecord] = []
        self._pairs: list[tuple[int, int, CoverageGeometryTargets, float, float]] = []
        self._sampling_diagnostics: dict[str, int | float | str] = {}
        self.resample(0)

    @property
    def samples(self) -> tuple[dict, ...]:
        return tuple(self._samples)

    @property
    def pair_records(self) -> tuple[GeometryPairRecord, ...]:
        return tuple(self._pair_records)

    @property
    def pair_fingerprint(self) -> str:
        entries = sorted(
            (
                str(self._samples[record.idx_a]["sample_id"]),
                str(self._samples[record.idx_b]["sample_id"]),
                record.stratum,
            )
            for record in self._pair_records
        )
        payload = "\n".join("\0".join(entry) for entry in entries).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def sampling_diagnostics(self) -> dict[str, int | float | str]:
        return dict(self._sampling_diagnostics)

    def resample(self, epoch: int) -> None:
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

        records: list[GeometryPairRecord] = []
        selected_pairs: set[tuple[int, int]] = set()
        eligible_indices: set[int] = set()
        candidate_pool_sizes: list[int] = []
        similarity_values: list[float] = []
        requested_by_stratum: Counter[str] = Counter()
        fulfilled_by_stratum: Counter[str] = Counter()
        duplicate_conflicts = 0
        for group_key, indices in sorted(by_source_module.items()):
            if len(indices) < 2:
                continue
            eligible_indices.update(indices)
            rng = random.Random(
                _stable_seed(self.sampling_seed, int(epoch), *group_key)
            )
            anchor_indices = sorted(
                indices, key=lambda idx: str(self._samples[idx]["sample_id"])
            )
            rng.shuffle(anchor_indices)
            for idx_a in anchor_indices:
                candidate_indices = sorted(
                    (idx for idx in indices if idx != idx_a),
                    key=lambda idx: str(self._samples[idx]["sample_id"]),
                )
                if len(candidate_indices) > self.candidate_pool_size:
                    candidate_indices = rng.sample(
                        candidate_indices, self.candidate_pool_size
                    )
                candidate_pool_sizes.append(len(candidate_indices))

                scored_candidates: list[
                    tuple[float, float, int, CoverageGeometryTargets]
                ] = []
                sample_a = self._samples[idx_a]
                for idx_b in candidate_indices:
                    sample_b = self._samples[idx_b]
                    targets = compute_coverage_geometry_targets(
                        sample_a["targets"]["coverage_vectors"],
                        sample_b["targets"]["coverage_vectors"],
                    )
                    scored_candidates.append(
                        (_coverage_similarity(targets), rng.random(), idx_b, targets)
                    )

                scored_candidates.sort(key=lambda candidate: candidate[:2])
                count = len(scored_candidates)
                low_end = max(1, math.ceil(count / 4))
                high_start = max(low_end, math.floor(3 * count / 4))
                strata = {
                    "relative-low": scored_candidates[:low_end],
                    "relative-mid": scored_candidates[low_end:high_start],
                    "relative-high": scored_candidates[high_start:],
                }
                for stratum, candidates in strata.items():
                    requested_by_stratum[stratum] += self.relative_quotas[stratum]
                    shuffled = list(candidates)
                    rng.shuffle(shuffled)
                    remaining = self.relative_quotas[stratum]
                    for similarity, _, idx_b, targets in shuffled:
                        if remaining <= 0:
                            break
                        pair_key = tuple(sorted((idx_a, idx_b)))
                        if pair_key in selected_pairs:
                            duplicate_conflicts += 1
                            continue
                        selected_pairs.add(pair_key)
                        records.append(
                            GeometryPairRecord(
                                idx_a=idx_a,
                                idx_b=idx_b,
                                targets=targets,
                                stratum=stratum,
                            )
                        )
                        fulfilled_by_stratum[stratum] += 1
                        similarity_values.append(similarity)
                        remaining -= 1

        degree_by_index: Counter[int] = Counter()
        for record in records:
            degree_by_index.update((record.idx_a, record.idx_b))
        mean_degree = (
            sum(degree_by_index.values()) / len(degree_by_index)
            if degree_by_index
            else 0.0
        )
        self._pair_records = [
            replace(
                record,
                degree_a=degree_by_index[record.idx_a],
                degree_b=degree_by_index[record.idx_b],
                endpoint_weight_a=mean_degree / degree_by_index[record.idx_a],
                endpoint_weight_b=mean_degree / degree_by_index[record.idx_b],
            )
            for record in records
        ]
        self._pairs = [
            (
                record.idx_a,
                record.idx_b,
                record.targets,
                record.endpoint_weight_a,
                record.endpoint_weight_b,
            )
            for record in self._pair_records
        ]
        degrees = list(degree_by_index.values())
        self._sampling_diagnostics = {
            "epoch": int(epoch),
            "eligible_anchors": len(eligible_indices),
            "positive_degree_samples": len(degree_by_index),
            "degree_zero_samples": len(eligible_indices - set(degree_by_index)),
            "unique_pairs": len(self._pair_records),
            "requested_relative_low": requested_by_stratum["relative-low"],
            "requested_relative_mid": requested_by_stratum["relative-mid"],
            "requested_relative_high": requested_by_stratum["relative-high"],
            "fulfilled_relative_low": fulfilled_by_stratum["relative-low"],
            "fulfilled_relative_mid": fulfilled_by_stratum["relative-mid"],
            "fulfilled_relative_high": fulfilled_by_stratum["relative-high"],
            "duplicate_conflicts": duplicate_conflicts,
            "candidate_pool_min": min(candidate_pool_sizes, default=0),
            "candidate_pool_mean": (
                sum(candidate_pool_sizes) / len(candidate_pool_sizes)
                if candidate_pool_sizes
                else 0.0
            ),
            "candidate_pool_max": max(candidate_pool_sizes, default=0),
            "degree_min": min(degrees, default=0),
            "degree_mean": mean_degree,
            "degree_max": max(degrees, default=0),
            "coverage_similarity_min": min(similarity_values, default=0.0),
            "coverage_similarity_mean": (
                sum(similarity_values) / len(similarity_values)
                if similarity_values
                else 0.0
            ),
            "coverage_similarity_max": max(similarity_values, default=0.0),
            "pair_fingerprint": self.pair_fingerprint,
        }
        logger.info(
            "ContrastivePairDataset sampling diagnostics: %s",
            self._sampling_diagnostics,
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[DualGraphData, DualGraphData, CoverageGeometryTargets, float, float]:
        idx_a, idx_b, targets, weight_a, weight_b = self._pairs[idx]
        return (
            self._build_data(idx_a),
            self._build_data(idx_b),
            targets,
            weight_a,
            weight_b,
        )

    def _build_data(self, sample_idx: int) -> DualGraphData:
        return self._graph_dataset[sample_idx]


def contrastive_pair_collate(
    pair_list: list[
        tuple[DualGraphData, DualGraphData, CoverageGeometryTargets, float, float]
    ],
) -> ContrastivePairBatch:
    items_a, items_b, targets, weights_a, weights_b = zip(*pair_list)
    follow = ["asm_node_type"]
    return ContrastivePairBatch(
        batch_a=Batch.from_data_list(list(items_a), follow_batch=follow),
        batch_b=Batch.from_data_list(list(items_b), follow_batch=follow),
        density_a=torch.tensor([target.density_a for target in targets]),
        density_b=torch.tensor([target.density_b for target in targets]),
        size_mask=torch.tensor([target.size_mask for target in targets]),
        jaccard=torch.tensor([target.jaccard for target in targets]),
        iou_mask=torch.tensor([target.iou_mask for target in targets]),
        endpoint_weight_a=torch.tensor(weights_a, dtype=torch.float32),
        endpoint_weight_b=torch.tensor(weights_b, dtype=torch.float32),
    )


class ContrastivePairDataModule:
    """DataModule for typed coverage-geometry pair batches."""

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = True,
        candidate_pool_size: int = 128,
        relative_low_quota: int = 2,
        relative_mid_quota: int = 1,
        relative_high_quota: int = 1,
        sampling_seed: int = 42,
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
        self.candidate_pool_size = candidate_pool_size
        self.relative_low_quota = relative_low_quota
        self.relative_mid_quota = relative_mid_quota
        self.relative_high_quota = relative_high_quota
        self.sampling_seed = sampling_seed
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
                candidate_pool_size=self.candidate_pool_size,
                relative_low_quota=self.relative_low_quota,
                relative_mid_quota=self.relative_mid_quota,
                relative_high_quota=self.relative_high_quota,
                sampling_seed=self.sampling_seed,
                text_encoder_config=self.text_encoder_config,
                encoding_cache_root=self.encoding_cache_root,
                processed_root=self.processed_root,
                sample_indices=self.sample_indices,
                asm_chunk_files=self.asm_chunk_files,
            )

    def set_epoch(self, epoch: int) -> None:
        if self._dataset is None:
            self.setup("fit")
        self._dataset.resample(epoch)

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
