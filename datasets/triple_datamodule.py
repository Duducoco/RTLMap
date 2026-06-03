#!/usr/bin/env python3
"""联合三元组对比学习数据管线

从 manifest.json/contrastive_samples.jsonlines 加载三元组 (test_a, test_b, merged)，
每条样本包含三张 DualGraphData（共享图拓扑，仅 edge_labels/y 不同），
用于单次 forward 内同时计算三路覆盖率预测损失和对比损失。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from .data_types import (
    DualGraphData,
    ContrastiveTripleBatch,
)
from .coverage_similarity import compute_mean_rate_jaccard
from .graph_builders import (
    build_asm_graph as _build_asm_graph,
    build_rtl_structure as _build_rtl_structure,
    empty_asm_graph as _empty_asm_graph,
)
from .manifest import (
    DatasetDirInput,
    normalize_dataset_dirs as _normalize_dataset_dirs,
)
from .targets import extract_targets_from_sample as _extract_targets_from_sample

logger = logging.getLogger(__name__)


def _sample_with_targets(entry: dict, targets: dict) -> dict:
    return {
        "targets": targets,
        "_rtl_path": entry["_rtl_path"],
    }


class ContrastiveTripleDataset(Dataset):
    """三元组对比学习数据集

    从 manifest.json 指向的 contrastive_samples.jsonlines 加载三元组。
    RTL 图结构（节点/边拓扑）在同一模块的所有测试中相同，边标签随测试不同。
    为降低磁盘读取，图结构只加载一次，三路数据共享结构、各自持有自己的标签。

    每条样本返回 (data_a, data_b, data_merged, similarity)：
    - data_a / data_b：完整双图数据（RTL 结构 + ASM + edge_labels_a / edge_labels_b）
    - data_merged：只含 RTL 部分（merged 不含 ASM），edge_labels 来自 merged 覆盖
    - similarity：data_a 与 data_b 的覆盖率 Jaccard 平均相似度（float）
    """

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        index_file: Optional[str | Path] = None,
        processed_dir: Optional[str | Path] = None,
    ):
        self.dataset_dirs = _normalize_dataset_dirs(dataset_dir)
        self.index_file = Path(index_file) if index_file else None
        # processed_dir 用于加载预处理好的 RTL/ASM .pt 图结构缓存（可选）
        self.processed_dir = Path(processed_dir) if processed_dir else None

        self._entries: list[dict] = []
        self._load_entries()

    def _load_entries(self):
        if not self.dataset_dirs:
            raise ValueError("未指定 dataset_dir")
        if self.index_file is not None:
            if len(self.dataset_dirs) != 1:
                raise ValueError("多 dataset_dir 模式不支持单个 index_file 覆盖")
            self._load_index(self.dataset_dirs[0], self.index_file)
        else:
            for dataset_dir in self.dataset_dirs:
                self._load_index(dataset_dir, self._default_index_file(dataset_dir))
        logger.info("ContrastiveTripleDataset: 加载 %d 条三元组", len(self._entries))

    def _load_index(self, dataset_dir: Path, index_file: Path):
        if not index_file.exists():
            raise FileNotFoundError(
                f"contrastive_samples.jsonlines 不存在: {index_file}"
            )
        with open(index_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("schema_version") != "contrastive_sample.v1":
                        raise ValueError(
                            f"不支持的样本 schema: {entry.get('schema_version')!r}"
                        )
                    entry["_rtl_path"] = str(dataset_dir / entry["rtl_graph"])
                    entry["_asm_a_path"] = (
                        str(dataset_dir / entry["asm_a"])
                        if entry.get("asm_a")
                        else None
                    )
                    entry["_asm_b_path"] = (
                        str(dataset_dir / entry["asm_b"])
                        if entry.get("asm_b")
                        else None
                    )
                    self._entries.append(entry)

    def _default_index_file(self, dataset_dir: Path) -> Path:
        manifest_file = dataset_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"manifest.json 不存在: {manifest_file}")
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != "contrastive_dataset.v1":
            raise ValueError(
                f"不支持的对比数据集 schema: {manifest.get('schema_version')!r}"
            )
        return dataset_dir / manifest.get("samples", "contrastive_samples.jsonlines")

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(
        self, idx: int
    ) -> tuple[DualGraphData, DualGraphData, DualGraphData, float]:
        entry = self._entries[idx]

        asm_a_path = entry.get("_asm_a_path")
        asm_b_path = entry.get("_asm_b_path")

        with open(entry["_rtl_path"], encoding="utf-8") as f:
            rtl_json = json.load(f)
        rtl_struct = _build_rtl_structure(rtl_json)
        valid_edge_mask = rtl_struct["_valid_edge_mask"]
        rtl_struct.pop("_valid_edge_mask", None)
        labels_a, y_a = _extract_targets_from_sample(
            _sample_with_targets(entry, entry["targets"]["a"]),
            valid_edge_mask,
        )
        labels_b, y_b = _extract_targets_from_sample(
            _sample_with_targets(entry, entry["targets"]["b"]),
            valid_edge_mask,
        )
        labels_merged, y_merged = _extract_targets_from_sample(
            _sample_with_targets(entry, entry["targets"]["merged"]),
            valid_edge_mask,
        )

        # 加载 ASM 图
        asm_a = _load_asm_graph(asm_a_path) if asm_a_path else _empty_asm()
        asm_b = _load_asm_graph(asm_b_path) if asm_b_path else _empty_asm()

        data_a = DualGraphData(
            **rtl_struct,
            **asm_a,
            edge_labels=labels_a,
            y=y_a,
        )
        data_b = DualGraphData(
            **rtl_struct,
            **asm_b,
            edge_labels=labels_b,
            y=y_b,
        )
        data_merged = DualGraphData(
            **rtl_struct,
            **_empty_asm(),
            edge_labels=labels_merged,
            y=y_merged,
        )

        similarity = compute_mean_rate_jaccard(
            y_a[0].tolist(),
            y_b[0].tolist(),
            y_merged[0].tolist(),
        )
        return data_a, data_b, data_merged, similarity


def contrastive_triple_collate(
    triple_list: list[tuple[DualGraphData, DualGraphData, DualGraphData, float]],
) -> ContrastiveTripleBatch:
    """将三元组列表 collate 为 ContrastiveTripleBatch"""
    items_a, items_b, items_merged, sims = zip(*triple_list)

    follow = ["asm_node_type"]
    batch_a = Batch.from_data_list(list(items_a), follow_batch=follow)
    batch_b = Batch.from_data_list(list(items_b), follow_batch=follow)
    batch_merged = Batch.from_data_list(list(items_merged), follow_batch=follow)
    similarity = torch.tensor(sims, dtype=torch.float32)

    return ContrastiveTripleBatch(
        batch_a=batch_a,
        batch_b=batch_b,
        batch_merged=batch_merged,
        similarity=similarity,
    )


class ContrastiveTripleDataModule:
    """三元组对比学习数据模块

    仅提供 train_dataloader；验证/测试沿用 DualGraphDataModule。
    """

    def __init__(
        self,
        dataset_dir: DatasetDirInput,
        index_file: Optional[str | Path] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = True,
    ):
        self.dataset_dir = dataset_dir
        self.index_file = index_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self._dataset: Optional[ContrastiveTripleDataset] = None

    def setup(self, stage: Optional[str] = None):
        if self._dataset is None:
            self._dataset = ContrastiveTripleDataset(
                dataset_dir=self.dataset_dir,
                index_file=self.index_file,
            )

    def train_dataloader(self) -> DataLoader:
        if self._dataset is None:
            self.setup()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=contrastive_triple_collate,
            pin_memory=True,
        )

    def __len__(self) -> int:
        if self._dataset is None:
            self.setup()
        return len(self._dataset)


def _load_asm_graph(asm_json_path: str) -> dict:
    """加载 ASM 图，复用 datamodule._build_asm_graph（编码回退零向量）"""
    return _build_asm_graph(asm_json_path, asm_encoding=None)


def _empty_asm() -> dict:
    """返回空 ASM 图（合并覆盖或无 ASM 时使用）"""
    return _empty_asm_graph()
