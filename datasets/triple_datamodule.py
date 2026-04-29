#!/usr/bin/env python3
"""联合三元组对比学习数据管线

从 contrastive_index.jsonlines 加载三元组 (test_a, test_b, merged)，
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

from .data_types import DualGraphData, ContrastiveTripleBatch, COVERAGE_KEYS
from .coverage_similarity import compute_edge_jaccard
from .datamodule import _build_rtl_structure, _build_asm_graph

logger = logging.getLogger(__name__)


def _extract_rtl_labels_from_json(rtl: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """从已加载 RTL JSON 中提取 edge_labels 和图级覆盖率 y。"""
    nodes = rtl["nodes"]
    node_id_set = {n["id"] for n in nodes}

    elabel_list = []
    for e in rtl["edges"]:
        if e["source"] not in node_id_set or e["target"] not in node_id_set:
            continue
        elabel_list.append(e["coverage_label"])

    y_vals = []
    for key in COVERAGE_KEYS:
        v = rtl.get(key)
        y_vals.append(float("nan") if v is None else float(v) / 100.0)

    return (
        torch.tensor(elabel_list, dtype=torch.long),
        torch.tensor([y_vals], dtype=torch.float),
    )


class ContrastiveTripleDataset(Dataset):
    """三元组对比学习数据集

    从 contrastive_index.jsonlines 每行加载 (rtl_a, asm_a, rtl_b, asm_b, rtl_merged)。
    RTL 图结构（节点/边拓扑）在同一模块的所有测试中相同，边标签随测试不同。
    为降低磁盘读取，图结构只加载一次，三路数据共享结构、各自持有自己的标签。

    每条样本返回 (data_a, data_b, data_merged, similarity)：
    - data_a / data_b：完整双图数据（RTL 结构 + ASM + edge_labels_a / edge_labels_b）
    - data_merged：只含 RTL 部分（merged 不含 ASM），edge_labels 来自 merged 覆盖
    - similarity：data_a 与 data_b 的 edge-level Jaccard 相似度（float）
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        index_file: Optional[str | Path] = None,
        processed_dir: Optional[str | Path] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.index_file = Path(index_file) if index_file else self.dataset_dir / "contrastive_index.jsonlines"
        # processed_dir 用于加载预处理好的 RTL/ASM .pt 图结构缓存（可选）
        self.processed_dir = Path(processed_dir) if processed_dir else None

        self._entries: list[dict] = []
        self._load_index()

    def _load_index(self):
        if not self.index_file.exists():
            raise FileNotFoundError(f"contrastive_index.jsonlines 不存在: {self.index_file}")
        with open(self.index_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._entries.append(json.loads(line))
        logger.info("ContrastiveTripleDataset: 加载 %d 条三元组", len(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> tuple[DualGraphData, DualGraphData, DualGraphData, float]:
        entry = self._entries[idx]
        dd = self.dataset_dir

        rtl_a_path = str(dd / entry["rtl_a"])
        rtl_b_path = str(dd / entry["rtl_b"])
        rtl_merged_path = str(dd / entry["rtl_merged"])
        asm_a_path = str(dd / entry["asm_a"]) if entry.get("asm_a") else None
        asm_b_path = str(dd / entry["asm_b"]) if entry.get("asm_b") else None

        # 一次读取 rtl_a：同时得到图结构与标签
        with open(rtl_a_path, encoding="utf-8") as f:
            rtl_a_json = json.load(f)
        rtl_struct = _build_rtl_structure(rtl_a_json)
        rtl_struct.pop("_valid_edge_mask", None)
        labels_a, y_a = _extract_rtl_labels_from_json(rtl_a_json)

        # 读取 rtl_b / merged 标签
        with open(rtl_b_path, encoding="utf-8") as f:
            rtl_b_json = json.load(f)
        labels_b, y_b = _extract_rtl_labels_from_json(rtl_b_json)

        with open(rtl_merged_path, encoding="utf-8") as f:
            rtl_merged_json = json.load(f)
        labels_merged, y_merged = _extract_rtl_labels_from_json(rtl_merged_json)

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

        similarity = float(compute_edge_jaccard(labels_a, labels_b))
        return data_a, data_b, data_merged, similarity


def contrastive_triple_collate(
    triple_list: list[tuple[DualGraphData, DualGraphData, DualGraphData, float]],
) -> ContrastiveTripleBatch:
    """将三元组列表 collate 为 ContrastiveTripleBatch"""
    items_a, items_b, items_merged, sims = zip(*triple_list)

    follow = ["asm_node_type"]
    batch_a = DualGraphData.from_data_list(list(items_a), follow_batch=follow)
    batch_b = DualGraphData.from_data_list(list(items_b), follow_batch=follow)
    batch_merged = DualGraphData.from_data_list(list(items_merged), follow_batch=follow)
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
        dataset_dir: str | Path,
        index_file: Optional[str | Path] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
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
    return {
        "asm_node_type": torch.zeros(1, dtype=torch.long),
        "asm_instruction_encoding": torch.zeros(1, 256),
        "asm_edge_index": torch.empty((2, 0), dtype=torch.long),
        "asm_edge_type": torch.empty(0, dtype=torch.long),
    }
