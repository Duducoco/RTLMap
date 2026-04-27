#!/usr/bin/env python3
"""对比学习数据管道 — 按同一 RTL 模块分组采样对比对"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.loader import DataLoader as PyGDataLoader

from .coverage_similarity import compute_edge_jaccard
from .data_types import DualGraphData, ContrastivePairBatch

logger = logging.getLogger(__name__)


@dataclass
class PairInfo:
    idx_a: int
    idx_b: int
    jaccard: float


class ContrastivePairDataset(Dataset):
    """对比学习对数据集 — 返回 (data_a, data_b, similarity)

    构建流程：
    1. 扫描基础数据集中所有样本的 _rtl_file，按 RTL 模块名分组
    2. 在同一模块组内枚举所有样本对，预计算 Jaccard 相似度
    3. 每个 epoch 随机采样 pairs_per_epoch 个对

    注意：_rtl_file 在 base_dataset.get() 合并图结构后会被删除，
    因此分组时需要从原始 .pt 文件读取 _rtl_file。
    """

    def __init__(
        self,
        base_dataset,
        pairs_per_epoch: int = 512,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.seed = seed
        self._epoch = 0
        self._module_groups: dict[str, list[int]] = {}
        self._all_pairs: list[PairInfo] = []
        self._epoch_pairs: list[PairInfo] = []

        self._build_groups()
        self._precompute_all_pairs()
        self._reshuffle_epoch_pairs()

    def _build_groups(self):
        """按 RTL 模块名分组样本索引

        直接从磁盘 .pt 文件读取 _rtl_file（未合并图结构前），
        因为 get() 会删除 _rtl_file。
        """
        processed_dir = Path(self.base_dataset.processed_dir)
        module_groups: dict[str, list[int]] = {}
        num_samples = len(self.base_dataset)

        for idx in range(num_samples):
            if self.base_dataset._idx_to_file:
                filename = self.base_dataset._idx_to_file[idx]
            else:
                filename = f"data_{idx}.pt"
            pt_path = processed_dir / filename
            if not pt_path.exists():
                continue
            # 从原始 .pt 读取 _rtl_file（不触发图结构合并）
            raw = torch.load(pt_path, weights_only=False)
            rtl_file = getattr(raw, "_rtl_file", None)
            if rtl_file is None:
                continue
            module_name = rtl_file.replace("rtl_", "").replace(".pt", "")
            module_groups.setdefault(module_name, []).append(idx)
            del raw

        # 过滤掉只有单个样本的模块
        self._module_groups = {
            k: v for k, v in module_groups.items() if len(v) >= 2
        }
        logger.info(
            "对比分组: %d 个 RTL 模块有 >= 2 个样本（共 %d 个模块）",
            len(self._module_groups),
            len(module_groups),
        )

    def _precompute_all_pairs(self):
        """预计算所有组内对的 Jaccard 相似度"""
        self._all_pairs = []
        for module_name, indices in self._module_groups.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_a, idx_b = indices[i], indices[j]
                    data_a = self.base_dataset[idx_a]
                    data_b = self.base_dataset[idx_b]
                    jaccard = compute_edge_jaccard(
                        data_a.edge_labels, data_b.edge_labels
                    )
                    self._all_pairs.append(PairInfo(idx_a, idx_b, jaccard))

        logger.info("预计算完成: %d 个对比对", len(self._all_pairs))

    def _reshuffle_epoch_pairs(self):
        """为当前 epoch 随机采样 pairs_per_epoch 个对"""
        rng = random.Random(self.seed + self._epoch)
        n = min(self.pairs_per_epoch, len(self._all_pairs))
        self._epoch_pairs = rng.sample(self._all_pairs, n)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._reshuffle_epoch_pairs()

    def __len__(self) -> int:
        return len(self._epoch_pairs)

    def __getitem__(self, idx: int):
        pair = self._epoch_pairs[idx]
        data_a = self.base_dataset[pair.idx_a]
        data_b = self.base_dataset[pair.idx_b]
        return data_a, data_b, pair.jaccard


def contrastive_pair_collate(pair_list) -> ContrastivePairBatch:
    """将对比对列表 collate 为 ContrastivePairBatch

    分别用 PyG 的 from_data_list 批量化两侧数据。
    """
    items_a, items_b, similarities = zip(*pair_list)

    batch_a = DualGraphData.from_data_list(
        list(items_a), follow_list=["asm_node_type"]
    )
    batch_b = DualGraphData.from_data_list(
        list(items_b), follow_list=["asm_node_type"]
    )
    similarity = torch.tensor(similarities, dtype=torch.float32)

    return ContrastivePairBatch(
        batch_a=batch_a, batch_b=batch_b, similarity=similarity
    )


class ContrastivePairDataModule:
    """对比学习数据模块 — 包装基础 DataModule 提供对比 DataLoader"""

    def __init__(
        self,
        base_dataset,
        pairs_per_epoch: int = 512,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pair_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.pair_dataset = ContrastivePairDataset(
            base_dataset=self.base_dataset,
            pairs_per_epoch=self.pairs_per_epoch,
            seed=self.seed,
        )

    def contrastive_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pair_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=contrastive_pair_collate,
            pin_memory=True,
        )