#!/usr/bin/env python3
"""数据加载模块"""

import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Optional, Callable
import lightning as L

from .data_types import DualGraphData


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    支持磁盘持久化，数据按索引存储在 processed_dir 中。

    Args:
        root: 数据集根目录，包含 raw/ 和 processed/ 子目录
        data_list: DualGraphData 对象列表（首次运行时提供）
        transform: 每次获取数据时应用的变换
        pre_transform: 处理前应用的变换（保存到磁盘）
        pre_filter: 处理前的过滤函数

    目录结构:
        root/
        ├── raw/              # 原始文件（可选）
        └── processed/        # 处理后的 .pt 文件
            ├── data_0.pt
            ├── data_1.pt
            └── ...

    Example:
        >>> # 首次创建数据集（处理并保存）
        >>> data_list = [DualGraphData(...), DualGraphData(...)]
        >>> dataset = DualGraphDataset(root='./data', data_list=data_list)

        >>> # 后续加载（从磁盘读取）
        >>> dataset = DualGraphDataset(root='./data')
        >>> len(dataset)
        2
        >>> dataset[0]
        DualGraphData(...)
    """

    def __init__(
        self,
        root: str,
        data_list: Optional[List[DualGraphData]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self._data_list = data_list
        self._num_samples: Optional[int] = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """原始文件列表（无需原始文件，直接从 data_list 处理）"""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """处理后的文件列表"""
        # 动态读取 processed_dir 中的文件
        if self._num_samples is not None:
            return [f"data_{i}.pt" for i in range(self._num_samples)]

        # 检查已存在的文件
        if osp.exists(self.processed_dir):
            files = sorted(
                [
                    f
                    for f in os.listdir(self.processed_dir)
                    if f.startswith("data_") and f.endswith(".pt")
                ]
            )
            if files:
                self._num_samples = len(files)
                return files

        # 如果提供了 data_list，返回预期文件名
        if self._data_list is not None:
            self._num_samples = len(self._data_list)
            return [f"data_{i}.pt" for i in range(self._num_samples)]

        return []

    def download(self):
        """无需下载（数据通过 data_list 参数提供）"""
        pass

    def process(self):
        """处理并保存数据到磁盘"""
        if self._data_list is None:
            return

        for idx, data in enumerate(self._data_list):
            # 应用预过滤
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            # 应用预变换
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # 保存到磁盘
            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))

        self._num_samples = len(self._data_list)
        # 清理内存中的数据列表
        self._data_list = None

    def len(self) -> int:
        """返回数据集大小"""
        if self._num_samples is not None:
            return self._num_samples

        # 从磁盘统计文件数量
        if osp.exists(self.processed_dir):
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith("data_") and f.endswith(".pt")
            ]
            self._num_samples = len(files)
            return self._num_samples

        return 0

    def get(self, idx: int) -> DualGraphData:
        """从磁盘加载指定索引的数据"""
        data = torch.load(
            osp.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False
        )
        return data


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        root: str,
        train_data: Optional[List[DualGraphData]] = None,
        val_data: Optional[List[DualGraphData]] = None,
        test_data: Optional[List[DualGraphData]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: 数据集根目录
            train_data: 训练数据列表（首次运行时提供）
            val_data: 验证数据列表（首次运行时提供）
            test_data: 测试数据列表（首次运行时提供）
            batch_size: 批大小
            num_workers: 数据加载线程数
            transform: 数据变换
        """
        super().__init__()
        self.root = root
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """初始化数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = DualGraphDataset(
                root=osp.join(self.root, "train"),
                data_list=self.train_data,
                transform=self.transform,
            )
            if self.val_data is not None or osp.exists(osp.join(self.root, "val", "processed")):
                self.val_dataset = DualGraphDataset(
                    root=osp.join(self.root, "val"),
                    data_list=self.val_data,
                    transform=self.transform,
                )

        if stage == "test" or stage is None:
            if self.test_data is not None or osp.exists(osp.join(self.root, "test", "processed")):
                self.test_dataset = DualGraphDataset(
                    root=osp.join(self.root, "test"),
                    data_list=self.test_data,
                    transform=self.transform,
                )

    def _create_loader(self, dataset, shuffle: bool) -> PyGDataLoader:
        """创建 PyG DataLoader"""
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            follow_batch=["asm_node_type"],  # 为 ASM 图生成 batch 索引
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> PyGDataLoader:
        return self._create_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[PyGDataLoader]:
        if self.val_dataset is None:
            return None
        return self._create_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[PyGDataLoader]:
        if self.test_dataset is None:
            return None
        return self._create_loader(self.test_dataset, shuffle=False)
