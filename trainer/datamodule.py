#!/usr/bin/env python3
"""数据加载模块"""

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Optional, Callable
import lightning as L

from models.data_types import DualGraphData


class DualGraphDataset(Dataset):
    """
    双图数据集 - 继承自 PyG Dataset

    用于存储 DualGraphData 列表，支持 PyG 的 transform。

    Args:
        data_list: DualGraphData 对象列表
        transform: 每次获取数据时应用的变换

    Example:
        >>> data_list = [DualGraphData(...), DualGraphData(...)]
        >>> dataset = DualGraphDataset(data_list)
        >>> len(dataset)
        2
        >>> dataset[0]
        DualGraphData(...)
    """

    def __init__(
        self,
        data_list: List[DualGraphData],
        transform: Optional[Callable] = None
    ):
        self._data_list = list(data_list)
        super().__init__(root=None, transform=transform)

    @property
    def raw_file_names(self) -> List[str]:
        """无原始文件"""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """无处理文件"""
        return []

    def download(self):
        """无需下载"""
        pass

    def process(self):
        """无需处理"""
        pass

    def len(self) -> int:
        """返回数据集大小"""
        return len(self._data_list)

    def get(self, idx: int) -> DualGraphData:
        """获取指定索引的数据"""
        return self._data_list[idx]


class DualGraphDataModule(L.LightningDataModule):
    """双图数据加载模块"""

    def __init__(
        self,
        train_data: List[DualGraphData],
        val_data: Optional[List[DualGraphData]] = None,
        test_data: Optional[List[DualGraphData]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None
    ):
        super().__init__()
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
                self.train_data,
                transform=self.transform
            )
            if self.val_data:
                self.val_dataset = DualGraphDataset(
                    self.val_data,
                    transform=self.transform
                )

        if stage == "test" or stage is None:
            if self.test_data:
                self.test_dataset = DualGraphDataset(
                    self.test_data,
                    transform=self.transform
                )

    def _create_loader(self, dataset, shuffle: bool) -> PyGDataLoader:
        """创建 PyG DataLoader"""
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            follow_batch=['asm_x'],  # 为 ASM 图生成 batch 索引
            pin_memory=True,
            persistent_workers=self.num_workers > 0
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
