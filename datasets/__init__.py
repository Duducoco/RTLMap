#!/usr/bin/env python3
"""数据集模块

提供双图数据类型和数据加载器：
- DualGraphData: 双图数据容器（PyG Data 子类）
- DualGraphDataset: PyG Dataset 封装（磁盘持久化）
- DualGraphDataModule: Lightning DataModule 封装
"""

from .data_types import DualGraphData, ContrastivePairBatch
from .datamodule import DualGraphDataset, DualGraphDataModule
from .coverage_similarity import compute_edge_jaccard

__all__ = [
    "DualGraphData",
    "DualGraphDataset",
    "DualGraphDataModule",
    "ContrastivePairBatch",
    "compute_edge_jaccard",
]
