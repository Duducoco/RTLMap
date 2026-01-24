#!/usr/bin/env python3
"""数据集模块

提供双图数据集和数据加载器：
- DualGraphDataset: PyG Dataset 封装
- DualGraphDataModule: Lightning DataModule 封装
"""

from .datamodule import DualGraphDataset, DualGraphDataModule

__all__ = [
    "DualGraphDataset",
    "DualGraphDataModule",
]
