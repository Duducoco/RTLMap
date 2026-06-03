#!/usr/bin/env python3
"""指标模块

提供三类任务的评估指标：
- EdgeClassificationMetrics: 边分类（覆盖/未覆盖）指标
- CoverageRegressionMetrics: 覆盖率回归指标
- ContrastiveMetrics: 对比学习对齐指标
"""

from .classification import EdgeClassificationMetrics
from .regression import CoverageRegressionMetrics
from .contrastive import ContrastiveMetrics

__all__ = [
    "EdgeClassificationMetrics",
    "CoverageRegressionMetrics",
    "ContrastiveMetrics",
]
