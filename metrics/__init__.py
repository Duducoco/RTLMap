#!/usr/bin/env python3
"""指标模块

提供两类任务的评估指标：
- CoverageRegressionMetrics: 覆盖率回归指标
- ContrastiveMetrics: 对比学习对齐指标
"""

from .regression import CoverageRegressionMetrics
from .contrastive import ContrastiveMetrics

__all__ = [
    "CoverageRegressionMetrics",
    "ContrastiveMetrics",
]
