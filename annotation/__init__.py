#!/usr/bin/env python3
"""
Annotation 模块 - 覆盖率解析和标注

提供 URG HTML 覆盖率报告解析和 CDFG 边标注功能。
"""

from .parser import CoverageParser, BranchCoverage, BranchStatus
from .annotator import CoverageAnnotator, AnnotationStats, annotate_cdfg_with_coverage

__all__ = [
    # 解析器
    "CoverageParser",
    "BranchCoverage",
    "BranchStatus",
    # 标注器
    "CoverageAnnotator",
    "AnnotationStats",
    "annotate_cdfg_with_coverage",
]
