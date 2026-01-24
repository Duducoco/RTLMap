#!/usr/bin/env python3
"""
RTL CDFG 模块 - RTL 控制数据流图提取和分析

提供从 RTLIL JSON 提取、分析、可视化和导出 CDFG 的功能。
"""

from .data_types import NodeType, EdgeType, Node, Edge, CDFG
from .classifier import CellClassifier
from .extractor import CDFGExtractor
from .analyzer import CDFGAnalyzer
from .visualizer import CDFGVisualizer
from .exporter import CDFGExporter

__all__ = [
    # 类型定义
    "NodeType",
    "EdgeType",
    "Node",
    "Edge",
    "CDFG",
    # 功能类
    "CellClassifier",
    "CDFGExtractor",
    "CDFGAnalyzer",
    "CDFGVisualizer",
    "CDFGExporter",
]
