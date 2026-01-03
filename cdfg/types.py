#!/usr/bin/env python3
"""
CDFG 基础类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from collections import defaultdict


class NodeType(Enum):
    """节点类型枚举"""

    INPUT = auto()  # 输入端口
    OUTPUT = auto()  # 输出端口
    CONSTANT = auto()  # 常数
    COMBINATIONAL = auto()  # 组合逻辑
    SEQUENTIAL = auto()  # 时序逻辑
    MEMORY = auto()  # 存储器
    MUX = auto()  # 多路选择器（控制节点）
    ARITHMETIC = auto()  # 算术运算
    LOGIC = auto()  # 逻辑运算
    COMPARE = auto()  # 比较运算
    SHIFT = auto()  # 移位运算
    UNKNOWN = auto()  # 未知类型


class EdgeType(Enum):
    """边类型枚举"""

    DATA = auto()  # 数据流
    CONTROL = auto()  # 控制流
    CLOCK = auto()  # 时钟
    RESET = auto()  # 复位
    ENABLE = auto()  # 使能


@dataclass
class Node:
    """图节点"""

    id: str
    name: str
    node_type: NodeType
    cell_type: str = ""
    width: int = 1
    parameters: Dict = field(default_factory=dict)
    attributes: Dict = field(default_factory=dict)
    input_ports: List[str] = field(default_factory=list)
    output_ports: List[str] = field(default_factory=list)
    # 覆盖率标注相关字段
    source_line: int = 0  # RTL 源码行号（从 src 属性解析）
    source_file: str = ""  # RTL 源文件名
    stmt_start_line: Optional[int] = None  # 语句起始行号（用于 CASE 语句匹配）


@dataclass
class Edge:
    """图边"""

    source: str  # 源节点 ID
    target: str  # 目标节点 ID
    source_port: str  # 源端口名
    target_port: str  # 目标端口名
    edge_type: EdgeType  # 边类型
    bits: List[int] = field(default_factory=list)  # 涉及的 bit 编号
    width: int = 1  # 信号宽度
    # 覆盖率标注相关字段
    source_line: int = 0  # RTL 源码行号
    branch_index: int = -1  # 分支索引（if-else-if 链中的位置，-1 表示不适用）
    coverage_label: int = -1  # 覆盖标签：-1=不适用, 0=未覆盖, 1=已覆盖
    coverage_type: str = (
        ""  # 覆盖类型："control", "data_true", "data_false", "" 表示不适用
    )


@dataclass
class CDFG:
    """控制数据流图"""

    module_name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # 索引结构
    bit_to_driver: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    bit_to_consumers: Dict[int, List[Tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
