#!/usr/bin/env python3
"""
CDFG 基础类型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from collections import defaultdict


# Cell 类型词汇表：cell_type 字符串 -> 整数索引
CELL_TYPE_VOCAB = {
    # 特殊类型（端口/常量）
    "INPUT": 0,
    "OUTPUT": 1,
    "CONSTANT": 2,
    # 控制类（MUX）
    "$mux": 3,
    "$pmux": 4,
    "$bmux": 5,
    "$demux": 6,
    "$tribuf": 7,
    # 时序类
    "$dff": 8,
    "$dffe": 9,
    "$adff": 10,
    "$adffe": 11,
    "$asdff": 12,
    "$asdffe": 13,
    "$sdff": 14,
    "$sdffe": 15,
    "$sdffce": 16,
    "$dlatch": 17,
    "$adlatch": 18,
    "$dlatchsr": 19,
    "$sr": 20,
    "$ff": 21,
    "$aldff": 22,
    "$aldffe": 23,
    # 存储器类
    "$mem": 24,
    "$mem_v2": 25,
    "$memrd": 26,
    "$memrd_v2": 27,
    "$memwr": 28,
    "$memwr_v2": 29,
    "$meminit": 30,
    "$meminit_v2": 31,
    # 算术类
    "$add": 32,
    "$sub": 33,
    "$mul": 34,
    "$div": 35,
    "$mod": 36,
    "$divfloor": 37,
    "$modfloor": 38,
    "$pow": 39,
    "$neg": 40,
    "$pos": 41,
    # 比较类
    "$eq": 42,
    "$ne": 43,
    "$lt": 44,
    "$le": 45,
    "$gt": 46,
    "$ge": 47,
    "$eqx": 48,
    "$nex": 49,
    # 逻辑类
    "$and": 50,
    "$or": 51,
    "$xor": 52,
    "$xnor": 53,
    "$not": 54,
    "$reduce_and": 55,
    "$reduce_or": 56,
    "$reduce_xor": 57,
    "$reduce_xnor": 58,
    "$reduce_bool": 59,
    "$logic_and": 60,
    "$logic_or": 61,
    "$logic_not": 62,
    # 移位类
    "$shl": 63,
    "$shr": 64,
    "$sshl": 65,
    "$sshr": 66,
    "$shift": 67,
    "$shiftx": 68,
    # 其他组合逻辑
    "$concat": 69,
    "$slice": 70,
    "$lut": 71,
    "$sop": 72,
    # 未知类型（兜底）
    "UNKNOWN": 73,
}

NUM_CELL_TYPES = len(CELL_TYPE_VOCAB)  # 74


def get_cell_type_index(cell_type: str) -> int:
    """将 cell_type 字符串转换为索引，未知类型返回 UNKNOWN 索引"""
    return CELL_TYPE_VOCAB.get(cell_type, CELL_TYPE_VOCAB["UNKNOWN"])


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
