#!/usr/bin/env python3
"""RTL CDFG 词汇表与边类型定义（供数据集加载使用）"""

from enum import Enum, auto


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
    return CELL_TYPE_VOCAB.get(cell_type, CELL_TYPE_VOCAB["UNKNOWN"])


class NodeType(Enum):
    """节点类型枚举"""

    INPUT = auto()
    OUTPUT = auto()
    CONSTANT = auto()
    COMBINATIONAL = auto()
    SEQUENTIAL = auto()
    MEMORY = auto()
    MUX = auto()
    ARITHMETIC = auto()
    LOGIC = auto()
    COMPARE = auto()
    SHIFT = auto()
    UNKNOWN = auto()


class EdgeType(Enum):
    DATA = auto()
    DATA_TRUE = auto()
    DATA_FALSE = auto()
    CONTROL = auto()
    CLOCK = auto()
    RESET = auto()
    ENABLE = auto()


NUM_NODE_TYPES = len(NodeType)

_CELL_TO_NODETYPE_MAP = {
    "INPUT":    NodeType.INPUT,
    "OUTPUT":   NodeType.OUTPUT,
    "CONSTANT": NodeType.CONSTANT,
    "$mux": NodeType.MUX,   "$pmux": NodeType.MUX,
    "$bmux": NodeType.MUX,  "$demux": NodeType.MUX,
    "$tribuf": NodeType.COMBINATIONAL,
    "$dff": NodeType.SEQUENTIAL,    "$dffe": NodeType.SEQUENTIAL,
    "$adff": NodeType.SEQUENTIAL,   "$adffe": NodeType.SEQUENTIAL,
    "$asdff": NodeType.SEQUENTIAL,  "$asdffe": NodeType.SEQUENTIAL,
    "$sdff": NodeType.SEQUENTIAL,   "$sdffe": NodeType.SEQUENTIAL,
    "$sdffce": NodeType.SEQUENTIAL, "$dlatch": NodeType.SEQUENTIAL,
    "$adlatch": NodeType.SEQUENTIAL,"$dlatchsr": NodeType.SEQUENTIAL,
    "$sr": NodeType.SEQUENTIAL,     "$ff": NodeType.SEQUENTIAL,
    "$aldff": NodeType.SEQUENTIAL,  "$aldffe": NodeType.SEQUENTIAL,
    "$mem": NodeType.MEMORY,      "$mem_v2": NodeType.MEMORY,
    "$memrd": NodeType.MEMORY,    "$memrd_v2": NodeType.MEMORY,
    "$memwr": NodeType.MEMORY,    "$memwr_v2": NodeType.MEMORY,
    "$meminit": NodeType.MEMORY,  "$meminit_v2": NodeType.MEMORY,
    "$add": NodeType.ARITHMETIC,  "$sub": NodeType.ARITHMETIC,
    "$mul": NodeType.ARITHMETIC,  "$div": NodeType.ARITHMETIC,
    "$mod": NodeType.ARITHMETIC,  "$divfloor": NodeType.ARITHMETIC,
    "$modfloor": NodeType.ARITHMETIC, "$pow": NodeType.ARITHMETIC,
    "$neg": NodeType.ARITHMETIC,  "$pos": NodeType.ARITHMETIC,
    "$eq": NodeType.COMPARE,  "$ne": NodeType.COMPARE,
    "$lt": NodeType.COMPARE,  "$le": NodeType.COMPARE,
    "$gt": NodeType.COMPARE,  "$ge": NodeType.COMPARE,
    "$eqx": NodeType.COMPARE, "$nex": NodeType.COMPARE,
    "$and": NodeType.LOGIC,       "$or": NodeType.LOGIC,
    "$xor": NodeType.LOGIC,       "$xnor": NodeType.LOGIC,
    "$not": NodeType.LOGIC,       "$reduce_and": NodeType.LOGIC,
    "$reduce_or": NodeType.LOGIC, "$reduce_xor": NodeType.LOGIC,
    "$reduce_xnor": NodeType.LOGIC,"$reduce_bool": NodeType.LOGIC,
    "$logic_and": NodeType.LOGIC, "$logic_or": NodeType.LOGIC,
    "$logic_not": NodeType.LOGIC,
    "$shl": NodeType.SHIFT,   "$shr": NodeType.SHIFT,
    "$sshl": NodeType.SHIFT,  "$sshr": NodeType.SHIFT,
    "$shift": NodeType.SHIFT, "$shiftx": NodeType.SHIFT,
    "$concat": NodeType.COMBINATIONAL, "$slice": NodeType.COMBINATIONAL,
    "$lut": NodeType.COMBINATIONAL,    "$sop": NodeType.COMBINATIONAL,
    "UNKNOWN": NodeType.UNKNOWN,
}

CELL_TYPE_TO_NODE_TYPE: list[int] = [
    _CELL_TO_NODETYPE_MAP.get(k, NodeType.UNKNOWN).value - 1
    for k, _ in sorted(CELL_TYPE_VOCAB.items(), key=lambda x: x[1])
]
