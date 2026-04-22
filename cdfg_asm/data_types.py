#!/usr/bin/env python3
"""ASM CDFG 枚举类型定义（供数据集加载使用）"""

from enum import Enum, auto


class AsmNodeType(Enum):
    """汇编 CDFG 节点类型"""

    ENTRY = auto()
    EXIT = auto()
    BRANCH = auto()
    JUMP = auto()
    COMPUTE = auto()
    ARITHMETIC = auto()
    LOGIC = auto()
    SHIFT = auto()
    COMPARE = auto()
    MAC = auto()
    SIMD = auto()
    LOAD = auto()
    STORE = auto()
    MEMORY = auto()
    CSR = auto()
    SYSTEM = auto()
    HWLOOP = auto()
    NOP = auto()
    UNKNOWN = auto()


class AsmEdgeType(Enum):
    """汇编 CDFG 边类型"""

    CONTROL_FLOW = auto()
    BRANCH_TAKEN = auto()
    BRANCH_NOT_TAKEN = auto()
    JUMP = auto()
    CALL = auto()
    RETURN = auto()
    DATA_DEP = auto()
    MEMORY_DEP = auto()
    ANTI_DEP = auto()
    OUTPUT_DEP = auto()
