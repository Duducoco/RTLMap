#!/usr/bin/env python3
"""
ASM CDFG 模块 - RISC-V 汇编代码到控制数据流图的转换

支持指令集：RV32IMFC + Xpulp（兼容 CV32E40P）
"""

from .data_types import (
    # 指令相关
    InstrFormat,
    InstrCategory,
    Instruction,
    BasicBlock,
    # CDFG 数据结构
    AsmNodeType,
    AsmEdgeType,
    AsmNode,
    AsmEdge,
    AsmCDFG,
    # 寄存器映射
    REGISTER_ABI_TO_X,
    FLOAT_REGISTER_ABI_TO_F,
)
from .classifier import InstructionClassifier
from .parser import AsmParser
from .bb_builder import BasicBlockBuilder
from .extractor import AsmCDFGExtractor
from .visualizer import AsmCDFGVisualizer

__all__ = [
    # 指令类型定义
    "InstrFormat",
    "InstrCategory",
    "Instruction",
    "BasicBlock",
    # CDFG 数据结构
    "AsmNodeType",
    "AsmEdgeType",
    "AsmNode",
    "AsmEdge",
    "AsmCDFG",
    # 寄存器映射
    "REGISTER_ABI_TO_X",
    "FLOAT_REGISTER_ABI_TO_F",
    # 功能类
    "InstructionClassifier",
    "AsmParser",
    "BasicBlockBuilder",
    "AsmCDFGExtractor",
    # 可视化
    "AsmCDFGVisualizer",
]
