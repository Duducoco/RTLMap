#!/usr/bin/env python3
"""
Cell 类型分类器
"""

import re
from typing import Tuple, Optional

from .types import NodeType, EdgeType


class CellClassifier:
    """Cell 类型分类器"""

    # 控制类（MUX）
    CONTROL_CELLS = {"$mux", "$pmux", "$bmux", "$demux", "$tribuf"}

    # 时序类
    SEQUENTIAL_CELLS = {
        "$dff",
        "$dffe",
        "$adff",
        "$adffe",
        "$asdff",
        "$asdffe",
        "$sdff",
        "$sdffe",
        "$sdffce",
        "$dlatch",
        "$adlatch",
        "$dlatchsr",
        "$sr",
        "$ff",
        "$aldff",
        "$aldffe",
    }

    # 存储器类
    MEMORY_CELLS = {
        "$mem",
        "$mem_v2",
        "$memrd",
        "$memrd_v2",
        "$memwr",
        "$memwr_v2",
        "$meminit",
        "$meminit_v2",
    }

    # 算术类
    ARITHMETIC_CELLS = {
        "$add",
        "$sub",
        "$mul",
        "$div",
        "$mod",
        "$divfloor",
        "$modfloor",
        "$pow",
        "$neg",
        "$pos",
    }

    # 比较类
    COMPARE_CELLS = {"$eq", "$ne", "$lt", "$le", "$gt", "$ge", "$eqx", "$nex"}

    # 逻辑类
    LOGIC_CELLS = {
        "$and",
        "$or",
        "$xor",
        "$xnor",
        "$not",
        "$reduce_and",
        "$reduce_or",
        "$reduce_xor",
        "$reduce_xnor",
        "$reduce_bool",
        "$logic_and",
        "$logic_or",
        "$logic_not",
    }

    # 移位类
    SHIFT_CELLS = {"$shl", "$shr", "$sshl", "$sshr", "$shift", "$shiftx"}

    # 控制信号端口名
    CONTROL_PORTS = {"S", "EN", "ARST", "SRST", "CLR", "SET", "CE", "ALOAD"}
    CLOCK_PORTS = {"CLK", "C"}
    RESET_PORTS = {"ARST", "SRST", "RST", "CLR", "R"}
    ENABLE_PORTS = {"EN", "CE", "E", "ALOAD"}

    @classmethod
    def classify(cls, cell_type: str) -> NodeType:
        """根据 cell 类型返回节点类型"""
        if cell_type in cls.CONTROL_CELLS:
            return NodeType.MUX
        elif cell_type in cls.SEQUENTIAL_CELLS:
            return NodeType.SEQUENTIAL
        elif cell_type in cls.MEMORY_CELLS:
            return NodeType.MEMORY
        elif cell_type in cls.ARITHMETIC_CELLS:
            return NodeType.ARITHMETIC
        elif cell_type in cls.COMPARE_CELLS:
            return NodeType.COMPARE
        elif cell_type in cls.LOGIC_CELLS:
            return NodeType.LOGIC
        elif cell_type in cls.SHIFT_CELLS:
            return NodeType.SHIFT
        else:
            return NodeType.COMBINATIONAL

    @classmethod
    def get_edge_type(cls, port_name: str) -> EdgeType:
        """根据端口名判断边类型"""
        if port_name in cls.CLOCK_PORTS:
            return EdgeType.CLOCK
        elif port_name in cls.RESET_PORTS:
            return EdgeType.RESET
        elif port_name in cls.ENABLE_PORTS:
            return EdgeType.ENABLE
        elif port_name in cls.CONTROL_PORTS:
            return EdgeType.CONTROL
        else:
            return EdgeType.DATA

    @classmethod
    def is_output_port(cls, cell_type: str, port_name: str) -> bool:
        """判断是否为输出端口"""
        output_ports = {"Y", "Q", "DATA", "RD_DATA", "DOUT", "CO", "X"}

        # 特殊处理某些 cell
        if cell_type in cls.MEMORY_CELLS:
            if "rd" in cell_type.lower():
                return port_name in {"DATA", "RD_DATA"}

        return port_name in output_ports

    @staticmethod
    def parse_source_location(src_attr: str) -> Tuple[str, int, Optional[int]]:
        """
        解析 src 属性，提取源文件名、行号和语句起始行号

        src 属性格式示例：
        - 简单格式: "..\\designs\\cv32e40p\\rtl\\cv32e40p_int_controller.sv:96.29-96.51"
        - 复合格式: "file.sv:177.14-177.73|file.sv:173.5-178.12" (多位置用 | 分隔)

        复合格式中，第一部分是具体分支项的位置，第二部分是整个语句（如 case/if）的范围。
        对于 CASE 语句，覆盖率报告记录的是 case 语句本身的行号，
        而 Yosys 生成的 MUX 节点记录的是具体 case 分支项的行号。

        Args:
            src_attr: Yosys JSON 中的 src 属性值

        Returns:
            (文件名, 行号, 语句起始行号) 元组
            语句起始行号用于 CASE 语句匹配，如果不存在则为 None
            如果解析失败返回 ("", 0, None)
        """
        if not src_attr:
            return "", 0, None

        parts = src_attr.split("|")
        first_loc = parts[0].strip()

        # 匹配格式: 文件路径:行号.列号 或 文件路径:行号.列号-行号.列号
        match = re.match(r"^(.+?):(\d+)\.", first_loc)
        if not match:
            return "", 0, None

        file_path = match.group(1)
        line_no = int(match.group(2))
        # 提取文件名（去掉路径）
        file_name = file_path.replace("\\", "/").split("/")[-1]

        # 尝试从第二部分提取语句起始行号（用于 case/if 语句匹配）
        stmt_start_line = None
        if len(parts) > 1:
            second_loc = parts[1].strip()
            range_match = re.match(r"^.+?:(\d+)\.", second_loc)
            if range_match:
                stmt_start_line = int(range_match.group(1))

        return file_name, line_no, stmt_start_line
