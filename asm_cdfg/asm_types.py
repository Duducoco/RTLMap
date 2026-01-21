#!/usr/bin/env python3
"""
ASM CDFG 类型定义

定义 RISC-V 汇编解析和 CDFG 提取所需的数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from enum import Enum, auto


class InstrFormat(Enum):
    """RISC-V 指令编码格式"""

    # RV32I 基本格式
    R_TYPE = auto()  # 寄存器-寄存器操作: add, sub, and, or, xor, sll, srl, sra, slt, sltu
    I_TYPE = auto()  # 立即数操作/加载: addi, ori, andi, lb, lh, lw, jalr
    S_TYPE = auto()  # 存储操作: sb, sh, sw
    B_TYPE = auto()  # 分支操作: beq, bne, blt, bge, bltu, bgeu
    U_TYPE = auto()  # 上立即数: lui, auipc
    J_TYPE = auto()  # 跳转: jal

    # RV32C 压缩格式
    CR_TYPE = auto()  # 寄存器型: c.add, c.mv, c.jr, c.jalr
    CI_TYPE = auto()  # 立即数型: c.addi, c.li, c.lui, c.lwsp
    CSS_TYPE = auto()  # 栈存储型: c.swsp
    CIW_TYPE = auto()  # 立即数宽型: c.addi4spn
    CL_TYPE = auto()  # 加载型: c.lw
    CS_TYPE = auto()  # 存储型: c.sw
    CB_TYPE = auto()  # 分支型: c.beqz, c.bnez
    CJ_TYPE = auto()  # 跳转型: c.j, c.jal

    # 特殊格式
    CSR_TYPE = auto()  # CSR 操作: csrrw, csrrs, csrrc, csrrwi, csrrsi, csrrci
    XPULP_TYPE = auto()  # Xpulp 扩展指令
    PSEUDO = auto()  # 伪指令: li, la, mv, not, neg, j, ret
    DIRECTIVE = auto()  # 汇编指令: .section, .globl, .align, .text, .data
    LABEL = auto()  # 标签定义


class InstrCategory(Enum):
    """指令语义类别（用于 NodeType 映射和基本块分析）"""

    ARITHMETIC = auto()  # 算术运算: add, sub, mul, div, lui, auipc
    LOGIC = auto()  # 逻辑运算: and, or, xor, not
    SHIFT = auto()  # 移位运算: sll, srl, sra
    COMPARE = auto()  # 比较运算: slt, sltu, feq, flt, fle
    BRANCH = auto()  # 条件分支（控制流决策点）: beq, bne, blt, bge
    JUMP = auto()  # 无条件跳转: jal, jalr, j, ret
    LOAD = auto()  # 加载: lb, lh, lw, flw
    STORE = auto()  # 存储: sb, sh, sw, fsw
    CSR = auto()  # CSR 操作: csrrw, csrrs, csrrc
    SYSTEM = auto()  # 系统指令: ecall, ebreak, fence, mret, wfi
    NOP = auto()  # 空操作: nop
    MAC = auto()  # 乘累加（Xpulp）: cv.mac, cv.msu
    SIMD = auto()  # SIMD 操作（Xpulp）: cv.add.h, cv.dotsp.h
    HWLOOP = auto()  # 硬件循环（Xpulp）: cv.starti, cv.endi, cv.count
    BITMANIP = auto()  # 位操作（Xpulp）: cv.extract, cv.insert, cv.bclr, cv.bset
    FLOAT = auto()  # 浮点运算（RV32F）: fadd.s, fsub.s, fmul.s
    DIRECTIVE = auto()  # 汇编指令
    UNKNOWN = auto()  # 未知指令


@dataclass
class Instruction:
    """解析后的单条指令"""

    line_no: int  # 源文件行号
    raw_text: str  # 原始文本
    mnemonic: str  # 指令助记符（小写）
    operands: List[str]  # 操作数列表（原始字符串）
    instr_format: InstrFormat  # 指令编码格式
    category: InstrCategory  # 语义类别

    # 解析后的操作数（可选）
    rd: Optional[str] = None  # 目标寄存器
    rs1: Optional[str] = None  # 源寄存器1
    rs2: Optional[str] = None  # 源寄存器2
    rs3: Optional[str] = None  # 源寄存器3（Xpulp MAC 指令用）
    imm: Optional[Union[int, str]] = None  # 立即数（数值或符号）
    label: Optional[str] = None  # 目标标签（分支/跳转目标）

    # 控制流属性
    is_terminator: bool = False  # 是否为基本块终结指令


@dataclass
class BasicBlock:
    """基本块 - 连续执行的指令序列"""

    id: str  # 基本块 ID（格式: bb_N）
    label: Optional[str]  # 标签名（如果有）
    instructions: List[Instruction]  # 包含的指令列表
    start_line: int  # 起始行号
    end_line: int  # 结束行号

    # 控制流图属性
    successors: List[str] = field(default_factory=list)  # 后继基本块 ID
    predecessors: List[str] = field(default_factory=list)  # 前驱基本块 ID

    # 数据流分析（寄存器活跃性）
    defs: Set[str] = field(default_factory=set)  # 定义的寄存器（写入）
    uses: Set[str] = field(default_factory=set)  # 使用的寄存器（读取）

    # 额外属性
    attributes: Dict = field(default_factory=dict)


# 寄存器 ABI 名称到 xN 格式的映射
REGISTER_ABI_TO_X = {
    "zero": "x0",
    "ra": "x1",
    "sp": "x2",
    "gp": "x3",
    "tp": "x4",
    "t0": "x5",
    "t1": "x6",
    "t2": "x7",
    "s0": "x8",
    "fp": "x8",  # frame pointer 是 s0 的别名
    "s1": "x9",
    "a0": "x10",
    "a1": "x11",
    "a2": "x12",
    "a3": "x13",
    "a4": "x14",
    "a5": "x15",
    "a6": "x16",
    "a7": "x17",
    "s2": "x18",
    "s3": "x19",
    "s4": "x20",
    "s5": "x21",
    "s6": "x22",
    "s7": "x23",
    "s8": "x24",
    "s9": "x25",
    "s10": "x26",
    "s11": "x27",
    "t3": "x28",
    "t4": "x29",
    "t5": "x30",
    "t6": "x31",
}

# 浮点寄存器 ABI 名称映射
FLOAT_REGISTER_ABI_TO_F = {
    "ft0": "f0",
    "ft1": "f1",
    "ft2": "f2",
    "ft3": "f3",
    "ft4": "f4",
    "ft5": "f5",
    "ft6": "f6",
    "ft7": "f7",
    "fs0": "f8",
    "fs1": "f9",
    "fa0": "f10",
    "fa1": "f11",
    "fa2": "f12",
    "fa3": "f13",
    "fa4": "f14",
    "fa5": "f15",
    "fa6": "f16",
    "fa7": "f17",
    "fs2": "f18",
    "fs3": "f19",
    "fs4": "f20",
    "fs5": "f21",
    "fs6": "f22",
    "fs7": "f23",
    "fs8": "f24",
    "fs9": "f25",
    "fs10": "f26",
    "fs11": "f27",
    "ft8": "f28",
    "ft9": "f29",
    "ft10": "f30",
    "ft11": "f31",
}


# =============================================================================
# ASM CDFG 专用数据结构（针对汇编代码特点设计）
# =============================================================================


class AsmNodeType(Enum):
    """汇编 CDFG 节点类型（基于基本块特征）"""

    # 控制流节点
    ENTRY = auto()  # 程序入口点
    EXIT = auto()  # 程序出口点（ret, mret, ecall 等）
    BRANCH = auto()  # 条件分支节点（beq, bne, blt 等）
    JUMP = auto()  # 无条件跳转节点（jal, j 等）

    # 计算节点
    COMPUTE = auto()  # 通用计算块（混合运算）
    ARITHMETIC = auto()  # 算术运算为主
    LOGIC = auto()  # 逻辑运算为主
    SHIFT = auto()  # 移位运算为主
    COMPARE = auto()  # 比较运算为主
    MAC = auto()  # 乘累加运算（Xpulp）
    SIMD = auto()  # SIMD 运算（Xpulp）

    # 内存访问节点
    LOAD = auto()  # 加载为主
    STORE = auto()  # 存储为主
    MEMORY = auto()  # 混合内存访问

    # 特殊节点
    CSR = auto()  # CSR 操作节点
    SYSTEM = auto()  # 系统调用节点
    HWLOOP = auto()  # 硬件循环节点（Xpulp）
    NOP = auto()  # 空操作节点

    # 默认
    UNKNOWN = auto()  # 未知类型


class AsmEdgeType(Enum):
    """汇编 CDFG 边类型"""

    # 控制流边
    CONTROL_FLOW = auto()  # 顺序控制流（fall-through）
    BRANCH_TAKEN = auto()  # 分支跳转（条件满足）
    BRANCH_NOT_TAKEN = auto()  # 分支不跳转（条件不满足）
    JUMP = auto()  # 无条件跳转
    CALL = auto()  # 函数调用
    RETURN = auto()  # 函数返回

    # 数据流边
    DATA_DEP = auto()  # 数据依赖（寄存器读写）
    MEMORY_DEP = auto()  # 内存依赖（load-store）
    ANTI_DEP = auto()  # 反依赖（WAR）
    OUTPUT_DEP = auto()  # 输出依赖（WAW）


@dataclass
class AsmNode:
    """汇编 CDFG 节点（对应一个基本块）"""

    id: str  # 节点 ID（如 bb_0）
    label: Optional[str]  # 标签名（如 main, loop）
    node_type: AsmNodeType  # 节点类型

    # 源码位置
    start_line: int = 0  # 起始行号
    end_line: int = 0  # 结束行号
    source_file: str = ""  # 源文件名

    # 基本块内容
    instructions: List[Instruction] = field(default_factory=list)  # 指令列表
    instr_count: int = 0  # 指令数量

    # 寄存器活跃性
    defs: Set[str] = field(default_factory=set)  # 定义的寄存器
    uses: Set[str] = field(default_factory=set)  # 使用的寄存器
    live_in: Set[str] = field(default_factory=set)  # 入口活跃寄存器
    live_out: Set[str] = field(default_factory=set)  # 出口活跃寄存器

    # 控制流信息
    successors: List[str] = field(default_factory=list)  # 后继节点 ID
    predecessors: List[str] = field(default_factory=list)  # 前驱节点 ID
    is_loop_header: bool = False  # 是否为循环头
    loop_depth: int = 0  # 循环嵌套深度

    # 分支信息（仅 BRANCH 类型节点）
    branch_condition: Optional[str] = None  # 分支条件（如 "beq x1, x2"）
    branch_target: Optional[str] = None  # 分支目标标签

    # 额外属性
    attributes: Dict = field(default_factory=dict)

    def get_dominant_category(self) -> InstrCategory:
        """获取块内主要指令类别"""
        if not self.instructions:
            return InstrCategory.UNKNOWN

        category_counts: Dict[InstrCategory, int] = {}
        for instr in self.instructions:
            cat = instr.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return max(category_counts, key=lambda c: category_counts[c])

    def get_instructions_text(self, indent: str = "") -> str:
        """
        获取当前块的指令内容，使用换行符连接

        Args:
            indent: 每行指令前的缩进字符串

        Returns:
            所有指令文本，以换行符连接
        """
        if not self.instructions:
            return ""

        lines = []
        for instr in self.instructions:
            if instr.operands:
                line = f"{indent}{instr.mnemonic} {', '.join(instr.operands)}"
            else:
                line = f"{indent}{instr.mnemonic}"
            lines.append(line)

        return "\n".join(lines)


@dataclass
class AsmEdge:
    """汇编 CDFG 边"""

    source: str  # 源节点 ID
    target: str  # 目标节点 ID
    edge_type: AsmEdgeType  # 边类型

    # 数据流信息（仅数据流边）
    register: Optional[str] = None  # 涉及的寄存器
    registers: Set[str] = field(default_factory=set)  # 涉及的多个寄存器

    # 控制流信息（仅控制流边）
    condition: Optional[str] = None  # 分支条件
    is_back_edge: bool = False  # 是否为回边（循环）

    # 源码位置
    source_line: int = 0  # 源码行号

    # 覆盖率标注（用于后续扩展）
    execution_count: int = -1  # 执行次数（-1 表示未标注）
    is_covered: Optional[bool] = None  # 是否被覆盖

    # 额外属性
    attributes: Dict = field(default_factory=dict)


@dataclass
class AsmCDFG:
    """汇编控制数据流图"""

    module_name: str  # 模块/程序名
    source_file: str = ""  # 源文件路径

    # 图结构
    nodes: Dict[str, AsmNode] = field(default_factory=dict)  # 节点字典
    edges: List[AsmEdge] = field(default_factory=list)  # 边列表

    # 入口和出口
    entry_node: Optional[str] = None  # 入口节点 ID
    exit_nodes: List[str] = field(default_factory=list)  # 出口节点 ID 列表

    # 统计信息
    total_instructions: int = 0  # 总指令数
    total_basic_blocks: int = 0  # 基本块数

    # 索引结构（加速查询）
    label_to_node: Dict[str, str] = field(default_factory=dict)  # 标签 -> 节点 ID
    reg_def_nodes: Dict[str, List[str]] = field(default_factory=dict)  # 寄存器 -> 定义节点列表
    reg_use_nodes: Dict[str, List[str]] = field(default_factory=dict)  # 寄存器 -> 使用节点列表

    def add_node(self, node: AsmNode):
        """添加节点"""
        self.nodes[node.id] = node
        if node.label:
            self.label_to_node[node.label] = node.id

        # 更新寄存器索引
        for reg in node.defs:
            if reg not in self.reg_def_nodes:
                self.reg_def_nodes[reg] = []
            self.reg_def_nodes[reg].append(node.id)

        for reg in node.uses:
            if reg not in self.reg_use_nodes:
                self.reg_use_nodes[reg] = []
            self.reg_use_nodes[reg].append(node.id)

    def add_edge(self, edge: AsmEdge):
        """添加边"""
        self.edges.append(edge)

    def get_node_by_label(self, label: str) -> Optional[AsmNode]:
        """通过标签获取节点"""
        node_id = self.label_to_node.get(label)
        return self.nodes.get(node_id) if node_id else None

    def get_successors(self, node_id: str) -> List[AsmNode]:
        """获取后继节点"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[succ_id] for succ_id in node.successors if succ_id in self.nodes]

    def get_predecessors(self, node_id: str) -> List[AsmNode]:
        """获取前驱节点"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[pred_id] for pred_id in node.predecessors if pred_id in self.nodes]

    def get_control_edges(self) -> List[AsmEdge]:
        """获取所有控制流边"""
        control_types = {
            AsmEdgeType.CONTROL_FLOW,
            AsmEdgeType.BRANCH_TAKEN,
            AsmEdgeType.BRANCH_NOT_TAKEN,
            AsmEdgeType.JUMP,
            AsmEdgeType.CALL,
            AsmEdgeType.RETURN,
        }
        return [e for e in self.edges if e.edge_type in control_types]

    def get_data_edges(self) -> List[AsmEdge]:
        """获取所有数据流边"""
        data_types = {
            AsmEdgeType.DATA_DEP,
            AsmEdgeType.MEMORY_DEP,
            AsmEdgeType.ANTI_DEP,
            AsmEdgeType.OUTPUT_DEP,
        }
        return [e for e in self.edges if e.edge_type in data_types]

    def get_back_edges(self) -> List[AsmEdge]:
        """获取所有回边（循环边）"""
        return [e for e in self.edges if e.is_back_edge]

    def compute_statistics(self):
        """计算统计信息"""
        self.total_basic_blocks = len(self.nodes)
        self.total_instructions = sum(node.instr_count for node in self.nodes.values())

