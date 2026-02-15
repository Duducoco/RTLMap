#!/usr/bin/env python3
"""
基本块构建器

将指令列表划分为基本块，并建立控制流图（CFG）。
"""

from typing import List, Dict, Optional, Set

from .data_types import Instruction, BasicBlock, InstrFormat, InstrCategory


class BasicBlockBuilder:
    """基本块构建器"""

    def __init__(self, verbose: bool = False):
        """
        初始化构建器

        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose

    def build(self, instructions: List[Instruction]) -> List[BasicBlock]:
        """
        将指令列表划分为基本块

        基本块划分规则：
        1. 标签开始一个新的基本块
        2. 分支/跳转指令结束当前基本块
        3. 分支/跳转的后续指令开始新的基本块

        Args:
            instructions: 指令列表

        Returns:
            基本块列表（已建立 predecessors/successors 关系）
        """
        # 阶段1: 识别基本块边界
        leaders = self._find_leaders(instructions)

        # 阶段2: 创建基本块
        blocks = self._create_blocks(instructions, leaders)

        # 阶段3: 建立控制流边
        self._link_blocks(blocks)

        if self.verbose:
            print(f"构建完成: {len(blocks)} 个基本块")
            for block in blocks:
                print(
                    f"  {block.id}: {block.label or '(无标签)'}, "
                    f"指令数={len(block.instructions)}, "
                    f"行号={block.start_line}-{block.end_line}"
                )

        return blocks

    def _find_leaders(self, instructions: List[Instruction]) -> Set[int]:
        """
        找出所有基本块的 leader（入口指令）

        Leader 规则：
        1. 程序第一条指令是 leader
        2. 分支/跳转目标是 leader
        3. 分支/跳转后的下一条指令是 leader
        """
        leaders = set()

        # 过滤掉汇编指令（directive），只保留实际指令
        real_instrs = [
            (i, instr)
            for i, instr in enumerate(instructions)
            if instr.instr_format != InstrFormat.DIRECTIVE
        ]

        if not real_instrs:
            return leaders

        # 规则1: 第一条实际指令是 leader
        leaders.add(real_instrs[0][0])

        # 收集所有标签
        label_to_idx: Dict[str, int] = {}
        for i, instr in enumerate(instructions):
            if instr.instr_format == InstrFormat.LABEL and instr.label:
                label_to_idx[instr.label] = i

        # 遍历找出其他 leader
        for i, instr in enumerate(instructions):
            if instr.instr_format == InstrFormat.DIRECTIVE:
                continue

            # 规则2: 分支/跳转后的指令是 leader
            if instr.is_terminator:
                # 后继指令
                for j in range(i + 1, len(instructions)):
                    if instructions[j].instr_format not in (
                        InstrFormat.DIRECTIVE,
                        InstrFormat.LABEL,
                    ):
                        leaders.add(j)
                        break

                # 分支目标
                if instr.label and instr.label in label_to_idx:
                    target_idx = label_to_idx[instr.label]
                    # 找到标签后的第一条实际指令
                    for j in range(target_idx + 1, len(instructions)):
                        if instructions[j].instr_format not in (
                            InstrFormat.DIRECTIVE,
                            InstrFormat.LABEL,
                        ):
                            leaders.add(j)
                            break

        return leaders

    def _create_blocks(
        self, instructions: List[Instruction], leaders: Set[int]
    ) -> List[BasicBlock]:
        """根据 leader 创建基本块"""
        blocks = []
        current_instrs: List[Instruction] = []
        current_labels: List[str] = []
        block_id = 0

        # 标签到块的映射（用于后续链接）
        self._label_to_block: Dict[str, str] = {}

        for i, instr in enumerate(instructions):
            # 处理汇编指令（跳过）
            if instr.instr_format == InstrFormat.DIRECTIVE:
                continue

            # 处理标签
            if instr.instr_format == InstrFormat.LABEL:
                # 如果当前有指令，先创建块
                if current_instrs:
                    primary_label = current_labels[0] if current_labels else None
                    block = self._create_single_block(
                        block_id,
                        primary_label,
                        current_instrs,
                        labels=list(current_labels),
                    )
                    blocks.append(block)
                    for lbl in current_labels:
                        self._label_to_block[lbl] = block.id
                    block_id += 1
                    current_instrs = []
                    current_labels = []

                # 累积标签（支持连续标签）
                if instr.label:
                    current_labels.append(instr.label)
                continue

            # 检查是否是新基本块的开始
            if i in leaders and current_instrs:
                primary_label = current_labels[0] if current_labels else None
                block = self._create_single_block(
                    block_id,
                    primary_label,
                    current_instrs,
                    labels=list(current_labels),
                )
                blocks.append(block)
                for lbl in current_labels:
                    self._label_to_block[lbl] = block.id
                block_id += 1
                current_instrs = []
                current_labels = []

            # 添加指令
            current_instrs.append(instr)

            # 终结指令结束当前块
            if instr.is_terminator:
                primary_label = current_labels[0] if current_labels else None
                block = self._create_single_block(
                    block_id,
                    primary_label,
                    current_instrs,
                    labels=list(current_labels),
                )
                blocks.append(block)
                for lbl in current_labels:
                    self._label_to_block[lbl] = block.id
                block_id += 1
                current_instrs = []
                current_labels = []

        # 处理最后一个块
        if current_instrs:
            primary_label = current_labels[0] if current_labels else None
            block = self._create_single_block(
                block_id,
                primary_label,
                current_instrs,
                labels=list(current_labels),
            )
            blocks.append(block)
            for lbl in current_labels:
                self._label_to_block[lbl] = block.id

        return blocks

    def _create_single_block(
        self,
        block_id: int,
        label: Optional[str],
        instrs: List[Instruction],
        labels: Optional[List[str]] = None,
    ) -> BasicBlock:
        """创建单个基本块并分析寄存器使用"""
        block = BasicBlock(
            id=f"bb_{block_id}",
            label=label,
            instructions=instrs,
            start_line=instrs[0].line_no if instrs else 0,
            end_line=instrs[-1].line_no if instrs else 0,
            labels=labels if labels else ([label] if label else []),
        )

        # 分析 defs/uses（按指令顺序，前面的 def 会遮蔽后面的 use）
        local_defs: Set[str] = set()

        for instr in instrs:
            # 先处理 uses（在 defs 之前）
            # 跳过 x0（RISC-V 硬连线零寄存器，读取恒为 0）
            for rs in [instr.rs1, instr.rs2, instr.rs3]:
                if rs and rs != "x0" and rs not in local_defs:
                    block.uses.add(rs)

            # 再处理 defs
            # 跳过 x0（写入 x0 无效果）
            if instr.rd and instr.rd != "x0":
                local_defs.add(instr.rd)
                block.defs.add(instr.rd)

        return block

    def _link_blocks(self, blocks: List[BasicBlock]):
        """建立基本块之间的控制流边"""
        if not blocks:
            return

        # 创建块 ID 到索引的映射
        id_to_idx = {block.id: i for i, block in enumerate(blocks)}

        for i, block in enumerate(blocks):
            if not block.instructions:
                # 空块，顺序连接到下一个
                if i + 1 < len(blocks):
                    block.successors.append(blocks[i + 1].id)
                continue

            last_instr = block.instructions[-1]

            if last_instr.category == InstrCategory.BRANCH:
                # 条件分支：有两个后继
                # 1. 分支目标
                if last_instr.label and last_instr.label in self._label_to_block:
                    target_block_id = self._label_to_block[last_instr.label]
                    if target_block_id not in block.successors:
                        block.successors.append(target_block_id)

                # 2. Fall-through
                if i + 1 < len(blocks):
                    fall_through_id = blocks[i + 1].id
                    if fall_through_id not in block.successors:
                        block.successors.append(fall_through_id)

            elif last_instr.category == InstrCategory.JUMP:
                # 无条件跳转
                if last_instr.label and last_instr.label in self._label_to_block:
                    target_block_id = self._label_to_block[last_instr.label]
                    if target_block_id not in block.successors:
                        block.successors.append(target_block_id)

                # jalr/ret 等间接跳转可能没有静态目标
                # 在这种情况下不添加后继

            elif last_instr.category == InstrCategory.SYSTEM:
                # 系统指令（如 mret, wfi）可能不返回
                # 但 ecall 等可能返回，保守处理
                if last_instr.mnemonic.lower() not in ("mret", "sret", "wfi", "dret"):
                    if i + 1 < len(blocks):
                        block.successors.append(blocks[i + 1].id)

            else:
                # 顺序执行
                if i + 1 < len(blocks):
                    block.successors.append(blocks[i + 1].id)

        # 建立前驱关系
        for block in blocks:
            for succ_id in block.successors:
                if succ_id in id_to_idx:
                    succ_block = blocks[id_to_idx[succ_id]]
                    if block.id not in succ_block.predecessors:
                        succ_block.predecessors.append(block.id)

    def get_label_to_block_map(self) -> Dict[str, str]:
        """获取标签到基本块 ID 的映射"""
        return self._label_to_block.copy() if hasattr(self, "_label_to_block") else {}
