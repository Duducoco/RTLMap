#!/usr/bin/env python3
"""
ASM CDFG 提取器

从基本块列表提取控制数据流图（CDFG）。
使用 asm_cdfg 模块独立的数据结构。
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Set

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 支持直接运行和模块导入两种方式
if __name__ == "__main__" or __package__ is None:
    from asm_cdfg.asm_types import (
        BasicBlock,
        Instruction,
        InstrCategory,
        AsmNode,
        AsmEdge,
        AsmCDFG,
        AsmNodeType,
        AsmEdgeType,
    )
else:
    from .asm_types import (
        BasicBlock,
        Instruction,
        InstrCategory,
        AsmNode,
        AsmEdge,
        AsmCDFG,
        AsmNodeType,
        AsmEdgeType,
    )


class AsmCDFGExtractor:
    """从基本块提取 CDFG"""

    def __init__(self, verbose: bool = False):
        """
        初始化提取器

        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose

    def extract(
        self, blocks: List[BasicBlock], module_name: str = "asm_program"
    ) -> AsmCDFG:
        """
        从基本块列表提取 CDFG

        Args:
            blocks: 基本块列表
            module_name: 模块名称

        Returns:
            AsmCDFG 对象
        """
        cdfg = AsmCDFG(module_name=module_name)

        # 阶段1: 为每个基本块创建节点
        self._create_nodes(blocks, cdfg)

        # 阶段2: 创建控制流边
        self._create_control_edges(blocks, cdfg)

        # 阶段3: 创建数据流边（基于寄存器依赖）
        self._create_data_edges(blocks, cdfg)

        # 阶段4: 检测回边和循环
        self._detect_back_edges(cdfg)

        # 计算统计信息
        cdfg.compute_statistics()

        if self.verbose:
            print(f"CDFG 提取完成:")
            print(f"  节点数: {len(cdfg.nodes)}")
            print(f"  边数: {len(cdfg.edges)}")
            control_edges = len(cdfg.get_control_edges())
            data_edges = len(cdfg.get_data_edges())
            back_edges = len(cdfg.get_back_edges())
            print(f"  - 控制流边: {control_edges}")
            print(f"  - 数据流边: {data_edges}")
            print(f"  - 回边（循环）: {back_edges}")

        return cdfg

    def _create_nodes(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """为每个基本块创建节点"""
        for i, block in enumerate(blocks):
            # 确定节点类型（根据基本块特征）
            node_type = self._get_block_node_type(block)

            # 确定节点名称
            label = block.label if block.label else None

            # 提取分支信息
            branch_condition = None
            branch_target = None
            if block.instructions and node_type == AsmNodeType.BRANCH:
                last_instr = block.instructions[-1]
                branch_condition = f"{last_instr.mnemonic} {', '.join(last_instr.operands)}"
                branch_target = last_instr.label

            # 创建节点
            node = AsmNode(
                id=block.id,
                label=label,
                node_type=node_type,
                start_line=block.start_line,
                end_line=block.end_line,
                instructions=block.instructions.copy(),
                instr_count=len(block.instructions),
                defs=block.defs.copy(),
                uses=block.uses.copy(),
                successors=block.successors.copy(),
                predecessors=block.predecessors.copy(),
                branch_condition=branch_condition,
                branch_target=branch_target,
            )

            cdfg.add_node(node)

            # 标记入口和出口
            if i == 0:
                cdfg.entry_node = node.id

            if node_type == AsmNodeType.EXIT:
                cdfg.exit_nodes.append(node.id)

    def _get_block_node_type(self, block: BasicBlock) -> AsmNodeType:
        """根据基本块特征确定节点类型"""
        if not block.instructions:
            return AsmNodeType.UNKNOWN

        # 根据最后一条指令的类别判断控制流类型
        last_instr = block.instructions[-1]

        # 退出指令
        if last_instr.mnemonic in {"ret", "mret", "sret", "uret"}:
            return AsmNodeType.EXIT

        # 系统调用
        if last_instr.mnemonic in {"ecall", "ebreak"}:
            return AsmNodeType.SYSTEM

        # 分支指令
        if last_instr.category == InstrCategory.BRANCH:
            return AsmNodeType.BRANCH

        # 跳转指令
        if last_instr.category == InstrCategory.JUMP:
            return AsmNodeType.JUMP

        # 根据块内主要操作类型判断
        categories = [instr.category for instr in block.instructions]
        category_counts: Dict[InstrCategory, int] = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if not category_counts:
            return AsmNodeType.COMPUTE

        dominant_category = max(category_counts, key=lambda c: category_counts[c])

        category_to_node_type = {
            InstrCategory.ARITHMETIC: AsmNodeType.ARITHMETIC,
            InstrCategory.LOGIC: AsmNodeType.LOGIC,
            InstrCategory.SHIFT: AsmNodeType.SHIFT,
            InstrCategory.COMPARE: AsmNodeType.COMPARE,
            InstrCategory.LOAD: AsmNodeType.LOAD,
            InstrCategory.STORE: AsmNodeType.STORE,
            InstrCategory.FLOAT: AsmNodeType.ARITHMETIC,
            InstrCategory.MAC: AsmNodeType.MAC,
            InstrCategory.SIMD: AsmNodeType.SIMD,
            InstrCategory.BITMANIP: AsmNodeType.LOGIC,
            InstrCategory.CSR: AsmNodeType.CSR,
            InstrCategory.SYSTEM: AsmNodeType.SYSTEM,
            InstrCategory.NOP: AsmNodeType.NOP,
            InstrCategory.HWLOOP: AsmNodeType.HWLOOP,
        }

        # 内存操作混合
        if InstrCategory.LOAD in category_counts and InstrCategory.STORE in category_counts:
            return AsmNodeType.MEMORY

        return category_to_node_type.get(dominant_category, AsmNodeType.COMPUTE)

    def _create_control_edges(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """创建控制流边"""
        # 构建块 ID 到块的映射
        block_map = {block.id: block for block in blocks}

        for block in blocks:
            node = cdfg.nodes[block.id]

            for succ_id in block.successors:
                # 确定边类型
                if node.node_type == AsmNodeType.BRANCH:
                    # 分支节点有两种出边
                    if node.branch_target:
                        target_node = cdfg.get_node_by_label(node.branch_target)
                        if target_node and target_node.id == succ_id:
                            edge_type = AsmEdgeType.BRANCH_TAKEN
                            condition = node.branch_condition
                        else:
                            edge_type = AsmEdgeType.BRANCH_NOT_TAKEN
                            condition = f"not ({node.branch_condition})" if node.branch_condition else None
                    else:
                        edge_type = AsmEdgeType.CONTROL_FLOW
                        condition = None
                elif node.node_type == AsmNodeType.JUMP:
                    # 检查是否为函数调用（jal ra, xxx）
                    if block.instructions:
                        last_instr = block.instructions[-1]
                        if last_instr.mnemonic in {"jal", "jalr"} and last_instr.rd == "x1":
                            edge_type = AsmEdgeType.CALL
                        else:
                            edge_type = AsmEdgeType.JUMP
                    else:
                        edge_type = AsmEdgeType.JUMP
                    condition = None
                elif node.node_type == AsmNodeType.EXIT:
                    edge_type = AsmEdgeType.RETURN
                    condition = None
                else:
                    edge_type = AsmEdgeType.CONTROL_FLOW
                    condition = None

                edge = AsmEdge(
                    source=block.id,
                    target=succ_id,
                    edge_type=edge_type,
                    condition=condition,
                    source_line=block.end_line,
                )
                cdfg.add_edge(edge)

    def _create_data_edges(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """
        基于 def-use 链创建数据流边

        使用简化的到达定值分析：
        - 遍历基本块，记录每个寄存器的最后定义位置
        - 当遇到使用时，创建从定义到使用的边
        """
        # 记录每个寄存器的最后定义块
        reg_last_def: Dict[str, str] = {}

        # 按程序顺序遍历（假设 blocks 已按顺序排列）
        for block in blocks:
            # 对于块中使用的每个寄存器，创建数据流边
            for used_reg in block.uses:
                if used_reg in reg_last_def:
                    def_block_id = reg_last_def[used_reg]

                    # 避免自环
                    if def_block_id != block.id:
                        edge = AsmEdge(
                            source=def_block_id,
                            target=block.id,
                            edge_type=AsmEdgeType.DATA_DEP,
                            register=used_reg,
                            registers={used_reg},
                        )
                        cdfg.add_edge(edge)

            # 更新寄存器定义
            for def_reg in block.defs:
                reg_last_def[def_reg] = block.id

    def _detect_back_edges(self, cdfg: AsmCDFG):
        """检测回边（使用 DFS）"""
        if not cdfg.entry_node:
            return

        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node_id: str):
            visited.add(node_id)
            in_stack.add(node_id)

            node = cdfg.nodes.get(node_id)
            if node:
                for succ_id in node.successors:
                    if succ_id in in_stack:
                        # 找到回边，标记对应的控制流边
                        for edge in cdfg.edges:
                            if edge.source == node_id and edge.target == succ_id:
                                edge.is_back_edge = True
                                # 标记目标节点为循环头
                                target_node = cdfg.nodes.get(succ_id)
                                if target_node:
                                    target_node.is_loop_header = True
                    elif succ_id not in visited:
                        dfs(succ_id)

            in_stack.remove(node_id)

        dfs(cdfg.entry_node)

    def extract_from_file(
        self, filepath: str, module_name: Optional[str] = None
    ) -> AsmCDFG:
        """
        从汇编文件直接提取 CDFG

        Args:
            filepath: 汇编文件路径
            module_name: 模块名（默认使用文件名）

        Returns:
            AsmCDFG 对象
        """
        if __package__ is None:
            from asm_cdfg.parser import AsmParser
            from asm_cdfg.bb_builder import BasicBlockBuilder
        else:
            from .parser import AsmParser
            from .bb_builder import BasicBlockBuilder

        if module_name is None:
            module_name = Path(filepath).stem

        # 解析
        parser = AsmParser(verbose=self.verbose)
        instructions = parser.parse_file(filepath)

        # 构建基本块
        builder = BasicBlockBuilder(verbose=self.verbose)
        blocks = builder.build(instructions)

        # 提取 CDFG
        cdfg = self.extract(blocks, module_name)
        cdfg.source_file = filepath

        return cdfg


def main():
    """命令行入口"""
    import argparse
    import json

    arg_parser = argparse.ArgumentParser(description="从 RISC-V 汇编提取 CDFG")
    arg_parser.add_argument("input", help="输入汇编文件 (.S)")
    arg_parser.add_argument("-o", "--output", help="输出 JSON 文件路径")
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    arg_parser.add_argument("--svg", action="store_true", help="生成 SVG 可视化")
    arg_parser.add_argument(
        "--simple", action="store_true", help="生成简化 SVG（仅控制流）"
    )

    args = arg_parser.parse_args()

    # 提取 CDFG
    extractor = AsmCDFGExtractor(verbose=args.verbose)
    cdfg = extractor.extract_from_file(args.input)

    # 输出统计信息
    print(f"\n汇编文件: {args.input}")
    print(f"模块名: {cdfg.module_name}")
    print(f"节点数: {len(cdfg.nodes)}")
    print(f"边数: {len(cdfg.edges)}")
    print(f"总指令数: {cdfg.total_instructions}")

    # 导出 JSON
    if args.output:
        export_data = {
            "module_name": cdfg.module_name,
            "source_file": cdfg.source_file,
            "entry_node": cdfg.entry_node,
            "exit_nodes": cdfg.exit_nodes,
            "total_instructions": cdfg.total_instructions,
            "total_basic_blocks": cdfg.total_basic_blocks,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type.name,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "instr_count": node.instr_count,
                    "instructions": [
                        {"mnemonic": i.mnemonic, "operands": i.operands}
                        for i in node.instructions
                    ],
                    "defs": list(node.defs),
                    "uses": list(node.uses),
                    "successors": node.successors,
                    "predecessors": node.predecessors,
                    "is_loop_header": node.is_loop_header,
                    "branch_condition": node.branch_condition,
                    "branch_target": node.branch_target,
                }
                for node_id, node in cdfg.nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type.name,
                    "register": e.register,
                    "condition": e.condition,
                    "is_back_edge": e.is_back_edge,
                }
                for e in cdfg.edges
            ],
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"已导出到: {args.output}")

    # 生成 SVG
    if args.svg:
        if __package__ is None:
            from asm_cdfg.visualizer import AsmCDFGVisualizer
        else:
            from .visualizer import AsmCDFGVisualizer

        svg_path = (
            args.output.replace(".json", "") if args.output else cdfg.module_name
        )

        visualizer = AsmCDFGVisualizer(
            show_instructions=True,
            max_instructions=8,
            show_registers=True,
        )

        if args.simple:
            output_path = visualizer.render_simple(cdfg, f"{svg_path}_cfg")
        else:
            output_path = visualizer.render(cdfg, f"{svg_path}_cdfg")

        print(f"已生成 SVG: {output_path}")


if __name__ == "__main__":
    main()
