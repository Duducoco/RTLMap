#!/usr/bin/env python3
"""
ASM CDFG 提取器

从基本块列表提取控制数据流图（CDFG）。
使用 asm_cdfg 模块独立的数据结构。
"""

import sys
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Set

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 支持直接运行和模块导入两种方式
if __name__ == "__main__" or __package__ is None:
    from cdfg_asm.data_types import (
        BasicBlock,
        Instruction,
        InstrCategory,
        InstrFormat,
        AsmNode,
        AsmEdge,
        AsmCDFG,
        AsmNodeType,
        AsmEdgeType,
    )
else:
    from .data_types import (
        BasicBlock,
        Instruction,
        InstrCategory,
        InstrFormat,
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
        self,
        blocks: List[BasicBlock],
        module_name: str = "asm_program",
        instructions: Optional[List[Instruction]] = None,
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

        # 阶段1.5: 基于 .globl 指令检测入口点
        if instructions:
            self._detect_entry_point(instructions, cdfg)

        # 阶段2: 创建控制流边
        self._create_control_edges(blocks, cdfg)

        # 阶段3: 创建数据流边（基于寄存器依赖）
        self._create_data_edges(blocks, cdfg)

        # 阶段3.5: 创建内存依赖边（load/store 依赖）
        self._create_memory_edges(blocks, cdfg)

        # 阶段4: 检测回边和循环
        self._detect_back_edges(cdfg)

        # 计算统计信息
        cdfg.compute_statistics()

        if self.verbose:
            print("CDFG 提取完成:")
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
                branch_condition = (
                    f"{last_instr.mnemonic} {', '.join(last_instr.operands)}"
                )
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

            # 注册连续标签中的非主标签到 label_to_node
            for lbl in block.labels:
                if lbl and lbl not in cdfg.label_to_node:
                    cdfg.label_to_node[lbl] = node.id

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
        if (
            InstrCategory.LOAD in category_counts
            and InstrCategory.STORE in category_counts
        ):
            return AsmNodeType.MEMORY

        return category_to_node_type.get(dominant_category, AsmNodeType.COMPUTE)

    def _create_control_edges(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """创建控制流边

        严格基于 block.successors（由 bb_builder 计算）创建边，
        避免与 CFG 的 successors/predecessors 不一致。
        """
        for block in blocks:
            node = cdfg.nodes[block.id]

            if node.node_type == AsmNodeType.BRANCH:
                # 分支节点：基于 successors 区分 TAKEN/NOT_TAKEN
                target_id = None
                if node.branch_target:
                    target_node = cdfg.get_node_by_label(node.branch_target)
                    target_id = target_node.id if target_node else None

                if target_id:
                    # 有明确的分支目标：区分 TAKEN/NOT_TAKEN
                    taken_created = False

                    for succ_id in block.successors:
                        if succ_id == target_id and not taken_created:
                            cdfg.add_edge(
                                AsmEdge(
                                    source=block.id,
                                    target=succ_id,
                                    edge_type=AsmEdgeType.BRANCH_TAKEN,
                                    condition=node.branch_condition,
                                    source_line=block.end_line,
                                )
                            )
                            taken_created = True
                        else:
                            cdfg.add_edge(
                                AsmEdge(
                                    source=block.id,
                                    target=succ_id,
                                    edge_type=AsmEdgeType.BRANCH_NOT_TAKEN,
                                    condition=(
                                        f"not ({node.branch_condition})"
                                        if node.branch_condition
                                        else None
                                    ),
                                    source_line=block.end_line,
                                )
                            )

                    # 目标 == fall-through（successors 只有一个元素）时补充 NOT_TAKEN
                    if len(block.successors) == 1 and target_id == block.successors[0]:
                        cdfg.add_edge(
                            AsmEdge(
                                source=block.id,
                                target=block.successors[0],
                                edge_type=AsmEdgeType.BRANCH_NOT_TAKEN,
                                condition=(
                                    f"not ({node.branch_condition})"
                                    if node.branch_condition
                                    else None
                                ),
                                source_line=block.end_line,
                            )
                        )
                else:
                    # 无明确目标（间接分支等）：所有 successors 标记为 BRANCH_TAKEN/NOT_TAKEN
                    # 第一个 successor 视为 TAKEN，其余为 NOT_TAKEN
                    for idx, succ_id in enumerate(block.successors):
                        if idx == 0:
                            cdfg.add_edge(
                                AsmEdge(
                                    source=block.id,
                                    target=succ_id,
                                    edge_type=AsmEdgeType.BRANCH_TAKEN,
                                    condition=node.branch_condition,
                                    source_line=block.end_line,
                                )
                            )
                        else:
                            cdfg.add_edge(
                                AsmEdge(
                                    source=block.id,
                                    target=succ_id,
                                    edge_type=AsmEdgeType.BRANCH_NOT_TAKEN,
                                    condition=(
                                        f"not ({node.branch_condition})"
                                        if node.branch_condition
                                        else None
                                    ),
                                    source_line=block.end_line,
                                )
                            )

            else:
                # 非分支节点
                for succ_id in block.successors:
                    if node.node_type == AsmNodeType.JUMP:
                        if block.instructions:
                            last_instr = block.instructions[-1]
                            if (
                                last_instr.mnemonic in {"jal", "jalr"}
                                and last_instr.rd == "x1"
                            ):
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

                    cdfg.add_edge(
                        AsmEdge(
                            source=block.id,
                            target=succ_id,
                            edge_type=edge_type,
                            condition=condition,
                            source_line=block.end_line,
                        )
                    )

    def _create_data_edges(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """
        基于到达定值分析（reaching definitions）创建数据流边

        使用 worklist 算法迭代至不动点：
        - reach_in[B] = ∪ reach_out[P], for P in predecessors(B)
        - reach_out[B] = gen[B] ∪ (reach_in[B] - kill[B])
        """
        if not blocks:
            return

        block_map = {b.id: b for b in blocks}

        # 计算 gen 和 kill 集合
        # gen[B] = {(reg, B) for reg in block.defs}
        # kill[B] = block.defs（杀死所有先前对这些寄存器的定义）
        gen: Dict[str, Set[tuple]] = {}
        kill_regs: Dict[str, Set[str]] = {}

        for block in blocks:
            gen[block.id] = {(reg, block.id) for reg in block.defs}
            kill_regs[block.id] = set(block.defs)

        # 初始化 reach_in / reach_out
        reach_in: Dict[str, Set[tuple]] = {b.id: set() for b in blocks}
        reach_out: Dict[str, Set[tuple]] = {b.id: set(gen[b.id]) for b in blocks}

        # Worklist 迭代至不动点
        worklist: deque = deque(b.id for b in blocks)
        in_worklist = set(worklist)

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)

            block = block_map[bid]

            # reach_in = ∪ reach_out[P]
            new_in: Set[tuple] = set()
            for pred_id in block.predecessors:
                if pred_id in reach_out:
                    new_in |= reach_out[pred_id]
            reach_in[bid] = new_in

            # reach_out = gen ∪ (reach_in - kill)
            killed = kill_regs[bid]
            surviving = {(r, d) for r, d in new_in if r not in killed}
            new_out = gen[bid] | surviving

            if new_out != reach_out[bid]:
                reach_out[bid] = new_out
                # 将后继加入 worklist
                for succ_id in block.successors:
                    if succ_id not in in_worklist:
                        worklist.append(succ_id)
                        in_worklist.add(succ_id)

        # 从 reach_in 创建 DATA_DEP 边（去重）
        seen_data_edges: Set[tuple] = set()  # (source_id, target_id, reg)

        for block in blocks:
            for used_reg in block.uses:
                for reg, def_block_id in reach_in[block.id]:
                    if reg == used_reg and def_block_id != block.id:
                        edge_key = (def_block_id, block.id, used_reg)
                        if edge_key not in seen_data_edges:
                            seen_data_edges.add(edge_key)
                            edge = AsmEdge(
                                source=def_block_id,
                                target=block.id,
                                edge_type=AsmEdgeType.DATA_DEP,
                                register=used_reg,
                                registers={used_reg},
                            )
                            cdfg.add_edge(edge)

    def _create_memory_edges(self, blocks: List[BasicBlock], cdfg: AsmCDFG):
        """
        创建内存依赖边（保守近似，无别名分析）

        使用 reaching-stores worklist 算法沿 CFG 拓扑传播，
        避免程序顺序相邻但 CFG 不可达的块之间产生虚假依赖。

        - STORE → LOAD: MEMORY_DEP 边 (RAW)
        - STORE → STORE: OUTPUT_DEP 边 (WAW)
        """
        if not blocks:
            return

        block_map = {b.id: b for b in blocks}

        # 识别含 STORE/LOAD 指令的块
        store_set: Set[str] = set()
        load_set: Set[str] = set()

        for block in blocks:
            if any(i.category == InstrCategory.STORE for i in block.instructions):
                store_set.add(block.id)
            if any(i.category == InstrCategory.LOAD for i in block.instructions):
                load_set.add(block.id)

        if not store_set:
            return

        # reaching-stores: reach_in[B] = 到达 B 入口的 store 块集合
        # gen[B] = {B} if B has store, else {}
        # kill: store 块不杀死先前 store（保守：无别名分析，所有 store 都可能别名）
        # reach_out[B] = gen[B] ∪ reach_in[B]
        reach_in: Dict[str, Set[str]] = {b.id: set() for b in blocks}
        reach_out: Dict[str, Set[str]] = {
            b.id: ({b.id} if b.id in store_set else set()) for b in blocks
        }

        worklist: deque = deque(b.id for b in blocks)
        in_worklist = set(worklist)

        while worklist:
            bid = worklist.popleft()
            in_worklist.discard(bid)

            block = block_map[bid]

            # reach_in = ∪ reach_out[P]
            new_in: Set[str] = set()
            for pred_id in block.predecessors:
                if pred_id in reach_out:
                    new_in |= reach_out[pred_id]
            reach_in[bid] = new_in

            # reach_out = gen ∪ reach_in
            gen = {bid} if bid in store_set else set()
            new_out = gen | new_in

            if new_out != reach_out[bid]:
                reach_out[bid] = new_out
                for succ_id in block.successors:
                    if succ_id not in in_worklist:
                        worklist.append(succ_id)
                        in_worklist.add(succ_id)

        # 从 reach_in 创建内存依赖边
        seen_mem_edges: Set[tuple] = set()

        for block in blocks:
            bid = block.id
            has_load = bid in load_set
            has_store = bid in store_set

            if not has_load and not has_store:
                continue

            for store_bid in reach_in[bid]:
                if store_bid == bid:
                    continue

                # STORE → LOAD: RAW
                if has_load:
                    key = (store_bid, bid, "MEMORY_DEP")
                    if key not in seen_mem_edges:
                        seen_mem_edges.add(key)
                        cdfg.add_edge(
                            AsmEdge(
                                source=store_bid,
                                target=bid,
                                edge_type=AsmEdgeType.MEMORY_DEP,
                            )
                        )

                # STORE → STORE: WAW
                if has_store:
                    key = (store_bid, bid, "OUTPUT_DEP")
                    if key not in seen_mem_edges:
                        seen_mem_edges.add(key)
                        cdfg.add_edge(
                            AsmEdge(
                                source=store_bid,
                                target=bid,
                                edge_type=AsmEdgeType.OUTPUT_DEP,
                            )
                        )

    def _detect_entry_point(self, instructions: List[Instruction], cdfg: AsmCDFG):
        """
        基于 .globl 指令检测入口点

        扫描 instructions 中的 .globl directive，提取全局符号名，
        在 label_to_node 中查找匹配节点作为入口。
        """
        for instr in instructions:
            if (
                instr.instr_format == InstrFormat.DIRECTIVE
                and instr.mnemonic == ".globl"
                and instr.operands
            ):
                symbol = instr.operands[0].strip()
                target_node = cdfg.get_node_by_label(symbol)
                if target_node:
                    cdfg.entry_node = target_node.id
                    if self.verbose:
                        print(f"  入口点: {symbol} -> {target_node.id}")
                    return

    def _detect_back_edges(self, cdfg: AsmCDFG):
        """检测回边（使用 DFS），覆盖不可达区域"""
        # 构建 (source, target) -> edges 索引，避免 DFS 中 O(E) 线性扫描
        edge_index: Dict[tuple, List[AsmEdge]] = {}
        for edge in cdfg.edges:
            key = (edge.source, edge.target)
            if key not in edge_index:
                edge_index[key] = []
            edge_index[key].append(edge)

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
                        for edge in edge_index.get((node_id, succ_id), []):
                            edge.is_back_edge = True
                            # 标记目标节点为循环头
                            target_node = cdfg.nodes.get(succ_id)
                            if target_node:
                                target_node.is_loop_header = True
                    elif succ_id not in visited:
                        dfs(succ_id)

            in_stack.remove(node_id)

        # 从入口节点开始 DFS
        if cdfg.entry_node:
            dfs(cdfg.entry_node)

        # 对未访问的节点启动新的 DFS（覆盖不可达区域）
        for node_id in list(cdfg.nodes.keys()):
            if node_id not in visited:
                dfs(node_id)

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
            from cdfg_asm.parser import AsmParser
            from cdfg_asm.bb_builder import BasicBlockBuilder
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
        cdfg = self.extract(blocks, module_name, instructions=instructions)
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
        export_data = cdfg.to_dict()

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"已导出到: {args.output}")

    # 生成 SVG
    if args.svg:
        if __package__ is None:
            from cdfg_asm.visualizer import AsmCDFGVisualizer
        else:
            from .visualizer import AsmCDFGVisualizer

        svg_path = args.output.replace(".json", "") if args.output else cdfg.module_name

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
