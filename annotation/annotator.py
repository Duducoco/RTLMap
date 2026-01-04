#!/usr/bin/env python3
"""
覆盖率标注器
将覆盖率报告中的分支覆盖数据映射到 CDFG 的边上
"""

from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict

from cdfg import CDFG, Edge, NodeType
from .parser import CoverageParser, BranchCoverage, ConditionCoverage


@dataclass
class AnnotationStats:
    """标注统计信息"""

    total_mux_nodes: int = 0  # MUX 节点总数
    annotated_mux_nodes: int = 0  # 已标注的 MUX 节点数
    total_edges: int = 0  # 边总数
    annotated_edges: int = 0  # 已标注的边数
    covered_edges: int = 0  # 已覆盖的边数
    uncovered_edges: int = 0  # 未覆盖的边数
    control_edges: int = 0  # 控制边数
    data_true_edges: int = 0  # data_true 边数
    data_false_edges: int = 0  # data_false 边数
    # 传播标注统计
    propagated_edges: int = 0  # 通过传播标注的边数
    always_executed_edges: int = 0  # 必然执行的边数（组合逻辑）
    # 条件覆盖率统计
    condition_annotated_mux: int = 0  # 通过条件覆盖率标注的 MUX 数
    condition_annotated_edges: int = 0  # 通过条件覆盖率标注的边数


class CoverageAnnotator:
    """覆盖率标注器"""

    # 行号搜索范围配置
    STMT_LINE_SEARCH_RANGE = range(-2, 6)  # stmt_start_line 搜索范围 [-2, +5]
    SOURCE_LINE_SEARCH_RANGE = range(-5, 40)  # source_line 搜索范围 [-5, +39]

    def __init__(
        self,
        cdfg: CDFG,
        coverage_data: Dict[int, BranchCoverage],
        source_file: str = None,
        condition_data: Dict[int, List[ConditionCoverage]] = None,
    ):
        """
        初始化标注器

        Args:
            cdfg: CDFG 对象
            coverage_data: 行号 -> BranchCoverage 的映射
            source_file: 覆盖率报告对应的源文件名（用于过滤 MUX 节点）
            condition_data: 行号 -> ConditionCoverage 列表的映射
        """
        self.cdfg = cdfg
        self.coverage_data = coverage_data
        self.source_file = source_file
        self.condition_data = condition_data or {}
        self.stats = AnnotationStats()

        # 追踪已标注的 MUX 节点（避免重复计数）
        self._annotated_mux_set: set = set()

        # 学习到的行号偏移量（VCS 与 Yosys 之间的偏移）
        self._learned_offset: int = None

        # 建立行号到 MUX 节点的映射
        self.line_to_mux_nodes: Dict[int, List[str]] = defaultdict(list)
        self._build_line_mapping()

    def _build_line_mapping(self):
        """建立行号到 MUX/时序节点的映射"""
        # MUX 节点映射（按 source_line）
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.MUX and node.source_line > 0:
                # 如果指定了源文件，只映射匹配的节点
                if self.source_file and node.source_file:
                    if node.source_file != self.source_file:
                        continue
                self.line_to_mux_nodes[node.source_line].append(node_id)
                self.stats.total_mux_nodes += 1

        # MUX 节点映射（按 stmt_start_line，用于 CASE 语句）
        # 当 src 属性包含复合位置时，stmt_start_line 记录了 case/if 语句的起始行
        self.stmt_line_to_mux_nodes: Dict[int, List[str]] = defaultdict(list)
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.MUX and node.stmt_start_line is not None:
                # 如果指定了源文件，只映射匹配的节点
                if self.source_file and node.source_file:
                    if node.source_file != self.source_file:
                        continue
                self.stmt_line_to_mux_nodes[node.stmt_start_line].append(node_id)

        # 时序节点映射（用于复位分支）
        self.line_to_seq_nodes: Dict[int, List[str]] = defaultdict(list)
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.SEQUENTIAL and node.source_line > 0:
                # 如果指定了源文件，只映射匹配的节点
                if self.source_file and node.source_file:
                    if node.source_file != self.source_file:
                        continue
                self.line_to_seq_nodes[node.source_line].append(node_id)

    def _find_mux_chain_for_coverage(
        self, coverage_line: int, total_branches: int
    ) -> List[str]:
        """
        为覆盖率行找到对应的 MUX 链

        对于 if-else-if 链，覆盖率报告只记录起始行号（如 96），
        但 CDFG 中每个 else-if 对应不同行号（96, 97, 98...）。

        对于 CASE 语句，覆盖率报告记录的是 case 语句本身的行号，
        而 CDFG 中 MUX 节点记录的是各个 case 分支项的行号。
        此时需要通过 stmt_start_line 映射来查找。

        注意：VCS 和 Yosys 之间可能存在较大的行号偏移（可达 30+ 行），
        这是由于 include 文件、宏展开等处理方式不同导致的。

        这个方法会找到从 coverage_line 开始的连续 MUX 链。

        Args:
            coverage_line: 覆盖率报告中的行号
            total_branches: 总分支数（用于确定链的长度）

        Returns:
            MUX 节点 ID 列表
        """
        # 如果已经学习到偏移量，优先使用学习到的偏移
        if self._learned_offset is not None:
            adjusted_line = coverage_line + self._learned_offset
            result = self._search_mux_at_line(adjusted_line, total_branches)
            if result:
                return result

        # 1. 首先尝试通过 stmt_start_line 查找（用于 CASE 语句）
        for offset in self.STMT_LINE_SEARCH_RANGE:
            candidate_line = coverage_line + offset
            if candidate_line in self.stmt_line_to_mux_nodes:
                stmt_mux = self.stmt_line_to_mux_nodes[candidate_line]
                if stmt_mux:
                    # 学习偏移量
                    if self._learned_offset is None and offset != 0:
                        self._learned_offset = offset
                        print(f"  学习到行号偏移: {offset}")
                    if len(stmt_mux) >= total_branches - 1:
                        return self._sort_mux_nodes_topologically(stmt_mux[: total_branches - 1])
                    else:
                        return self._sort_mux_nodes_topologically(stmt_mux)

        # 2. 然后在扩大的范围内搜索 source_line（用于 IF 语句）
        for offset in self.SOURCE_LINE_SEARCH_RANGE:
            candidate_line = coverage_line + offset
            if candidate_line in self.line_to_mux_nodes:
                nearby_mux = self.line_to_mux_nodes[candidate_line]
                if nearby_mux:
                    # 学习偏移量
                    if self._learned_offset is None and offset != 0:
                        self._learned_offset = offset
                        print(f"  学习到行号偏移: {offset}")

                    # 如果找到足够的 MUX，直接返回
                    if len(nearby_mux) >= total_branches - 1:
                        return self._sort_mux_nodes_topologically(nearby_mux[: total_branches - 1])

                    # 尝试从这个位置开始收集更多 MUX
                    return self._collect_mux_chain_from_line(candidate_line, total_branches)

        # 3. 如果仍未找到，尝试更宽松的匹配：查找所有 MUX 的 stmt_start_line
        for offset in self.SOURCE_LINE_SEARCH_RANGE:
            candidate_line = coverage_line + offset
            # 检查是否有任何 MUX 的 stmt_start_line 匹配
            matching_mux = []
            for node_id, node in self.cdfg.nodes.items():
                if node.node_type.name == 'MUX' and node.stmt_start_line == candidate_line:
                    if self.source_file and node.source_file:
                        if node.source_file != self.source_file:
                            continue
                    matching_mux.append(node_id)

            if matching_mux:
                if self._learned_offset is None and offset != 0:
                    self._learned_offset = offset
                    print(f"  学习到行号偏移: {offset}")
                return self._sort_mux_nodes_topologically(matching_mux[:total_branches - 1] if len(matching_mux) >= total_branches - 1 else matching_mux)

        return []

    def _search_mux_at_line(self, line: int, total_branches: int) -> List[str]:
        """在指定行搜索 MUX 节点"""
        # 先检查 stmt_start_line
        if line in self.stmt_line_to_mux_nodes:
            stmt_mux = self.stmt_line_to_mux_nodes[line]
            if stmt_mux:
                if len(stmt_mux) >= total_branches - 1:
                    return self._sort_mux_nodes_topologically(stmt_mux[: total_branches - 1])
                return self._sort_mux_nodes_topologically(stmt_mux)

        # 再检查 source_line
        if line in self.line_to_mux_nodes:
            nearby_mux = self.line_to_mux_nodes[line]
            if nearby_mux:
                if len(nearby_mux) >= total_branches - 1:
                    return self._sort_mux_nodes_topologically(nearby_mux[: total_branches - 1])
                return self._collect_mux_chain_from_line(line, total_branches)

        return []

    def _collect_mux_chain_from_line(self, start_line: int, total_branches: int) -> List[str]:
        """从指定行开始收集 MUX 链"""
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())

        # 收集足够数量的 MUX
        mux_nodes = []
        expected_mux_count = total_branches - 1

        for line in all_mux_lines:
            if line >= start_line:
                mux_nodes.extend(self.line_to_mux_nodes[line])
                if len(mux_nodes) >= expected_mux_count:
                    break

        return self._sort_mux_nodes_topologically(mux_nodes[:expected_mux_count])

    def annotate(self) -> AnnotationStats:
        """
        执行标注

        Returns:
            标注统计信息
        """
        self.stats.total_edges = len(self.cdfg.edges)

        # 对于每个有覆盖数据的行
        for line_no, coverage in self.coverage_data.items():
            # 使用新的方法找到 MUX 链
            mux_nodes = self._find_mux_chain_for_coverage(
                line_no, coverage.total_branches
            )

            if mux_nodes:
                print(
                    f"行 {line_no}: 找到 {len(mux_nodes)} 个 MUX 节点，对应 {coverage.total_branches} 个分支"
                )
                # 标注每个 MUX 节点
                self._annotate_mux_chain(mux_nodes, coverage)
            else:
                # 尝试查找时序元件（复位分支）
                # 复位逻辑 (if rst) 可能被综合成带异步复位的触发器
                seq_nodes = self._find_sequential_nodes_for_coverage(line_no)
                if seq_nodes:
                    print(
                        f"行 {line_no}: 找到 {len(seq_nodes)} 个时序节点（复位分支），对应 {coverage.total_branches} 个分支"
                    )
                    self._annotate_sequential_reset(seq_nodes, coverage)
                else:
                    print(
                        f"警告: 行 {line_no} ({coverage.branch_type}) 没有找到对应的节点"
                    )

        return self.stats

    def propagate_coverage(self) -> AnnotationStats:
        """
        传播覆盖标注到非分支边

        基于可达性分析：
        1. 组合逻辑节点（AND/OR/NOT等）：如果任一输入已标注 → 输出边已执行
        2. MUX 节点：如果 A 或 B 端口有已标注输入 → 输出边继承覆盖状态
        3. 时序节点：D 端口已标注 → Q 输出边已执行
        4. 输入端口边：标记为必然执行（always）

        Returns:
            更新后的标注统计信息
        """
        # 建立图的邻接表
        # node_id -> 输入边列表
        node_inputs: Dict[str, List[Edge]] = defaultdict(list)
        # node_id -> 输出边列表
        node_outputs: Dict[str, List[Edge]] = defaultdict(list)

        for edge in self.cdfg.edges:
            node_inputs[edge.target].append(edge)
            node_outputs[edge.source].append(edge)

        # 标记输入端口边为"必然执行"
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.INPUT:
                for edge in node_outputs.get(node_id, []):
                    if edge.coverage_label == -1:
                        edge.coverage_label = 1
                        edge.coverage_type = "always"
                        self.stats.always_executed_edges += 1
                        self.stats.annotated_edges += 1
                        self.stats.covered_edges += 1

        # 常量节点边也标记为必然执行
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.CONSTANT:
                for edge in node_outputs.get(node_id, []):
                    if edge.coverage_label == -1:
                        edge.coverage_label = 1
                        edge.coverage_type = "always"
                        self.stats.always_executed_edges += 1
                        self.stats.annotated_edges += 1
                        self.stats.covered_edges += 1

        # 迭代传播直到收敛
        changed = True
        iteration = 0
        max_iterations = 100  # 防止无限循环

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for node_id, node in self.cdfg.nodes.items():
                # 获取该节点的输入边和输出边
                inputs = node_inputs.get(node_id, [])
                outputs = node_outputs.get(node_id, [])

                if not outputs:
                    continue

                # 根据节点类型决定传播规则
                if node.node_type == NodeType.LOGIC:
                    # 组合逻辑：任一输入已标注 → 输出已执行
                    # 如果有已覆盖输入，输出为已覆盖；否则如果所有输入都未覆盖，输出也未覆盖
                    any_input_covered = any(e.coverage_label == 1 for e in inputs)
                    any_input_labeled = any(e.coverage_label >= 0 for e in inputs)
                    if any_input_labeled:
                        new_label = 1 if any_input_covered else 0
                        for edge in outputs:
                            if edge.coverage_label == -1:
                                edge.coverage_label = new_label
                                edge.coverage_type = "propagated"
                                self.stats.propagated_edges += 1
                                self.stats.annotated_edges += 1
                                if new_label == 1:
                                    self.stats.covered_edges += 1
                                else:
                                    self.stats.uncovered_edges += 1
                                changed = True

                elif node.node_type == NodeType.MUX:
                    # MUX：检查 A 或 B 端口是否有已标注的输入
                    data_inputs = [e for e in inputs if e.target_port in ("A", "B")]
                    any_data_covered = any(e.coverage_label == 1 for e in data_inputs)
                    any_data_labeled = any(e.coverage_label >= 0 for e in data_inputs)
                    if any_data_labeled:
                        new_label = 1 if any_data_covered else 0
                        for edge in outputs:
                            if edge.coverage_label == -1:
                                edge.coverage_label = new_label
                                edge.coverage_type = "propagated"
                                self.stats.propagated_edges += 1
                                self.stats.annotated_edges += 1
                                if new_label == 1:
                                    self.stats.covered_edges += 1
                                else:
                                    self.stats.uncovered_edges += 1
                                changed = True

                elif node.node_type == NodeType.SEQUENTIAL:
                    # 时序节点：D/AD 端口已标注 或 CLK 端口已标注 → Q 输出已执行
                    relevant_inputs = [
                        e for e in inputs if e.target_port in ("D", "AD", "CLK")
                    ]
                    any_input_covered = any(
                        e.coverage_label == 1 for e in relevant_inputs
                    )
                    any_input_labeled = any(
                        e.coverage_label >= 0 for e in relevant_inputs
                    )
                    if any_input_labeled:
                        new_label = 1 if any_input_covered else 0
                        for edge in outputs:
                            if edge.coverage_label == -1:
                                edge.coverage_label = new_label
                                edge.coverage_type = "propagated"
                                self.stats.propagated_edges += 1
                                self.stats.annotated_edges += 1
                                if new_label == 1:
                                    self.stats.covered_edges += 1
                                else:
                                    self.stats.uncovered_edges += 1
                                changed = True

                elif node.node_type == NodeType.OUTPUT:
                    # 输出端口：不产生输出边，跳过
                    pass

                else:
                    # 其他节点类型：任一输入已标注 → 输出继承状态
                    any_input_covered = any(e.coverage_label == 1 for e in inputs)
                    any_input_labeled = any(e.coverage_label >= 0 for e in inputs)
                    if any_input_labeled:
                        new_label = 1 if any_input_covered else 0
                        for edge in outputs:
                            if edge.coverage_label == -1:
                                edge.coverage_label = new_label
                                edge.coverage_type = "propagated"
                                self.stats.propagated_edges += 1
                                self.stats.annotated_edges += 1
                                if new_label == 1:
                                    self.stats.covered_edges += 1
                                else:
                                    self.stats.uncovered_edges += 1
                                changed = True

        print(f"传播完成，迭代 {iteration} 次")
        return self.stats

    def _find_sequential_nodes_for_coverage(self, coverage_line: int) -> List[str]:
        """
        为覆盖率行找到对应的时序节点

        复位逻辑 (if rst_n == 1'b0) 可能被综合成带异步复位的触发器 ($aldff, $adff 等)

        Args:
            coverage_line: 覆盖率报告中的行号

        Returns:
            时序节点 ID 列表
        """
        # 检查直接匹配和附近行（行号可能有偏差）
        for offset in range(-2, 3):
            candidate_line = coverage_line + offset
            if candidate_line in self.line_to_seq_nodes:
                return self.line_to_seq_nodes[candidate_line]
        return []

    def _annotate_sequential_reset(
        self, seq_nodes: List[str], coverage: BranchCoverage
    ):
        """
        标注时序元件的复位分支

        对于 always_ff @(posedge clk, negedge rst_n) begin
            if (rst_n == 1'b0) begin ... end
            else begin ... end
        end

        综合成 $aldff 时：
        - ALOAD 端口 = 复位信号（对应 if 条件）
        - AD 端口 = 复位值（对应 if-true 分支）
        - D 端口 = 正常数据（对应 else 分支）

        Args:
            seq_nodes: 时序节点 ID 列表
            coverage: 分支覆盖数据
        """
        # 对于简单的 if-else 复位逻辑，通常有 2 个分支
        # 分支 0: rst_n == 0 (复位激活) -> ALOAD=1, 使用 AD 值
        # 分支 1: rst_n == 1 (正常运行) -> ALOAD=0, 使用 D 值

        for branch_idx, branch_status in enumerate(coverage.branches):
            is_covered = branch_status.is_covered

            # 确定是复位分支还是正常分支
            # 条件1=1 表示复位激活，条件1=0 表示正常运行
            is_reset_branch = branch_status.condition_values.get(1, 0) == 1

            for seq_node_id in seq_nodes:
                self._annotate_seq_edges(
                    seq_node_id,
                    branch_idx,
                    is_covered,
                    is_reset_branch,
                    coverage.line_no,
                )

    def _annotate_seq_edges(
        self,
        seq_node_id: str,
        branch_index: int,
        is_covered: bool,
        is_reset_branch: bool,
        coverage_line: int,
    ):
        """
        标注单个时序元件的边

        Args:
            seq_node_id: 时序节点 ID
            branch_index: 分支索引
            is_covered: 是否已覆盖
            is_reset_branch: 是否是复位分支
            coverage_line: 覆盖率报告中的行号
        """
        coverage_label = 1 if is_covered else 0

        for edge in self.cdfg.edges:
            if edge.target != seq_node_id:
                continue

            # ALOAD/ARST 端口 - 复位控制信号
            if edge.target_port in ("ALOAD", "ARST", "SRST", "RST"):
                edge.source_line = coverage_line
                edge.branch_index = branch_index
                edge.coverage_label = coverage_label
                edge.coverage_type = "control"
                self.stats.annotated_edges += 1
                self.stats.control_edges += 1
                if is_covered:
                    self.stats.covered_edges += 1
                else:
                    self.stats.uncovered_edges += 1

            # AD 端口 - 复位值（复位分支）
            elif edge.target_port == "AD" and is_reset_branch:
                edge.source_line = coverage_line
                edge.branch_index = branch_index
                edge.coverage_label = coverage_label
                edge.coverage_type = "data_true"  # 复位分支 = true 分支
                self.stats.annotated_edges += 1
                self.stats.data_true_edges += 1
                if is_covered:
                    self.stats.covered_edges += 1
                else:
                    self.stats.uncovered_edges += 1

            # D 端口 - 正常数据（非复位分支）
            elif edge.target_port == "D" and not is_reset_branch:
                edge.source_line = coverage_line
                edge.branch_index = branch_index
                edge.coverage_label = coverage_label
                edge.coverage_type = "data_false"  # 正常分支 = false 分支
                self.stats.annotated_edges += 1
                self.stats.data_false_edges += 1
                if is_covered:
                    self.stats.covered_edges += 1
                else:
                    self.stats.uncovered_edges += 1

    def _sort_mux_nodes_topologically(self, mux_nodes: List[str]) -> List[str]:
        """
        对 MUX 节点进行拓扑排序

        对于 if-else-if 链，Yosys 会生成串联的 MUX：
        第一个条件的 MUX 输出连接到第二个条件 MUX 的 A 端口（false 分支）

        Returns:
            按拓扑顺序排列的 MUX 节点 ID 列表
        """
        if len(mux_nodes) <= 1:
            return mux_nodes

        # 建立 MUX 之间的依赖关系
        # 如果 MUX_A 的输出连接到 MUX_B 的输入，则 MUX_A 在 MUX_B 之前
        mux_set = set(mux_nodes)
        dependencies: Dict[str, set] = {node: set() for node in mux_nodes}

        for edge in self.cdfg.edges:
            if edge.source in mux_set and edge.target in mux_set:
                # edge.source 应该在 edge.target 之前
                dependencies[edge.target].add(edge.source)

        # 拓扑排序
        result = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            result.append(node)

        for node in mux_nodes:
            visit(node)

        return result

    def _annotate_mux_chain(self, mux_nodes: List[str], coverage: BranchCoverage):
        """
        标注 MUX 链

        Args:
            mux_nodes: MUX 节点 ID 列表（已排序）
            coverage: 分支覆盖数据
        """
        # 对于简单的 if-else（2个分支），只有一个 MUX
        # 对于 if-else-if 链（n个分支），有 n-1 个 MUX

        # 分支覆盖数据中的每个分支对应一个条件组合
        # 分支 i 表示：条件 1..i-1 为假，条件 i 为真

        # 建立条件编号到 MUX 索引的映射
        # 条件编号可能不从 1 开始，也可能不连续（如 CASE 语句）
        all_conditions = set()
        for br in coverage.branches:
            all_conditions.update(br.condition_values.keys())

        # 按条件编号排序，建立映射
        sorted_conditions = sorted(all_conditions)
        cond_to_mux_idx = {cond: idx for idx, cond in enumerate(sorted_conditions)}

        for branch_idx, branch_status in enumerate(coverage.branches):
            # 找到这个分支对应的 MUX
            # 对于 if-else-if 链：
            # - 分支 0: 条件1=1 -> MUX 0 的 B 端口（true 分支）
            # - 分支 1: 条件1=0, 条件2=1 -> MUX 1 的 B 端口
            # - ...
            # - 最后一个分支: 所有条件为 0 -> 最后一个 MUX 的 A 端口（else 分支）

            # 对于复杂的 CASE 语句（嵌套多层 if-else）:
            # condition_values 包含路径上所有条件的值
            # 需要标注路径上所有为真的条件对应的 MUX

            # 收集所有条件值
            true_conditions = []
            false_conditions = []
            for cond_idx, cond_val in sorted(branch_status.condition_values.items()):
                if cond_val == 1:
                    true_conditions.append(cond_idx)
                elif cond_val == 0:
                    false_conditions.append(cond_idx)
                # cond_val == -1 表示无关，跳过

            if true_conditions:
                # 标注所有为真的条件对应的 MUX 的 B 端口
                for cond in true_conditions:
                    # 使用映射表获取 MUX 索引
                    mux_idx = cond_to_mux_idx.get(cond, cond - 1)
                    if 0 <= mux_idx < len(mux_nodes):
                        self._annotate_mux_edges(
                            mux_nodes[mux_idx],
                            branch_idx,
                            branch_status.is_covered,
                            is_true_branch=True,
                            coverage_line=coverage.line_no,
                        )

                # 标注所有为假的条件对应的 MUX 的 A 端口
                for cond in false_conditions:
                    mux_idx = cond_to_mux_idx.get(cond, cond - 1)
                    if 0 <= mux_idx < len(mux_nodes):
                        self._annotate_mux_edges(
                            mux_nodes[mux_idx],
                            branch_idx,
                            branch_status.is_covered,
                            is_true_branch=False,
                            coverage_line=coverage.line_no,
                        )
            else:
                # 所有条件都为 0 或无关，这是 else 分支
                # 执行 else 分支时，整个链上所有 MUX 的 A 端口都被执行
                for mux_node in mux_nodes:
                    self._annotate_mux_edges(
                        mux_node,
                        branch_idx,
                        branch_status.is_covered,
                        is_true_branch=False,
                        coverage_line=coverage.line_no,
                    )

    def _annotate_mux_edges(
        self,
        mux_node_id: str,
        branch_index: int,
        is_covered: bool,
        is_true_branch: bool,
        coverage_line: int,
    ):
        """
        标注单个 MUX 的边

        Args:
            mux_node_id: MUX 节点 ID
            branch_index: 分支索引
            is_covered: 是否已覆盖
            is_true_branch: 是 true 分支还是 false 分支
            coverage_line: 覆盖率报告中的行号
        """
        coverage_label = 1 if is_covered else 0

        for edge in self.cdfg.edges:
            if edge.target != mux_node_id:
                continue

            # S 端口 - 控制信号
            if edge.target_port == "S":
                # 使用或逻辑：如果已经是已覆盖(1)，不要被未覆盖(0)覆盖
                if edge.coverage_label == -1:
                    # 首次标注
                    edge.source_line = coverage_line
                    edge.branch_index = branch_index
                    edge.coverage_label = coverage_label
                    edge.coverage_type = "control"
                    self.stats.annotated_edges += 1
                    self.stats.control_edges += 1
                    if is_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1
                elif edge.coverage_label == 0 and is_covered:
                    # 之前标注为未覆盖，现在发现有覆盖，更新为已覆盖
                    edge.coverage_label = 1
                    self.stats.covered_edges += 1
                    self.stats.uncovered_edges -= 1

            # A 端口 - false 分支
            elif edge.target_port == "A" and not is_true_branch:
                if edge.coverage_label == -1:
                    edge.source_line = coverage_line
                    edge.branch_index = branch_index
                    edge.coverage_label = coverage_label
                    edge.coverage_type = "data_false"
                    self.stats.annotated_edges += 1
                    self.stats.data_false_edges += 1
                    if is_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1
                elif edge.coverage_label == 0 and is_covered:
                    edge.coverage_label = 1
                    self.stats.covered_edges += 1
                    self.stats.uncovered_edges -= 1

            # B 端口 - true 分支
            elif edge.target_port == "B" and is_true_branch:
                if edge.coverage_label == -1:
                    edge.source_line = coverage_line
                    edge.branch_index = branch_index
                    edge.coverage_label = coverage_label
                    edge.coverage_type = "data_true"
                    self.stats.annotated_edges += 1
                    self.stats.data_true_edges += 1
                    if is_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1
                elif edge.coverage_label == 0 and is_covered:
                    edge.coverage_label = 1
                    self.stats.covered_edges += 1
                    self.stats.uncovered_edges -= 1

        # 只在第一次标注该 MUX 时计数
        if mux_node_id not in self._annotated_mux_set:
            self._annotated_mux_set.add(mux_node_id)
            self.stats.annotated_mux_nodes += 1

    def annotate_condition_coverage(self) -> AnnotationStats:
        """
        标注条件覆盖率数据

        条件覆盖率对应三元表达式和逻辑表达式，这些在 CDFG 中通常表示为 MUX 节点。
        三元表达式 `cond ? a : b` 对应单个 MUX。
        逻辑表达式 `a && b` 或 `a || b` 可能对应多个 MUX 的组合。

        Returns:
            更新后的标注统计信息
        """
        if not self.condition_data:
            return self.stats

        print(f"\n--- 条件覆盖率标注 ---")
        print(f"共有 {len(self.condition_data)} 行的条件覆盖数据")

        annotated_count = 0
        for line_no, cond_list in self.condition_data.items():
            # 跳过已被分支覆盖率标注过的行
            if line_no in self.coverage_data:
                continue

            for cond_cov in cond_list:
                # 只处理主表达式，跳过子表达式（避免重复标注）
                if cond_cov.expression_type != "EXPRESSION":
                    continue

                # 查找对应的 MUX 节点
                mux_nodes = self._find_mux_for_condition(line_no, cond_cov)
                if not mux_nodes:
                    continue

                # 标注 MUX 节点
                annotated = self._annotate_condition_mux(mux_nodes, cond_cov)
                if annotated:
                    annotated_count += 1
                    print(f"  行 {line_no}: 标注 {len(mux_nodes)} 个 MUX (条件覆盖)")

        print(f"条件覆盖率标注完成: {annotated_count} 个表达式")
        return self.stats

    def _find_mux_for_condition(
        self, coverage_line: int, cond_cov: ConditionCoverage
    ) -> List[str]:
        """
        为条件覆盖率找到对应的 MUX 节点

        Args:
            coverage_line: 覆盖率报告中的行号
            cond_cov: 条件覆盖信息

        Returns:
            MUX 节点 ID 列表
        """
        # 使用学习到的偏移量
        if self._learned_offset is not None:
            adjusted_line = coverage_line + self._learned_offset
            if adjusted_line in self.line_to_mux_nodes:
                mux_list = self.line_to_mux_nodes[adjusted_line]
                # 条件表达式通常只对应一个或少量 MUX
                return mux_list[:cond_cov.num_conditions]

        # 搜索 source_line
        for offset in self.SOURCE_LINE_SEARCH_RANGE:
            candidate_line = coverage_line + offset
            if candidate_line in self.line_to_mux_nodes:
                mux_list = self.line_to_mux_nodes[candidate_line]
                if mux_list:
                    # 学习偏移量
                    if self._learned_offset is None and offset != 0:
                        self._learned_offset = offset
                        print(f"  学习到行号偏移: {offset}")
                    return mux_list[:cond_cov.num_conditions]

        return []

    def _annotate_condition_mux(
        self, mux_nodes: List[str], cond_cov: ConditionCoverage
    ) -> bool:
        """
        标注条件覆盖率对应的 MUX 节点

        Args:
            mux_nodes: MUX 节点 ID 列表
            cond_cov: 条件覆盖信息

        Returns:
            是否成功标注
        """
        if not mux_nodes:
            return False

        # 分析覆盖状态
        # 对于条件覆盖率，我们关注整体表达式的 true/false 覆盖情况
        true_covered = False
        false_covered = False

        for cond_status in cond_cov.conditions:
            if cond_status.is_covered:
                # 检查条件组合的结果
                # 对于单条件：{1: 0} 表示 false, {1: 1} 表示 true
                # 对于多条件：需要根据运算符判断（暂时简化处理）
                if cond_cov.num_conditions == 1:
                    val = cond_status.condition_values.get(1, 0)
                    if val == 1:
                        true_covered = True
                    else:
                        false_covered = True
                else:
                    # 多条件情况，检查是否有任一条件为真
                    any_true = any(v == 1 for v in cond_status.condition_values.values())
                    if any_true:
                        true_covered = True
                    else:
                        false_covered = True

        # 标注 MUX 节点的边
        for mux_node_id in mux_nodes:
            for edge in self.cdfg.edges:
                if edge.target != mux_node_id:
                    continue

                # 跳过已标注的边
                if edge.coverage_label >= 0:
                    continue

                # S 端口 - 控制信号
                if edge.target_port == "S":
                    # 控制信号已被评估（无论 true 还是 false）
                    is_covered = true_covered or false_covered
                    edge.source_line = cond_cov.line_no
                    edge.coverage_label = 1 if is_covered else 0
                    edge.coverage_type = "condition"
                    self.stats.condition_annotated_edges += 1
                    self.stats.annotated_edges += 1
                    if is_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1

                # B 端口 - true 分支
                elif edge.target_port == "B":
                    edge.source_line = cond_cov.line_no
                    edge.coverage_label = 1 if true_covered else 0
                    edge.coverage_type = "condition_true"
                    self.stats.condition_annotated_edges += 1
                    self.stats.annotated_edges += 1
                    if true_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1

                # A 端口 - false 分支
                elif edge.target_port == "A":
                    edge.source_line = cond_cov.line_no
                    edge.coverage_label = 1 if false_covered else 0
                    edge.coverage_type = "condition_false"
                    self.stats.condition_annotated_edges += 1
                    self.stats.annotated_edges += 1
                    if false_covered:
                        self.stats.covered_edges += 1
                    else:
                        self.stats.uncovered_edges += 1

            # 统计 MUX 节点
            if mux_node_id not in self._annotated_mux_set:
                self._annotated_mux_set.add(mux_node_id)
                self.stats.annotated_mux_nodes += 1
                self.stats.condition_annotated_mux += 1

        return True


def annotate_cdfg_with_coverage(
    cdfg: CDFG,
    html_path: str,
    instance_tag: str = None,
    propagate: bool = False,
    use_condition_coverage: bool = True,
) -> AnnotationStats:
    """
    便捷函数：用覆盖率数据标注 CDFG

    Args:
        cdfg: CDFG 对象
        html_path: 覆盖率报告 HTML 文件路径
        instance_tag: 实例标签，None 则使用第一个实例
        propagate: 是否启用覆盖率传播（标注非分支边）
        use_condition_coverage: 是否使用条件覆盖率（标注三元表达式等）

    Returns:
        标注统计信息
    """
    # 解析覆盖率报告
    parser = CoverageParser(html_path)

    if instance_tag is None:
        instance_tag = parser.get_last_instance()

    # 解析分支覆盖率
    coverage_data = parser.parse_branch_coverage(instance_tag)

    # 解析条件覆盖率
    condition_data = {}
    if use_condition_coverage:
        condition_data = parser.parse_condition_coverage(instance_tag)

    # 获取源文件名用于过滤 MUX 节点
    source_file = parser.get_source_file()

    # 执行标注
    annotator = CoverageAnnotator(
        cdfg, coverage_data, source_file=source_file, condition_data=condition_data
    )
    stats = annotator.annotate()

    # 标注条件覆盖率
    if use_condition_coverage and condition_data:
        stats = annotator.annotate_condition_coverage()

    # 可选：传播覆盖率到非分支边
    if propagate:
        stats = annotator.propagate_coverage()

    return stats
