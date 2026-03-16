#!/usr/bin/env python3
"""
覆盖率标注器
将覆盖率报告中的分支覆盖数据映射到 CDFG 的边上
"""

from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
import os
import re
from cdfg_rtl import CDFG, NodeType
from .parser import CoverageParser, BranchCoverage


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


class CoverageAnnotator:
    """覆盖率标注器"""

    def __init__(
        self,
        cdfg: CDFG,
        coverage_data: List[BranchCoverage],
        source_file: str = None,
        source_dir: str = None,
    ):
        """
        初始化标注器

        Args:
            cdfg: CDFG 对象
            coverage_data: BranchCoverage 列表（支持同行多实例）
            source_file: 覆盖率报告对应的源文件名（用于过滤 MUX 节点）
            source_dir: 设计源码目录路径（用于精确范围匹配）
        """
        self.cdfg = cdfg
        self.coverage_data = coverage_data
        self.source_file = source_file
        self.source_dir = source_dir
        self.stats = AnnotationStats()

        # 追踪已标注的 MUX 节点（避免重复计数）
        self._annotated_mux_set: set = set()

        # 源文件行缓存（避免重复读取）
        self._source_lines_cache: Dict[str, List[str]] = {}

        # 建立行号到 MUX 节点的映射
        self.line_to_mux_nodes: Dict[int, List[str]] = defaultdict(list)
        self._build_line_mapping()

    def _apply_edge_label(
        self,
        edge,
        coverage_line: int,
        branch_index: int,
        is_covered: bool,
        coverage_type: str,
    ):
        """
        标注单条边并更新统计计数器（带覆盖状态保护）

        - 首次标注（coverage_label == -1）：设置标签并计数
        - 升级（0 → 1）：仅当新状态为已覆盖时更新
        - 其他情况：跳过（防止降级）
        """
        coverage_label = 1 if is_covered else 0

        if edge.coverage_label == -1:
            edge.source_line = coverage_line
            edge.branch_index = branch_index
            edge.coverage_label = coverage_label
            edge.coverage_type = coverage_type
            self.stats.annotated_edges += 1
            if coverage_type == "control":
                self.stats.control_edges += 1
            elif coverage_type == "data_true":
                self.stats.data_true_edges += 1
            elif coverage_type == "data_false":
                self.stats.data_false_edges += 1
            if is_covered:
                self.stats.covered_edges += 1
            else:
                self.stats.uncovered_edges += 1
        elif edge.coverage_label == 0 and is_covered:
            edge.coverage_label = 1
            self.stats.covered_edges += 1
            self.stats.uncovered_edges -= 1

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

        # 建立 MUX 节点到其控制信号（S 端口）驱动源的映射
        self._mux_to_control_source: Dict[str, str] = {}
        self._build_mux_control_mapping()

    def _build_mux_control_mapping(self):
        """建立 MUX 节点到其控制信号源的映射"""
        for edge in self.cdfg.edges:
            if edge.target_port == "S":
                # edge.source 是控制信号的驱动节点
                self._mux_to_control_source[edge.target] = edge.source

    def _get_mux_control_source(self, mux_node_id: str) -> str:
        """获取 MUX 节点的控制信号源节点 ID"""
        return self._mux_to_control_source.get(mux_node_id, "")

    def _group_mux_by_control_signal(
        self, mux_nodes: List[str]
    ) -> Dict[str, List[str]]:
        """
        按控制信号源分组 MUX 节点

        当一个 IF 条件同时驱动多个数据路径时，Yosys 会为每个输出生成独立的 MUX，
        但这些 MUX 共享相同的控制信号（S 端口）。

        Args:
            mux_nodes: MUX 节点 ID 列表

        Returns:
            控制信号源 -> MUX 节点列表的映射
        """
        groups: Dict[str, List[str]] = defaultdict(list)
        for mux_id in mux_nodes:
            control_source = self._get_mux_control_source(mux_id)
            groups[control_source].append(mux_id)
        return groups

    def _count_unique_control_signals(self, mux_nodes: List[str]) -> int:
        """
        统计 MUX 节点中唯一控制信号源的数量

        这代表了实际的独立分支条件数量，而非 MUX 节点总数。

        Args:
            mux_nodes: MUX 节点 ID 列表

        Returns:
            唯一控制信号源的数量
        """
        groups = self._group_mux_by_control_signal(mux_nodes)
        return len(groups)

    def _find_mux_chain_for_coverage(
        self, coverage_line: int, total_branches: int, branch_type: str = "IF"
    ) -> List[str]:
        """
        为覆盖率行找到对应的 MUX 链

        对于 if-else-if 链，覆盖率报告只记录起始行号（如 96），
        但 CDFG 中每个 else-if 对应不同行号（96, 97, 98...）。

        对于 CASE 语句，覆盖率报告记录的是 case 语句本身的行号，
        而 CDFG 中 MUX 节点记录的是各个 case 分支项的行号。
        VCS 会将 CASE 语句内部的所有 if-else 合并到一个覆盖率条目中。

        对于三元表达式，覆盖率报告记录的是表达式所在的行号，
        需要解析 ? : 语法来确定范围。

        Args:
            coverage_line: 覆盖率报告中的行号
            total_branches: 总分支数（用于确定链的长度）
            branch_type: 分支类型 ("IF", "CASE" 或 "TERNARY")

        Returns:
            MUX 节点 ID 列表
        """
        expected_mux_count = total_branches - 1
        best_match = []
        best_match_score = 0

        # 收集所有 MUX 的行号
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())
        if not all_mux_lines:
            return []

        # 根据分支类型使用不同的收集策略
        if branch_type == "CASE":
            # CASE 语句：使用 case/endcase 范围解析
            collected = self._collect_case_mux_nodes(coverage_line, expected_mux_count)
            if collected:
                return collected
        elif branch_type == "IF":
            # IF 语句：使用 if/begin/end 范围解析
            collected = self._collect_if_mux_nodes(coverage_line, expected_mux_count)
            if collected:
                return collected
            elif collected is None:
                # 成功解析到范围但没有 MUX（可能是时序逻辑），不回退到启发式方法
                return []
        elif branch_type == "TERNARY":
            # 三元表达式：使用 ? : 范围解析
            collected = self._collect_ternary_mux_nodes(
                coverage_line, expected_mux_count
            )
            if collected:
                return collected

        # 如果精确范围解析失败，回退到启发式方法
        # 直接在 coverage_line 行搜索 MUX
        collected = self._collect_continuous_mux_chain(
            coverage_line, expected_mux_count
        )

        if len(collected) >= expected_mux_count:
            # 找到足够的 MUX
            # 添加合理性检查：如果找到的 MUX 数量远超期望，可能是匹配错误
            if expected_mux_count > 0 and len(collected) > expected_mux_count * 3:
                pass  # 数量不合理，跳过
            else:
                # 返回所有收集到的 MUX，而不是只取 expected_mux_count 个
                # 因为 Yosys 可能为数据路径复制生成额外的 MUX
                return self._sort_mux_nodes_for_annotation(collected)

        # 记录最佳匹配（即使不够）
        if len(collected) > best_match_score:
            if expected_mux_count == 0 or len(collected) <= expected_mux_count * 3:
                best_match = collected
                best_match_score = len(collected)

        # 尝试通过 stmt_start_line 查找
        if coverage_line in self.stmt_line_to_mux_nodes:
            stmt_mux = self.stmt_line_to_mux_nodes[coverage_line]
            if len(stmt_mux) >= expected_mux_count:
                return self._sort_mux_nodes_for_annotation(
                    stmt_mux[:expected_mux_count]
                )
            if len(stmt_mux) > best_match_score:
                best_match = stmt_mux
                best_match_score = len(stmt_mux)

        # 返回最佳匹配（即使不完整）
        if best_match:
            print(
                f"  警告：期望 {expected_mux_count} 个 MUX，但只找到 {len(best_match)} 个"
            )
            return self._sort_mux_nodes_for_annotation(best_match)

        return []

    def _collect_mux_in_range(self, start_line: int, end_line: int) -> List[str]:
        """
        收集指定行号范围内的所有 MUX 节点

        同时使用 source_line 和 stmt_start_line 两种映射，
        避免因行号来源不同而遗漏节点。

        Args:
            start_line: 起始行号（1-based，含）
            end_line: 结束行号（1-based，含）

        Returns:
            去重的 MUX 节点 ID 列表（保持插入顺序）
        """
        collected = []
        collected_set: set = set()

        # 1. 通过 source_line 收集
        for line in sorted(self.line_to_mux_nodes.keys()):
            if start_line <= line <= end_line:
                for mux_id in self.line_to_mux_nodes[line]:
                    if mux_id not in collected_set:
                        collected.append(mux_id)
                        collected_set.add(mux_id)

        # 2. 通过 stmt_start_line 收集
        for line in sorted(self.stmt_line_to_mux_nodes.keys()):
            if start_line <= line <= end_line:
                for mux_id in self.stmt_line_to_mux_nodes[line]:
                    if mux_id not in collected_set:
                        collected.append(mux_id)
                        collected_set.add(mux_id)

        return collected

    def _collect_case_mux_nodes(
        self, coverage_line: int, expected_mux_count: int
    ) -> List[str]:
        """
        为 CASE 语句收集所有相关的 MUX 节点

        VCS 将 CASE 语句和其内部的所有 if-else 合并成一个覆盖率条目。
        需要收集从 case 语句开始到 endcase 之间的所有 MUX。

        对于 CASE 语句，CDFG 中可能有比条件数更多的 MUX，因为：
        1. 每个 case 分支内的 if-else 会生成 MUX
        2. 数据路径复制会产生多个 MUX

        我们需要收集所有相关的 MUX，而不是只取 expected_mux_count 个。

        Args:
            coverage_line: 覆盖率报告中的行号（case 语句的行号）
            expected_mux_count: 期望的 MUX 数量（仅用于参考）

        Returns:
            MUX 节点 ID 列表（按行号排序）
        """
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())
        if not all_mux_lines:
            return []

        # 尝试从源文件中找到 case/endcase 的范围
        case_range = self._find_case_range_from_source(coverage_line)

        if case_range:
            case_start, case_end = case_range
            print(f"  从源码解析到 case 范围: {case_start}-{case_end}")

            collected = self._collect_mux_in_range(case_start, case_end)

            if collected:
                return self._sort_mux_nodes_for_annotation(collected)

        # 如果无法从源码解析，使用启发式方法
        # 收集从 coverage_line 开始的一定范围内的所有 MUX
        # CASE 语句通常跨越较大的行号范围（可能 100+ 行）
        collected = []
        for line in all_mux_lines:
            if line >= coverage_line and line <= coverage_line + 150:
                collected.extend(self.line_to_mux_nodes[line])

        # 返回所有收集到的 MUX（不限制数量）
        if collected:
            # 按拓扑顺序排序并返回所有 MUX
            return self._sort_mux_nodes_for_annotation(collected)

        return []

    def _collect_if_mux_nodes(
        self, coverage_line: int, expected_mux_count: int
    ) -> List[str]:
        """
        为 IF 语句收集所有相关的 MUX 节点

        通过解析源码中的 if/else/begin/end 关键字来确定精确范围，
        然后收集该范围内的所有 MUX 节点。

        验证策略（方案 B）：按控制信号分组
        - 当一个 IF 条件同时驱动多个数据路径时，Yosys 为每个输出生成独立的 MUX
        - 这些 MUX 共享相同的控制信号（S 端口），应被视为一组
        - 验证时比较"唯一控制信号数"而非"MUX 节点总数"

        Args:
            coverage_line: 覆盖率报告中的行号（if 语句的行号）
            expected_mux_count: 期望的 MUX 数量（仅用于参考）

        Returns:
            MUX 节点 ID 列表（按拓扑顺序排序）
            返回 None 表示成功解析到范围但没有 MUX（可能是时序逻辑）
        """
        if not self.line_to_mux_nodes:
            return []

        # 尝试从源文件中找到 if 语句的范围
        if_range = self._find_if_range_from_source(coverage_line)

        if if_range:
            if_start, if_end = if_range
            print(f"  从源码解析到 if 范围: {if_start}-{if_end}")

            collected = self._collect_mux_in_range(if_start, if_end)

            if collected:
                # 方案 B：按控制信号分组验证
                # 统计唯一控制信号数，而非 MUX 节点总数
                unique_control_count = self._count_unique_control_signals(collected)

                # 如果唯一控制信号数与期望 MUX 数匹配或接近，说明匹配正确
                # 允许一定容差（3 倍）用于处理嵌套 if-else 等情况
                if (
                    expected_mux_count > 0
                    and unique_control_count > expected_mux_count * 3
                ):
                    print(
                        f"  警告: 收集到 {len(collected)} 个 MUX（{unique_control_count} 个独立控制信号），"
                        f"但期望 {expected_mux_count} 个，可能是行号匹配错误"
                    )
                    # 检查是否有更精确的匹配
                    # 尝试在更小的范围内查找
                    narrow_collected = self._collect_mux_in_range(
                        if_start, if_start + 10
                    )

                    if narrow_collected:
                        narrow_control_count = self._count_unique_control_signals(
                            narrow_collected
                        )
                        if narrow_control_count <= expected_mux_count * 2:
                            print(
                                f"  使用精确范围匹配: 收集到 {len(narrow_collected)} 个 MUX（{narrow_control_count} 个独立控制信号）"
                            )
                            return self._sort_mux_nodes_for_annotation(narrow_collected)

                    # 如果仍然不匹配，返回 None 表示已解析但无有效 MUX
                    return None

                # 匹配成功：打印详细信息
                if (
                    len(collected) > expected_mux_count
                    and unique_control_count <= expected_mux_count
                ):
                    print(
                        f"  数据路径复制检测: {len(collected)} 个 MUX 共享 {unique_control_count} 个控制信号（期望 {expected_mux_count} 个）"
                    )

                return self._sort_mux_nodes_for_annotation(collected)
            else:
                # 成功解析到 IF 范围，但范围内没有 MUX
                # 这可能是时序逻辑（如 always_ff 中的复位），返回 None 避免启发式错误匹配
                print(
                    f"  IF 范围 {if_start}-{if_end} 内没有 MUX 节点（可能是时序逻辑）"
                )
                return None

        # 如果无法从源码解析，返回空列表，让调用者回退到启发式方法
        return []

    def _collect_ternary_mux_nodes(
        self, coverage_line: int, expected_mux_count: int
    ) -> List[str]:
        """
        为三元表达式收集所有相关的 MUX 节点

        三元表达式 (cond ? true_val : false_val) 通常在单行或跨越少数几行。
        通过解析源码中的 ? : 语法来确定范围。

        对于嵌套的三元表达式，需要找到所有相关的 MUX。

        Args:
            coverage_line: 覆盖率报告中的行号（三元表达式的行号）
            expected_mux_count: 期望的 MUX 数量（仅用于参考）

        Returns:
            MUX 节点 ID 列表（按拓扑顺序排序）
        """
        if not self.line_to_mux_nodes:
            return []

        # 尝试从源文件中找到三元表达式的范围，带 MUX 数量验证
        ternary_range = self._find_ternary_range_from_source(
            coverage_line, expected_mux_count
        )

        if ternary_range:
            ternary_start, ternary_end = ternary_range
            print(f"  从源码解析到三元表达式范围: {ternary_start}-{ternary_end}")

            collected = self._collect_mux_in_range(ternary_start, ternary_end)

            if collected:
                return self._sort_mux_nodes_for_annotation(collected)

        # 如果无法从源码解析，返回空列表，让调用者回退到启发式方法
        return []

    def _find_ternary_range_from_source(
        self, coverage_line: int, expected_mux_count: int = 0
    ) -> tuple:
        """
        从源文件中找到三元表达式的范围

        通过解析源码中的 ? : 语法来确定范围。
        处理嵌套的三元表达式和跨行的情况。

        Args:
            coverage_line: 覆盖率报告中的行号
            expected_mux_count: 期望的 MUX 数量（用于验证匹配正确性）

        Returns:
            (ternary_start, ternary_end) 元组，如果找不到则返回 None
        """
        lines = self._read_source_lines()
        if not lines:
            return None

        all_mux_lines = sorted(self.line_to_mux_nodes.keys())

        # 直接在 coverage_line 行搜索
        search_line = coverage_line - 1  # 转换为 0-based 索引

        if search_line < 0 or search_line >= len(lines):
            return None

        line_content = lines[search_line]

        # 检查是否包含三元表达式的 ? 操作符
        # 注意：
        # 1. 需要排除注释中的 ?
        # 2. 三元表达式可能跨行，: 可能在后续行，所以只检查 ? 即可
        if "?" in line_content:
            # 简单检查：确保 ? 不在注释中
            comment_pos = line_content.find("//")
            question_pos = line_content.find("?")

            if comment_pos == -1 or question_pos < comment_pos:
                ternary_start = search_line + 1  # 转换回 1-based 行号

                # 找到三元表达式的结束位置
                ternary_end = self._find_ternary_end(lines, search_line)

                if ternary_end:
                    # 计算该范围内的 MUX 数量
                    mux_count = 0
                    for line in all_mux_lines:
                        if line >= ternary_start and line <= ternary_end:
                            mux_count += len(self.line_to_mux_nodes[line])

                    # 只有当 MUX 数量大于 0 时才返回
                    if mux_count > 0:
                        return (ternary_start, ternary_end)

        return None

    def _find_ternary_end(self, lines: List[str], start_line: int) -> int:
        """
        从三元表达式开始位置找到其结束位置

        处理嵌套的三元表达式和跨行的情况。

        嵌套三元表达式示例：
        1. 简单嵌套：a ? b : (c ? d : e)
        2. 多层嵌套：a ? (b ? c : d) : (e ? f : g)
        3. 跨行嵌套：
           sel1 ? val1 :
           sel2 ? val2 :
                  val3

        Args:
            lines: 源文件行列表
            start_line: 三元表达式起始行索引（0-based）

        Returns:
            三元表达式结束的行号（1-based），如果找不到则返回 None
        """
        # 三元表达式可能跨越多行
        # 需要匹配括号和 ? : 的配对

        # 策略：
        # 1. 跟踪括号深度 () 和 []
        # 2. 对于 ?，增加待匹配的冒号计数
        # 3. 对于 :，需要区分三元表达式的冒号和位选择的冒号
        #    - 位选择的冒号在 [] 内部
        #    - 三元表达式的冒号在 [] 外部
        # 4. 当所有 ? 都找到匹配的 : 时，表达式结束

        paren_depth = 0
        bracket_depth = 0
        ternary_depth = 0  # 未匹配的 ? 数量（嵌套深度）

        for i in range(start_line, min(start_line + 30, len(lines))):
            line_content = lines[i]

            # 跳过注释部分
            comment_pos = line_content.find("//")
            if comment_pos != -1:
                line_content = line_content[:comment_pos]

            # 处理字符串字面量（简单处理：跳过引号内的内容）
            in_string = False
            j = 0
            while j < len(line_content):
                char = line_content[j]

                # 处理字符串
                if char == '"' and (j == 0 or line_content[j - 1] != "\\"):
                    in_string = not in_string
                    j += 1
                    continue

                if in_string:
                    j += 1
                    continue

                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "?":
                    # 新的三元表达式开始
                    ternary_depth += 1
                elif char == ":":
                    # 区分三元表达式的冒号和位选择的冒号
                    # 位选择的冒号在 [] 内部，如 data[7:0]
                    if bracket_depth == 0:
                        # 这是三元表达式的冒号
                        if ternary_depth > 0:
                            ternary_depth -= 1
                    # 如果 bracket_depth > 0，这是位选择的冒号，忽略
                elif char == ";":
                    # 分号表示语句结束
                    return i + 1  # 转换为 1-based 行号
                elif char == ",":
                    # 逗号可能表示表达式结束（在函数参数或数组初始化中）
                    # 只有当所有三元表达式都已匹配时才结束
                    if ternary_depth == 0 and paren_depth <= 0:
                        return i + 1

                j += 1

            # 检查行尾是否有悬挂的操作符（表示表达式继续到下一行）
            # 链式三元表达式通常以 `:` 结尾继续到下一行
            line_stripped = line_content.strip()
            has_trailing_colon = line_stripped.endswith(":")
            has_trailing_question = line_stripped.endswith("?")

            # 检查是否找到了完整的三元表达式
            # 条件：所有 ? 都找到了匹配的 :，且括号平衡，且没有悬挂操作符
            if ternary_depth == 0 and paren_depth <= 0 and i > start_line:
                if not has_trailing_colon and not has_trailing_question:
                    return i + 1  # 转换为 1-based 行号

        # 如果没有找到明确的结束，返回起始行后的几行
        return min(start_line + 10, len(lines))

    def _find_case_range_from_source(self, coverage_line: int) -> tuple:
        """
        从源文件中找到 case 语句的范围

        通过解析源码中的 case 和 endcase 关键字来确定范围。

        Args:
            coverage_line: 覆盖率报告中的行号

        Returns:
            (case_start, case_end) 元组，如果找不到则返回 None
        """
        lines = self._read_source_lines()
        if not lines:
            return None

        # 直接在 coverage_line 行搜索
        search_line = coverage_line - 1  # 转换为 0-based 索引

        if search_line < 0 or search_line >= len(lines):
            return None

        line_content = lines[search_line].strip().lower()

        # 检查是否是 case 语句
        if "case" in line_content and "endcase" not in line_content:
            case_start = search_line + 1  # 转换回 1-based 行号

            # 向下搜索 endcase
            endcase_line = self._find_endcase(lines, search_line)
            if endcase_line:
                case_end = endcase_line
                return (case_start, case_end)

        return None

    def _find_endcase(self, lines: List[str], case_line: int) -> int:
        """
        从 case 语句开始向下搜索 endcase

        处理嵌套的 case 语句。

        Args:
            lines: 源文件行列表
            case_line: case 语句的行索引（0-based）

        Returns:
            endcase 的行号（1-based），如果找不到则返回 None
        """
        nesting_level = 1

        for i in range(case_line + 1, len(lines)):
            line_content = lines[i].strip().lower()

            # 跳过注释
            if line_content.startswith("//"):
                continue

            # 检查嵌套的 case
            if "case" in line_content and "endcase" not in line_content:
                # 确保是 case 关键字而不是变量名的一部分
                if re.search(r"\bcase[xz]?\s*\(", line_content):
                    nesting_level += 1

            # 检查 endcase
            if "endcase" in line_content:
                nesting_level -= 1
                if nesting_level == 0:
                    return i + 1  # 转换为 1-based 行号

        return None

    def _find_if_range_from_source(self, coverage_line: int) -> tuple:
        """
        从源文件中找到 if 语句的范围

        通过解析源码中的 if/else/end 关键字来确定范围。

        Args:
            coverage_line: 覆盖率报告中的行号

        Returns:
            (if_start, if_end) 元组，如果找不到则返回 None
        """
        lines = self._read_source_lines()
        if not lines:
            return None

        # 直接在 coverage_line 行搜索
        search_line = coverage_line - 1  # 转换为 0-based 索引

        if search_line < 0 or search_line >= len(lines):
            return None

        line_content = lines[search_line].strip().lower()

        # 检查是否是 if 语句（排除 else if）
        # 匹配 "if (" 或 "if(" 但不匹配 "else if"
        if re.search(r"(?<!else\s)\bif\s*\(", line_content):
            if_start = search_line + 1  # 转换回 1-based 行号

            # 向下搜索 if 语句的结束
            if_end = self._find_if_end(lines, search_line)
            if if_end:
                return (if_start, if_end)

        return None

    def _find_if_end(self, lines: List[str], if_line: int) -> int:
        """
        从 if 语句开始向下搜索 if 块的结束

        处理嵌套的 if-else-if 结构，包括没有 begin/end 的链式 if-else-if。

        Args:
            lines: 源文件行列表
            if_line: if 语句的行索引（0-based）

        Returns:
            if 块结束的行号（1-based），如果找不到则返回 None
        """
        # 检查 if 语句是否包含 begin
        if_line_content = lines[if_line].split("//")[0].strip().lower()
        has_begin = "begin" in if_line_content

        # 如果没有 begin，可能是单行 if、三元表达式，或 if-else-if 链
        if not has_begin:
            # 检查下一行是否有 begin
            if if_line + 1 < len(lines):
                next_line = lines[if_line + 1].split("//")[0].strip().lower()
                if "begin" in next_line:
                    has_begin = True
                elif "else" in next_line:
                    # 这是一个 if-else-if 链（没有 begin/end）
                    # 继续搜索直到找到最后的 else 分支或非 else-if 行
                    return self._find_if_else_chain_end(lines, if_line)
                else:
                    # 检查是否是单行 if 后面跟着 else if
                    # 例如: if (cond) val = x; else if (cond2) val = y;
                    # 在同一行上可能有 else if
                    if "else" in if_line_content:  # 注释已在第840行排除
                        # 当前行就有 else，检查是否是 if-else-if 链的一部分
                        return self._find_if_else_chain_end(lines, if_line)

                    # 单行语句后面可能还有 else 分支
                    # 例如:
                    #   if (!rst_n)
                    #     stmt1;
                    #   else             <-- 需要继续检查这里
                    #     if (cond)
                    #       stmt2;
                    # 查找语句结尾（分号），然后检查下一行
                    stmt_end_line = if_line + 1
                    for i in range(if_line + 1, min(if_line + 5, len(lines))):
                        check_line = lines[i].strip()
                        if ";" in check_line:
                            stmt_end_line = i
                            break

                    # 检查语句后是否有 else
                    if stmt_end_line + 1 < len(lines):
                        after_stmt_line = lines[stmt_end_line + 1].strip().lower()
                        if re.search(r"\belse\b", after_stmt_line):
                            # else 后面可能跟着 if，需要递归处理
                            # 检查是否是 else if
                            if re.search(r"\belse\s+if\b", after_stmt_line):
                                # else if，继续作为 if-else-if 链处理
                                return self._find_if_else_chain_end(lines, if_line)
                            else:
                                # 纯 else 分支，需要找到 else 分支的结束
                                else_start = stmt_end_line + 1
                                # 检查 else 后面是什么
                                # 可能是: else begin ... end
                                # 或者: else if (cond) ...
                                # 或者: else stmt;
                                else_content = after_stmt_line
                                if "begin" in else_content:
                                    # else begin ... end
                                    # 需要找到对应的 end
                                    else_end = self._find_begin_end_block(
                                        lines, else_start
                                    )
                                    return else_end
                                elif re.search(r"\bif\s*\(", else_content):
                                    # else 行内有 if（不是 else if，而是 else\n  if）
                                    # 递归查找内部 if 的结束
                                    inner_if_end = self._find_if_end(lines, else_start)
                                    return inner_if_end
                                else:
                                    # 检查下一行是否有 if（else 后换行跟 if）
                                    if else_start + 1 < len(lines):
                                        next_to_else = (
                                            lines[else_start + 1].strip().lower()
                                        )
                                        if re.search(r"\bif\s*\(", next_to_else):
                                            # else 后面跟着 if
                                            inner_if_end = self._find_if_end(
                                                lines, else_start + 1
                                            )
                                            return inner_if_end
                                        else:
                                            # else 后面是单行语句
                                            for j in range(
                                                else_start + 1,
                                                min(else_start + 5, len(lines)),
                                            ):
                                                if ";" in lines[j]:
                                                    return j + 1
                                            return else_start + 2
                                    return else_start + 2

                    # 真正的单行 if（没有 else）
                    return stmt_end_line + 1

        if not has_begin:
            # 最后检查是否可能是 if-else-if 链的开始
            # 向下看几行，看是否有 else if
            for i in range(if_line + 1, min(if_line + 5, len(lines))):
                check_line = lines[i].strip().lower()
                if re.search(r"\belse\s+if\b", check_line):
                    # 这是 if-else-if 链
                    return self._find_if_else_chain_end(lines, if_line)
                elif check_line and not check_line.startswith("//"):
                    # 遇到非空非注释行，且不是 else if，说明是单行 if
                    break
            return if_line + 1

        # 有 begin，需要找到对应的 end
        nesting_level = 1

        for i in range(if_line + 1, len(lines)):
            line_content = lines[i].strip().lower()

            # 跳过注释
            if line_content.startswith("//"):
                continue

            # 检查 begin（增加嵌套层级）
            # 注意：begin 可能在 if/else/case 等语句的同一行
            begin_count = len(re.findall(r"\bbegin\b", line_content))

            # 检查 end（减少嵌套层级）
            # 注意：end 可能后跟 else
            end_count = len(re.findall(r"\bend\b", line_content))

            # 更新嵌套层级
            nesting_level += begin_count - end_count

            # 当嵌套层级回到 0 时，找到了 if 块的结束
            if nesting_level <= 0:
                return i + 1  # 转换为 1-based 行号

        # 如果没有找到，返回文件末尾
        return len(lines)

    def _find_begin_end_block(self, lines: List[str], start_line: int) -> int:
        """
        从包含 begin 的行开始，找到对应的 end 行

        Args:
            lines: 源文件行列表
            start_line: 包含 begin 的行索引（0-based）

        Returns:
            end 所在的行号（1-based）
        """
        nesting_level = 0
        for i in range(start_line, len(lines)):
            line_content = lines[i].strip().lower()

            # 跳过注释
            if line_content.startswith("//"):
                continue

            # 移除行内注释
            if "//" in line_content:
                line_content = line_content.split("//")[0].strip()

            # 统计 begin 和 end
            begin_count = len(re.findall(r"\bbegin\b", line_content))
            end_count = len(re.findall(r"\bend\b", line_content))

            nesting_level += begin_count - end_count

            if nesting_level <= 0:
                return i + 1  # 1-based

        return len(lines)

    def _find_if_else_chain_end(self, lines: List[str], if_line: int) -> int:
        """
        查找 if-else-if 链的结束位置（没有 begin/end 块的情况）

        这种结构通常出现在优先级编码器中：
        if (cond1) val = x;
        else if (cond2) val = y;
        else if (cond3) val = z;
        else val = default;

        Args:
            lines: 源文件行列表
            if_line: if 语句的行索引（0-based）

        Returns:
            if-else-if 链结束的行号（1-based）
        """
        last_valid_line = if_line

        for i in range(if_line, len(lines)):
            line_content = lines[i].strip().lower()

            # 跳过空行和纯注释行
            if not line_content or line_content.startswith("//"):
                continue

            # 去掉行内注释
            if "//" in line_content:
                line_content = line_content.split("//")[0].strip()

            # 检查是否是 if/else if/else 行
            is_if_else = (
                re.search(r"\bif\s*\(", line_content)
                or re.search(r"\belse\s+if\s*\(", line_content)
                or re.search(r"\belse\b", line_content)
            )

            if is_if_else:
                last_valid_line = i
                continue

            # 检查是否是赋值语句（if 分支的执行语句）
            # 通常是 xxx = yyy; 形式
            if "=" in line_content and ";" in line_content:
                last_valid_line = i
                continue

            # 检查是否到达了 end 或其他块结束标记
            if re.search(r"\bend\b", line_content):
                last_valid_line = i
                break

            # 如果遇到其他语句（不是 if/else/赋值），链结束
            # 但要排除缩进的赋值语句
            if line_content and not line_content.startswith("//"):
                # 检查这一行是否是前一行语句的延续
                prev_line = lines[i - 1].strip() if i > 0 else ""
                if not prev_line.endswith(";") and ";" in line_content:
                    # 这是前一行语句的延续
                    last_valid_line = i
                    continue

                # 否则链结束
                break

        return last_valid_line + 1  # 转换为 1-based 行号

    def _get_source_file_path(self) -> str:
        """
        获取源文件的完整路径

        Returns:
            源文件路径，如果找不到则返回 None
        """
        if not self.source_file:
            return None

        possible_paths = [self.source_file]
        if self.source_dir:
            possible_paths.append(os.path.join(self.source_dir, self.source_file))

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _read_source_lines(self) -> List[str]:
        """
        读取源文件并缓存结果

        Returns:
            源文件行列表，如果无法读取则返回空列表
        """
        source_file_path = self._get_source_file_path()
        if not source_file_path:
            return []

        if source_file_path in self._source_lines_cache:
            return self._source_lines_cache[source_file_path]

        try:
            with open(source_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self._source_lines_cache[source_file_path] = lines
            return lines
        except Exception as e:
            print(f"  警告: 无法读取源文件 {source_file_path}: {e}")
            return []

    def _collect_continuous_mux_chain(
        self, start_line: int, expected_count: int
    ) -> List[str]:
        """
        从指定行开始收集连续的 MUX 链

        对于 if-else-if 结构，收集所有相关的 MUX，而不是只取期望数量。
        因为 Yosys 可能为数据路径复制生成额外的 MUX。

        Args:
            start_line: 起始行号
            expected_count: 期望的 MUX 数量（仅用于参考）

        Returns:
            MUX 节点 ID 列表
        """
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())
        collected = []

        # 允许行号有小的跳跃（最多跳过10行，以覆盖嵌套的 if 语句）
        current_line = start_line
        max_gap = 10

        for line in all_mux_lines:
            if line >= current_line and line <= current_line + max_gap:
                collected.extend(self.line_to_mux_nodes[line])
                current_line = line + 1
                # 不再限制数量，收集所有连续的 MUX

        return collected

    def _sort_mux_nodes_for_annotation(self, mux_nodes: List[str]) -> List[str]:
        """
        为标注排序 MUX 节点

        对于 if-else-if 链，需要按逻辑顺序（从外到内）排列
        以便条件编号与 MUX 正确对应

        Args:
            mux_nodes: MUX 节点 ID 列表

        Returns:
            排序后的 MUX 节点 ID 列表
        """
        if len(mux_nodes) <= 1:
            return mux_nodes

        # 先进行拓扑排序
        sorted_nodes = self._sort_mux_nodes_topologically(mux_nodes)

        # 然后反转，使其按逻辑顺序（从外到内）
        # 这样条件1对应最外层的MUX（索引0）
        return sorted_nodes[::-1]

    def _search_mux_at_line(self, line: int, total_branches: int) -> List[str]:
        """在指定行搜索 MUX 节点"""
        expected_mux_count = total_branches - 1

        # 先检查 stmt_start_line
        if line in self.stmt_line_to_mux_nodes:
            stmt_mux = self.stmt_line_to_mux_nodes[line]
            if stmt_mux:
                if len(stmt_mux) >= expected_mux_count:
                    return self._sort_mux_nodes_for_annotation(
                        stmt_mux[:expected_mux_count]
                    )
                return self._sort_mux_nodes_for_annotation(stmt_mux)

        # 尝试收集连续的 MUX 链
        collected = self._collect_continuous_mux_chain(line, expected_mux_count)
        if collected:
            return self._sort_mux_nodes_for_annotation(collected[:expected_mux_count])

        return []

    def annotate(self) -> AnnotationStats:
        """
        执行标注

        Returns:
            标注统计信息
        """
        self.stats.total_edges = len(self.cdfg.edges)

        # 按 (行号, 分支类型) 分组覆盖率数据，以便处理 generate 展开的同行多实例
        line_type_coverages: Dict[tuple, List[BranchCoverage]] = defaultdict(list)
        for cov in self.coverage_data:
            key = (cov.line_no, cov.branch_type)
            line_type_coverages[key].append(cov)

        # 对于每组同行同类型的覆盖率数据
        for (line_no, branch_type), coverages in line_type_coverages.items():
            # 计算该行所有实例需要的总 MUX 数
            total_instances = len(coverages)
            mux_per_instance = coverages[0].total_branches - 1  # 每个实例需要的 MUX 数

            # 查找该行所有 MUX 节点
            all_mux_nodes = self._find_mux_chain_for_coverage(
                line_no, coverages[0].total_branches, branch_type
            )

            if all_mux_nodes:
                # 检测是否是 generate 展开场景
                if total_instances > 1:
                    print(
                        f"行 {line_no}: 找到 {len(all_mux_nodes)} 个 MUX 节点，对应 {total_instances} 个实例 × {coverages[0].total_branches} 分支"
                    )

                    # 按控制信号分组 MUX
                    mux_groups = self._group_mux_by_control_signal(all_mux_nodes)
                    unique_control_sources = list(mux_groups.keys())

                    # 如果 MUX 组数等于实例数，每个实例对应一组 MUX
                    if len(unique_control_sources) == total_instances:
                        for idx, coverage in enumerate(coverages):
                            if idx < len(unique_control_sources):
                                control_source = unique_control_sources[idx]
                                instance_mux_nodes = mux_groups[control_source]
                                self._annotate_mux_chain(instance_mux_nodes, coverage)
                    else:
                        # 按顺序分配 MUX 给每个实例
                        sorted_all_mux = self._sort_mux_nodes_for_annotation(
                            all_mux_nodes
                        )
                        for idx, coverage in enumerate(coverages):
                            start_idx = idx * mux_per_instance
                            end_idx = start_idx + mux_per_instance
                            if start_idx < len(sorted_all_mux):
                                instance_mux_nodes = sorted_all_mux[start_idx:end_idx]
                                if instance_mux_nodes:
                                    self._annotate_mux_chain(
                                        instance_mux_nodes, coverage
                                    )
                else:
                    # 单实例场景（普通情况）
                    print(
                        f"行 {line_no}: 找到 {len(all_mux_nodes)} 个 MUX 节点，对应 {coverages[0].total_branches} 个分支"
                    )
                    self._annotate_mux_chain(all_mux_nodes, coverages[0])
            else:
                # 尝试查找时序元件（复位分支）
                # 复位逻辑 (if rst) 可能被综合成带异步复位的触发器
                seq_nodes = self._find_sequential_nodes_for_coverage(line_no)
                if seq_nodes:
                    print(
                        f"行 {line_no}: 找到 {len(seq_nodes)} 个时序节点（复位分支），对应 {coverages[0].total_branches} 个分支"
                    )
                    # 对于时序节点，也需要处理多实例情况
                    for coverage in coverages:
                        self._annotate_sequential_reset(seq_nodes, coverage)
                else:
                    print(f"警告: 行 {line_no} ({branch_type}) 没有找到对应的节点")

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
        for edge in self.cdfg.edges:
            if edge.target != seq_node_id:
                continue

            # ALOAD/ARST 端口 - 复位控制信号（仅复位分支标注）
            if edge.target_port in ("ALOAD", "ARST", "SRST", "RST") and is_reset_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "control"
                )

            # AD 端口 - 复位值（复位分支）
            elif edge.target_port == "AD" and is_reset_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "data_true"
                )

            # D 端口 - 正常数据（非复位分支）
            elif edge.target_port == "D" and not is_reset_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "data_false"
                )

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

    def _annotate_mux_chain(
        self,
        mux_nodes: List[str],
        coverage: BranchCoverage,
        skip_reset_detection: bool = False,
    ):
        """
        标注 MUX 链

        Args:
            mux_nodes: MUX 节点 ID 列表（已按逻辑顺序排序）
            coverage: 分支覆盖数据
            skip_reset_detection: 跳过复位逻辑检测（用于 CASE 回退标注）
        """
        # 检测是否是 CASE 语句（条件值中包含 CASE 标签值 2）
        is_case_statement = any(
            2 in br.condition_values.values() for br in coverage.branches
        )

        if is_case_statement:
            # 使用 CASE 语句专用标注方法
            self._annotate_case_statement(mux_nodes, coverage)
            return

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

        # 检测数据路径复制：按控制信号分组 MUX
        # 当多个 MUX 共享同一控制信号时，它们应该被视为一组
        mux_groups = self._group_mux_by_control_signal(mux_nodes)
        unique_control_sources = list(mux_groups.keys())

        # 建立条件编号到 MUX 组的映射
        # 策略：始终按控制信号分组映射
        #
        # 特殊情况处理：复位逻辑
        # 当 MUX 组数 < 条件数时，可能是因为第一个条件是复位逻辑（如 ~rst_n）
        # 复位逻辑会被综合成时序节点的异步复位，而不是 MUX
        # 在这种情况下，需要跳过第一个条件，从第二个条件开始映射
        #
        # 检测条件：
        # 1. 分支类型是 IF（不是 TERNARY，因为三元表达式不会有复位逻辑）
        # 2. MUX 组数 < 条件数
        # 3. 第一个分支只有条件 1 为真（典型的复位分支模式）
        skip_first_condition = False
        if (
            not skip_reset_detection
            and coverage.branch_type == "IF"
            and len(unique_control_sources) < len(sorted_conditions)
        ):
            # 检查是否是复位逻辑模式
            # 复位分支通常是：条件1=1，其他条件=-1（无关）
            first_branch = coverage.branches[0] if coverage.branches else None
            if first_branch:
                cond_vals = first_branch.condition_values
                first_cond = sorted_conditions[0]
                # 检查第一个条件是否为真，且其他条件都是无关
                if cond_vals.get(first_cond) == 1:
                    other_vals = [cond_vals.get(c, -1) for c in sorted_conditions[1:]]
                    if all(v == -1 for v in other_vals):
                        # 这是复位逻辑模式，跳过第一个条件
                        skip_first_condition = True
                        print(
                            f"  检测到复位逻辑: 条件 {first_cond} 对应时序节点复位，跳过 MUX 映射"
                        )

        cond_to_mux_group = {}
        conditions_to_map = (
            sorted_conditions[1:] if skip_first_condition else sorted_conditions
        )

        for idx, cond in enumerate(conditions_to_map):
            if idx < len(unique_control_sources):
                control_source = unique_control_sources[idx]
                cond_to_mux_group[cond] = mux_groups[control_source]
            else:
                cond_to_mux_group[cond] = []

        # 如果跳过了第一个条件，为它创建空映射
        if skip_first_condition:
            cond_to_mux_group[sorted_conditions[0]] = []

        # 调试信息
        if len(mux_nodes) > len(sorted_conditions):
            print(
                f"  注意：{len(mux_nodes)} 个 MUX 分组为 {len(unique_control_sources)} 组，对应 {len(sorted_conditions)} 个条件"
            )

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
            #
            # CASE 语句特殊处理：
            # 当 condition_values={1: -1, 2: 1, 3: -1} 时，表示：
            # - 条件 2 为真（true_branch）
            # - 条件 1 和 3 为"无关"，但在 MUX 链中，要到达条件 2，
            #   必须先经过条件 1 的 A 端口（因为 MUX 按条件编号顺序串联）
            #
            # 因此对于 CASE 语句，-1 值的条件如果编号小于某个 true 条件，
            # 应该被视为 false（走 A 端口）

            # 收集所有条件值
            true_conditions = []
            false_conditions = []
            dont_care_conditions = []
            for cond_idx, cond_val in sorted(branch_status.condition_values.items()):
                if cond_val == 1:
                    true_conditions.append(cond_idx)
                elif cond_val == 0:
                    false_conditions.append(cond_idx)
                elif cond_val == -1:
                    dont_care_conditions.append(cond_idx)

            # CASE 语句特殊处理：将位于 true 条件之前的 don't care 条件视为 false
            if true_conditions and dont_care_conditions:
                max_true_cond = max(true_conditions)
                for dc_cond in dont_care_conditions:
                    if dc_cond < max_true_cond:
                        # 这个 don't care 条件在 MUX 链中位于 true 条件之前
                        # 要到达 true 条件，必须走这个 MUX 的 A 端口
                        false_conditions.append(dc_cond)

            if true_conditions:
                # 标注所有为真的条件对应的 MUX 组的 B 端口
                for cond in true_conditions:
                    mux_group = cond_to_mux_group.get(cond, [])
                    for mux_node in mux_group:
                        self._annotate_mux_edges(
                            mux_node,
                            branch_idx,
                            branch_status.is_covered,
                            is_true_branch=True,
                            coverage_line=coverage.line_no,
                        )

                # 标注所有为假的条件对应的 MUX 组的 A 端口
                for cond in false_conditions:
                    mux_group = cond_to_mux_group.get(cond, [])
                    for mux_node in mux_group:
                        self._annotate_mux_edges(
                            mux_node,
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

    def _annotate_case_statement(self, mux_nodes: List[str], coverage: BranchCoverage):
        """
        CASE 语句标注策略

        VCS 将 CASE 语句内部的所有 IF 分支合并到一个覆盖率条目中：
        - 条件 1: CASE 选择条件（值为 2，表示 CASE 标签被选中）
        - 条件 2+: 各 CASE 分支内部的 IF 条件或嵌套 CASE 条件

        策略：
        1. 检测是否是嵌套 CASE（多个条件有 case_labels）
        2. 对于嵌套 CASE，使用简化的标注策略
        3. 对于单层 CASE，使用行号范围匹配

        Args:
            mux_nodes: MUX 节点 ID 列表
            coverage: 分支覆盖数据
        """

        # 步骤 1：识别所有 CASE 选择器条件（条件值为 2 的条件编号）
        case_selector_conditions = set()
        if_cond_indices = []

        for br in coverage.branches:
            for cond_idx, cond_val in br.condition_values.items():
                if cond_val == 2:  # CASE 标签值
                    case_selector_conditions.add(cond_idx)
                elif cond_val in (0, 1):
                    if cond_idx not in if_cond_indices:
                        if_cond_indices.append(cond_idx)

        # 检测是否是复杂 CASE（需要聚合策略的情况）
        # 1. 嵌套 CASE（多个 CASE 选择器条件）
        # 2. 大量内部 IF 条件（超过 5 个），精确映射困难
        is_nested_case = len(case_selector_conditions) > 1
        is_complex_case = len(if_cond_indices) > 5

        if is_nested_case or is_complex_case:
            reason = (
                "嵌套 CASE"
                if is_nested_case
                else f"复杂 CASE ({len(if_cond_indices)} 个内部 IF)"
            )
            print(
                f"  CASE 语句: {reason}，选择器条件 {sorted(case_selector_conditions)}，内部 IF 条件 {if_cond_indices}"
            )
            self._annotate_nested_case(
                mux_nodes, coverage, case_selector_conditions, if_cond_indices
            )
            return

        # 单层简单 CASE 的原有逻辑
        case_cond_idx = (
            min(case_selector_conditions) if case_selector_conditions else None
        )

        print(
            f"  CASE 语句: 条件 {case_cond_idx} 是 CASE 选择器，条件 {if_cond_indices} 是内部 IF"
        )

        # 步骤 2：解析 CASE 分支的行号范围
        case_branch_ranges = self._parse_case_branch_ranges(coverage.line_no)

        if not case_branch_ranges:
            print("  警告: 无法解析 CASE 分支范围，回退到普通标注")
            self._annotate_mux_chain(mux_nodes, coverage, skip_reset_detection=True)
            return

        print(f"  CASE 分支范围: {case_branch_ranges}")

        # 步骤 3：按行号范围分配 MUX 到 CASE 分支
        branch_to_mux: Dict[str, List[str]] = defaultdict(list)
        unassigned_mux = []

        for mux_id in mux_nodes:
            mux_line = self._get_node_source_line(mux_id)
            assigned = False
            for label, (start, end) in case_branch_ranges.items():
                if start <= mux_line <= end:
                    branch_to_mux[label].append(mux_id)
                    assigned = True
                    break
            if not assigned:
                unassigned_mux.append(mux_id)

        if unassigned_mux:
            print(f"  未分配到分支的 MUX: {len(unassigned_mux)} 个")

        # 步骤 4：根据分支路径标注
        for branch_idx, branch_status in enumerate(coverage.branches):
            # 获取此分支的 CASE 标签
            case_label = branch_status.case_labels.get(case_cond_idx)
            if not case_label:
                if all(v == -1 for v in branch_status.condition_values.values()):
                    case_label = "default"
                else:
                    continue

            # 获取此 CASE 分支内的 MUX
            branch_mux_nodes = branch_to_mux.get(case_label, [])

            if not branch_mux_nodes:
                # 尝试模糊匹配
                case_label_lower = case_label.lower().strip()
                # 移除包名前缀（如 cv32e40p_pkg::TRAP_MACHINE -> TRAP_MACHINE）
                if "::" in case_label:
                    case_label_stripped = case_label.split("::")[-1].lower().strip()
                else:
                    case_label_stripped = case_label_lower

                for label in branch_to_mux.keys():
                    label_lower = label.lower().strip()
                    if (
                        label_lower == case_label_lower
                        or label_lower == case_label_stripped
                    ):
                        branch_mux_nodes = branch_to_mux[label]
                        break

            if not branch_mux_nodes:
                # 如果该分支没有对应的 MUX，跳过
                continue

            # 按控制信号分组
            mux_groups_in_branch = self._group_mux_by_control_signal(branch_mux_nodes)

            # 获取此分支的内部 IF 条件值
            internal_if_values = {}
            for if_cond_idx in if_cond_indices:
                if_cond_val = branch_status.condition_values.get(if_cond_idx, -1)
                if if_cond_val in (0, 1):
                    internal_if_values[if_cond_idx] = if_cond_val

            # 获取该 CASE 分支对应的内部 IF 条件
            if internal_if_values and mux_groups_in_branch:
                # 有内部 IF 条件，需要精确标注
                # 策略：区分 CASE 选择器 MUX 和内部 IF MUX
                # - 内部 IF MUX：控制信号是输入端口（以 port_ 开头）
                # - CASE 选择器 MUX：控制信号是内部信号（以 $ 开头）

                # 分离 CASE 选择器 MUX 和内部 IF MUX
                case_selector_mux = []
                internal_if_mux = []

                for ctrl_src, mux_list in mux_groups_in_branch.items():
                    if ctrl_src.startswith("port_"):
                        # 控制信号是输入端口，这是内部 IF MUX
                        internal_if_mux.extend(mux_list)
                    else:
                        # 控制信号是内部信号，这是 CASE 选择器相关 MUX
                        case_selector_mux.extend(mux_list)

                # 标注 CASE 选择器 MUX（每个分支路径都需要标注）
                for mux_node in case_selector_mux:
                    self._annotate_mux_edges(
                        mux_node,
                        branch_idx,
                        branch_status.is_covered,
                        is_true_branch=True,
                        coverage_line=coverage.line_no,
                    )

                # 标注内部 IF MUX
                for if_cond_idx, if_cond_val in internal_if_values.items():
                    is_true_branch = if_cond_val == 1

                    # 使用内部 IF MUX
                    if_mux_nodes = (
                        internal_if_mux if internal_if_mux else branch_mux_nodes
                    )

                    for mux_node in if_mux_nodes:
                        self._annotate_mux_edges(
                            mux_node,
                            branch_idx,
                            branch_status.is_covered,
                            is_true_branch,
                            coverage.line_no,
                        )
            else:
                # 没有内部 IF（default 分支或纯 CASE 选择）
                # 对于该 CASE 分支内的所有 MUX，标注 B 端口（分支被选中）
                for mux_node in branch_mux_nodes:
                    self._annotate_mux_edges(
                        mux_node,
                        branch_idx,
                        branch_status.is_covered,
                        is_true_branch=True,
                        coverage_line=coverage.line_no,
                    )

            # 标注未分配的 MUX（全局 CASE 选择器，如 $pmux）
            for mux_node in unassigned_mux:
                self._annotate_mux_edges(
                    mux_node,
                    branch_idx,
                    branch_status.is_covered,
                    is_true_branch=True,
                    coverage_line=coverage.line_no,
                )

    def _annotate_nested_case(
        self,
        mux_nodes: List[str],
        coverage: BranchCoverage,
        case_selector_conditions: set,
        if_cond_indices: List[int],
    ):
        """
        嵌套 CASE 语句的标注策略

        对于嵌套 CASE（如 case 内嵌套 case 或 case 内嵌套 if 再嵌套 case），
        VCS 会将所有层级的条件合并到一个覆盖率条目中。

        策略：
        1. 分析分支覆盖情况，找出哪些分支是覆盖的，哪些是未覆盖的
        2. 按控制信号分组 MUX
        3. 对于每个 MUX 组，根据关联分支的覆盖状态进行标注
        4. 保守策略：如果有任何覆盖的分支经过该 MUX，则标注为已覆盖

        Args:
            mux_nodes: MUX 节点 ID 列表
            coverage: 分支覆盖数据
            case_selector_conditions: CASE 选择器条件编号集合
            if_cond_indices: 内部 IF 条件编号列表
        """
        # 统计覆盖情况
        covered_count = sum(1 for br in coverage.branches if br.is_covered)
        uncovered_count = len(coverage.branches) - covered_count

        # 对于嵌套 CASE，我们使用聚合标注策略：
        # - B 端口（true 分支）：如果有任何已覆盖的分支，则标注为已覆盖
        # - A 端口（false 分支）：如果有任何已覆盖的分支，则标注为已覆盖
        # - 只有当某个端口只关联未覆盖的分支时，才标注为未覆盖

        # 收集所有有 true 条件（激活）的分支
        branches_with_true = []
        # 收集所有有 false 条件（未激活）的分支
        branches_with_false = []

        for branch_idx, branch_status in enumerate(coverage.branches):
            has_active = any(
                v in (1, 2) for v in branch_status.condition_values.values()
            )
            has_inactive = any(v == 0 for v in branch_status.condition_values.values())

            if has_active:
                branches_with_true.append((branch_idx, branch_status))
            if has_inactive:
                branches_with_false.append((branch_idx, branch_status))

        # 判断 B 端口（true 分支）的整体覆盖状态
        b_port_has_covered = any(br.is_covered for _, br in branches_with_true)

        # 判断 A 端口（false 分支）的整体覆盖状态
        a_port_has_covered = any(br.is_covered for _, br in branches_with_false)

        # 标注所有 MUX 的 B 端口
        if branches_with_true:
            rep_branch_idx = branches_with_true[0][0]
            for mux_id in mux_nodes:
                self._annotate_mux_edges(
                    mux_id,
                    rep_branch_idx,
                    b_port_has_covered,
                    is_true_branch=True,
                    coverage_line=coverage.line_no,
                )

        # 标注所有 MUX 的 A 端口
        if branches_with_false:
            rep_branch_idx = branches_with_false[0][0]
            for mux_id in mux_nodes:
                self._annotate_mux_edges(
                    mux_id,
                    rep_branch_idx,
                    a_port_has_covered,
                    is_true_branch=False,
                    coverage_line=coverage.line_no,
                )

        # 如果有未覆盖的分支，需要确保这些信息被记录
        # 在统计信息中反映出来
        if uncovered_count > 0:
            print(
                f"  注意: 嵌套 CASE 有 {uncovered_count}/{len(coverage.branches)} 个未覆盖分支"
            )

    def _parse_case_branch_ranges(self, case_line: int) -> Dict[str, tuple]:
        """
        从源码解析每个 CASE 分支的行号范围

        Args:
            case_line: CASE 语句的起始行号

        Returns:
            标签名 -> (起始行, 结束行) 的映射（1-based 行号）
        """
        lines = self._read_source_lines()
        if not lines:
            return {}

        # SystemVerilog 关键字，不应被识别为 CASE 标签
        # 包括 begin/end 块标签语法: begin: label_name
        sv_keywords = {
            "begin",
            "end",
            "if",
            "else",
            "for",
            "while",
            "do",
            "foreach",
            "case",
            "casex",
            "casez",
            "endcase",
            "function",
            "endfunction",
            "task",
            "endtask",
            "module",
            "endmodule",
            "always",
            "always_ff",
            "always_comb",
            "always_latch",
            "assign",
            "wire",
            "reg",
            "logic",
            "input",
            "output",
            "inout",
            "parameter",
            "localparam",
            "generate",
            "endgenerate",
            "initial",
            "final",
            "fork",
            "join",
            "disable",
            "wait",
            "event",
            "posedge",
            "negedge",
            "or",
            "and",
            "not",
            "unique",
            "priority",
            "return",
            "break",
            "continue",
        }

        def remove_comments(line: str) -> str:
            """移除行内注释"""
            # 简单处理：找到 // 并截断
            comment_idx = line.find("//")
            if comment_idx >= 0:
                return line[:comment_idx]
            return line

        ranges = {}
        current_label = None
        current_start = None
        case_depth = 0
        found_case_start = False  # 是否已找到真正的 case 语句起始

        # 从 case 语句开始行向下扫描
        start_idx = case_line - 1  # 转换为 0-based 索引

        for i in range(start_idx, len(lines)):
            line = lines[i]
            line_stripped = line.strip()

            # 跳过纯注释行
            if line_stripped.startswith("//"):
                continue

            # 移除行内注释后再分析
            line_no_comment = remove_comments(line_stripped)
            line_lower = line_no_comment.lower()

            # 检测 case 语句起始
            # 对于 IF+CASE 复合结构，覆盖率报告可能指向 IF 行而非 CASE 行
            if re.search(r"\b(unique\s+|priority\s+)?case[xz]?\s*\(", line_lower):
                if not found_case_start:
                    # 找到真正的 case 语句起始
                    found_case_start = True
                else:
                    # 嵌套的 case 语句
                    case_depth += 1

            # 检测 endcase
            if re.search(r"\bendcase\b", line_lower):
                if case_depth > 0:
                    case_depth -= 1
                elif found_case_start:
                    # 当前 case 语句结束
                    if current_label:
                        ranges[current_label] = (current_start, i + 1)  # 1-based 行号
                    break

            # 只在找到 case 语句起始后、且非嵌套 case 内解析分支标签
            if found_case_start and case_depth == 0:
                # 检测 case 分支标签
                # 标签必须在行首（或只有空白），格式: IDLE: 或 IDLE : 或 IDLE: begin 或 3'b000: 或 default:
                label_match = re.match(
                    r"^\s*([A-Za-z_][A-Za-z0-9_]*|\d+'[bhd][0-9a-fA-F_]+|default)\s*:",
                    line_no_comment,
                )
                if label_match:
                    label_name = label_match.group(1)
                    label_name_lower = label_name.lower()

                    # 排除 SystemVerilog 关键字（如 begin: block_name）
                    if label_name_lower in sv_keywords:
                        continue

                    # 保存前一个分支的范围
                    if current_label:
                        # 结束行是当前行的前一行（1-based）
                        ranges[current_label] = (
                            current_start,
                            i,
                        )  # i 是下一个标签的 0-based 索引，数值上等于前一行的 1-based 行号

                    current_label = label_name
                    current_start = i + 1  # 1-based 行号（包含标签行本身）

        return ranges

    def _get_node_source_line(self, node_id: str) -> int:
        """获取节点的源代码行号"""
        node = self.cdfg.nodes.get(node_id)
        if node:
            return node.source_line
        return 0

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
        for edge in self.cdfg.edges:
            if edge.target != mux_node_id:
                continue

            # S 端口 - 控制信号
            # 控制边的覆盖状态反映"这个条件的真分支是否被执行过"
            # 只有当 is_true_branch=True 时才更新 S 端口的覆盖状态
            if edge.target_port == "S" and is_true_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "control"
                )

            # A 端口 - false 分支
            elif edge.target_port == "A" and not is_true_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "data_false"
                )

            # B 端口 - true 分支
            elif edge.target_port == "B" and is_true_branch:
                self._apply_edge_label(
                    edge, coverage_line, branch_index, is_covered, "data_true"
                )

        # 只在第一次标注该 MUX 时计数
        if mux_node_id not in self._annotated_mux_set:
            self._annotated_mux_set.add(mux_node_id)
            self.stats.annotated_mux_nodes += 1


def annotate_cdfg_with_coverage(
    cdfg: CDFG,
    html_path: str,
    instance_tag: str = None,
    source_dir: str = None,
) -> AnnotationStats:
    """
    便捷函数：用覆盖率数据标注 CDFG

    Args:
        cdfg: CDFG 对象
        html_path: 覆盖率报告 HTML 文件路径
        instance_tag: 实例标签，None 则使用第一个实例
        source_dir: 设计源码目录路径（用于精确范围匹配）

    Returns:
        标注统计信息
    """
    # 解析覆盖率报告
    parser = CoverageParser(html_path)

    if instance_tag is None:
        instance_tag = parser.get_last_instance()

    # 解析分支覆盖率
    coverage_data = parser.parse_branch_coverage(instance_tag)

    # 获取源文件名用于过滤 MUX 节点
    source_file = parser.get_source_file()

    # 执行标注
    annotator = CoverageAnnotator(
        cdfg, coverage_data, source_file=source_file, source_dir=source_dir
    )
    stats = annotator.annotate()

    return stats
