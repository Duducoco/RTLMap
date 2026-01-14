#!/usr/bin/env python3
"""
覆盖率标注器
将覆盖率报告中的分支覆盖数据映射到 CDFG 的边上
"""

from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict

from cdfg import CDFG, Edge, NodeType
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
    # 传播标注统计
    propagated_edges: int = 0  # 通过传播标注的边数
    always_executed_edges: int = 0  # 必然执行的边数（组合逻辑）


class CoverageAnnotator:
    """覆盖率标注器"""

    # 行号搜索范围配置
    STMT_LINE_SEARCH_RANGE = range(-2, 6)  # stmt_start_line 搜索范围 [-2, +5]
    SOURCE_LINE_SEARCH_RANGE = range(-5, 40)  # source_line 搜索范围 [-5, +39]
    
    # 偏移量搜索配置
    DEFAULT_OFFSET_SEARCH_RANGE = (-5, 40)  # 默认搜索范围
    LEARNED_OFFSET_TOLERANCE = 5  # 学习到偏移量后的容差范围

    def __init__(
        self,
        cdfg: CDFG,
        coverage_data: Dict[int, BranchCoverage],
        source_file: str = None,
    ):
        """
        初始化标注器

        Args:
            cdfg: CDFG 对象
            coverage_data: 行号 -> BranchCoverage 的映射
            source_file: 覆盖率报告对应的源文件名（用于过滤 MUX 节点）
        """
        self.cdfg = cdfg
        self.coverage_data = coverage_data
        self.source_file = source_file
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
    
    def _get_offset_search_order(self, search_range: tuple = None) -> List[int]:
        """
        获取优化后的偏移量搜索顺序
        
        如果已经学习到偏移量，优先搜索该偏移量附近的范围；
        否则使用默认的搜索范围。
        
        Args:
            search_range: 搜索范围元组 (min_offset, max_offset)，默认使用 DEFAULT_OFFSET_SEARCH_RANGE
            
        Returns:
            偏移量列表，按优先级排序
        """
        if search_range is None:
            search_range = self.DEFAULT_OFFSET_SEARCH_RANGE
        
        min_offset, max_offset = search_range
        
        if self._learned_offset is not None:
            # 已学习到偏移量，优先搜索该偏移量附近
            # 1. 首先尝试学习到的偏移量
            # 2. 然后尝试容差范围内的偏移量
            # 3. 最后尝试其他偏移量
            
            learned = self._learned_offset
            tolerance = self.LEARNED_OFFSET_TOLERANCE
            
            # 优先级1：学习到的偏移量
            priority_offsets = [learned]
            
            # 优先级2：容差范围内的偏移量（按距离排序）
            for delta in range(1, tolerance + 1):
                if min_offset <= learned - delta <= max_offset:
                    priority_offsets.append(learned - delta)
                if min_offset <= learned + delta <= max_offset:
                    priority_offsets.append(learned + delta)
            
            # 优先级3：其他偏移量
            other_offsets = [
                o for o in range(min_offset, max_offset + 1)
                if o not in priority_offsets
            ]
            
            return priority_offsets + other_offsets
        else:
            # 未学习到偏移量，使用默认顺序
            # 优先尝试 0 偏移，然后按距离递增
            offsets = [0]
            for delta in range(1, max(abs(min_offset), abs(max_offset)) + 1):
                if min_offset <= -delta:
                    offsets.append(-delta)
                if delta <= max_offset:
                    offsets.append(delta)
            
            # 过滤掉超出范围的偏移量
            return [o for o in offsets if min_offset <= o <= max_offset]
    
    def _update_learned_offset(self, offset: int):
        """
        更新学习到的偏移量
        
        如果是首次学习，直接设置；
        如果已有偏移量，验证新偏移量是否在容差范围内。
        
        Args:
            offset: 新发现的偏移量
        """
        if self._learned_offset is None:
            if offset != 0:
                self._learned_offset = offset
                print(f"  学习到行号偏移: {offset}")
        else:
            # 已有偏移量，检查新偏移量是否在容差范围内
            diff = abs(offset - self._learned_offset)
            if diff > self.LEARNED_OFFSET_TOLERANCE:
                print(f"  警告: 新偏移量 {offset} 与已学习偏移量 {self._learned_offset} 差异较大")

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

        注意：VCS 和 Yosys 之间可能存在较大的行号偏移（可达 30+ 行），
        这是由于 include 文件、宏展开等处理方式不同导致的。

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
        elif branch_type == "TERNARY":
            # 三元表达式：使用 ? : 范围解析
            collected = self._collect_ternary_mux_nodes(coverage_line, expected_mux_count)
            if collected:
                return collected
        
        # 如果精确范围解析失败，回退到启发式方法
        # 1. 首先尝试找到连续的 MUX 链（if-else-if 结构）
        for offset in range(-5, 40):  # 扩大搜索范围
            candidate_line = coverage_line + offset
            
            # 收集从 candidate_line 开始的连续 MUX
            collected = self._collect_continuous_mux_chain(candidate_line, expected_mux_count)
            
            if len(collected) >= expected_mux_count:
                # 找到足够的 MUX
                if self._learned_offset is None and offset != 0:
                    self._learned_offset = offset
                    print(f"  学习到行号偏移: {offset}")
                # 返回所有收集到的 MUX，而不是只取 expected_mux_count 个
                # 因为 Yosys 可能为数据路径复制生成额外的 MUX
                return self._sort_mux_nodes_for_annotation(collected)
            
            # 记录最佳匹配（即使不够）
            if len(collected) > best_match_score:
                best_match = collected
                best_match_score = len(collected)
        
        # 2. 如果找不到连续的链，尝试通过 stmt_start_line 查找
        for offset in self.STMT_LINE_SEARCH_RANGE:
            candidate_line = coverage_line + offset
            if candidate_line in self.stmt_line_to_mux_nodes:
                stmt_mux = self.stmt_line_to_mux_nodes[candidate_line]
                if len(stmt_mux) >= expected_mux_count:
                    if self._learned_offset is None and offset != 0:
                        self._learned_offset = offset
                        print(f"  学习到行号偏移: {offset}")
                    return self._sort_mux_nodes_for_annotation(stmt_mux[:expected_mux_count])
                if len(stmt_mux) > best_match_score:
                    best_match = stmt_mux
                    best_match_score = len(stmt_mux)
        
        # 3. 返回最佳匹配（即使不完整）
        if best_match:
            print(f"  警告：期望 {expected_mux_count} 个 MUX，但只找到 {len(best_match)} 个")
            return self._sort_mux_nodes_for_annotation(best_match)
        
        return []
    
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
            
            # 收集范围内的所有 MUX
            collected = []
            for line in all_mux_lines:
                if line >= case_start and line <= case_end:
                    collected.extend(self.line_to_mux_nodes[line])
            
            if collected:
                return self._sort_mux_nodes_for_annotation(collected)
        
        # 如果无法从源码解析，使用启发式方法
        best_collected = []
        best_offset = 0
        
        for offset in range(-5, 20):
            case_start_line = coverage_line + offset
            
            # 收集从 case_start_line 开始的一定范围内的所有 MUX
            # CASE 语句通常跨越较大的行号范围（可能 100+ 行）
            collected = []
            for line in all_mux_lines:
                if line >= case_start_line and line <= case_start_line + 150:
                    collected.extend(self.line_to_mux_nodes[line])
            
            if len(collected) > len(best_collected):
                best_collected = collected
                best_offset = offset
        
        # 返回所有收集到的 MUX（不限制数量）
        if best_collected:
            if self._learned_offset is None and best_offset != 0:
                self._learned_offset = best_offset
                print(f"  学习到行号偏移: {best_offset}")
            
            # 按拓扑顺序排序并返回所有 MUX
            return self._sort_mux_nodes_for_annotation(best_collected)
        
        return []
    
    def _collect_if_mux_nodes(
        self, coverage_line: int, expected_mux_count: int
    ) -> List[str]:
        """
        为 IF 语句收集所有相关的 MUX 节点
        
        通过解析源码中的 if/else/begin/end 关键字来确定精确范围，
        然后收集该范围内的所有 MUX 节点。
        
        Args:
            coverage_line: 覆盖率报告中的行号（if 语句的行号）
            expected_mux_count: 期望的 MUX 数量（仅用于参考）
            
        Returns:
            MUX 节点 ID 列表（按拓扑顺序排序）
        """
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())
        if not all_mux_lines:
            return []
        
        # 尝试从源文件中找到 if 语句的范围
        if_range = self._find_if_range_from_source(coverage_line)
        
        if if_range:
            if_start, if_end = if_range
            print(f"  从源码解析到 if 范围: {if_start}-{if_end}")
            
            # 收集范围内的所有 MUX
            collected = []
            for line in all_mux_lines:
                if line >= if_start and line <= if_end:
                    collected.extend(self.line_to_mux_nodes[line])
            
            if collected:
                return self._sort_mux_nodes_for_annotation(collected)
        
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
        all_mux_lines = sorted(self.line_to_mux_nodes.keys())
        if not all_mux_lines:
            return []
        
        # 尝试从源文件中找到三元表达式的范围
        ternary_range = self._find_ternary_range_from_source(coverage_line)
        
        if ternary_range:
            ternary_start, ternary_end = ternary_range
            print(f"  从源码解析到三元表达式范围: {ternary_start}-{ternary_end}")
            
            # 收集范围内的所有 MUX
            collected = []
            for line in all_mux_lines:
                if line >= ternary_start and line <= ternary_end:
                    collected.extend(self.line_to_mux_nodes[line])
            
            if collected:
                return self._sort_mux_nodes_for_annotation(collected)
        
        # 如果无法从源码解析，返回空列表，让调用者回退到启发式方法
        return []
    
    def _find_ternary_range_from_source(self, coverage_line: int) -> tuple:
        """
        从源文件中找到三元表达式的范围
        
        通过解析源码中的 ? : 语法来确定范围。
        处理嵌套的三元表达式和跨行的情况。
        
        Args:
            coverage_line: 覆盖率报告中的行号
            
        Returns:
            (ternary_start, ternary_end) 元组，如果找不到则返回 None
        """
        # 获取源文件路径
        source_file_path = self._get_source_file_path()
        if not source_file_path:
            return None
        
        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  警告: 无法读取源文件 {source_file_path}: {e}")
            return None
        
        # 使用优化后的偏移量搜索顺序
        for offset in self._get_offset_search_order((-5, 20)):
            search_line = coverage_line + offset - 1  # 转换为 0-based 索引
            
            if search_line < 0 or search_line >= len(lines):
                continue
            
            line_content = lines[search_line]
            
            # 检查是否包含三元表达式的 ? 操作符
            # 注意：需要排除注释中的 ?
            if '?' in line_content and ':' in line_content:
                # 简单检查：确保 ? 不在注释中
                comment_pos = line_content.find('//')
                question_pos = line_content.find('?')
                
                if comment_pos == -1 or question_pos < comment_pos:
                    ternary_start = search_line + 1  # 转换回 1-based 行号
                    
                    # 找到三元表达式的结束位置
                    ternary_end = self._find_ternary_end(lines, search_line)
                    
                    if ternary_end:
                        # 学习偏移量
                        self._update_learned_offset(offset)
                        
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
            comment_pos = line_content.find('//')
            if comment_pos != -1:
                line_content = line_content[:comment_pos]
            
            # 处理字符串字面量（简单处理：跳过引号内的内容）
            in_string = False
            j = 0
            while j < len(line_content):
                char = line_content[j]
                
                # 处理字符串
                if char == '"' and (j == 0 or line_content[j-1] != '\\'):
                    in_string = not in_string
                    j += 1
                    continue
                
                if in_string:
                    j += 1
                    continue
                
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == '?':
                    # 新的三元表达式开始
                    ternary_depth += 1
                elif char == ':':
                    # 区分三元表达式的冒号和位选择的冒号
                    # 位选择的冒号在 [] 内部，如 data[7:0]
                    if bracket_depth == 0:
                        # 这是三元表达式的冒号
                        if ternary_depth > 0:
                            ternary_depth -= 1
                    # 如果 bracket_depth > 0，这是位选择的冒号，忽略
                elif char == ';':
                    # 分号表示语句结束
                    return i + 1  # 转换为 1-based 行号
                elif char == ',':
                    # 逗号可能表示表达式结束（在函数参数或数组初始化中）
                    # 只有当所有三元表达式都已匹配时才结束
                    if ternary_depth == 0 and paren_depth <= 0:
                        return i + 1
                
                j += 1
            
            # 检查是否找到了完整的三元表达式
            # 所有 ? 都找到了匹配的 :，且括号平衡
            if ternary_depth == 0 and paren_depth <= 0 and i > start_line:
                # 确保至少处理了一个 ?
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
        # 获取源文件路径
        source_file_path = self._get_source_file_path()
        if not source_file_path:
            return None
        
        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  警告: 无法读取源文件 {source_file_path}: {e}")
            return None
        
        # 使用优化后的偏移量搜索顺序
        for offset in self._get_offset_search_order((-5, 20)):
            search_line = coverage_line + offset - 1  # 转换为 0-based 索引
            
            if search_line < 0 or search_line >= len(lines):
                continue
            
            line_content = lines[search_line].strip().lower()
            
            # 检查是否是 case 语句
            if 'case' in line_content and 'endcase' not in line_content:
                case_start = search_line + 1  # 转换回 1-based 行号
                
                # 向下搜索 endcase
                endcase_line = self._find_endcase(lines, search_line)
                if endcase_line:
                    case_end = endcase_line
                    
                    # 学习偏移量
                    self._update_learned_offset(offset)
                    
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
            if line_content.startswith('//'):
                continue
            
            # 检查嵌套的 case
            if 'case' in line_content and 'endcase' not in line_content:
                # 确保是 case 关键字而不是变量名的一部分
                import re
                if re.search(r'\bcase[xz]?\s*\(', line_content):
                    nesting_level += 1
            
            # 检查 endcase
            if 'endcase' in line_content:
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
        import re
        
        # 获取源文件路径
        source_file_path = self._get_source_file_path()
        if not source_file_path:
            return None
        
        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  警告: 无法读取源文件 {source_file_path}: {e}")
            return None
        
        # 使用优化后的偏移量搜索顺序
        for offset in self._get_offset_search_order((-5, 20)):
            search_line = coverage_line + offset - 1  # 转换为 0-based 索引
            
            if search_line < 0 or search_line >= len(lines):
                continue
            
            line_content = lines[search_line].strip().lower()
            
            # 检查是否是 if 语句（排除 else if）
            # 匹配 "if (" 或 "if(" 但不匹配 "else if"
            if re.search(r'(?<!else\s)\bif\s*\(', line_content):
                if_start = search_line + 1  # 转换回 1-based 行号
                
                # 向下搜索 if 语句的结束
                if_end = self._find_if_end(lines, search_line)
                if if_end:
                    # 学习偏移量
                    self._update_learned_offset(offset)
                    
                    return (if_start, if_end)
        
        return None
    
    def _find_if_end(self, lines: List[str], if_line: int) -> int:
        """
        从 if 语句开始向下搜索 if 块的结束
        
        处理嵌套的 if-else-if 结构。
        
        Args:
            lines: 源文件行列表
            if_line: if 语句的行索引（0-based）
            
        Returns:
            if 块结束的行号（1-based），如果找不到则返回 None
        """
        import re
        
        # 检查 if 语句是否包含 begin
        if_line_content = lines[if_line].strip().lower()
        has_begin = 'begin' in if_line_content
        
        # 如果没有 begin，可能是单行 if 或三元表达式
        if not has_begin:
            # 检查下一行是否有 begin
            if if_line + 1 < len(lines):
                next_line = lines[if_line + 1].strip().lower()
                if 'begin' in next_line:
                    has_begin = True
                else:
                    # 单行 if 或三元表达式，返回当前行
                    return if_line + 1
        
        if not has_begin:
            return if_line + 1
        
        # 有 begin，需要找到对应的 end
        nesting_level = 1
        in_else_branch = False
        
        for i in range(if_line + 1, len(lines)):
            line_content = lines[i].strip().lower()
            
            # 跳过注释
            if line_content.startswith('//'):
                continue
            
            # 检查 begin（增加嵌套层级）
            # 注意：begin 可能在 if/else/case 等语句的同一行
            begin_count = len(re.findall(r'\bbegin\b', line_content))
            
            # 检查 end（减少嵌套层级）
            # 注意：end 可能后跟 else
            end_count = len(re.findall(r'\bend\b', line_content))
            
            # 更新嵌套层级
            nesting_level += begin_count - end_count
            
            # 检查是否是 else 分支
            if 'else' in line_content:
                in_else_branch = True
            
            # 当嵌套层级回到 0 时，找到了 if 块的结束
            if nesting_level <= 0:
                return i + 1  # 转换为 1-based 行号
        
        # 如果没有找到，返回文件末尾
        return len(lines)
    
    def _get_source_file_path(self) -> str:
        """
        获取源文件的完整路径
        
        Returns:
            源文件路径，如果找不到则返回 None
        """
        if not self.source_file:
            return None
        
        # 尝试从 CDFG 节点中获取源文件路径
        for node in self.cdfg.nodes.values():
            if node.source_file == self.source_file:
                # 尝试构建完整路径
                # 假设源文件在 designs/<design_name>/source/rtl/ 目录下
                import os
                
                # 尝试几种可能的路径
                possible_paths = [
                    # 直接使用源文件名
                    self.source_file,
                    # 在当前目录下查找
                    os.path.join("designs", "cv32e40p", "source", "rtl", self.source_file),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
                
                break
        
        return None
    
    def _collect_continuous_mux_chain(self, start_line: int, expected_count: int) -> List[str]:
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
                    return self._sort_mux_nodes_for_annotation(stmt_mux[:expected_mux_count])
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

        # 对于每个有覆盖数据的行
        for line_no, coverage in self.coverage_data.items():
            # 使用新的方法找到 MUX 链，传递分支类型
            mux_nodes = self._find_mux_chain_for_coverage(
                line_no, coverage.total_branches, coverage.branch_type
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
            mux_nodes: MUX 节点 ID 列表（已按逻辑顺序排序）
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

        # 调试信息
        if len(mux_nodes) < len(sorted_conditions):
            print(f"  注意：MUX 数量 ({len(mux_nodes)}) 少于条件数量 ({len(sorted_conditions)})")

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
            # 控制边的覆盖状态反映"这个条件的真分支是否被执行过"
            # 只有当 is_true_branch=True 时才更新 S 端口的覆盖状态
            if edge.target_port == "S" and is_true_branch:
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



def annotate_cdfg_with_coverage(
    cdfg: CDFG,
    html_path: str,
    instance_tag: str = None,
    propagate: bool = False,
) -> AnnotationStats:
    """
    便捷函数：用覆盖率数据标注 CDFG

    Args:
        cdfg: CDFG 对象
        html_path: 覆盖率报告 HTML 文件路径
        instance_tag: 实例标签，None 则使用第一个实例
        propagate: 是否启用覆盖率传播（标注非分支边）

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
        cdfg, coverage_data, source_file=source_file
    )
    stats = annotator.annotate()

    # 可选：传播覆盖率到非分支边
    if propagate:
        stats = annotator.propagate_coverage()

    return stats
