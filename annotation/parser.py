#!/usr/bin/env python3
"""
URG (Unified Report Generator) HTML 覆盖率报告解析器
支持 Synopsys VCS 生成的覆盖率报告格式
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class BranchStatus:
    """单个分支的覆盖状态"""

    condition_values: Dict[int, int] = field(
        default_factory=dict
    )  # 条件编号 -> 值 (0/1/-1), -1 表示 "-", 2 表示 CASE 标签被选中
    is_covered: bool = False
    case_labels: Dict[int, str] = field(
        default_factory=dict
    )  # 条件编号 -> CASE 标签名（如 "IDLE", "DIVIDE"）


@dataclass
class BranchCoverage:
    """单行代码的分支覆盖信息"""

    line_no: int  # 行号
    branch_type: str  # 分支类型 (IF, CASE 等)
    total_branches: int  # 总分支数
    covered_branches: int  # 已覆盖分支数
    percent: float  # 覆盖率百分比
    branches: List[BranchStatus] = field(default_factory=list)  # 每个分支的详细状态
    instance_index: int = 0  # 同行多实例时的实例索引（用于 generate 展开）




class CoverageParser:
    """覆盖率报告解析器"""

    def __init__(self, html_path: str):
        """
        初始化解析器

        Args:
            html_path: HTML 覆盖率报告文件路径
        """
        self.html_path = Path(html_path)
        if not self.html_path.exists():
            raise FileNotFoundError(f"覆盖率报告文件不存在: {html_path}")

        with open(self.html_path, "r", encoding="utf-8", errors="ignore") as f:
            self.html_content = f.read()

    def parse_branch_coverage(
        self, instance_tag: str = None
    ) -> List[BranchCoverage]:
        """
        解析分支覆盖数据

        Args:
            instance_tag: 要解析的实例标签（如 "inst_tag_58"），None 则解析模块级数据

        Returns:
            BranchCoverage 列表，支持同一行号有多个实例（如 generate 展开）
        """
        # 找到对应的 Branch 部分
        if instance_tag:
            # 实例级数据
            anchor_pattern = f'<a name="{instance_tag}_Branch"></a>'
            next_anchor_pattern = r'<a name="inst_tag_\d+_\w+"></a>|<hr>'
        else:
            # 模块级数据
            anchor_pattern = '<a name="Branch"></a>'
            next_anchor_pattern = (
                r'<a name="inst_tag_\d+"></a>|<a name="inst_tag_\d+_\w+"></a>'
            )

        # 找到 Branch 部分的起始位置
        start_match = re.search(re.escape(anchor_pattern), self.html_content)
        if not start_match:
            return {}

        start_pos = start_match.end()

        # 找到下一个部分的起始位置作为结束
        remaining = self.html_content[start_pos:]
        end_match = re.search(next_anchor_pattern, remaining)
        if end_match:
            branch_section = remaining[: end_match.start()]
        else:
            branch_section = remaining

        return self._parse_branch_section(branch_section)

    def _parse_branch_section(self, section_html: str) -> List[BranchCoverage]:
        """解析 Branch 部分的 HTML

        支持同一行号有多个实例（如 generate 展开），每个实例创建独立的 BranchCoverage 对象。
        """
        result: List[BranchCoverage] = []

        # 用于跟踪同一行号的实例索引
        line_instance_count: Dict[int, int] = {}

        # 1. 解析摘要表格，获取每行的基本信息
        # 匹配 IF 或 CASE 行: <td>IF</td><td class="rt">60</td><td class="rt">2</td>...
        summary_pattern = r'<tr class="s\d+">\s*<td>(IF|CASE|TERNARY)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>([\d.]+)'

        for match in re.finditer(summary_pattern, section_html):
            branch_type = match.group(1)
            line_no = int(match.group(2))
            total = int(match.group(3))
            covered = int(match.group(4))
            percent = float(match.group(5))

            # 计算这是该行号的第几个实例
            instance_idx = line_instance_count.get(line_no, 0)
            line_instance_count[line_no] = instance_idx + 1

            result.append(BranchCoverage(
                line_no=line_no,
                branch_type=branch_type,
                total_branches=total,
                covered_branches=covered,
                percent=percent,
                branches=[],
                instance_index=instance_idx,
            ))

        # 2. 解析每个分支的详细状态
        # 找到所有 "Branches:" 后的表格
        branches_sections = re.split(
            r"<span class=repname>Branches:</span>", section_html
        )

        # 按顺序遍历，每个 Branches 段落对应 result 中的一个 BranchCoverage
        for i, section in enumerate(
            branches_sections[1:], 0
        ):  # 从 0 开始索引，跳过第一个（摘要部分前的内容）
            if i >= len(result):
                break  # 没有更多的 BranchCoverage 需要填充

            # 解析分支状态表
            # 匹配: <tr class="uGreen"> 或 <tr class="uRed">
            branch_rows = re.findall(
                r'<tr class="(uGreen|uRed)">(.*?)</tr>', section, re.DOTALL
            )

            for row_class, row_content in branch_rows:
                is_covered = row_class == "uGreen"

                # 提取条件值
                # 格式1 (IF语句): <td align=center>1</td> 或 <td align=center nowrap>-</td>
                # 格式2 (CASE语句): <td nowrap>IDLE</td> 或 <td nowrap>DIVIDE</td>
                # 格式3 (旧版CASE): <td nowrap>(1.RESET )->(2)->...</td><td>Covered</td>
                #
                # 更宽泛的正则表达式：捕获所有 td 内容（除了空标签）
                td_values = re.findall(
                    r"<td[^>]*>([^<]+)</td>", row_content
                )

                condition_values = {}
                case_labels = {}

                # 首先检查是否是复杂路径格式：(1.STATE)->(2)->(!3)->...
                # 这种格式会被提取为单个 td，需要特殊处理
                path_match = re.search(r'<td[^>]*>\((\d+\.[^)]+)\)->', row_content)
                if path_match:
                    # 复杂路径格式
                    # 匹配整个 td 内容
                    full_path_match = re.search(r'<td[^>]*>(\([^<]+)</td>', row_content)
                    if full_path_match:
                        path_str = full_path_match.group(1)
                        # 解析路径中的条件: (1.STATE)->(2)->(!3)->(4.-)
                        # 数字表示条件编号，! 表示取反，- 表示无关
                        cond_matches = re.findall(r'\((!?)(\d+)(?:\.([^)]*))?\)', path_str)
                        for cond_match in cond_matches:
                            negated = cond_match[0] == '!'  # 是否取反
                            cond_num = int(cond_match[1])   # 条件编号
                            label = cond_match[2] if len(cond_match) > 2 else None  # 标签名

                            if label == '-':
                                # 无关条件
                                condition_values[cond_num] = -1
                            elif label:
                                # CASE 标签（如 "RESET", "DECODE"）
                                condition_values[cond_num] = 2
                                case_labels[cond_num] = label.strip()
                            elif negated:
                                # !n 表示条件 n 为假
                                condition_values[cond_num] = 0
                            else:
                                # n 表示条件 n 为真
                                condition_values[cond_num] = 1
                else:
                    # 简单表格格式
                    for idx, val in enumerate(td_values):
                        val = val.strip()
                        if val == "1":
                            condition_values[idx + 1] = 1
                        elif val == "0":
                            condition_values[idx + 1] = 0
                        elif val == "-":
                            condition_values[idx + 1] = -1
                        elif val in ("Covered", "Not Covered", ""):
                            # 跳过覆盖状态标签和空值
                            pass
                        else:
                            # CASE 标签值（如 "IDLE", "DIVIDE", "default"）
                            # 使用特殊值 2 表示 CASE 分支被选中
                            condition_values[idx + 1] = 2
                            case_labels[idx + 1] = val

                # 如果没有找到简单的条件值，尝试解析旧版 CASE 格式
                # 旧版 CASE 格式: (1.STATE_NAME)->(2)->(!3)->...
                if not condition_values:
                    # 匹配整个分支路径字符串
                    path_match = re.search(r'<td[^>]*>\(([^<]+)\)</td>', row_content)
                    if path_match:
                        path_str = path_match.group(1)
                        # 解析路径中的条件: (1.STATE)->(2)->(!3)->(4.-)
                        # 数字表示条件编号，! 表示取反，- 表示无关
                        cond_matches = re.findall(r'\((!?\d+)(?:\.([^)]+))?\)', path_str)
                        for cond_match in cond_matches:
                            cond_num_str = cond_match[0]  # 条件编号（可能带 !）
                            label = cond_match[1] if len(cond_match) > 1 else None  # 标签名

                            if cond_num_str.startswith('!'):
                                cond_num = int(cond_num_str[1:])
                                condition_values[cond_num] = 0  # !n 表示条件 n 为假
                            else:
                                cond_num = int(cond_num_str)
                                # 检查是否有标签（非 "-" 的标签表示 CASE 选择）
                                if label and label != "-":
                                    condition_values[cond_num] = 2  # CASE 标签被选中
                                    case_labels[cond_num] = label
                                else:
                                    condition_values[cond_num] = 1  # n 表示条件 n 为真

                if condition_values:  # 只有当有条件值时才添加
                    result[i].branches.append(
                        BranchStatus(
                            condition_values=condition_values,
                            is_covered=is_covered,
                            case_labels=case_labels
                        )
                    )

        return result

    def get_available_instances(self) -> List[str]:
        """获取报告中所有可用的实例标签"""
        pattern = r'<a name="(inst_tag_\d+)_Branch"></a>'
        matches = re.findall(pattern, self.html_content)
        return list(set(matches))

    def get_last_instance(self) -> Optional[str]:
        """获取第一个实例标签"""
        instances = self.get_available_instances()
        if instances:
            # 按数字排序
            instances.sort(key=lambda x: int(re.search(r"\d+", x).group()))
            return instances[-1]
        return None

    def get_module_name(self) -> Optional[str]:
        """
        获取覆盖率报告对应的模块名

        从 HTML title 中解析，格式如：
        <title>Unified Coverage Report :: Module :: cv32e40p_int_controller</title>

        Returns:
            模块名，如 "cv32e40p_int_controller"
        """
        match = re.search(r"<title>.*?Module\s*::\s*(\w+)</title>", self.html_content)
        if match:
            return match.group(1)
        return None

    def get_source_file(self) -> Optional[str]:
        """
        获取覆盖率报告对应的源文件名

        从 HTML 中解析，格式如：
        <span class=repname>Source File(s) : </span>
        ...openSrcFile('/.../cv32e40p_int_controller.sv')">...

        Returns:
            源文件名（不含路径），如 "cv32e40p_int_controller.sv"
        """
        # 查找 Source File(s) 部分
        match = re.search(
            r'<span class=repname>Source File\(s\)\s*:\s*</span>.*?openSrcFile\([\'"]([^\'"]+)[\'"]\)',
            self.html_content,
            re.DOTALL,
        )
        if match:
            full_path = match.group(1)
            # 提取文件名
            return full_path.replace("\\", "/").split("/")[-1]
        return None



def main():
    """命令行入口，用于调试和测试覆盖率解析"""
    import argparse

    parser = argparse.ArgumentParser(
        description="解析 VCS/URG 覆盖率 HTML 报告"
    )
    parser.add_argument("html_path", help="覆盖率 HTML 文件路径")
    parser.add_argument("--instance", "-i", help="实例标签 (如 inst_tag_58)")
    parser.add_argument("--branch", "-b", action="store_true", help="只解析分支覆盖")

    args = parser.parse_args()

    # 如果没有指定类型，则解析所有
    parse_all = not args.branch

    try:
        cov_parser = CoverageParser(args.html_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    # 基本信息
    print("=" * 60)
    print(f"文件: {cov_parser.html_path}")
    print(f"模块: {cov_parser.get_module_name()}")
    print(f"源文件: {cov_parser.get_source_file()}")

    # 可用实例
    instances = cov_parser.get_available_instances()
    if instances:
        instances.sort(key=lambda x: int(re.search(r"\d+", x).group()))
        print(f"可用实例 ({len(instances)}): {', '.join(instances[:5])}" +
              (f" ... (共 {len(instances)} 个)" if len(instances) > 5 else ""))
        print(f"最后实例: {cov_parser.get_last_instance()}")

    instance_tag = args.instance
    if instance_tag:
        print(f"\n使用实例: {instance_tag}")

    # 分支覆盖
    if parse_all or args.branch:
        print("\n" + "=" * 60)
        print("分支覆盖 (Branch Coverage)")
        print("=" * 60)

        branch_cov_list = cov_parser.parse_branch_coverage(instance_tag)
        if not branch_cov_list:
            print("  (无分支覆盖数据)")
        else:
            # 按行号分组显示
            from collections import defaultdict
            by_line: Dict[int, List[BranchCoverage]] = defaultdict(list)
            for bc in branch_cov_list:
                by_line[bc.line_no].append(bc)

            for line_no in sorted(by_line.keys()):
                coverages = by_line[line_no]
                for bc in coverages:
                    instance_str = f" [实例 {bc.instance_index}]" if len(coverages) > 1 else ""
                    status = "✓" if bc.covered_branches == bc.total_branches else "✗"
                    print(f"\n行 {line_no}{instance_str} [{bc.branch_type}] {status} "
                          f"{bc.covered_branches}/{bc.total_branches} ({bc.percent:.1f}%)")

                    for idx, branch in enumerate(bc.branches):
                        cov_mark = "✓" if branch.is_covered else "✗"
                        cond_str = ", ".join(
                            f"C{k}={'T' if v == 1 else 'F' if v == 0 else '-'}"
                            for k, v in sorted(branch.condition_values.items())
                        )
                        print(f"  分支 {idx + 1}: [{cov_mark}] {cond_str}")


    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
