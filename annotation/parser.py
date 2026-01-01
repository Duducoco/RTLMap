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
    condition_values: Dict[int, int] = field(default_factory=dict)  # 条件编号 -> 值 (0/1), -1 表示 "-"
    is_covered: bool = False


@dataclass
class BranchCoverage:
    """单行代码的分支覆盖信息"""
    line_no: int                   # 行号
    branch_type: str               # 分支类型 (IF, CASE 等)
    total_branches: int            # 总分支数
    covered_branches: int          # 已覆盖分支数
    percent: float                 # 覆盖率百分比
    branches: List[BranchStatus] = field(default_factory=list)  # 每个分支的详细状态


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

        with open(self.html_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.html_content = f.read()

    def parse_branch_coverage(self, instance_tag: str = None) -> Dict[int, BranchCoverage]:
        """
        解析分支覆盖数据

        Args:
            instance_tag: 要解析的实例标签（如 "inst_tag_58"），None 则解析模块级数据

        Returns:
            行号 -> BranchCoverage 的映射字典
        """
        # 找到对应的 Branch 部分
        if instance_tag:
            # 实例级数据
            anchor_pattern = f'<a name="{instance_tag}_Branch"></a>'
            next_anchor_pattern = r'<a name="inst_tag_\d+_\w+"></a>|<hr>'
        else:
            # 模块级数据
            anchor_pattern = '<a name="Branch"></a>'
            next_anchor_pattern = r'<a name="inst_tag_\d+"></a>|<a name="inst_tag_\d+_\w+"></a>'

        # 找到 Branch 部分的起始位置
        start_match = re.search(re.escape(anchor_pattern), self.html_content)
        if not start_match:
            return {}

        start_pos = start_match.end()

        # 找到下一个部分的起始位置作为结束
        remaining = self.html_content[start_pos:]
        end_match = re.search(next_anchor_pattern, remaining)
        if end_match:
            branch_section = remaining[:end_match.start()]
        else:
            branch_section = remaining

        return self._parse_branch_section(branch_section)

    def _parse_branch_section(self, section_html: str) -> Dict[int, BranchCoverage]:
        """解析 Branch 部分的 HTML"""
        result = {}

        # 1. 解析摘要表格，获取每行的基本信息
        # 匹配 IF 或 CASE 行: <td>IF</td><td class="rt">60</td><td class="rt">2</td>...
        summary_pattern = r'<tr class="s\d+">\s*<td>(IF|CASE|TERNARY)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>([\d.]+)'

        for match in re.finditer(summary_pattern, section_html):
            branch_type = match.group(1)
            line_no = int(match.group(2))
            total = int(match.group(3))
            covered = int(match.group(4))
            percent = float(match.group(5))

            result[line_no] = BranchCoverage(
                line_no=line_no,
                branch_type=branch_type,
                total_branches=total,
                covered_branches=covered,
                percent=percent,
                branches=[]
            )

        # 2. 解析每个分支的详细状态
        # 找到所有 "Branches:" 后的表格
        branches_sections = re.split(r'<span class=repname>Branches:</span>', section_html)

        current_line_no = None
        for i, section in enumerate(branches_sections[1:], 1):  # 跳过第一个（摘要部分前的内容）
            # 找到这个 Branches 对应的行号
            # 往前找最近的行号信息
            prev_section = branches_sections[i-1] if i > 0 else ""
            # 查找 pre class="code" 中的行号
            code_match = re.search(r'<pre class="code">.*?(\d+)\s+(?:if|else if|case|assign)', prev_section, re.DOTALL | re.IGNORECASE)
            if code_match:
                current_line_no = int(code_match.group(1))

            if current_line_no is None or current_line_no not in result:
                # 尝试从摘要表中按顺序获取
                if i <= len(result):
                    sorted_lines = sorted(result.keys())
                    if i <= len(sorted_lines):
                        current_line_no = sorted_lines[i-1]

            if current_line_no is None or current_line_no not in result:
                continue

            # 解析分支状态表
            # 匹配: <tr class="uGreen"> 或 <tr class="uRed">
            branch_rows = re.findall(
                r'<tr class="(uGreen|uRed)">(.*?)</tr>',
                section,
                re.DOTALL
            )

            for row_class, row_content in branch_rows:
                is_covered = row_class == 'uGreen'

                # 提取条件值
                # <td align=center>1</td> 或 <td align=center nowrap>-</td>
                td_values = re.findall(r'<td[^>]*>([01-]|Covered|Not Covered)</td>', row_content)

                condition_values = {}
                for idx, val in enumerate(td_values):
                    if val == '1':
                        condition_values[idx + 1] = 1
                    elif val == '0':
                        condition_values[idx + 1] = 0
                    elif val == '-':
                        condition_values[idx + 1] = -1
                    # 跳过 "Covered" / "Not Covered"

                if condition_values:  # 只有当有条件值时才添加
                    result[current_line_no].branches.append(BranchStatus(
                        condition_values=condition_values,
                        is_covered=is_covered
                    ))

        return result

    def get_available_instances(self) -> List[str]:
        """获取报告中所有可用的实例标签"""
        pattern = r'<a name="(inst_tag_\d+)_Branch"></a>'
        matches = re.findall(pattern, self.html_content)
        return list(set(matches))

    def get_first_instance(self) -> Optional[str]:
        """获取第一个实例标签"""
        instances = self.get_available_instances()
        if instances:
            # 按数字排序
            instances.sort(key=lambda x: int(re.search(r'\d+', x).group()))
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
        match = re.search(r'<title>.*?Module\s*::\s*(\w+)</title>', self.html_content)
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
            re.DOTALL
        )
        if match:
            full_path = match.group(1)
            # 提取文件名
            return full_path.replace('\\', '/').split('/')[-1]
        return None

