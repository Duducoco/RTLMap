#!/usr/bin/env python3
"""
数据集统计分析脚本

分析 simulation_results 下测试用例的 RTL 标注 JSON 和 ASM CDFG JSON，
输出分布特征、覆盖率标签统计、图结构特征和数据质量检查。

使用示例：
    uv run python analyze_dataset.py --sim-dir designs/cv32e40p/simulation_results_2
    uv run python analyze_dataset.py --sim-dir designs/cv32e40p/simulation_results_2 --sample 500
    uv run python analyze_dataset.py --sim-dir designs/cv32e40p/simulation_results_2 --plot --output-dir analysis_output
    uv run python analyze_dataset.py --sim-dir designs/cv32e40p/simulation_results_2 --json c
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class RTLStats:
    """单个 RTL JSON 文件的轻量统计"""

    module_name: str = ""
    test_case: str = ""
    num_nodes: int = 0
    num_edges: int = 0
    branch_coverage: float = -1.0  # 原始百分比值

    # 分布计数
    node_type_counts: dict[str, int] = field(default_factory=dict)
    cell_type_counts: dict[str, int] = field(default_factory=dict)
    node_width_counts: dict[int, int] = field(default_factory=dict)
    edge_type_counts: dict[str, int] = field(default_factory=dict)
    edge_width_counts: dict[int, int] = field(default_factory=dict)

    # 覆盖率标签
    label_minus1: int = 0  # 未标注
    label_0: int = 0  # 未覆盖
    label_1: int = 0  # 已覆盖

    # 按边类型的标签分布: edge_type -> {-1: N, 0: N, 1: N}
    edge_type_label_counts: dict[str, dict[int, int]] = field(default_factory=dict)

    # 错误标记
    error: str = ""


@dataclass
class ASMStats:
    """单个 ASM JSON 文件的轻量统计"""

    test_case: str = ""
    num_nodes: int = 0
    num_edges: int = 0
    total_instructions: int = 0

    node_type_counts: dict[str, int] = field(default_factory=dict)
    edge_type_counts: dict[str, int] = field(default_factory=dict)

    # 每个基本块的指令数
    block_instr_counts: list[int] = field(default_factory=list)

    error: str = ""


# ---------------------------------------------------------------------------
# 单文件统计提取（worker 函数，在子进程执行）
# ---------------------------------------------------------------------------


def extract_rtl_stats(json_path: str, test_case: str) -> RTLStats:
    """从单个 RTL annotated JSON 提取统计信息"""
    stats = RTLStats(test_case=test_case)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        stats.error = str(e)
        return stats

    stats.module_name = data.get("module_name", "")
    stats.branch_coverage = data.get("branch_coverage", -1.0)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    stats.num_nodes = len(nodes)
    stats.num_edges = len(edges)

    # 节点统计
    node_type_c: dict[str, int] = {}
    cell_type_c: dict[str, int] = {}
    node_width_c: dict[int, int] = {}
    for n in nodes:
        nt = n.get("type", "UNKNOWN")
        node_type_c[nt] = node_type_c.get(nt, 0) + 1
        ct = n.get("cell_type", "UNKNOWN")
        cell_type_c[ct] = cell_type_c.get(ct, 0) + 1
        w = n.get("width", 1)
        node_width_c[w] = node_width_c.get(w, 0) + 1

    stats.node_type_counts = node_type_c
    stats.cell_type_counts = cell_type_c
    stats.node_width_counts = node_width_c

    # 边统计
    edge_type_c: dict[str, int] = {}
    edge_width_c: dict[int, int] = {}
    edge_type_label_c: dict[str, dict[int, int]] = {}
    lm1 = l0 = l1 = 0
    for e in edges:
        et = e.get("type", "DATA")
        edge_type_c[et] = edge_type_c.get(et, 0) + 1
        w = e.get("width", 1)
        edge_width_c[w] = edge_width_c.get(w, 0) + 1
        label = e.get("coverage_label", -1)
        if label == -1:
            lm1 += 1
        elif label == 0:
            l0 += 1
        elif label == 1:
            l1 += 1
        # 按边类型统计标签分布
        if et not in edge_type_label_c:
            edge_type_label_c[et] = {-1: 0, 0: 0, 1: 0}
        edge_type_label_c[et][label] = edge_type_label_c[et].get(label, 0) + 1

    stats.edge_type_counts = edge_type_c
    stats.edge_width_counts = edge_width_c
    stats.edge_type_label_counts = edge_type_label_c
    stats.label_minus1 = lm1
    stats.label_0 = l0
    stats.label_1 = l1

    return stats


def extract_asm_stats(json_path: str, test_case: str) -> ASMStats:
    """从单个 ASM CDFG JSON 提取统计信息"""
    stats = ASMStats(test_case=test_case)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        stats.error = str(e)
        return stats

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    stats.num_nodes = len(nodes)
    stats.num_edges = len(edges)

    node_type_c: dict[str, int] = {}
    block_instr_counts: list[int] = []
    total_instr = 0
    for _nid, node in nodes.items():
        nt = node.get("node_type", "UNKNOWN")
        node_type_c[nt] = node_type_c.get(nt, 0) + 1
        instrs = node.get("instructions", [])
        ic = len(instrs)
        block_instr_counts.append(ic)
        total_instr += ic

    stats.node_type_counts = node_type_c
    stats.block_instr_counts = block_instr_counts
    stats.total_instructions = total_instr

    edge_type_c: dict[str, int] = {}
    for e in edges:
        et = e.get("edge_type", "UNKNOWN")
        edge_type_c[et] = edge_type_c.get(et, 0) + 1
    stats.edge_type_counts = edge_type_c

    return stats


def _process_test_case(
    args: tuple[str, str],
) -> tuple[list[RTLStats], ASMStats | None]:
    """处理单个测试用例，返回 (rtl_stats_list, asm_stats)"""
    sim_dir, tc = args
    tc_dir = os.path.join(sim_dir, tc)
    annotated_dir = os.path.join(tc_dir, "annotated")

    # RTL 文件
    rtl_list: list[RTLStats] = []
    if os.path.isdir(annotated_dir):
        for fn in os.listdir(annotated_dir):
            if fn.endswith(".json"):
                rtl_list.append(extract_rtl_stats(os.path.join(annotated_dir, fn), tc))

    # ASM 文件
    asm_path = os.path.join(tc_dir, f"{tc}.json")
    asm_stats = None
    if os.path.isfile(asm_path):
        asm_stats = extract_asm_stats(asm_path, tc)

    return rtl_list, asm_stats


# ---------------------------------------------------------------------------
# 发现与收集
# ---------------------------------------------------------------------------


def discover_test_cases(sim_dir: str) -> list[str]:
    """发现 simulation_results 下所有测试用例目录"""
    entries = sorted(os.listdir(sim_dir))
    return [e for e in entries if os.path.isdir(os.path.join(sim_dir, e))]


def collect_all_stats(
    sim_dir: str, sample_n: int | None = None, workers: int | None = None
) -> tuple[list[RTLStats], list[ASMStats]]:
    """并行收集所有测试用例的统计信息"""
    test_cases = discover_test_cases(sim_dir)
    total = len(test_cases)
    console.print(f"发现 [bold]{total}[/bold] 个测试用例")

    if sample_n and sample_n < total:
        random.seed(42)
        test_cases = random.sample(test_cases, sample_n)
        console.print(f"随机采样 [bold]{sample_n}[/bold] 个")

    if workers is None:
        workers = min(os.cpu_count() or 4, 32)

    all_rtl: list[RTLStats] = []
    all_asm: list[ASMStats] = []

    task_args = [(sim_dir, tc) for tc in test_cases]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("收集统计信息", total=len(task_args))

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_test_case, arg): arg for arg in task_args}
            for future in as_completed(futures):
                rtl_list, asm_stats = future.result()
                all_rtl.extend(rtl_list)
                if asm_stats is not None:
                    all_asm.append(asm_stats)
                progress.advance(task)

    return all_rtl, all_asm


# ---------------------------------------------------------------------------
# 统计辅助
# ---------------------------------------------------------------------------


def _describe(values: list[float | int]) -> dict[str, float]:
    """计算描述统计量"""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "count": 0}
    n = len(values)
    s = sorted(values)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in s) / n if n > 1 else 0
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(math.sqrt(var), 2),
        "count": n,
    }


def _percentiles(values: list[float | int]) -> dict[str, float]:
    """计算分位数"""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    result = {}
    for p in [5, 25, 50, 75, 95]:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = round(s[idx], 2)
    return result


def _top_n(counter: Counter, n: int = 20) -> list[tuple[str, int]]:
    """返回 Top-N"""
    return counter.most_common(n)


# ---------------------------------------------------------------------------
# 输出函数
# ---------------------------------------------------------------------------


def print_overview(
    rtl_stats: list[RTLStats],
    asm_stats: list[ASMStats],
    test_cases_total: int,
) -> list[str]:
    """样本概览"""
    lines: list[str] = []
    console.rule("[bold]1. 样本概览[/bold]")

    # RTL 模块列表
    module_counts: Counter[str] = Counter()
    tc_set: set[str] = set()
    for r in rtl_stats:
        if not r.error:
            module_counts[r.module_name] += 1
            tc_set.add(r.test_case)

    asm_tc_set = {a.test_case for a in asm_stats if not a.error}

    table = Table(title="概览")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("测试用例总数（扫描）", str(test_cases_total))
    table.add_row("有 RTL 标注的测试用例", str(len(tc_set)))
    table.add_row("有 ASM CDFG 的测试用例", str(len(asm_tc_set)))
    table.add_row("RTL JSON 文件总数", str(len([r for r in rtl_stats if not r.error])))
    table.add_row("RTL 模块数", str(len(module_counts)))
    console.print(table)

    lines.append("## 1. 样本概览\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 测试用例总数 | {test_cases_total} |")
    lines.append(f"| 有 RTL 标注 | {len(tc_set)} |")
    lines.append(f"| 有 ASM CDFG | {len(asm_tc_set)} |")
    lines.append(f"| RTL JSON 总数 | {len([r for r in rtl_stats if not r.error])} |")
    lines.append(f"| RTL 模块数 | {len(module_counts)} |\n")

    # 各模块样本数
    mod_table = Table(title="各模块样本数")
    mod_table.add_column("模块", style="cyan")
    mod_table.add_column("样本数", style="green")
    lines.append("### RTL 模块样本数\n")
    lines.append("| 模块 | 样本数 |")
    lines.append("|------|--------|")
    for mod, cnt in sorted(module_counts.items()):
        mod_table.add_row(mod, str(cnt))
        lines.append(f"| {mod} | {cnt} |")
    console.print(mod_table)
    lines.append("")

    # ASM 统计概览
    if asm_stats:
        asm_nodes = [a.num_nodes for a in asm_stats if not a.error]
        asm_instrs = [a.total_instructions for a in asm_stats if not a.error]
        desc_nodes = _describe(asm_nodes)
        desc_instrs = _describe(asm_instrs)

        asm_table = Table(title="ASM 概览")
        asm_table.add_column("指标", style="cyan")
        for k in ["min", "max", "mean", "median", "std"]:
            asm_table.add_column(k, style="green")
        asm_table.add_row(
            "节点数（基本块）",
            *[str(desc_nodes[k]) for k in ["min", "max", "mean", "median", "std"]],
        )
        asm_table.add_row(
            "总指令数",
            *[str(desc_instrs[k]) for k in ["min", "max", "mean", "median", "std"]],
        )
        console.print(asm_table)

        lines.append("### ASM 概览\n")
        lines.append("| 指标 | min | max | mean | median | std |")
        lines.append("|------|-----|-----|------|--------|-----|")
        lines.append(
            f"| 节点数 | {desc_nodes['min']} | {desc_nodes['max']} | "
            f"{desc_nodes['mean']} | {desc_nodes['median']} | {desc_nodes['std']} |"
        )
        lines.append(
            f"| 总指令数 | {desc_instrs['min']} | {desc_instrs['max']} | "
            f"{desc_instrs['mean']} | {desc_instrs['median']} | {desc_instrs['std']} |"
        )
        lines.append("")

    return lines


def print_rtl_stats(rtl_stats: list[RTLStats]) -> list[str]:
    """RTL 图统计（按模块分组）"""
    lines: list[str] = []
    console.rule("[bold]2. RTL 图统计[/bold]")
    lines.append("## 2. RTL 图统计\n")

    # 按模块分组
    by_module: dict[str, list[RTLStats]] = defaultdict(list)
    for r in rtl_stats:
        if not r.error:
            by_module[r.module_name].append(r)

    # 节点数/边数统计
    table = Table(title="节点数/边数（按模块）")
    table.add_column("模块", style="cyan")
    for col in [
        "N_min",
        "N_max",
        "N_mean",
        "N_med",
        "E_min",
        "E_max",
        "E_mean",
        "E_med",
    ]:
        table.add_column(col, style="green")

    lines.append("### 节点/边数量\n")
    lines.append(
        "| 模块 | N_min | N_max | N_mean | N_med | E_min | E_max | E_mean | E_med |"
    )
    lines.append(
        "|------|-------|-------|--------|-------|-------|-------|--------|-------|"
    )

    for mod in sorted(by_module):
        stats_list = by_module[mod]
        nd = _describe([s.num_nodes for s in stats_list])
        ed = _describe([s.num_edges for s in stats_list])
        table.add_row(
            mod,
            str(nd["min"]),
            str(nd["max"]),
            str(nd["mean"]),
            str(nd["median"]),
            str(ed["min"]),
            str(ed["max"]),
            str(ed["mean"]),
            str(ed["median"]),
        )
        lines.append(
            f"| {mod} | {nd['min']} | {nd['max']} | {nd['mean']} | {nd['median']} "
            f"| {ed['min']} | {ed['max']} | {ed['mean']} | {ed['median']} |"
        )
    console.print(table)
    lines.append("")

    # 全局节点 type 分布
    global_node_type: Counter[str] = Counter()
    global_cell_type: Counter[str] = Counter()
    global_node_width: Counter[int] = Counter()
    global_edge_type: Counter[str] = Counter()
    global_edge_width: Counter[int] = Counter()

    for r in rtl_stats:
        if r.error:
            continue
        for k, v in r.node_type_counts.items():
            global_node_type[k] += v
        for k, v in r.cell_type_counts.items():
            global_cell_type[k] += v
        for k, v in r.node_width_counts.items():
            global_node_width[k] += v
        for k, v in r.edge_type_counts.items():
            global_edge_type[k] += v
        for k, v in r.edge_width_counts.items():
            global_edge_width[k] += v

    # 节点 type 分布
    total_nodes = sum(global_node_type.values())
    nt_table = Table(title="节点 type 分布")
    nt_table.add_column("类型", style="cyan")
    nt_table.add_column("数量", style="green")
    nt_table.add_column("占比", style="yellow")
    lines.append("### 节点 type 分布\n")
    lines.append("| 类型 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for nt, cnt in global_node_type.most_common():
        pct = f"{cnt / total_nodes * 100:.1f}%" if total_nodes else "0%"
        nt_table.add_row(nt, str(cnt), pct)
        lines.append(f"| {nt} | {cnt} | {pct} |")
    console.print(nt_table)
    lines.append("")

    # cell_type Top-20
    ct_table = Table(title="cell_type Top-20")
    ct_table.add_column("cell_type", style="cyan")
    ct_table.add_column("数量", style="green")
    ct_table.add_column("占比", style="yellow")
    lines.append("### cell_type Top-20\n")
    lines.append("| cell_type | 数量 | 占比 |")
    lines.append("|-----------|------|------|")
    for ct, cnt in _top_n(global_cell_type, 20):
        pct = f"{cnt / total_nodes * 100:.1f}%" if total_nodes else "0%"
        ct_table.add_row(ct, str(cnt), pct)
        lines.append(f"| {ct} | {cnt} | {pct} |")
    console.print(ct_table)
    lines.append("")

    # 节点 width 分布（Top-15）
    nw_table = Table(title="节点 width 分布 Top-15")
    nw_table.add_column("位宽", style="cyan")
    nw_table.add_column("数量", style="green")
    lines.append("### 节点 width 分布 Top-15\n")
    lines.append("| 位宽 | 数量 |")
    lines.append("|------|------|")
    for w, cnt in global_node_width.most_common(15):
        nw_table.add_row(str(w), str(cnt))
        lines.append(f"| {w} | {cnt} |")
    console.print(nw_table)
    lines.append("")

    # 边 type 分布
    total_edges = sum(global_edge_type.values())
    et_table = Table(title="边 type 分布")
    et_table.add_column("类型", style="cyan")
    et_table.add_column("数量", style="green")
    et_table.add_column("占比", style="yellow")
    lines.append("### 边 type 分布\n")
    lines.append("| 类型 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for et, cnt in global_edge_type.most_common():
        pct = f"{cnt / total_edges * 100:.1f}%" if total_edges else "0%"
        et_table.add_row(et, str(cnt), pct)
        lines.append(f"| {et} | {cnt} | {pct} |")
    console.print(et_table)
    lines.append("")

    # 边 width 分布（Top-15）
    ew_table = Table(title="边 width 分布 Top-15")
    ew_table.add_column("位宽", style="cyan")
    ew_table.add_column("数量", style="green")
    lines.append("### 边 width 分布 Top-15\n")
    lines.append("| 位宽 | 数量 |")
    lines.append("|------|------|")
    for w, cnt in global_edge_width.most_common(15):
        ew_table.add_row(str(w), str(cnt))
        lines.append(f"| {w} | {cnt} |")
    console.print(ew_table)
    lines.append("")

    return lines


def print_asm_stats(asm_stats: list[ASMStats]) -> list[str]:
    """ASM 图统计"""
    lines: list[str] = []
    console.rule("[bold]3. ASM 图统计[/bold]")
    lines.append("## 3. ASM 图统计\n")

    valid = [a for a in asm_stats if not a.error]
    if not valid:
        console.print("[yellow]无有效 ASM 数据[/yellow]")
        lines.append("无有效 ASM 数据\n")
        return lines

    # 节点数/边数/指令数
    desc_table = Table(title="ASM 图描述统计")
    desc_table.add_column("指标", style="cyan")
    for col in ["min", "max", "mean", "median", "std"]:
        desc_table.add_column(col, style="green")

    lines.append("### 描述统计\n")
    lines.append("| 指标 | min | max | mean | median | std |")
    lines.append("|------|-----|-----|------|--------|-----|")

    for label, vals in [
        ("节点数（基本块）", [a.num_nodes for a in valid]),
        ("边数", [a.num_edges for a in valid]),
        ("总指令数", [a.total_instructions for a in valid]),
    ]:
        d = _describe(vals)
        desc_table.add_row(
            label, *[str(d[k]) for k in ["min", "max", "mean", "median", "std"]]
        )
        lines.append(
            f"| {label} | {d['min']} | {d['max']} | {d['mean']} | {d['median']} | {d['std']} |"
        )
    console.print(desc_table)
    lines.append("")

    # 每基本块平均指令数
    all_block_counts: list[int] = []
    for a in valid:
        all_block_counts.extend(a.block_instr_counts)
    if all_block_counts:
        bd = _describe(all_block_counts)
        console.print(
            f"每基本块指令数: mean={bd['mean']}, median={bd['median']}, "
            f"min={bd['min']}, max={bd['max']}"
        )
        lines.append(
            f"每基本块指令数: mean={bd['mean']}, median={bd['median']}, "
            f"min={bd['min']}, max={bd['max']}\n"
        )

    # 节点 type 分布
    global_asm_nt: Counter[str] = Counter()
    for a in valid:
        for k, v in a.node_type_counts.items():
            global_asm_nt[k] += v

    total_asm_nodes = sum(global_asm_nt.values())
    ant_table = Table(title="ASM 节点 type 分布")
    ant_table.add_column("类型", style="cyan")
    ant_table.add_column("数量", style="green")
    ant_table.add_column("占比", style="yellow")
    lines.append("### ASM 节点 type 分布\n")
    lines.append("| 类型 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for nt, cnt in global_asm_nt.most_common():
        pct = f"{cnt / total_asm_nodes * 100:.1f}%" if total_asm_nodes else "0%"
        ant_table.add_row(nt, str(cnt), pct)
        lines.append(f"| {nt} | {cnt} | {pct} |")
    console.print(ant_table)
    lines.append("")

    # 边 type 分布
    global_asm_et: Counter[str] = Counter()
    for a in valid:
        for k, v in a.edge_type_counts.items():
            global_asm_et[k] += v

    total_asm_edges = sum(global_asm_et.values())
    aet_table = Table(title="ASM 边 type 分布")
    aet_table.add_column("类型", style="cyan")
    aet_table.add_column("数量", style="green")
    aet_table.add_column("占比", style="yellow")
    lines.append("### ASM 边 type 分布\n")
    lines.append("| 类型 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for et, cnt in global_asm_et.most_common():
        pct = f"{cnt / total_asm_edges * 100:.1f}%" if total_asm_edges else "0%"
        aet_table.add_row(et, str(cnt), pct)
        lines.append(f"| {et} | {cnt} | {pct} |")
    console.print(aet_table)
    lines.append("")

    return lines


def print_label_stats(rtl_stats: list[RTLStats]) -> list[str]:
    """覆盖率标签分布"""
    lines: list[str] = []
    console.rule("[bold]4. 覆盖率标签分布[/bold]")
    lines.append("## 4. 覆盖率标签分布\n")

    valid = [r for r in rtl_stats if not r.error]
    if not valid:
        return lines

    # 全局边级标签分布
    total_m1 = sum(r.label_minus1 for r in valid)
    total_0 = sum(r.label_0 for r in valid)
    total_1 = sum(r.label_1 for r in valid)
    total_all = total_m1 + total_0 + total_1

    label_table = Table(title="全局边级标签分布")
    label_table.add_column("标签", style="cyan")
    label_table.add_column("数量", style="green")
    label_table.add_column("占比", style="yellow")
    lines.append("### 全局边级标签分布\n")
    lines.append("| 标签 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for label_name, cnt in [
        ("-1（未标注）", total_m1),
        ("0（未覆盖）", total_0),
        ("1（已覆盖）", total_1),
    ]:
        pct = f"{cnt / total_all * 100:.1f}%" if total_all else "0%"
        label_table.add_row(label_name, str(cnt), pct)
        lines.append(f"| {label_name} | {cnt} | {pct} |")
    console.print(label_table)
    lines.append("")

    # 有效标签率（每样本）
    effective_rates: list[float] = []
    for r in valid:
        total = r.label_minus1 + r.label_0 + r.label_1
        if total > 0:
            effective_rates.append((r.label_0 + r.label_1) / total * 100)
    if effective_rates:
        erd = _describe(effective_rates)
        console.print(
            f"有效标签率（%）: mean={erd['mean']}, median={erd['median']}, "
            f"min={erd['min']}, max={erd['max']}, std={erd['std']}"
        )
        lines.append(
            f"有效标签率（%）: mean={erd['mean']}, median={erd['median']}, "
            f"min={erd['min']}, max={erd['max']}, std={erd['std']}\n"
        )

    # 正负样本比（按模块分组）
    by_module: dict[str, list[RTLStats]] = defaultdict(list)
    for r in valid:
        by_module[r.module_name].append(r)

    ratio_table = Table(title="正负样本比（按模块）")
    ratio_table.add_column("模块", style="cyan")
    ratio_table.add_column("已覆盖(1)", style="green")
    ratio_table.add_column("未覆盖(0)", style="red")
    ratio_table.add_column("比值(1:0)", style="yellow")
    ratio_table.add_column("未标注(-1)", style="dim")
    lines.append("### 正负样本比（按模块）\n")
    lines.append("| 模块 | 已覆盖(1) | 未覆盖(0) | 比值(1:0) | 未标注(-1) |")
    lines.append("|------|-----------|-----------|-----------|------------|")

    for mod in sorted(by_module):
        m1 = sum(r.label_minus1 for r in by_module[mod])
        l0 = sum(r.label_0 for r in by_module[mod])
        l1 = sum(r.label_1 for r in by_module[mod])
        ratio = f"{l1 / l0:.2f}" if l0 > 0 else "inf"
        ratio_table.add_row(mod, str(l1), str(l0), ratio, str(m1))
        lines.append(f"| {mod} | {l1} | {l0} | {ratio} | {m1} |")
    console.print(ratio_table)
    lines.append("")

    # 图级 branch_coverage 统计（按模块）
    console.print()
    cov_table = Table(title="branch_coverage 分布（按模块）")
    cov_table.add_column("模块", style="cyan")
    for col in ["mean", "std", "min", "p25", "median", "p75", "max"]:
        cov_table.add_column(col, style="green")
    lines.append("### branch_coverage 分布（按模块）\n")
    lines.append("| 模块 | mean | std | min | p25 | median | p75 | max |")
    lines.append("|------|------|-----|-----|-----|--------|-----|-----|")

    for mod in sorted(by_module):
        covs = [r.branch_coverage for r in by_module[mod] if r.branch_coverage >= 0]
        if not covs:
            continue
        d = _describe(covs)
        p = _percentiles(covs)
        cov_table.add_row(
            mod,
            str(d["mean"]),
            str(d["std"]),
            str(d["min"]),
            str(p.get("p25", "")),
            str(d["median"]),
            str(p.get("p75", "")),
            str(d["max"]),
        )
        lines.append(
            f"| {mod} | {d['mean']} | {d['std']} | {d['min']} | {p.get('p25', '')} "
            f"| {d['median']} | {p.get('p75', '')} | {d['max']} |"
        )
    console.print(cov_table)
    lines.append("")

    return lines


def print_structure_stats(
    rtl_stats: list[RTLStats], asm_stats: list[ASMStats]
) -> list[str]:
    """图结构特征"""
    lines: list[str] = []
    console.rule("[bold]5. 图结构特征[/bold]")
    lines.append("## 5. 图结构特征\n")

    valid_rtl = [r for r in rtl_stats if not r.error]
    valid_asm = {a.test_case: a for a in asm_stats if not a.error}

    # RTL 图密度 (E / N²)
    densities: list[float] = []
    for r in valid_rtl:
        if r.num_nodes > 1:
            densities.append(r.num_edges / (r.num_nodes**2))
    if densities:
        dd = _describe(densities)
        console.print(
            f"RTL 图密度 (E/N²): mean={dd['mean']}, median={dd['median']}, "
            f"min={dd['min']}, max={dd['max']}"
        )
        lines.append(
            f"RTL 图密度 (E/N²): mean={dd['mean']}, median={dd['median']}, "
            f"min={dd['min']}, max={dd['max']}\n"
        )

    # RTL/ASM 节点数比值
    ratios: list[float] = []
    for r in valid_rtl:
        asm = valid_asm.get(r.test_case)
        if asm and asm.num_nodes > 0:
            ratios.append(r.num_nodes / asm.num_nodes)
    if ratios:
        rd = _describe(ratios)
        console.print(
            f"RTL/ASM 节点数比值: mean={rd['mean']}, median={rd['median']}, "
            f"min={rd['min']}, max={rd['max']}"
        )
        lines.append(
            f"RTL/ASM 节点数比值: mean={rd['mean']}, median={rd['median']}, "
            f"min={rd['min']}, max={rd['max']}\n"
        )

    # ASM 指令数 vs RTL 覆盖率相关性（Pearson）
    pairs_x: list[float] = []
    pairs_y: list[float] = []
    for r in valid_rtl:
        if r.branch_coverage < 0:
            continue
        asm = valid_asm.get(r.test_case)
        if asm:
            pairs_x.append(float(asm.total_instructions))
            pairs_y.append(r.branch_coverage)

    if len(pairs_x) > 2:
        n = len(pairs_x)
        r_val = _pearson(pairs_x, pairs_y)
        console.print(
            f"ASM 指令数 vs RTL branch_coverage Pearson r = [bold]{r_val:.4f}[/bold] "
            f"(n={n})"
        )
        lines.append(
            f"ASM 指令数 vs RTL branch_coverage Pearson r = {r_val:.4f} (n={n})\n"
        )

    return lines


def print_quality_check(
    rtl_stats: list[RTLStats],
    asm_stats: list[ASMStats],
    sim_dir: str,
) -> list[str]:
    """数据质量检查"""
    lines: list[str] = []
    console.rule("[bold]6. 数据质量检查[/bold]")
    lines.append("## 6. 数据质量检查\n")

    all_tc = set(discover_test_cases(sim_dir))
    rtl_tc = {r.test_case for r in rtl_stats if not r.error}
    asm_tc = {a.test_case for a in asm_stats if not a.error}

    missing_rtl = all_tc - rtl_tc
    missing_asm = all_tc - asm_tc

    # 读取错误
    rtl_errors = [r for r in rtl_stats if r.error]
    asm_errors = [a for a in asm_stats if a.error]

    table = Table(title="数据质量检查")
    table.add_column("检查项", style="cyan")
    table.add_column("数量", style="green")
    table.add_column("说明", style="yellow")

    checks = [
        ("缺失 annotated JSON", len(missing_rtl), "无 annotated/ 目录或无 JSON"),
        ("缺失 ASM JSON", len(missing_asm), "无 {test_case}.json"),
        ("RTL 读取错误", len(rtl_errors), "JSON 解析失败"),
        ("ASM 读取错误", len(asm_errors), "JSON 解析失败"),
    ]

    # branch_coverage 异常
    valid_rtl = [r for r in rtl_stats if not r.error and r.branch_coverage >= 0]
    cov_0 = [r for r in valid_rtl if r.branch_coverage == 0]
    cov_100 = [r for r in valid_rtl if r.branch_coverage == 100]
    checks.append(("branch_coverage = 0", len(cov_0), "完全未覆盖"))
    checks.append(("branch_coverage = 100", len(cov_100), "完全覆盖"))

    # 全部边 -1 的样本
    all_minus1 = [
        r
        for r in rtl_stats
        if not r.error and r.label_0 == 0 and r.label_1 == 0 and r.num_edges > 0
    ]
    checks.append(("全部边标签为 -1", len(all_minus1), "无有效覆盖标注"))

    lines.append("| 检查项 | 数量 | 说明 |")
    lines.append("|--------|------|------|")
    for name, cnt, desc in checks:
        table.add_row(name, str(cnt), desc)
        lines.append(f"| {name} | {cnt} | {desc} |")
    console.print(table)
    lines.append("")

    # 如果有缺失，列出前几个示例
    if missing_rtl:
        examples = sorted(missing_rtl)[:5]
        console.print(
            f"缺失 RTL 示例: {', '.join(examples)}{'...' if len(missing_rtl) > 5 else ''}"
        )
    if missing_asm:
        examples = sorted(missing_asm)[:5]
        console.print(
            f"缺失 ASM 示例: {', '.join(examples)}{'...' if len(missing_asm) > 5 else ''}"
        )

    return lines


# ---------------------------------------------------------------------------
# 逐模块详细分析
# ---------------------------------------------------------------------------


def print_per_module_detail(rtl_stats: list[RTLStats]) -> list[str]:
    """针对每个模块单独进行详细数据分析"""
    lines: list[str] = []
    console.rule("[bold]7. 逐模块详细分析[/bold]")
    lines.append("## 7. 逐模块详细分析\n")

    # 按模块分组
    by_module: dict[str, list[RTLStats]] = defaultdict(list)
    for r in rtl_stats:
        if not r.error:
            by_module[r.module_name].append(r)

    for mod in sorted(by_module):
        samples = by_module[mod]
        representative = samples[0]  # 图拓扑固定，取首个样本
        n_samples = len(samples)

        console.rule(f"[bold cyan]{mod}[/bold cyan]", style="cyan")
        lines.append(f"### {mod}\n")

        # ── 7.1 结构概览 ──
        density = (
            representative.num_edges / (representative.num_nodes**2)
            if representative.num_nodes > 1
            else 0
        )
        mux_count = representative.node_type_counts.get("MUX", 0)
        mux_ratio = (
            mux_count / representative.num_nodes * 100
            if representative.num_nodes > 0
            else 0
        )
        # 平均度
        avg_degree = (
            2 * representative.num_edges / representative.num_nodes
            if representative.num_nodes > 0
            else 0
        )

        overview_table = Table(title=f"{mod} — 结构概览")
        overview_table.add_column("指标", style="cyan")
        overview_table.add_column("值", style="green")
        struct_items = [
            ("样本数", str(n_samples)),
            ("|V| (节点数)", str(representative.num_nodes)),
            ("|E| (边数)", str(representative.num_edges)),
            ("图密度 (E/N²)", f"{density:.4f}"),
            ("平均度 (2E/N)", f"{avg_degree:.2f}"),
            ("MUX 节点数", str(mux_count)),
            ("MUX 占比", f"{mux_ratio:.1f}%"),
        ]
        lines.append("#### 结构概览\n")
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        for label, val in struct_items:
            overview_table.add_row(label, val)
            lines.append(f"| {label} | {val} |")
        console.print(overview_table)
        lines.append("")

        # ── 7.2 节点 type 分布（单样本，因拓扑固定） ──
        total_nodes = representative.num_nodes
        nt_table = Table(title="节点 type 分布")
        nt_table.add_column("类型", style="cyan")
        nt_table.add_column("数量", style="green")
        nt_table.add_column("占比", style="yellow")
        lines.append("#### 节点 type 分布\n")
        lines.append("| 类型 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for nt, cnt in sorted(
            representative.node_type_counts.items(), key=lambda x: -x[1]
        ):
            pct = f"{cnt / total_nodes * 100:.1f}%" if total_nodes else "0%"
            nt_table.add_row(nt, str(cnt), pct)
            lines.append(f"| {nt} | {cnt} | {pct} |")
        console.print(nt_table)
        lines.append("")

        # ── 7.3 cell_type 分布 ──
        ct_counter = Counter(representative.cell_type_counts)
        ct_table = Table(title="cell_type 分布")
        ct_table.add_column("cell_type", style="cyan")
        ct_table.add_column("数量", style="green")
        ct_table.add_column("占比", style="yellow")
        lines.append("#### cell_type 分布\n")
        lines.append("| cell_type | 数量 | 占比 |")
        lines.append("|-----------|------|------|")
        for ct, cnt in ct_counter.most_common():
            pct = f"{cnt / total_nodes * 100:.1f}%" if total_nodes else "0%"
            ct_table.add_row(ct, str(cnt), pct)
            lines.append(f"| {ct} | {cnt} | {pct} |")
        console.print(ct_table)
        lines.append("")

        # ── 7.4 边 type 分布 ──
        total_edges = representative.num_edges
        et_table = Table(title="边 type 分布")
        et_table.add_column("类型", style="cyan")
        et_table.add_column("数量", style="green")
        et_table.add_column("占比", style="yellow")
        lines.append("#### 边 type 分布\n")
        lines.append("| 类型 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for et, cnt in sorted(
            representative.edge_type_counts.items(), key=lambda x: -x[1]
        ):
            pct = f"{cnt / total_edges * 100:.1f}%" if total_edges else "0%"
            et_table.add_row(et, str(cnt), pct)
            lines.append(f"| {et} | {cnt} | {pct} |")
        console.print(et_table)
        lines.append("")

        # ── 7.5 节点 width 分布 ──
        nw_table = Table(title="节点 width 分布")
        nw_table.add_column("位宽", style="cyan")
        nw_table.add_column("数量", style="green")
        nw_table.add_column("占比", style="yellow")
        lines.append("#### 节点 width 分布\n")
        lines.append("| 位宽 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        nw_counter = Counter(representative.node_width_counts)
        for w, cnt in nw_counter.most_common():
            pct = f"{cnt / total_nodes * 100:.1f}%" if total_nodes else "0%"
            nw_table.add_row(str(w), str(cnt), pct)
            lines.append(f"| {w} | {cnt} | {pct} |")
        console.print(nw_table)
        lines.append("")

        # ── 7.6 边 width 分布 ──
        ew_table = Table(title="边 width 分布")
        ew_table.add_column("位宽", style="cyan")
        ew_table.add_column("数量", style="green")
        ew_table.add_column("占比", style="yellow")
        lines.append("#### 边 width 分布\n")
        lines.append("| 位宽 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        ew_counter = Counter(representative.edge_width_counts)
        for w, cnt in ew_counter.most_common():
            pct = f"{cnt / total_edges * 100:.1f}%" if total_edges else "0%"
            ew_table.add_row(str(w), str(cnt), pct)
            lines.append(f"| {w} | {cnt} | {pct} |")
        console.print(ew_table)
        lines.append("")

        # ── 7.7 覆盖率标签分布（跨所有测试用例聚合） ──
        total_m1 = sum(s.label_minus1 for s in samples)
        total_0 = sum(s.label_0 for s in samples)
        total_1 = sum(s.label_1 for s in samples)
        total_all = total_m1 + total_0 + total_1
        has_labels = total_0 + total_1 > 0

        label_table = Table(title="覆盖率标签分布（聚合）")
        label_table.add_column("标签", style="cyan")
        label_table.add_column("总数", style="green")
        label_table.add_column("占比", style="yellow")
        label_table.add_column("每样本均值", style="dim")
        lines.append("#### 覆盖率标签分布\n")
        lines.append("| 标签 | 总数 | 占比 | 每样本均值 |")
        lines.append("|------|------|------|------------|")
        for lbl_name, lbl_total in [
            ("-1 (未标注)", total_m1),
            ("0 (未覆盖)", total_0),
            ("1 (已覆盖)", total_1),
        ]:
            pct = f"{lbl_total / total_all * 100:.1f}%" if total_all else "0%"
            avg = f"{lbl_total / n_samples:.1f}"
            label_table.add_row(lbl_name, str(lbl_total), pct, avg)
            lines.append(f"| {lbl_name} | {lbl_total} | {pct} | {avg} |")
        console.print(label_table)
        lines.append("")

        if has_labels:
            # 有效标签比值
            ratio_str = f"{total_1 / total_0:.2f}" if total_0 > 0 else "inf"
            eff_rate = (total_0 + total_1) / total_all * 100 if total_all else 0
            console.print(f"  有效标签率: {eff_rate:.1f}% | 正负比(1:0): {ratio_str}")
            lines.append(f"有效标签率: {eff_rate:.1f}% | 正负比(1:0): {ratio_str}\n")

            # 单样本级标签变化统计
            per_sample_l0 = [s.label_0 for s in samples]
            per_sample_l1 = [s.label_1 for s in samples]
            d_l0 = _describe(per_sample_l0)
            d_l1 = _describe(per_sample_l1)

            var_table = Table(title="单样本标签变化")
            var_table.add_column("标签", style="cyan")
            for col in ["min", "max", "mean", "median", "std"]:
                var_table.add_column(col, style="green")
            lines.append("#### 单样本标签变化\n")
            lines.append("| 标签 | min | max | mean | median | std |")
            lines.append("|------|-----|-----|------|--------|-----|")
            for lbl, d in [("label=0", d_l0), ("label=1", d_l1)]:
                var_table.add_row(
                    lbl,
                    *[str(d[k]) for k in ["min", "max", "mean", "median", "std"]],
                )
                lines.append(
                    f"| {lbl} | {d['min']} | {d['max']} | {d['mean']} "
                    f"| {d['median']} | {d['std']} |"
                )
            console.print(var_table)
            lines.append("")
        else:
            console.print("  [dim]该模块无边级覆盖标签（全部为 -1）[/dim]")
            lines.append("该模块无边级覆盖标签（全部为 -1）\n")

        # ── 7.8 按边类型的标签分布 ──
        # 聚合所有样本的 edge_type_label_counts
        agg_et_label: dict[str, dict[int, int]] = {}
        for s in samples:
            for et, lbl_map in s.edge_type_label_counts.items():
                if et not in agg_et_label:
                    agg_et_label[et] = {-1: 0, 0: 0, 1: 0}
                for lbl, cnt in lbl_map.items():
                    agg_et_label[et][lbl] = agg_et_label[et].get(lbl, 0) + cnt

        if agg_et_label:
            etl_table = Table(title="按边类型的标签分布（聚合）")
            etl_table.add_column("边类型", style="cyan")
            etl_table.add_column("-1", style="dim")
            etl_table.add_column("0", style="red")
            etl_table.add_column("1", style="green")
            etl_table.add_column("有效标签率", style="yellow")
            lines.append("#### 按边类型的标签分布\n")
            lines.append("| 边类型 | -1 | 0 | 1 | 有效标签率 |")
            lines.append("|--------|-----|-----|-----|------------|")
            for et in sorted(agg_et_label):
                m1 = agg_et_label[et].get(-1, 0)
                z = agg_et_label[et].get(0, 0)
                o = agg_et_label[et].get(1, 0)
                t = m1 + z + o
                eff = f"{(z + o) / t * 100:.1f}%" if t > 0 else "0%"
                etl_table.add_row(et, str(m1), str(z), str(o), eff)
                lines.append(f"| {et} | {m1} | {z} | {o} | {eff} |")
            console.print(etl_table)
            lines.append("")

        # ── 7.9 branch_coverage 分布 ──
        covs = [s.branch_coverage for s in samples if s.branch_coverage >= 0]
        if covs:
            d = _describe(covs)
            p = _percentiles(covs)
            cov_table = Table(title="branch_coverage 分布")
            cov_table.add_column("指标", style="cyan")
            cov_table.add_column("值", style="green")
            cov_items = [
                ("mean", str(d["mean"])),
                ("std", str(d["std"])),
                ("min", str(d["min"])),
                ("p5", str(p.get("p5", ""))),
                ("p25", str(p.get("p25", ""))),
                ("median", str(d["median"])),
                ("p75", str(p.get("p75", ""))),
                ("p95", str(p.get("p95", ""))),
                ("max", str(d["max"])),
            ]
            lines.append("#### branch_coverage 分布\n")
            lines.append("| 指标 | 值 |")
            lines.append("|------|------|")
            for label, val in cov_items:
                cov_table.add_row(label, val)
                lines.append(f"| {label} | {val} |")
            console.print(cov_table)
            lines.append("")

            # 覆盖率区间分布
            if d["std"] > 0:
                bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100.01)]
                bin_table = Table(title="覆盖率区间分布")
                bin_table.add_column("区间", style="cyan")
                bin_table.add_column("数量", style="green")
                bin_table.add_column("占比", style="yellow")
                lines.append("#### 覆盖率区间分布\n")
                lines.append("| 区间 | 数量 | 占比 |")
                lines.append("|------|------|------|")
                for lo, hi in bins:
                    cnt = sum(1 for c in covs if lo <= c < hi)
                    pct = f"{cnt / len(covs) * 100:.1f}%"
                    label = f"[{lo:.0f}, {hi:.0f})" if hi <= 100 else f"[{lo:.0f}, 100]"
                    bin_table.add_row(label, str(cnt), pct)
                    lines.append(f"| {label} | {cnt} | {pct} |")
                console.print(bin_table)
                lines.append("")

        console.print()

    return lines


# ---------------------------------------------------------------------------
# 覆盖率 vs ASM 指令数（逐模块）
# ---------------------------------------------------------------------------


import re


def _extract_instr_count_from_name(test_case: str) -> int | None:
    """从测试用例名称中提取指令数量。

    命名模式: {test_type}_{instr_count}_{seed_index}
    例如 arithmetic_base_200_0000 -> 200
         debug_single_step_400_0008 -> 400
    """
    # 匹配倒数第二个 _ 分隔的数字段
    m = re.match(r"^(.+)_(\d+)_(\d+)$", test_case)
    if m:
        return int(m.group(2))
    return None


def _pearson(xs: list[float], ys: list[float]) -> float:
    """计算 Pearson 相关系数，样本不足时返回 0"""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    return cov_xy / (sx * sy) if sx > 0 and sy > 0 else 0.0


def print_coverage_vs_asm_per_module(
    rtl_stats: list[RTLStats],
    asm_stats: list[ASMStats],
    max_instructions: int | None = None,
) -> list[str]:
    """逐模块分析覆盖率与 ASM 指令数量（从测试名称提取）的关系"""
    lines: list[str] = []
    filter_label = f"（指令数 <= {max_instructions}）" if max_instructions else ""
    console.rule(f"[bold]8. 覆盖率 vs 指令数（逐模块）{filter_label}[/bold]")
    lines.append(f"## 8. 覆盖率 vs 指令数（逐模块）{filter_label}\n")

    # 按模块分组，构建 (instr_count_from_name, branch_coverage) 配对
    by_module: dict[str, list[tuple[float, float]]] = defaultdict(list)
    skipped = 0
    for r in rtl_stats:
        if r.error or r.branch_coverage < 0:
            continue
        instr_count = _extract_instr_count_from_name(r.test_case)
        if instr_count is None:
            skipped += 1
            continue
        if max_instructions and instr_count > max_instructions:
            continue
        by_module[r.module_name].append(
            (float(instr_count), r.branch_coverage)
        )

    if skipped:
        console.print(f"[dim]跳过 {skipped} 个无法从名称提取指令数的样本[/dim]")

    if not by_module:
        console.print("[yellow]无有效配对数据[/yellow]")
        lines.append("无有效配对数据\n")
        return lines

    # 汇总表格
    summary_table = Table(title="覆盖率 vs 指令数（逐模块相关性）")
    summary_table.add_column("模块", style="cyan")
    summary_table.add_column("样本数", style="green")
    summary_table.add_column("Pearson r", style="yellow")
    summary_table.add_column("指令数 mean", style="green")
    summary_table.add_column("指令数 std", style="green")
    summary_table.add_column("Coverage mean", style="green")
    summary_table.add_column("Coverage std", style="green")

    lines.append(
        "| 模块 | 样本数 | Pearson r | 指令数 mean | 指令数 std "
        "| Coverage mean | Coverage std |"
    )
    lines.append(
        "|------|--------|-----------|-------------|------------|"
        "---------------|--------------|"
    )

    for mod in sorted(by_module):
        pairs = by_module[mod]
        n = len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r_val = _pearson(xs, ys)
        d_x = _describe(xs)
        d_y = _describe(ys)

        r_str = f"{r_val:.4f}" if n >= 3 else "N/A"
        summary_table.add_row(
            mod,
            str(n),
            r_str,
            str(d_x["mean"]),
            str(d_x["std"]),
            str(d_y["mean"]),
            str(d_y["std"]),
        )
        lines.append(
            f"| {mod} | {n} | {r_str} | {d_x['mean']} | {d_x['std']} "
            f"| {d_y['mean']} | {d_y['std']} |"
        )

    console.print(summary_table)
    lines.append("")

    # 逐模块详细：按指令数分桶统计覆盖率
    for mod in sorted(by_module):
        pairs = by_module[mod]
        if len(pairs) < 10:
            continue

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        # 等频分桶（四分位）
        sorted_pairs = sorted(pairs, key=lambda p: p[0])
        n = len(sorted_pairs)
        quartile_size = n // 4
        if quartile_size < 2:
            continue

        console.print(f"\n[bold cyan]{mod}[/bold cyan] — 指令数分桶覆盖率统计")
        lines.append(f"### {mod} — 指令数分桶覆盖率统计\n")

        bucket_table = Table(title=f"{mod} 分桶统计")
        bucket_table.add_column("分桶", style="cyan")
        bucket_table.add_column("样本数", style="green")
        bucket_table.add_column("指令数范围", style="green")
        bucket_table.add_column("Coverage mean", style="yellow")
        bucket_table.add_column("Coverage median", style="yellow")
        bucket_table.add_column("Coverage std", style="dim")

        lines.append(
            "| 分桶 | 样本数 | ASM指令范围 | Coverage mean | Coverage median | Coverage std |"
        )
        lines.append(
            "|------|--------|------------|---------------|-----------------|--------------|"
        )

        quartile_labels = ["Q1 (最少)", "Q2", "Q3", "Q4 (最多)"]
        for qi in range(4):
            start = qi * quartile_size
            end = (qi + 1) * quartile_size if qi < 3 else n
            bucket = sorted_pairs[start:end]
            bx = [p[0] for p in bucket]
            by_ = [p[1] for p in bucket]
            d_by = _describe(by_)
            x_range = f"[{min(bx):.0f}, {max(bx):.0f}]"
            bucket_table.add_row(
                quartile_labels[qi],
                str(len(bucket)),
                x_range,
                str(d_by["mean"]),
                str(d_by["median"]),
                str(d_by["std"]),
            )
            lines.append(
                f"| {quartile_labels[qi]} | {len(bucket)} | {x_range} "
                f"| {d_by['mean']} | {d_by['median']} | {d_by['std']} |"
            )

        console.print(bucket_table)
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_distributions(
    rtl_stats: list[RTLStats],
    asm_stats: list[ASMStats],
    output_dir: str,
    max_instructions: int | None = None,
) -> list[str]:
    """生成 matplotlib 图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[red]matplotlib 未安装，跳过可视化。运行: uv add matplotlib[/red]"
        )
        return []

    lines: list[str] = []
    os.makedirs(output_dir, exist_ok=True)

    valid_rtl = [r for r in rtl_stats if not r.error]

    # 1. 覆盖率分布直方图
    covs = [r.branch_coverage for r in valid_rtl if r.branch_coverage >= 0]
    if covs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(covs, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.set_xlabel("Branch Coverage (%)")
        ax.set_ylabel("Count")
        ax.set_title("Branch Coverage Distribution")
        ax.axvline(
            sum(covs) / len(covs),
            color="red",
            linestyle="--",
            label=f"Mean={sum(covs) / len(covs):.1f}%",
        )
        ax.legend()
        path = os.path.join(output_dir, "coverage_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  已保存: {path}")
        lines.append(f"![Coverage Distribution]({path})\n")

    # 2. 边标签饼图
    total_m1 = sum(r.label_minus1 for r in valid_rtl)
    total_0 = sum(r.label_0 for r in valid_rtl)
    total_1 = sum(r.label_1 for r in valid_rtl)
    if total_m1 + total_0 + total_1 > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 全部标签
        labels_all = ["-1 (未标注)", "0 (未覆盖)", "1 (已覆盖)"]
        sizes_all = [total_m1, total_0, total_1]
        colors_all = ["#cccccc", "#e74c3c", "#2ecc71"]
        axes[0].pie(
            sizes_all,
            labels=labels_all,
            colors=colors_all,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("All Edge Labels")

        # 仅有效标签
        if total_0 + total_1 > 0:
            labels_eff = ["0 (未覆盖)", "1 (已覆盖)"]
            sizes_eff = [total_0, total_1]
            colors_eff = ["#e74c3c", "#2ecc71"]
            axes[1].pie(
                sizes_eff,
                labels=labels_eff,
                colors=colors_eff,
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[1].set_title("Effective Labels Only (0 & 1)")

        path = os.path.join(output_dir, "label_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  已保存: {path}")
        lines.append(f"![Label Distribution]({path})\n")

    # 3. 节点/边 type 条形图
    global_node_type: Counter[str] = Counter()
    global_edge_type: Counter[str] = Counter()
    for r in valid_rtl:
        for k, v in r.node_type_counts.items():
            global_node_type[k] += v
        for k, v in r.edge_type_counts.items():
            global_edge_type[k] += v

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 节点 type
    nt_items = global_node_type.most_common()
    if nt_items:
        names, counts = zip(*nt_items)
        axes[0].barh(list(reversed(names)), list(reversed(counts)), color="#4C72B0")
        axes[0].set_title("Node Type Distribution")
        axes[0].set_xlabel("Count")

    # 边 type
    et_items = global_edge_type.most_common()
    if et_items:
        names, counts = zip(*et_items)
        axes[1].barh(list(reversed(names)), list(reversed(counts)), color="#55A868")
        axes[1].set_title("Edge Type Distribution")
        axes[1].set_xlabel("Count")

    path = os.path.join(output_dir, "type_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  已保存: {path}")
    lines.append(f"![Type Distribution]({path})\n")

    # 4. 节点 width 分布
    global_node_width: Counter[int] = Counter()
    for r in valid_rtl:
        for k, v in r.node_width_counts.items():
            global_node_width[k] += v

    if global_node_width:
        fig, ax = plt.subplots(figsize=(10, 5))
        widths_sorted = sorted(global_node_width.items())
        ws, cs = zip(*widths_sorted)
        ax.bar([str(w) for w in ws[:30]], cs[:30], color="#C44E52")
        ax.set_xlabel("Width (bits)")
        ax.set_ylabel("Count")
        ax.set_title("Node Width Distribution (Top 30)")
        plt.xticks(rotation=45, ha="right")
        path = os.path.join(output_dir, "width_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  已保存: {path}")
        lines.append(f"![Width Distribution]({path})\n")

    # 5. 按模块的覆盖率箱线图
    by_module: dict[str, list[float]] = defaultdict(list)
    for r in valid_rtl:
        if r.branch_coverage >= 0:
            by_module[r.module_name].append(r.branch_coverage)

    if by_module:
        fig, ax = plt.subplots(figsize=(12, 6))
        mod_names = sorted(by_module.keys())
        data = [by_module[m] for m in mod_names]
        ax.boxplot(data, tick_labels=[m.replace("cv32e40p_", "") for m in mod_names])
        ax.set_ylabel("Branch Coverage (%)")
        ax.set_title("Coverage by Module")
        plt.xticks(rotation=30, ha="right")
        path = os.path.join(output_dir, "coverage_by_module.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  已保存: {path}")
        lines.append(f"![Coverage by Module]({path})\n")

    # 6. 覆盖率 vs 指令数散点图（逐模块，指令数从测试名称提取）
    mod_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in valid_rtl:
        if r.branch_coverage < 0:
            continue
        instr_count = _extract_instr_count_from_name(r.test_case)
        if instr_count is None:
            continue
        if max_instructions and instr_count > max_instructions:
            continue
        mod_pairs[r.module_name].append(
            (float(instr_count), r.branch_coverage)
        )

    if mod_pairs:
        # 筛选有足够样本的模块
        plot_mods = {m: ps for m, ps in mod_pairs.items() if len(ps) >= 10}
        if plot_mods:
            n_mods = len(plot_mods)
            cols = min(n_mods, 3)
            rows = math.ceil(n_mods / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False
            )

            for idx, mod in enumerate(sorted(plot_mods)):
                ax = axes[idx // cols][idx % cols]
                pairs = plot_mods[mod]
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                ax.scatter(xs, ys, alpha=0.3, s=8, color="#4C72B0")
                ax.set_xlabel("Instructions (from test name)")
                ax.set_ylabel("Branch Coverage (%)")
                r_val = _pearson(xs, ys)
                short_name = mod.replace("cv32e40p_", "")
                ax.set_title(f"{short_name} (r={r_val:.3f}, n={len(pairs)})")

            # 隐藏多余子图
            for idx in range(n_mods, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)

            fig.suptitle("Coverage vs Instructions (per Module)", fontsize=14)
            fig.tight_layout()
            path = os.path.join(output_dir, "coverage_vs_asm_per_module.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            console.print(f"  已保存: {path}")
            lines.append(f"![Coverage vs ASM per Module]({path})\n")

    return lines


# ---------------------------------------------------------------------------
# JSON 导出
# ---------------------------------------------------------------------------


def export_json(
    rtl_stats: list[RTLStats],
    asm_stats: list[ASMStats],
    output_path: str,
) -> None:
    """导出统计摘要为 JSON"""
    valid_rtl = [r for r in rtl_stats if not r.error]
    valid_asm = [a for a in asm_stats if not a.error]

    # 按模块聚合
    by_module: dict[str, dict[str, Any]] = {}
    for r in valid_rtl:
        mod = r.module_name
        if mod not in by_module:
            by_module[mod] = {
                "node_counts": [],
                "edge_counts": [],
                "branch_coverages": [],
                "label_minus1": 0,
                "label_0": 0,
                "label_1": 0,
            }
        by_module[mod]["node_counts"].append(r.num_nodes)
        by_module[mod]["edge_counts"].append(r.num_edges)
        if r.branch_coverage >= 0:
            by_module[mod]["branch_coverages"].append(r.branch_coverage)
        by_module[mod]["label_minus1"] += r.label_minus1
        by_module[mod]["label_0"] += r.label_0
        by_module[mod]["label_1"] += r.label_1

    summary: dict[str, Any] = {
        "total_rtl_samples": len(valid_rtl),
        "total_asm_samples": len(valid_asm),
        "modules": {},
    }

    for mod, d in sorted(by_module.items()):
        summary["modules"][mod] = {
            "samples": len(d["node_counts"]),
            "nodes": _describe(d["node_counts"]),
            "edges": _describe(d["edge_counts"]),
            "branch_coverage": _describe(d["branch_coverages"]),
            "label_distribution": {
                "-1": d["label_minus1"],
                "0": d["label_0"],
                "1": d["label_1"],
            },
        }

    # ASM 汇总
    if valid_asm:
        summary["asm"] = {
            "samples": len(valid_asm),
            "nodes": _describe([a.num_nodes for a in valid_asm]),
            "edges": _describe([a.num_edges for a in valid_asm]),
            "instructions": _describe([a.total_instructions for a in valid_asm]),
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    console.print(f"JSON 摘要已保存: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="数据集统计分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sim-dir",
        required=True,
        help="simulation_results 目录路径",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="随机采样 N 个测试用例（默认全量）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行 worker 数（默认自动检测）",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成 matplotlib 可视化图表",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="图表输出目录（默认 analysis_output）",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="输出 JSON 统计摘要",
    )
    parser.add_argument(
        "--report",
        default="analysis_report.md",
        metavar="PATH",
        help="Markdown 报告输出路径（默认 analysis_report.md）",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不生成 Markdown 报告",
    )
    parser.add_argument(
        "--per-module",
        action="store_true",
        help="启用逐模块详细分析",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=None,
        metavar="N",
        help="覆盖率 vs ASM 分析仅包含指令数 <= N 的测试用例",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_dir = args.sim_dir

    if not os.path.isdir(sim_dir):
        console.print(f"[red]目录不存在: {sim_dir}[/red]")
        sys.exit(1)

    total_tc = len(discover_test_cases(sim_dir))
    console.print(f"[bold]数据集统计分析[/bold] - {sim_dir}")
    console.print()

    # 收集统计
    rtl_stats, asm_stats = collect_all_stats(
        sim_dir, sample_n=args.sample, workers=args.workers
    )
    console.print(f"收集完成: RTL={len(rtl_stats)} 条, ASM={len(asm_stats)} 条\n")

    # 报告行
    report_lines: list[str] = ["# 数据集统计分析报告\n"]

    # 各统计模块
    report_lines.extend(print_overview(rtl_stats, asm_stats, total_tc))
    report_lines.extend(print_rtl_stats(rtl_stats))
    report_lines.extend(print_asm_stats(asm_stats))
    report_lines.extend(print_label_stats(rtl_stats))
    report_lines.extend(print_structure_stats(rtl_stats, asm_stats))
    report_lines.extend(print_quality_check(rtl_stats, asm_stats, sim_dir))

    # 逐模块详细分析
    if args.per_module:
        report_lines.extend(print_per_module_detail(rtl_stats))

    # 覆盖率 vs ASM 指令数（逐模块）
    report_lines.extend(
        print_coverage_vs_asm_per_module(rtl_stats, asm_stats, args.max_instructions)
    )

    # 可选可视化
    if args.plot:
        console.rule("[bold]可视化[/bold]")
        plot_lines = plot_distributions(
            rtl_stats, asm_stats, args.output_dir, args.max_instructions
        )
        report_lines.extend(["\n## 可视化\n"] + plot_lines)

    # JSON 导出
    if args.json:
        export_json(rtl_stats, asm_stats, args.json)

    # Markdown 报告
    if not args.no_report:
        report_path = args.report
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        console.print(f"\nMarkdown 报告已保存: [bold]{report_path}[/bold]")


if __name__ == "__main__":
    main()
