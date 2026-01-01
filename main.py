#!/usr/bin/env python3
"""
RTLMap - RTL Coverage Annotation Tool
将覆盖率数据标注到 CDFG 上，用于神经网络训练
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

from cdfg import CDFGExtractor, CDFGExporter, NodeType
from annotation import CoverageParser, annotate_cdfg_with_coverage, AnnotationStats


def find_coverage_files(design_dir: Path, coverage_dir: Path, cdfg) -> list:
    """
    根据 module2html.json 映射和 CDFG 中的源文件自动查找覆盖率文件

    Args:
        design_dir: 设计目录（包含 module2html.json）
        coverage_dir: 覆盖率报告目录
        cdfg: CDFG 对象

    Returns:
        覆盖率文件路径列表
    """
    # 读取 module2html.json
    mapping_file = design_dir / "module2html.json"
    if not mapping_file.exists():
        print(f"警告: 未找到映射文件 {mapping_file}")
        return []

    with open(mapping_file, 'r', encoding='utf-8') as f:
        module2html = json.load(f)

    # 收集 CDFG 中所有源文件对应的模块名
    source_files = set()
    for node in cdfg.nodes.values():
        if node.source_file:
            # 从文件名推断模块名（去掉 .sv 后缀）
            module_name = node.source_file.replace('.sv', '').replace('.v', '')
            source_files.add(module_name)

    print(f"\nCDFG 中包含 {len(source_files)} 个源文件的节点")

    # 查找对应的覆盖率文件
    coverage_files = []
    matched_modules = []

    for module_name, html_file in module2html.items():
        # 检查模块名是否匹配 CDFG 中的源文件
        # 支持精确匹配和模糊匹配（模块名可能是层次化的如 cv32e40p_wrapper.core_i.xxx）
        base_module = module_name.split('.')[-1] if '.' in module_name else module_name

        if base_module in source_files or module_name in source_files:
            html_path = coverage_dir / html_file
            if html_path.exists():
                coverage_files.append(str(html_path))
                matched_modules.append(module_name)

    print(f"匹配到 {len(coverage_files)} 个覆盖率报告文件")

    return coverage_files


def main():
    parser = argparse.ArgumentParser(
        description='RTLMap - 将覆盖率数据标注到 CDFG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 指定设计 JSON 和覆盖率目录，自动匹配模块
  uv run main.py designs/cv32e40p/json/cv32e40p_core.json -c designs/cv32e40p/coverage_reports/coverage_report1

  # 仅提取 CDFG（不标注覆盖率）
  uv run main.py design.json --no-coverage

  # 生成 SVG 可视化
  uv run main.py design.json -c coverage_dir --svg

  # 启用覆盖率传播（标注所有可达边）
  uv run main.py design.json -c coverage_dir --propagate
        """
    )

    parser.add_argument('design_json', help='Yosys 生成的 RTLIL JSON 文件')
    parser.add_argument('-c', '--coverage-dir', dest='coverage_dir',
                        help='覆盖率报告目录（自动从 module2html.json 匹配文件）')
    parser.add_argument('-o', '--output', default='cdfg_annotated.json',
                        help='输出 JSON 文件路径 (默认: cdfg_annotated.json)')
    parser.add_argument('-i', '--instance', default=None,
                        help='覆盖率实例标签 (默认: 使用第一个实例)')
    parser.add_argument('--no-coverage', action='store_true',
                        help='仅提取 CDFG，不标注覆盖率')
    parser.add_argument('--propagate', action='store_true',
                        help='启用覆盖率传播，标注所有可达边（非分支边）')
    parser.add_argument('--svg', action='store_true',
                        help='生成 SVG 可视化图')
    parser.add_argument('--svg-output', default='cdfg_output.svg',
                        help='SVG 输出文件路径 (默认: cdfg_output.svg)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')

    args = parser.parse_args()

    # 检查输入文件
    design_json_path = Path(args.design_json)
    if not design_json_path.exists():
        print(f"错误: 设计文件不存在: {args.design_json}")
        sys.exit(1)

    # 提取 CDFG
    print(f"正在从 {args.design_json} 提取 CDFG...")
    extractor = CDFGExtractor(args.design_json)
    cdfg = extractor.extract()

    print(f"模块: {cdfg.module_name}")
    print(f"节点数: {len(cdfg.nodes)}")
    print(f"边数: {len(cdfg.edges)}")

    if args.verbose:
        # 显示节点类型分布
        type_counts = {}
        for node in cdfg.nodes.values():
            type_name = node.node_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        print("\n节点类型分布:")
        for type_name, count in sorted(type_counts.items()):
            print(f"  {type_name}: {count}")

        # 显示源文件分布
        file_counts = defaultdict(int)
        for node in cdfg.nodes.values():
            if node.source_file:
                file_counts[node.source_file] += 1
        if file_counts:
            print("\n源文件分布:")
            for file_name, count in sorted(file_counts.items()):
                print(f"  {file_name}: {count} 个节点")

    # 确定覆盖率文件列表
    coverage_files = []

    if not args.no_coverage:
        if args.coverage_dir:
            # 从目录和 module2html.json 查找覆盖率文件
            coverage_dir = Path(args.coverage_dir)
            if not coverage_dir.exists():
                print(f"错误: 覆盖率报告目录不存在: {args.coverage_dir}")
                sys.exit(1)

            # 设计目录是 JSON 文件的上两级目录（json/../）
            design_dir = design_json_path.parent.parent
            coverage_files = find_coverage_files(design_dir, coverage_dir, cdfg)

    # 标注覆盖率
    total_stats = AnnotationStats()
    total_stats.total_edges = len(cdfg.edges)

    if coverage_files:
        print(f"\n找到 {len(coverage_files)} 个覆盖率报告文件")

        for html_path in coverage_files:
            print(f"\n--- 正在处理: {Path(html_path).name} ---")

            # 解析覆盖率
            try:
                cov_parser = CoverageParser(html_path)
            except Exception as e:
                print(f"  错误: 无法解析文件: {e}")
                continue

            # 获取模块信息
            module_name = cov_parser.get_module_name()
            source_file = cov_parser.get_source_file()

            if module_name:
                print(f"  模块: {module_name}")
            if source_file:
                print(f"  源文件: {source_file}")

            # 获取实例标签
            instance_tag = args.instance
            if instance_tag is None:
                instance_tag = cov_parser.get_first_instance()
                if instance_tag:
                    print(f"  实例: {instance_tag}")

            if args.verbose:
                instances = cov_parser.get_available_instances()
                if instances:
                    print(f"  可用实例: {instances}")

            # 执行标注（不传播，最后统一传播）
            stats = annotate_cdfg_with_coverage(cdfg, html_path, instance_tag,
                                                propagate=False)

            # 累积统计
            total_stats.total_mux_nodes += stats.total_mux_nodes
            total_stats.annotated_mux_nodes += stats.annotated_mux_nodes
            total_stats.annotated_edges += stats.annotated_edges
            total_stats.covered_edges += stats.covered_edges
            total_stats.uncovered_edges += stats.uncovered_edges
            total_stats.control_edges += stats.control_edges
            total_stats.data_true_edges += stats.data_true_edges
            total_stats.data_false_edges += stats.data_false_edges

            print(f"  标注: {stats.annotated_mux_nodes} MUX, {stats.annotated_edges} 边")

        # 如果需要传播，在所有标注完成后统一传播
        if args.propagate:
            print("\n正在传播覆盖率...")
            from annotation.annotator import CoverageAnnotator
            annotator = CoverageAnnotator(cdfg, {})  # 空覆盖数据，只做传播
            annotator.propagate_coverage()

        # 从 CDFG 边的实际状态统计（而不是累加各文件的操作次数）
        final_stats = AnnotationStats()
        final_stats.total_edges = len(cdfg.edges)
        final_stats.total_mux_nodes = total_stats.total_mux_nodes
        final_stats.annotated_mux_nodes = total_stats.annotated_mux_nodes

        for edge in cdfg.edges:
            if edge.coverage_label >= 0:
                final_stats.annotated_edges += 1
                if edge.coverage_label == 1:
                    final_stats.covered_edges += 1
                else:
                    final_stats.uncovered_edges += 1

                # 按类型统计
                if edge.coverage_type == "control":
                    final_stats.control_edges += 1
                elif edge.coverage_type == "data_true":
                    final_stats.data_true_edges += 1
                elif edge.coverage_type == "data_false":
                    final_stats.data_false_edges += 1
                elif edge.coverage_type == "always":
                    final_stats.always_executed_edges += 1
                elif edge.coverage_type == "propagated":
                    final_stats.propagated_edges += 1

        # 显示总统计
        print("\n" + "=" * 50)
        print("总体标注统计:")
        print(f"  处理文件数: {len(coverage_files)}")
        print(f"  MUX 节点: {final_stats.annotated_mux_nodes}/{final_stats.total_mux_nodes}")
        print(f"  已标注边: {final_stats.annotated_edges}/{final_stats.total_edges}")
        print(f"  已覆盖边: {final_stats.covered_edges}")
        print(f"  未覆盖边: {final_stats.uncovered_edges}")

        if args.verbose:
            print(f"  控制边: {final_stats.control_edges}")
            print(f"  data_true 边: {final_stats.data_true_edges}")
            print(f"  data_false 边: {final_stats.data_false_edges}")

            if args.propagate:
                print(f"  必然执行边: {final_stats.always_executed_edges}")
                print(f"  传播标注边: {final_stats.propagated_edges}")

        # 计算覆盖率
        if final_stats.annotated_edges > 0:
            coverage_rate = final_stats.covered_edges / final_stats.annotated_edges * 100
            print(f"\n边覆盖率: {coverage_rate:.2f}%")

        total_stats = final_stats  # 用于后续 SVG 生成判断

    # 导出 JSON
    exporter = CDFGExporter(cdfg)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(exporter.to_json())
    print(f"\n已导出 CDFG: {args.output}")

    # 生成 SVG
    if args.svg:
        try:
            # 如果有覆盖率数据，启用覆盖率显示
            show_coverage = total_stats.annotated_edges > 0
            exporter.to_graphviz(args.svg_output, show_coverage=show_coverage)
            print(f"已导出 SVG: {args.svg_output}")
        except Exception as e:
            print(f"警告: 无法生成 SVG: {e}")

    # 显示示例标注
    if args.verbose and total_stats.annotated_edges > 0:
        print("\n标注示例（前 5 条已标注边）:")
        count = 0
        for edge in cdfg.edges:
            if edge.coverage_label >= 0:
                status = "已覆盖" if edge.coverage_label == 1 else "未覆盖"
                print(f"  {edge.source} -> {edge.target} [{edge.target_port}]")
                print(f"    类型: {edge.coverage_type}, 分支: {edge.branch_index}, 状态: {status}")
                count += 1
                if count >= 5:
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
