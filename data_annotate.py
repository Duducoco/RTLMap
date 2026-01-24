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
from dataclasses import dataclass, field
from typing import Optional

from cdfg_rtl import CDFGExtractor, CDFGExporter, CDFG
from annotation import CoverageParser, annotate_cdfg_with_coverage, AnnotationStats
from tools import YosysRunner


@dataclass
class DataAnnotatorConfig:
    """DataAnnotator 配置类"""

    design_json: str
    coverage_dir: Optional[str] = None
    output: str = "cdfg_annotated.json"
    instance: Optional[str] = None
    no_coverage: bool = False
    propagate: bool = True
    svg: bool = False
    svg_output: str = "cdfg_output.svg"
    verbose: bool = False


class DataAnnotator:
    """
    RTLMap 数据标注器

    将覆盖率数据标注到 CDFG 上，用于神经网络训练。
    支持从 VCS/URG 覆盖率报告解析分支覆盖率并映射到 CDFG 边。
    """

    def __init__(self, config: DataAnnotatorConfig):
        """
        初始化 DataAnnotator

        Args:
            config: DataAnnotatorConfig 配置对象
        """
        self.config = config
        self.cdfg: Optional[CDFG] = None
        self.coverage_files: list = []
        self.stats: AnnotationStats = AnnotationStats()

    @classmethod
    def from_args(
        cls,
        design_json: str,
        coverage_dir: Optional[str] = None,
        output: str = "cdfg_annotated.json",
        instance: Optional[str] = None,
        no_coverage: bool = False,
        propagate: bool = False,
        svg: bool = False,
        svg_output: str = "cdfg_output.svg",
        verbose: bool = False,
    ) -> "DataAnnotator":
        """
        从参数创建 DataAnnotator 实例

        Args:
            design_json: Yosys 生成的 RTLIL JSON 文件路径
            coverage_dir: 覆盖率报告目录
            output: 输出 JSON 文件路径
            instance: 覆盖率实例标签
            no_coverage: 是否仅提取 CDFG 不标注覆盖率
            propagate: 是否启用覆盖率传播
            svg: 是否生成 SVG 可视化
            svg_output: SVG 输出文件路径
            verbose: 是否显示详细信息

        Returns:
            DataAnnotator 实例
        """
        config = DataAnnotatorConfig(
            design_json=design_json,
            coverage_dir=coverage_dir,
            output=output,
            instance=instance,
            no_coverage=no_coverage,
            propagate=propagate,
            svg=svg,
            svg_output=svg_output,
            verbose=verbose,
        )
        return cls(config)

    def _build_source_to_html_cache(
        self, design_dir: Path, coverage_dir: Path
    ) -> dict[str, str]:
        """
        构建或加载 源文件名 -> HTML文件名 的缓存

        缓存文件 source2html.json 保存在设计目录下（与 module2html.json 同级），格式：
        {
            "_meta": {"html_count": 33, "generated_at": "2024-01-16T12:00:00"},
            "cv32e40p_sleep_unit.sv": "mod17.html",
            "cv32e40p_sim_clock_gate.sv": "mod3.html",
            ...
        }

        缓存失效条件：HTML 文件数量变化

        Args:
            design_dir: 设计目录（与 module2html.json 同级）
            coverage_dir: 覆盖率报告目录

        Returns:
            源文件名 -> HTML文件名 的映射字典
        """
        cache_file = design_dir / "source2html.json"
        html_files = list(coverage_dir.glob("mod*.html"))
        current_html_count = len(html_files)

        # 尝试加载缓存
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                # 检查缓存是否有效（HTML 文件数量一致）
                meta = cached_data.get("_meta", {})
                if meta.get("html_count") == current_html_count:
                    # 移除 _meta 键，返回纯映射
                    cached_data.pop("_meta", None)
                    if self.config.verbose:
                        print(f"使用缓存的源文件映射 ({len(cached_data)} 条)")
                    return cached_data
                else:
                    if self.config.verbose:
                        print(
                            f"缓存失效: HTML 文件数量变化 ({meta.get('html_count')} -> {current_html_count})"
                        )
            except (json.JSONDecodeError, KeyError):
                pass  # 缓存损坏，重新生成

        # 遍历 HTML 文件，解析源文件名
        source_to_html: dict[str, str] = {}
        for html_path in html_files:
            try:
                parser = CoverageParser(str(html_path))
                source_file = parser.get_source_file()
                if source_file and source_file not in source_to_html:
                    # 只保存文件名，不保存完整路径
                    source_to_html[source_file] = html_path.name
            except Exception:
                continue

        # 保存缓存
        cache_data = {
            "_meta": {
                "html_count": current_html_count,
                "generated_at": __import__("datetime").datetime.now().isoformat(),
            },
            **source_to_html,
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            if self.config.verbose:
                print(f"已生成源文件映射缓存: {cache_file}")
        except Exception as e:
            if self.config.verbose:
                print(f"警告: 无法保存缓存文件: {e}")

        return source_to_html

    def _find_coverage_files(self, design_dir: Path, coverage_dir: Path) -> list:
        """
        根据覆盖率报告中的源文件信息和 CDFG 中的源文件自动查找覆盖率文件

        使用 source2html.json 缓存加速查找，避免每次都解析所有 HTML 文件。

        Args:
            design_dir: 设计目录
            coverage_dir: 覆盖率报告目录

        Returns:
            覆盖率文件路径列表
        """
        # 收集 CDFG 中所有源文件名
        cdfg_source_files = set()
        for node in self.cdfg.nodes.values():
            if node.source_file:
                cdfg_source_files.add(node.source_file)

        print(f"\nCDFG 中包含 {len(cdfg_source_files)} 个源文件的节点")

        # 获取源文件到 HTML 的映射（优先使用缓存）
        source_to_html = self._build_source_to_html_cache(design_dir, coverage_dir)

        if self.config.verbose:
            print(f"覆盖率目录中找到 {len(source_to_html)} 个有效的源文件映射")

        # 匹配 CDFG 源文件与覆盖率报告
        coverage_files = []
        matched_sources = []

        for source_file in cdfg_source_files:
            if source_file in source_to_html:
                # 将相对文件名转换为完整路径
                html_path = coverage_dir / source_to_html[source_file]
                if html_path.exists():
                    coverage_files.append(str(html_path))
                    matched_sources.append(source_file)

        print(f"匹配到 {len(coverage_files)} 个覆盖率报告文件")

        if self.config.verbose and matched_sources:
            for src in matched_sources:
                print(f"  {src} -> {source_to_html[src]}")

        return coverage_files

    def _print_verbose_info(self):
        """打印详细信息（节点类型和源文件分布）"""
        if not self.config.verbose or not self.cdfg:
            return

        # 显示节点类型分布
        type_counts = {}
        for node in self.cdfg.nodes.values():
            type_name = node.node_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        print("\n节点类型分布:")
        for type_name, count in sorted(type_counts.items()):
            print(f"  {type_name}: {count}")

        # 显示源文件分布
        file_counts = defaultdict(int)
        for node in self.cdfg.nodes.values():
            if node.source_file:
                file_counts[node.source_file] += 1
        if file_counts:
            print("\n源文件分布:")
            for file_name, count in sorted(file_counts.items()):
                print(f"  {file_name}: {count} 个节点")

    def extract_cdfg(self) -> CDFG:
        """
        从设计 JSON 提取 CDFG

        Returns:
            提取的 CDFG 对象

        Raises:
            FileNotFoundError: 设计文件不存在
        """
        design_json_path = Path(self.config.design_json)
        if not design_json_path.exists():
            raise FileNotFoundError(f"设计文件不存在: {self.config.design_json}")

        print(f"正在从 {self.config.design_json} 提取 CDFG...")
        extractor = CDFGExtractor(self.config.design_json)
        self.cdfg = extractor.extract()

        print(f"模块: {self.cdfg.module_name}")
        print(f"节点数: {len(self.cdfg.nodes)}")
        print(f"边数: {len(self.cdfg.edges)}")

        self._print_verbose_info()

        return self.cdfg

    def find_coverage_files(self) -> list:
        """
        查找覆盖率文件

        Returns:
            覆盖率文件路径列表
        """
        if self.config.no_coverage or not self.config.coverage_dir:
            return []

        if not self.cdfg:
            raise RuntimeError("请先调用 extract_cdfg() 提取 CDFG")

        coverage_dir = Path(self.config.coverage_dir)
        if not coverage_dir.exists():
            raise FileNotFoundError(f"覆盖率报告目录不存在: {self.config.coverage_dir}")

        # 设计目录是 JSON 文件的上两级目录（json/../）
        design_json_path = Path(self.config.design_json)
        design_dir = design_json_path.parent.parent

        self.coverage_files = self._find_coverage_files(design_dir, coverage_dir)
        return self.coverage_files

    def annotate(self) -> AnnotationStats:
        """
        执行覆盖率标注

        Returns:
            标注统计信息

        Raises:
            RuntimeError: 未提取 CDFG
        """
        if not self.cdfg:
            raise RuntimeError("请先调用 extract_cdfg() 提取 CDFG")

        # 初始化统计
        total_stats = AnnotationStats()
        total_stats.total_edges = len(self.cdfg.edges)

        if not self.coverage_files:
            self.stats = total_stats
            return total_stats

        print(f"\n找到 {len(self.coverage_files)} 个覆盖率报告文件")

        for html_path in self.coverage_files:
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
            instance_tag = self.config.instance
            if instance_tag is None:
                instance_tag = cov_parser.get_last_instance()
                if instance_tag:
                    print(f"  实例: {instance_tag}")

            if self.config.verbose:
                instances = cov_parser.get_available_instances()
                if instances:
                    print(f"  可用实例: {instances}")

            # 执行标注（不传播，最后统一传播）
            stats = annotate_cdfg_with_coverage(
                self.cdfg, html_path, instance_tag, propagate=False
            )

            # 累积统计
            total_stats.total_mux_nodes += stats.total_mux_nodes
            total_stats.annotated_mux_nodes += stats.annotated_mux_nodes
            total_stats.annotated_edges += stats.annotated_edges
            total_stats.covered_edges += stats.covered_edges
            total_stats.uncovered_edges += stats.uncovered_edges
            total_stats.control_edges += stats.control_edges
            total_stats.data_true_edges += stats.data_true_edges
            total_stats.data_false_edges += stats.data_false_edges

            print(
                f"  标注: {stats.annotated_mux_nodes} MUX, {stats.annotated_edges} 边"
            )

        # 如果需要传播，在所有标注完成后统一传播
        if self.config.propagate:
            print("\n正在传播覆盖率...")
            from annotation.annotator import CoverageAnnotator

            annotator = CoverageAnnotator(self.cdfg, {})  # 空覆盖数据，只做传播
            annotator.propagate_coverage()

        # 从 CDFG 边的实际状态统计
        self.stats = self._calculate_final_stats(total_stats)

        return self.stats

    def _calculate_final_stats(self, total_stats: AnnotationStats) -> AnnotationStats:
        """
        从 CDFG 边的实际状态计算最终统计

        Args:
            total_stats: 累积的原始统计

        Returns:
            最终统计信息
        """
        final_stats = AnnotationStats()
        final_stats.total_edges = len(self.cdfg.edges)
        final_stats.total_mux_nodes = total_stats.total_mux_nodes
        final_stats.annotated_mux_nodes = total_stats.annotated_mux_nodes

        for edge in self.cdfg.edges:
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

        return final_stats

    def print_stats(self):
        """打印标注统计信息"""
        if not self.coverage_files:
            return

        print("\n" + "=" * 50)
        print("总体标注统计:")
        print(f"  处理文件数: {len(self.coverage_files)}")
        print(
            f"  MUX 节点: {self.stats.annotated_mux_nodes}/{self.stats.total_mux_nodes} (直接匹配覆盖率分支)"
        )
        print(f"  已标注边: {self.stats.annotated_edges}/{self.stats.total_edges}")
        print(f"  已覆盖边: {self.stats.covered_edges}")
        print(f"  未覆盖边: {self.stats.uncovered_edges}")

        if self.config.verbose:
            print(f"  控制边: {self.stats.control_edges}")
            print(f"  data_true 边: {self.stats.data_true_edges}")
            print(f"  data_false 边: {self.stats.data_false_edges}")

            if self.config.propagate:
                print(f"  必然执行边: {self.stats.always_executed_edges}")
                print(f"  传播标注边: {self.stats.propagated_edges}")

        # 计算覆盖率
        if self.stats.annotated_edges > 0:
            coverage_rate = self.stats.covered_edges / self.stats.annotated_edges * 100
            print(f"\n边覆盖率: {coverage_rate:.2f}%")

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        导出 CDFG 到 JSON 文件

        Args:
            output_path: 输出文件路径，默认使用配置中的路径

        Returns:
            实际输出的文件路径
        """
        if not self.cdfg:
            raise RuntimeError("请先调用 extract_cdfg() 提取 CDFG")

        output = output_path or self.config.output
        exporter = CDFGExporter(self.cdfg)
        with open(output, "w", encoding="utf-8") as f:
            f.write(exporter.to_json())
        print(f"\n已导出 CDFG: {output}")

        return output

    def export_svg(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        导出 CDFG 到 SVG 文件

        Args:
            output_path: 输出文件路径，默认使用配置中的路径

        Returns:
            实际输出的文件路径，失败返回 None
        """
        if not self.cdfg:
            raise RuntimeError("请先调用 extract_cdfg() 提取 CDFG")

        output = output_path or self.config.svg_output
        try:
            # 如果有覆盖率数据，启用覆盖率显示
            show_coverage = self.stats.annotated_edges > 0
            exporter = CDFGExporter(self.cdfg)
            exporter.to_graphviz(output, show_coverage=show_coverage)
            print(f"已导出 SVG: {output}")
            return output
        except Exception as e:
            print(f"警告: 无法生成 SVG: {e}")
            return None

    def print_annotated_examples(self, count: int = 5):
        """
        打印标注示例

        Args:
            count: 显示的边数量
        """
        if not self.config.verbose or self.stats.annotated_edges == 0:
            return

        print(f"\n标注示例（前 {count} 条已标注边）:")
        shown = 0
        for edge in self.cdfg.edges:
            if edge.coverage_label >= 0:
                status = "已覆盖" if edge.coverage_label == 1 else "未覆盖"
                print(f"  {edge.source} -> {edge.target} [{edge.target_port}]")
                print(
                    f"    类型: {edge.coverage_type}, 分支: {edge.branch_index}, 状态: {status}"
                )
                shown += 1
                if shown >= count:
                    break

    def run(self) -> int:
        """
        执行完整的标注流程

        Returns:
            退出码，0 表示成功
        """
        try:
            # 提取 CDFG
            self.extract_cdfg()

            # 查找覆盖率文件
            self.find_coverage_files()

            # 执行标注
            self.annotate()

            # 打印统计
            self.print_stats()

            # 导出 JSON
            self.export_json()

            # 生成 SVG（如果需要）
            if self.config.svg:
                self.export_svg()

            # 打印示例
            self.print_annotated_examples()

            return 0

        except FileNotFoundError as e:
            print(f"错误: {e}")
            return 1
        except Exception as e:
            print(f"错误: {e}")
            return 1


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="RTLMap - 将覆盖率数据标注到 CDFG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 指定设计 JSON 和覆盖率目录，自动匹配模块
  uv run data_annotate.py --design_name cv32e40p --module_name cv32e40p_core -c designs/cv32e40p/coverage_reports/coverage_report1

  # 仅提取 CDFG（不标注覆盖率）
  uv run data_annotate.py --design_name cv32e40p --module_name cv32e40p_core --no-coverage

  # 生成 SVG 可视化
  uv run data_annotate.py --design_name cv32e40p --module_name cv32e40p_core -c designs/cv32e40p/coverage_reports/coverage_report1 --svg

  # 启用覆盖率传播（标注所有可达边）
  uv run data_annotate.py --design_name cv32e40p --module_name cv32e40p_core -c designs/cv32e40p/coverage_reports/coverage_report1 --propagate
        """,
    )
    parser.add_argument("--design_name", required=True, help="Design name")
    parser.add_argument("--module_name", required=True, help="Module name")
    parser.add_argument(
        "-c",
        "--coverage-dir",
        default="designs/cv32e40p/coverage_reports/coverage_report1",
        dest="coverage_dir",
        help="覆盖率报告目录（自动从 module2html.json 匹配文件）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cdfg_annotated.json",
        help="输出 JSON 文件路径 (默认: cdfg_annotated.json)",
    )
    parser.add_argument(
        "-i", "--instance", default=None, help="覆盖率实例标签 (默认: 使用最后一个实例)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="仅提取 CDFG，不标注覆盖率"
    )
    parser.add_argument(
        "--propagate",
        action="store_true",
        help="启用覆盖率传播，标注所有可达边（非分支边）",
    )
    parser.add_argument("--svg", action="store_true", help="生成 SVG 可视化图")
    parser.add_argument(
        "--svg-output",
        default="cdfg_output.svg",
        help="SVG 输出文件路径 (默认: cdfg_output.svg)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 检查design_name与rtlil_json的一致性
    design_dir = Path("designs", args.design_name).absolute()
    rtlil_json_path = design_dir / "RTLIL_json" / f"{args.module_name}.json"
    if not rtlil_json_path.exists():
        # 调用yosys runner来进行生成
        runner = YosysRunner()
        filelist = design_dir / f"{args.design_name}.flist"
        try:
            runner.get_json(
                top_module=args.module_name,
                output_dir=rtlil_json_path.parent,
                flist=filelist,
                files=None,
            )
        except Exception as e:
            print(f"Failed to run Yosys: {e}")
    if not rtlil_json_path.exists():
        print("没有找到rtlil文件")
        exit()

    # 创建配置
    config = DataAnnotatorConfig(
        design_json=rtlil_json_path,
        coverage_dir=args.coverage_dir,
        output=args.output,
        instance=args.instance,
        no_coverage=args.no_coverage,
        propagate=args.propagate,
        svg=args.svg,
        svg_output=args.svg_output,
        verbose=args.verbose,
    )

    # 创建并运行标注器
    annotator = DataAnnotator(config)
    return annotator.run()


if __name__ == "__main__":
    sys.exit(main())
