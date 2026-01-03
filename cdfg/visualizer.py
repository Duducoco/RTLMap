#!/usr/bin/env python3
"""
CDFG 可视化器
"""

from typing import Dict
import graphviz

from .types import CDFG, Node, Edge, NodeType, EdgeType


class CDFGVisualizer:
    """CDFG 可视化器"""

    # 节点样式配置
    NODE_STYLES = {
        NodeType.INPUT: {
            "shape": "ellipse",
            "fillcolor": "lightgreen",
            "style": "filled",
        },
        NodeType.OUTPUT: {
            "shape": "ellipse",
            "fillcolor": "lightcoral",
            "style": "filled",
        },
        NodeType.CONSTANT: {
            "shape": "diamond",
            "fillcolor": "lightyellow",
            "style": "filled",
        },
        NodeType.SEQUENTIAL: {
            "shape": "box",
            "fillcolor": "lightblue",
            "style": "filled,bold",
        },
        NodeType.MUX: {"shape": "trapezium", "fillcolor": "wheat", "style": "filled"},
        NodeType.ARITHMETIC: {"shape": "box", "fillcolor": "white", "style": "filled"},
        NodeType.LOGIC: {"shape": "box", "fillcolor": "lavender", "style": "filled"},
        NodeType.COMPARE: {"shape": "box", "fillcolor": "mistyrose", "style": "filled"},
        NodeType.MEMORY: {
            "shape": "box3d",
            "fillcolor": "lightgray",
            "style": "filled",
        },
        NodeType.SHIFT: {"shape": "box", "fillcolor": "honeydew", "style": "filled"},
        NodeType.COMBINATIONAL: {
            "shape": "box",
            "fillcolor": "white",
            "style": "filled",
        },
        NodeType.UNKNOWN: {"shape": "box", "fillcolor": "white", "style": "filled"},
    }

    # 边样式配置
    EDGE_STYLES = {
        EdgeType.DATA: {"color": "black", "style": "solid"},
        EdgeType.CONTROL: {"color": "red", "style": "dashed"},
        EdgeType.CLOCK: {"color": "blue", "style": "dotted"},
        EdgeType.RESET: {"color": "orange", "style": "dashed"},
        EdgeType.ENABLE: {"color": "green", "style": "dashed"},
    }

    # 覆盖率边样式配置
    COVERAGE_EDGE_STYLES = {
        # 已覆盖/已执行的边 - 绿色加粗
        "covered": {"color": "green", "penwidth": "2.5", "style": "solid"},
        # 未覆盖的边 - 红色虚线
        "uncovered": {"color": "red", "penwidth": "1.5", "style": "dashed"},
        # 未标注的边 - 灰色
        "unknown": {"color": "gray", "penwidth": "1.0", "style": "dotted"},
    }

    def __init__(self, cdfg: CDFG):
        self.cdfg = cdfg
        # 建立原始 ID 到清理后 ID 的映射
        self.id_map: Dict[str, str] = {}

    def _sanitize_id(self, node_id: str) -> str:
        """
        清理节点 ID，使其符合 Graphviz DOT 语法
        """
        if node_id in self.id_map:
            return self.id_map[node_id]

        # 替换特殊字符
        sanitized = node_id
        replacements = [
            ("$", "S"),
            (":", "_"),
            ("/", "_"),
            ("\\", "_"),
            (".", "_"),
            ("-", "_"),
            (" ", "_"),
            ("[", "_"),
            ("]", "_"),
        ]
        for char, replacement in replacements:
            sanitized = sanitized.replace(char, replacement)

        # 确保不以数字开头
        if sanitized and sanitized[0].isdigit():
            sanitized = "n_" + sanitized

        # 确保唯一性
        base_id = sanitized
        counter = 1
        while sanitized in self.id_map.values():
            sanitized = f"{base_id}_{counter}"
            counter += 1

        self.id_map[node_id] = sanitized
        return sanitized

    def render(
        self,
        output_file: str = "cdfg",
        format: str = "svg",
        show_ports: bool = True,
        show_widths: bool = True,
        rankdir: str = "TB",
        separate_control: bool = False,
        engine: str = "auto",
        show_coverage: bool = False,
    ) -> str:
        """
        渲染 CDFG

        Args:
            output_file: 输出文件名（不含扩展名）
            format: 输出格式（svg, png, pdf 等）
            show_ports: 是否显示端口名
            show_widths: 是否显示位宽
            rankdir: 布局方向（TB=上到下, LR=左到右）
            separate_control: 是否分离控制路径
            engine: graphviz 布局引擎，可选值：
                    'auto' - 自动选择（小图用 dot，大图用 sfdp）
                    'dot' - 层次布局，适合小型 DAG（<100 节点）
                    'sfdp' - 力导向布局，适合大型图（推荐）
                    'neato' - 力导向布局
                    'fdp' - 力导向布局
                    'circo' - 圆形布局
                    'twopi' - 径向布局
            show_coverage: 是否显示覆盖率标注（绿色=已执行，红色=未覆盖）

        Returns:
            输出文件路径
        """
        # 重置 ID 映射
        self.id_map = {}

        # 自动选择引擎
        if engine == "auto":
            node_count = len(self.cdfg.nodes)
            edge_count = len(self.cdfg.edges)
            # 对于大型图，使用 sfdp 引擎（更快）
            if node_count > 100 or edge_count > 300:
                engine = "sfdp"
                print(
                    f"[INFO] 图较大 ({node_count} 节点, {edge_count} 边)，使用 sfdp 引擎加速渲染"
                )
            else:
                engine = "dot"

        dot = graphviz.Digraph(
            name="CDFG",
            comment=f"CDFG: {self.cdfg.module_name}",
            format=format,
            engine=engine,
        )
        # 只有 dot 引擎支持 rankdir
        if engine == "dot":
            dot.attr(rankdir=rankdir, compound="true")
        else:
            dot.attr(compound="true")
        dot.attr("node", fontname="Helvetica", fontsize="10")
        dot.attr("edge", fontname="Helvetica", fontsize="8")

        if separate_control:
            self._render_separated(dot, show_ports, show_widths, show_coverage)
        else:
            self._render_unified(dot, show_ports, show_widths, show_coverage)

        # 如果显示覆盖率，添加图例
        if show_coverage:
            self._add_coverage_legend(dot)

        output_path = dot.render(output_file, cleanup=True)
        return output_path

    def _add_coverage_legend(self, dot: graphviz.Digraph):
        """添加覆盖率图例"""
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Coverage Legend", style="rounded", color="gray")
            legend.attr(rank="sink")

            # 创建图例节点
            legend.node("legend_start", "", shape="point", width="0")
            legend.node("legend_covered", "已执行 (Covered)", shape="plaintext")
            legend.node("legend_uncovered", "未覆盖 (Uncovered)", shape="plaintext")
            legend.node("legend_unknown", "未标注 (Unknown)", shape="plaintext")

            # 创建示例边
            legend.edge(
                "legend_start", "legend_covered", **self.COVERAGE_EDGE_STYLES["covered"]
            )
            legend.edge(
                "legend_start",
                "legend_uncovered",
                **self.COVERAGE_EDGE_STYLES["uncovered"],
            )
            legend.edge(
                "legend_start", "legend_unknown", **self.COVERAGE_EDGE_STYLES["unknown"]
            )

    def _get_coverage_edge_style(self, edge: Edge) -> dict:
        """根据边的覆盖标签获取样式"""
        if edge.coverage_label == 1:
            return self.COVERAGE_EDGE_STYLES["covered"].copy()
        elif edge.coverage_label == 0:
            return self.COVERAGE_EDGE_STYLES["uncovered"].copy()
        else:
            return self.COVERAGE_EDGE_STYLES["unknown"].copy()

    def _render_unified(
        self,
        dot: graphviz.Digraph,
        show_ports: bool,
        show_widths: bool,
        show_coverage: bool = False,
    ):
        """统一渲染"""
        # 添加节点
        for node_id, node in self.cdfg.nodes.items():
            safe_id = self._sanitize_id(node_id)
            label = self._create_node_label(node, show_widths)
            style = self.NODE_STYLES.get(
                node.node_type, self.NODE_STYLES[NodeType.UNKNOWN]
            )
            dot.node(safe_id, label=label, **style)

        # 添加边
        for edge in self.cdfg.edges:
            safe_source = self._sanitize_id(edge.source)
            safe_target = self._sanitize_id(edge.target)
            label = self._create_edge_label(
                edge, show_ports, show_widths, show_coverage
            )

            if show_coverage:
                # 使用覆盖率样式
                style = self._get_coverage_edge_style(edge)
            else:
                # 使用默认边类型样式
                style = self.EDGE_STYLES.get(
                    edge.edge_type, self.EDGE_STYLES[EdgeType.DATA]
                ).copy()

            dot.edge(safe_source, safe_target, label=label, **style)

    def _render_separated(
        self,
        dot: graphviz.Digraph,
        show_ports: bool,
        show_widths: bool,
        show_coverage: bool = False,
    ):
        """分离控制路径和数据路径渲染"""
        # 数据路径子图
        with dot.subgraph(name="cluster_datapath") as dp:
            dp.attr(label="Data Path", style="rounded", color="blue")

            for node_id, node in self.cdfg.nodes.items():
                if node.node_type not in [NodeType.CONSTANT]:
                    safe_id = self._sanitize_id(node_id)
                    label = self._create_node_label(node, show_widths)
                    style = self.NODE_STYLES.get(
                        node.node_type, self.NODE_STYLES[NodeType.UNKNOWN]
                    )
                    dp.node(safe_id, label=label, **style)

        # 常数节点单独添加
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.CONSTANT:
                safe_id = self._sanitize_id(node_id)
                label = self._create_node_label(node, show_widths)
                style = self.NODE_STYLES[NodeType.CONSTANT]
                dot.node(safe_id, label=label, **style)

        # 添加边
        for edge in self.cdfg.edges:
            safe_source = self._sanitize_id(edge.source)
            safe_target = self._sanitize_id(edge.target)
            label = self._create_edge_label(
                edge, show_ports, show_widths, show_coverage
            )

            if show_coverage:
                # 使用覆盖率样式
                style = self._get_coverage_edge_style(edge)
            else:
                # 使用默认边类型样式
                style = self.EDGE_STYLES.get(
                    edge.edge_type, self.EDGE_STYLES[EdgeType.DATA]
                ).copy()

            dot.edge(safe_source, safe_target, label=label, **style)

    def _create_node_label(self, node: Node, show_widths: bool) -> str:
        """创建节点标签"""
        if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
            if show_widths and node.width > 1:
                return f"{node.name}[{node.width - 1}:0]"
            return node.name

        if node.node_type == NodeType.CONSTANT:
            return node.name

        # Cell 节点 - 显示简化的类型名
        type_name = node.cell_type.replace("$", "")
        if show_widths and node.width > 1:
            return f"{type_name}\n[{node.width}bit]"
        return type_name

    def _create_edge_label(
        self,
        edge: Edge,
        show_ports: bool,
        show_widths: bool,
        show_coverage: bool = False,
    ) -> str:
        """创建边标签"""
        parts = []

        if show_ports and edge.target_port:
            parts.append(edge.target_port)

        if show_widths and edge.width > 1:
            parts.append(f"[{edge.width}]")

        # 如果显示覆盖率，添加覆盖类型标签
        if show_coverage and edge.coverage_type:
            # 简化标签
            ctype_short = {
                "always": "A",
                "control": "C",
                "data_true": "T",
                "data_false": "F",
                "propagated": "P",
            }.get(edge.coverage_type, edge.coverage_type[0].upper())
            parts.append(f"({ctype_short})")

        return " ".join(parts) if parts else ""

    def to_dot_string(self) -> str:
        """返回 DOT 格式字符串"""
        self.id_map = {}
        dot = graphviz.Digraph(name="CDFG")
        self._render_unified(dot, True, True)
        return dot.source
