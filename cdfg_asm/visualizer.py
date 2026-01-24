#!/usr/bin/env python3
"""
ASM CDFG 可视化器

专门为汇编基本块级 CDFG 设计的可视化器，支持：
- 基本块内指令列表显示
- 控制流边和数据流边的区分
- 寄存器数据流可视化
- 分支条件标注
- 循环回边高亮
"""

from typing import Dict, Optional
import graphviz

# 支持直接运行和模块导入两种方式
if __name__ == "__main__" or __package__ is None:
    from cdfg_asm.data_types import (
        AsmCDFG,
        AsmNode,
        AsmEdge,
        AsmNodeType,
        AsmEdgeType,
        InstrCategory,
    )
else:
    from .data_types import (
        AsmCDFG,
        AsmNode,
        AsmEdge,
        AsmNodeType,
        AsmEdgeType,
        InstrCategory,
    )


class AsmCDFGVisualizer:
    """汇编 CDFG 可视化器"""

    # 节点样式配置（基于 AsmNodeType）
    NODE_STYLES = {
        # 控制流节点
        AsmNodeType.ENTRY: {
            "shape": "record",
            "fillcolor": "#90EE90",  # lightgreen - 入口
            "style": "filled,rounded,bold",
            "color": "#228B22",
        },
        AsmNodeType.EXIT: {
            "shape": "record",
            "fillcolor": "#FFB6C1",  # lightpink - 出口
            "style": "filled,rounded,bold",
            "color": "#DC143C",
        },
        AsmNodeType.BRANCH: {
            "shape": "record",
            "fillcolor": "#FFE4B5",  # moccasin - 分支
            "style": "filled,rounded",
            "color": "#D2691E",
        },
        AsmNodeType.JUMP: {
            "shape": "record",
            "fillcolor": "#FFDAB9",  # peachpuff - 跳转
            "style": "filled,rounded",
            "color": "#CD853F",
        },
        # 计算节点
        AsmNodeType.COMPUTE: {
            "shape": "record",
            "fillcolor": "#FFFFFF",  # white
            "style": "filled,rounded",
            "color": "#808080",
        },
        AsmNodeType.ARITHMETIC: {
            "shape": "record",
            "fillcolor": "#E0FFFF",  # lightcyan
            "style": "filled,rounded",
            "color": "#4682B4",
        },
        AsmNodeType.LOGIC: {
            "shape": "record",
            "fillcolor": "#E6E6FA",  # lavender
            "style": "filled,rounded",
            "color": "#6A5ACD",
        },
        AsmNodeType.SHIFT: {
            "shape": "record",
            "fillcolor": "#F0FFF0",  # honeydew
            "style": "filled,rounded",
            "color": "#228B22",
        },
        AsmNodeType.COMPARE: {
            "shape": "record",
            "fillcolor": "#FFE4E1",  # mistyrose
            "style": "filled,rounded",
            "color": "#CD5C5C",
        },
        AsmNodeType.MAC: {
            "shape": "record",
            "fillcolor": "#DDA0DD",  # plum
            "style": "filled,rounded",
            "color": "#8B008B",
        },
        AsmNodeType.SIMD: {
            "shape": "record",
            "fillcolor": "#DA70D6",  # orchid
            "style": "filled,rounded",
            "color": "#9400D3",
        },
        # 内存节点
        AsmNodeType.LOAD: {
            "shape": "record",
            "fillcolor": "#B0E0E6",  # powderblue
            "style": "filled,rounded",
            "color": "#4169E1",
        },
        AsmNodeType.STORE: {
            "shape": "record",
            "fillcolor": "#87CEEB",  # skyblue
            "style": "filled,rounded",
            "color": "#1E90FF",
        },
        AsmNodeType.MEMORY: {
            "shape": "record",
            "fillcolor": "#D3D3D3",  # lightgray
            "style": "filled,rounded",
            "color": "#696969",
        },
        # 特殊节点
        AsmNodeType.CSR: {
            "shape": "record",
            "fillcolor": "#ADD8E6",  # lightblue
            "style": "filled,rounded",
            "color": "#4169E1",
        },
        AsmNodeType.SYSTEM: {
            "shape": "record",
            "fillcolor": "#FFA07A",  # lightsalmon
            "style": "filled,rounded",
            "color": "#FF4500",
        },
        AsmNodeType.HWLOOP: {
            "shape": "record",
            "fillcolor": "#98FB98",  # palegreen
            "style": "filled,rounded",
            "color": "#32CD32",
        },
        AsmNodeType.NOP: {
            "shape": "record",
            "fillcolor": "#F5F5F5",  # whitesmoke
            "style": "filled,rounded",
            "color": "#C0C0C0",
        },
        AsmNodeType.UNKNOWN: {
            "shape": "record",
            "fillcolor": "#F5F5F5",
            "style": "filled,rounded",
            "color": "#A9A9A9",
        },
    }

    # 边样式配置（基于 AsmEdgeType）
    EDGE_STYLES = {
        # 控制流边
        AsmEdgeType.CONTROL_FLOW: {
            "color": "#2F4F4F",  # darkslategray
            "style": "solid",
            "penwidth": "1.5",
        },
        AsmEdgeType.BRANCH_TAKEN: {
            "color": "#228B22",  # forestgreen
            "style": "bold",
            "penwidth": "2.0",
        },
        AsmEdgeType.BRANCH_NOT_TAKEN: {
            "color": "#DC143C",  # crimson
            "style": "dashed",
            "penwidth": "1.5",
        },
        AsmEdgeType.JUMP: {
            "color": "#FF8C00",  # darkorange
            "style": "bold",
            "penwidth": "2.0",
        },
        AsmEdgeType.CALL: {
            "color": "#4169E1",  # royalblue
            "style": "bold",
            "penwidth": "2.5",
        },
        AsmEdgeType.RETURN: {
            "color": "#9370DB",  # mediumpurple
            "style": "dashed",
            "penwidth": "2.0",
        },
        # 数据流边
        AsmEdgeType.DATA_DEP: {
            "color": "#4682B4",  # steelblue
            "style": "dotted",
            "penwidth": "1.0",
        },
        AsmEdgeType.MEMORY_DEP: {
            "color": "#708090",  # slategray
            "style": "dotted",
            "penwidth": "1.0",
        },
        AsmEdgeType.ANTI_DEP: {
            "color": "#B8860B",  # darkgoldenrod
            "style": "dotted",
            "penwidth": "1.0",
        },
        AsmEdgeType.OUTPUT_DEP: {
            "color": "#8B4513",  # saddlebrown
            "style": "dotted",
            "penwidth": "1.0",
        },
    }

    # 回边样式
    BACK_EDGE_STYLE = {
        "color": "#FF0000",  # red
        "style": "bold",
        "penwidth": "2.5",
        "constraint": "false",  # 不影响布局
    }

    def __init__(
        self,
        show_instructions: bool = True,
        max_instructions: int = 10,
        show_registers: bool = True,
        compact_mode: bool = False,
        show_data_edges: bool = True,
    ):
        """
        初始化可视化器

        Args:
            show_instructions: 是否显示基本块内的指令列表
            max_instructions: 每个节点最多显示的指令数
            show_registers: 是否显示 def/use 寄存器集合
            compact_mode: 紧凑模式（只显示指令助记符）
            show_data_edges: 是否显示数据流边
        """
        self.show_instructions = show_instructions
        self.max_instructions = max_instructions
        self.show_registers = show_registers
        self.compact_mode = compact_mode
        self.show_data_edges = show_data_edges
        self.id_map: Dict[str, str] = {}

    def _sanitize_id(self, node_id: str) -> str:
        """清理节点 ID，使其符合 Graphviz DOT 语法"""
        if node_id in self.id_map:
            return self.id_map[node_id]

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

        if sanitized and sanitized[0].isdigit():
            sanitized = "n_" + sanitized

        base_id = sanitized
        counter = 1
        while sanitized in self.id_map.values():
            sanitized = f"{base_id}_{counter}"
            counter += 1

        self.id_map[node_id] = sanitized
        return sanitized

    def _escape_html(self, text: str) -> str:
        """转义 HTML 特殊字符"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _create_node_label(self, node: AsmNode) -> str:
        """
        创建节点标签（使用 HTML-like 格式支持多行和结构化显示）
        """
        # 标题
        if node.label:
            title = self._escape_html(node.label)
        else:
            title = f"block_{node.id}"

        line_info = f"L{node.start_line}" if node.start_line else ""

        # 循环头标记
        loop_marker = " ↻" if node.is_loop_header else ""

        if self.compact_mode:
            return f"{title}{loop_marker}\\n({node.instr_count} instrs)"

        # 构建 HTML 表格标签
        html_parts = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">']

        # 标题行
        type_badge = (
            f'<FONT POINT-SIZE="7" COLOR="#666666">[{node.node_type.name}]</FONT>'
        )
        html_parts.append(
            f'<TR><TD COLSPAN="2" BGCOLOR="#E8E8E8"><B>{title}</B>{loop_marker} '
            f'<FONT POINT-SIZE="8"> {line_info}</FONT><BR/>{type_badge}</TD></TR>'
        )

        # 指令列表
        if self.show_instructions and node.instructions:
            display_instrs = node.instructions[: self.max_instructions]
            for instr in display_instrs:
                mnemonic = self._escape_html(instr.mnemonic)
                operands = ", ".join(self._escape_html(op) for op in instr.operands)

                if operands:
                    html_parts.append(
                        f'<TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="9">'
                        f"{mnemonic}</FONT></TD>"
                        f'<TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="9">'
                        f"{operands}</FONT></TD></TR>"
                    )
                else:
                    html_parts.append(
                        f'<TR><TD COLSPAN="2" ALIGN="LEFT">'
                        f'<FONT FACE="Courier" POINT-SIZE="9">{mnemonic}</FONT></TD></TR>'
                    )

            if len(node.instructions) > self.max_instructions:
                remaining = len(node.instructions) - self.max_instructions
                html_parts.append(
                    f'<TR><TD COLSPAN="2" ALIGN="CENTER">'
                    f'<FONT POINT-SIZE="8">... +{remaining} more</FONT></TD></TR>'
                )

        # 寄存器信息
        if self.show_registers and (node.defs or node.uses):
            html_parts.append('<TR><TD COLSPAN="2" HEIGHT="1"></TD></TR>')

            if node.defs:
                def_str = ", ".join(sorted(node.defs)[:5])
                if len(node.defs) > 5:
                    def_str += f" +{len(node.defs) - 5}"
                html_parts.append(
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#228B22">'
                    f"def:</FONT></TD>"
                    f'<TD ALIGN="LEFT"><FONT POINT-SIZE="8">{def_str}</FONT></TD></TR>'
                )

            if node.uses:
                use_str = ", ".join(sorted(node.uses)[:5])
                if len(node.uses) > 5:
                    use_str += f" +{len(node.uses) - 5}"
                html_parts.append(
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#4169E1">'
                    f"use:</FONT></TD>"
                    f'<TD ALIGN="LEFT"><FONT POINT-SIZE="8">{use_str}</FONT></TD></TR>'
                )

        # 分支条件
        if node.branch_condition and node.node_type == AsmNodeType.BRANCH:
            cond = self._escape_html(node.branch_condition)
            html_parts.append(
                f'<TR><TD COLSPAN="2" BGCOLOR="#FFF8DC">'
                f'<FONT POINT-SIZE="8" COLOR="#8B4513">cond: {cond}</FONT></TD></TR>'
            )

        html_parts.append("</TABLE>>")
        return "".join(html_parts)

    def _create_edge_label(self, edge: AsmEdge) -> str:
        """创建边标签"""
        # 控制流边
        if edge.edge_type == AsmEdgeType.BRANCH_TAKEN:
            return "T"
        elif edge.edge_type == AsmEdgeType.BRANCH_NOT_TAKEN:
            return "F"
        elif edge.edge_type == AsmEdgeType.CALL:
            return "call"
        elif edge.edge_type == AsmEdgeType.RETURN:
            return "ret"

        # 数据流边 - 显示寄存器
        if edge.edge_type == AsmEdgeType.DATA_DEP and edge.register:
            return edge.register

        return ""

    def render(
        self,
        cdfg: AsmCDFG,
        output_file: str = "asm_cdfg",
        format: str = "svg",
        rankdir: str = "TB",
        engine: str = "dot",
    ) -> str:
        """
        渲染 ASM CDFG

        Args:
            cdfg: AsmCDFG 对象
            output_file: 输出文件名（不含扩展名）
            format: 输出格式（svg, png, pdf 等）
            rankdir: 布局方向（TB=上到下, LR=左到右）
            engine: graphviz 布局引擎

        Returns:
            输出文件路径
        """
        self.id_map = {}

        dot = graphviz.Digraph(
            name="AsmCDFG",
            comment=f"ASM CDFG: {cdfg.module_name}",
            format=format,
            engine=engine,
        )

        # 图属性
        dot.attr(rankdir=rankdir, compound="true", splines="spline")
        dot.attr("node", fontname="Helvetica", fontsize="10")
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # 添加节点
        for node_id, node in cdfg.nodes.items():
            safe_id = self._sanitize_id(node_id)
            label = self._create_node_label(node)
            style = self.NODE_STYLES.get(
                node.node_type, self.NODE_STYLES[AsmNodeType.UNKNOWN]
            )
            dot.node(safe_id, label=label, **style)

        # 添加边
        for edge in cdfg.edges:
            # 是否显示数据流边
            if not self.show_data_edges and edge.edge_type in {
                AsmEdgeType.DATA_DEP,
                AsmEdgeType.MEMORY_DEP,
                AsmEdgeType.ANTI_DEP,
                AsmEdgeType.OUTPUT_DEP,
            }:
                continue

            safe_source = self._sanitize_id(edge.source)
            safe_target = self._sanitize_id(edge.target)
            label = self._create_edge_label(edge)

            # 回边使用特殊样式
            if edge.is_back_edge:
                style = self.BACK_EDGE_STYLE.copy()
            else:
                style = self.EDGE_STYLES.get(
                    edge.edge_type, self.EDGE_STYLES[AsmEdgeType.CONTROL_FLOW]
                ).copy()

            dot.edge(safe_source, safe_target, label=label, **style)

        # 添加图例
        self._add_legend(dot)

        # 渲染
        output_path = dot.render(output_file, cleanup=True)
        return output_path

    def _add_legend(self, dot: graphviz.Digraph):
        """添加图例"""
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", style="rounded", color="gray", fontsize="10")
            legend.attr(rank="sink")

            # 控制流边图例
            legend.node("l_ctrl_src", "", shape="point", width="0.1")
            legend.node("l_ctrl_dst", "Control Flow", shape="plaintext", fontsize="9")
            legend.edge(
                "l_ctrl_src",
                "l_ctrl_dst",
                **self.EDGE_STYLES[AsmEdgeType.CONTROL_FLOW],
            )

            legend.node("l_taken_src", "", shape="point", width="0.1")
            legend.node("l_taken_dst", "Branch Taken", shape="plaintext", fontsize="9")
            legend.edge(
                "l_taken_src",
                "l_taken_dst",
                **self.EDGE_STYLES[AsmEdgeType.BRANCH_TAKEN],
            )

            legend.node("l_ntaken_src", "", shape="point", width="0.1")
            legend.node(
                "l_ntaken_dst", "Branch Not Taken", shape="plaintext", fontsize="9"
            )
            legend.edge(
                "l_ntaken_src",
                "l_ntaken_dst",
                **self.EDGE_STYLES[AsmEdgeType.BRANCH_NOT_TAKEN],
            )

            legend.node("l_back_src", "", shape="point", width="0.1")
            legend.node(
                "l_back_dst", "Back Edge (Loop)", shape="plaintext", fontsize="9"
            )
            legend.edge("l_back_src", "l_back_dst", **self.BACK_EDGE_STYLE)

            if self.show_data_edges:
                legend.node("l_data_src", "", shape="point", width="0.1")
                legend.node(
                    "l_data_dst", "Data Dependency", shape="plaintext", fontsize="9"
                )
                legend.edge(
                    "l_data_src",
                    "l_data_dst",
                    **self.EDGE_STYLES[AsmEdgeType.DATA_DEP],
                )

    def render_simple(
        self,
        cdfg: AsmCDFG,
        output_file: str = "asm_cfg",
        format: str = "svg",
    ) -> str:
        """
        简化渲染（只显示控制流图，不显示指令详情）

        Args:
            cdfg: AsmCDFG 对象
            output_file: 输出文件名
            format: 输出格式

        Returns:
            输出文件路径
        """
        self.id_map = {}

        dot = graphviz.Digraph(
            name="AsmCFG",
            comment=f"ASM CFG: {cdfg.module_name}",
            format=format,
            engine="dot",
        )

        dot.attr(rankdir="TB", splines="spline")
        dot.attr(
            "node",
            fontname="Helvetica",
            fontsize="11",
            shape="box",
            style="rounded,filled",
        )
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # 简化节点颜色
        simple_colors = {
            AsmNodeType.ENTRY: "#90EE90",
            AsmNodeType.EXIT: "#FFB6C1",
            AsmNodeType.BRANCH: "#FFE4B5",
            AsmNodeType.JUMP: "#FFDAB9",
            AsmNodeType.CSR: "#ADD8E6",
            AsmNodeType.SYSTEM: "#FFA07A",
        }

        # 添加节点
        for node_id, node in cdfg.nodes.items():
            safe_id = self._sanitize_id(node_id)
            name = node.label if node.label else f"bb_{node_id}"
            loop_marker = " ↻" if node.is_loop_header else ""
            label = f"{name}{loop_marker}\\n({node.instr_count} instrs)"
            fillcolor = simple_colors.get(node.node_type, "#FFFFFF")
            dot.node(safe_id, label=label, fillcolor=fillcolor)

        # 只添加控制流边
        for edge in cdfg.get_control_edges():
            safe_source = self._sanitize_id(edge.source)
            safe_target = self._sanitize_id(edge.target)

            if edge.is_back_edge:
                style = self.BACK_EDGE_STYLE.copy()
            elif edge.edge_type == AsmEdgeType.BRANCH_TAKEN:
                style = {"color": "#228B22", "penwidth": "1.5", "label": "T"}
            elif edge.edge_type == AsmEdgeType.BRANCH_NOT_TAKEN:
                style = {
                    "color": "#DC143C",
                    "style": "dashed",
                    "penwidth": "1.5",
                    "label": "F",
                }
            else:
                style = {"color": "#2F4F4F", "penwidth": "1.5"}

            dot.edge(safe_source, safe_target, **style)

        return dot.render(output_file, cleanup=True)
