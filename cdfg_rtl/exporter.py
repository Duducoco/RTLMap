#!/usr/bin/env python3
"""
CDFG 导出器
"""

import json

from .data_types import CDFG
from .visualizer import CDFGVisualizer


class CDFGExporter:
    """CDFG 导出器"""

    def __init__(self, cdfg: CDFG):
        self.cdfg = cdfg

    def _get_port_idx(self, node_id: str, port_name: str, is_output: bool) -> int:
        """获取端口在节点端口列表中的索引

        Args:
            node_id: 节点 ID
            port_name: 端口名称
            is_output: True 表示查找 output_ports，False 表示查找 input_ports

        Returns:
            端口索引，未找到时返回 0
        """
        node = self.cdfg.nodes.get(node_id)
        if node is None:
            return 0

        ports = node.output_ports if is_output else node.input_ports
        try:
            return ports.index(port_name)
        except ValueError:
            return 0  # 未找到时返回 0

    def to_dict(self) -> dict:
        """导出为字典（精简版本，仅保留 GNN 训练必需字段）"""
        return {
            "module_name": self.cdfg.module_name,
            "nodes": [
                {
                    "id": n.id,
                    # "name": n.name,  # 仅调试用
                    "type": n.node_type.name,
                    "cell_type": n.cell_type,
                    "width": n.width,
                    # "parameters": n.parameters,  # 仅调试用
                    "input_ports": n.input_ports,
                    "output_ports": n.output_ports,
                    # "source_line": n.source_line,  # 仅调试/标注用
                    # "stmt_start_line": n.stmt_start_line,  # 仅标注过程使用
                    # "source_file": n.source_file,  # 仅调试用
                }
                for n in self.cdfg.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "source_port": e.source_port,
                    "target_port": e.target_port,
                    "source_port_idx": self._get_port_idx(e.source, e.source_port, is_output=True),
                    "target_port_idx": self._get_port_idx(e.target, e.target_port, is_output=False),
                    "type": e.edge_type.name,
                    "width": e.width,
                    # "source_line": e.source_line,  # 仅调试用
                    # "branch_index": e.branch_index,  # 仅标注过程使用
                    "coverage_label": e.coverage_label,
                    # "coverage_type": e.coverage_type,  # 仅调试用
                }
                for e in self.cdfg.edges
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """导出为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_networkx(self):
        """导出为 NetworkX 图（需要安装 networkx）"""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("需要安装 networkx: pip install networkx")

        G = nx.MultiDiGraph(name=self.cdfg.module_name)

        for node_id, node in self.cdfg.nodes.items():
            G.add_node(
                node_id,
                name=node.name,
                node_type=node.node_type.name,
                cell_type=node.cell_type,
                width=node.width,
            )

        for edge in self.cdfg.edges:
            G.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type.name,
                source_port=edge.source_port,
                target_port=edge.target_port,
                width=edge.width,
            )

        return G

    def to_graphviz(
        self,
        output_file: str = "cdfg_output",
        format: str = "svg",
        show_coverage: bool = True,
        **kwargs,
    ) -> str:
        """
        导出为 Graphviz 图形

        Args:
            output_file: 输出文件名（不含扩展名，或带 .svg 等扩展名）
            format: 输出格式（svg, png, pdf 等）
            show_coverage: 是否显示覆盖率标注
            **kwargs: 传递给 CDFGVisualizer.render() 的其他参数

        Returns:
            输出文件路径
        """
        # 处理文件名（移除扩展名）
        if output_file.endswith(f".{format}"):
            output_file = output_file[: -len(format) - 1]

        visualizer = CDFGVisualizer(self.cdfg)
        return visualizer.render(
            output_file, format=format, show_coverage=show_coverage, **kwargs
        )
