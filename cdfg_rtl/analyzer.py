#!/usr/bin/env python3
"""
CDFG 分析器
"""

from typing import List, Tuple
from collections import defaultdict

from .data_types import CDFG, EdgeType, NodeType


class CDFGAnalyzer:
    """CDFG 分析器"""

    def __init__(self, cdfg: CDFG):
        self.cdfg = cdfg

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {
            "module_name": self.cdfg.module_name,
            "total_nodes": len(self.cdfg.nodes),
            "total_edges": len(self.cdfg.edges),
            "node_types": defaultdict(int),
            "edge_types": defaultdict(int),
            "cell_types": defaultdict(int),
        }

        for node in self.cdfg.nodes.values():
            stats["node_types"][node.node_type.name] += 1
            if node.cell_type:
                stats["cell_types"][node.cell_type] += 1

        for edge in self.cdfg.edges:
            stats["edge_types"][edge.edge_type.name] += 1

        return stats

    def get_dataflow_paths(self, from_node: str, to_node: str) -> List[List[str]]:
        """获取两个节点之间的所有数据流路径"""
        paths = []
        visited = set()

        def dfs(current: str, path: List[str]):
            if current == to_node:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)

            for edge in self.cdfg.edges:
                if edge.source == current and edge.edge_type in (
                    EdgeType.DATA,
                    EdgeType.DATA_TRUE,
                    EdgeType.DATA_FALSE,
                ):
                    path.append(edge.target)
                    dfs(edge.target, path)
                    path.pop()

            visited.remove(current)

        dfs(from_node, [from_node])
        return paths

    def get_fanin(self, node_id: str) -> List[Tuple[str, EdgeType]]:
        """获取节点的所有扇入"""
        fanin = []
        for edge in self.cdfg.edges:
            if edge.target == node_id:
                fanin.append((edge.source, edge.edge_type))
        return fanin

    def get_fanout(self, node_id: str) -> List[Tuple[str, EdgeType]]:
        """获取节点的所有扇出"""
        fanout = []
        for edge in self.cdfg.edges:
            if edge.source == node_id:
                fanout.append((edge.target, edge.edge_type))
        return fanout

    def get_critical_path(self) -> List[str]:
        """获取最长组合逻辑路径（简化版）"""
        # 找到所有输入和寄存器输出作为起点
        start_nodes = set()
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.INPUT:
                start_nodes.add(node_id)
            elif node.node_type == NodeType.SEQUENTIAL:
                start_nodes.add(node_id)

        # 找到所有输出和寄存器输入作为终点
        end_nodes = set()
        for node_id, node in self.cdfg.nodes.items():
            if node.node_type == NodeType.OUTPUT:
                end_nodes.add(node_id)
            elif node.node_type == NodeType.SEQUENTIAL:
                end_nodes.add(node_id)

        longest_path = []

        for start in start_nodes:
            for end in end_nodes:
                if start != end:
                    paths = self.get_dataflow_paths(start, end)
                    for path in paths:
                        if len(path) > len(longest_path):
                            longest_path = path

        return longest_path
