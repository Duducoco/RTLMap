#!/usr/bin/env python3
"""双图数据类型定义"""

from torch_geometric.data import Data


class DualGraphData(Data):
    """
    双图数据 - 继承自 PyG Data 类

    属性：
    - 主图（RTL）: x, edge_index, edge_type, edge_labels, edge_attr, node_type
    - 辅助图（ASM）: asm_x, asm_edge_index, asm_edge_type, asm_edge_attr, asm_node_type
    - 标签: y (图级标签), edge_labels (边标签)
    - Batching: batch (主图), asm_x_batch (辅助图，需 follow_batch=['asm_x'])

    边类型枚举：
    - 0: DATA
    - 1: CONTROL
    - 2: CLOCK
    - 3: RESET
    - 4: ENABLE

    边标签：
    - -1: 未标注（训练时 mask）
    - 0: 未覆盖
    - 1: 已覆盖
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        """定义 batching 时的节点索引增量"""
        if key == "asm_edge_index":
            return self.asm_x.size(0) if self.asm_x is not None else 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """定义 batching 时的拼接维度"""
        if key in [
            "edge_type",
            "edge_labels",
            "asm_edge_type",
            "node_type",
            "asm_node_type",
        ]:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def num_nodes(self) -> int:
        """主图节点数"""
        return self.x.size(0) if self.x is not None else 0

    @property
    def num_asm_nodes(self) -> int:
        """辅助图节点数"""
        return self.asm_x.size(0) if self.asm_x is not None else 0

    @property
    def num_edges(self) -> int:
        """主图边数"""
        return self.edge_index.size(1) if self.edge_index is not None else 0

    @property
    def num_asm_edges(self) -> int:
        """辅助图边数"""
        asm_ei = getattr(self, "asm_edge_index", None)
        return asm_ei.size(1) if asm_ei is not None else 0
