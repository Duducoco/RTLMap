#!/usr/bin/env python3
"""双图数据类型定义"""

from torch_geometric.data import Data


class DualGraphData(Data):
    """
    双图数据 - 继承自 PyG Data 类

    属性：
    - 主图（RTL）:
        - node_cell_type: [N] 节点的 cell_type 索引（对应 CELL_TYPE_VOCAB）
        - node_width: [N] 节点的 width（信号位宽）
        - edge_index: [2, E] 边索引
        - edge_type: [E] 边类型（0: DATA, 1: DATA_TRUE, 2: DATA_FALSE, 3: CONTROL, 4: CLOCK, 5: RESET, 6: ENABLE）
        - edge_width: [E] 边的 width（信号位宽）
        - edge_labels: [E] 边标签（-1: 未标注, 0: 未覆盖, 1: 已覆盖）

    - 辅助图（ASM）:
        - asm_node_type: [M] ASM 节点类型索引（对应 AsmNodeType 枚举）
        - asm_instruction_encoding: [M, D] ASM 节点的指令编码（由外部模型生成）
        - asm_edge_index: [2, F] ASM 边索引
        - asm_edge_type: [F] ASM 边类型（对应 AsmEdgeType 枚举）

    - 标签:
        - y: [B, 1] 图级标签（整体覆盖率）

    - Batching:
        - batch: [N] 主图 batch 索引（自动生成）
        - asm_node_type_batch: [M] 辅助图 batch 索引（需 follow_batch=['asm_node_type']）

    ASM 节点类型枚举 (AsmNodeType, 22 种):
        0: ENTRY, 1: EXIT, 2: BRANCH, 3: JUMP,
        4: COMPUTE, 5: ARITHMETIC, 6: LOGIC, 7: SHIFT, 8: COMPARE,
        9: MAC, 10: SIMD, 11: LOAD, 12: STORE, 13: MEMORY,
        14: CSR, 15: SYSTEM, 16: HWLOOP, 17: NOP, 18: UNKNOWN, ...

    ASM 边类型枚举 (AsmEdgeType, 10 种):
        0: CONTROL_FLOW, 1: BRANCH_TAKEN, 2: BRANCH_NOT_TAKEN, 3: JUMP,
        4: CALL, 5: RETURN, 6: DATA_DEP, 7: MEMORY_DEP, 8: ANTI_DEP, 9: OUTPUT_DEP
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        """定义 batching 时的节点索引增量"""
        if key == "asm_edge_index":
            # ASM 边索引基于 ASM 节点数
            asm_node_type = getattr(self, "asm_node_type", None)
            return asm_node_type.size(0) if asm_node_type is not None else 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """定义 batching 时的拼接维度"""
        # 1D 张量在 dim=0 拼接
        if key in [
            # RTL 节点属性
            "node_cell_type",
            "node_width",
            # RTL 边属性
            "edge_type",
            "edge_width",
            "edge_labels",
            # ASM 节点属性
            "asm_node_type",
            # ASM 边属性
            "asm_edge_type",
        ]:
            return 0
        # 2D 张量（节点特征）在 dim=0 拼接
        if key == "asm_instruction_encoding":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def num_nodes(self) -> int:
        """主图节点数"""
        node_cell_type = getattr(self, "node_cell_type", None)
        return node_cell_type.size(0) if node_cell_type is not None else 0

    @property
    def num_asm_nodes(self) -> int:
        """辅助图节点数"""
        asm_node_type = getattr(self, "asm_node_type", None)
        return asm_node_type.size(0) if asm_node_type is not None else 0

    @property
    def num_edges(self) -> int:
        """主图边数"""
        edge_index = getattr(self, "edge_index", None)
        return edge_index.size(1) if edge_index is not None else 0

    @property
    def num_asm_edges(self) -> int:
        """辅助图边数"""
        asm_ei = getattr(self, "asm_edge_index", None)
        return asm_ei.size(1) if asm_ei is not None else 0
