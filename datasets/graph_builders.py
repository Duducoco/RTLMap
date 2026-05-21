#!/usr/bin/env python3
"""RTL/ASM 图张量构建工具。"""

from typing import Optional

import torch

from cdfg_asm.data_types import AsmEdgeType, AsmNodeType
from cdfg_rtl.data_types import CELL_TYPE_TO_NODE_TYPE, EdgeType, get_cell_type_index

# 枚举名 -> 0-indexed 映射表
EDGE_TYPE_MAP = {e.name: e.value - 1 for e in EdgeType}
ASM_NODE_TYPE_MAP = {e.name: e.value - 1 for e in AsmNodeType}
ASM_EDGE_TYPE_MAP = {e.name: e.value - 1 for e in AsmEdgeType}
NODE_TYPE_MAP = {idx: nt for idx, nt in enumerate(CELL_TYPE_TO_NODE_TYPE)}


def make_edge_index(src: list, tgt: list) -> torch.Tensor:
    """构建 edge_index 张量，空列表时返回 shape=(2,0) 的空张量。"""
    if src:
        return torch.tensor([src, tgt], dtype=torch.long)
    return torch.empty((2, 0), dtype=torch.long)


def normalize_cell_type(cell_type: str) -> str:
    if cell_type == "input":
        return "INPUT"
    if cell_type == "output":
        return "OUTPUT"
    return cell_type


def build_rtl_structure(rtl: dict) -> dict:
    """构建 RTL 图结构张量。

    返回值额外包含 ``_valid_edge_mask``，用于标签提取时和过滤后的边保持对齐。
    """
    nodes = rtl["nodes"]
    node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    node_cell_type = [
        get_cell_type_index(normalize_cell_type(n["cell_type"])) for n in nodes
    ]
    node_width = [n["width"] for n in nodes]

    src_list, tgt_list = [], []
    etype_list, ewidth_list = [], []
    esrc_port_list, etgt_port_list = [], []
    valid_edge_mask: list[bool] = []

    for e in rtl["edges"]:
        si = node_id_to_idx.get(e["source"])
        ti = node_id_to_idx.get(e["target"])
        if si is None or ti is None:
            valid_edge_mask.append(False)
            continue
        valid_edge_mask.append(True)
        src_list.append(si)
        tgt_list.append(ti)
        etype_list.append(EDGE_TYPE_MAP.get(e["type"], 0))
        ewidth_list.append(e["width"])
        esrc_port_list.append(e["source_port_idx"])
        etgt_port_list.append(e["target_port_idx"])

    return {
        "node_cell_type": torch.tensor(node_cell_type, dtype=torch.long),
        "node_type": torch.tensor(
            [NODE_TYPE_MAP[ct] for ct in node_cell_type], dtype=torch.long
        ),
        "node_width": torch.tensor(node_width, dtype=torch.long),
        "edge_index": make_edge_index(src_list, tgt_list),
        "edge_type": torch.tensor(etype_list, dtype=torch.long),
        "edge_width": torch.tensor(ewidth_list, dtype=torch.long),
        "edge_source_port_idx": torch.tensor(esrc_port_list, dtype=torch.long),
        "edge_target_port_idx": torch.tensor(etgt_port_list, dtype=torch.long),
        "_valid_edge_mask": valid_edge_mask,
    }


def build_asm_graph(asm_json_path: str, asm_encoding: Optional[torch.Tensor]) -> dict:
    """构建 ASM 图张量。"""
    import json

    with open(asm_json_path, encoding="utf-8") as f:
        asm = json.load(f)

    asm_node_ids = list(asm["nodes"].keys())
    asm_id_to_idx = {nid: i for i, nid in enumerate(asm_node_ids)}

    asm_nt = [
        ASM_NODE_TYPE_MAP.get(
            asm["nodes"][nid]["node_type"],
            AsmNodeType.UNKNOWN.value - 1,
        )
        for nid in asm_node_ids
    ]

    asm_src, asm_tgt, asm_et = [], [], []
    for ae in asm["edges"]:
        s = asm_id_to_idx.get(ae["source"])
        t = asm_id_to_idx.get(ae["target"])
        if s is None or t is None:
            continue
        asm_src.append(s)
        asm_tgt.append(t)
        asm_et.append(ASM_EDGE_TYPE_MAP.get(ae["edge_type"], 0))

    asm_instr_enc = asm_encoding
    if asm_instr_enc is None:
        asm_instr_enc = torch.zeros(len(asm_node_ids), 256)

    return {
        "asm_node_type": torch.tensor(asm_nt, dtype=torch.long),
        "asm_instruction_encoding": asm_instr_enc,
        "asm_edge_index": make_edge_index(asm_src, asm_tgt),
        "asm_edge_type": torch.tensor(asm_et, dtype=torch.long),
    }


def empty_asm_graph() -> dict:
    """返回一个占位空 ASM 图。"""
    return {
        "asm_node_type": torch.zeros(1, dtype=torch.long),
        "asm_instruction_encoding": torch.zeros(1, 256),
        "asm_edge_index": torch.empty((2, 0), dtype=torch.long),
        "asm_edge_type": torch.empty(0, dtype=torch.long),
    }
