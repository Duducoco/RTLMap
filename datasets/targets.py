#!/usr/bin/env python3
"""覆盖率监督目标解析工具。"""

import torch

from .data_types import COVERAGE_KEYS


def extract_targets_from_sample(
    sample: dict, valid_edge_mask: list[bool]
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = sample.get("targets", {})
    edge_target = targets.get("edge_coverage", {})
    default_label = int(edge_target.get("default", -1))
    labels_by_edge_id = [default_label] * len(valid_edge_mask)
    for item in edge_target.get("labels", []):
        edge_id = int(item["edge_id"])
        if 0 <= edge_id < len(labels_by_edge_id):
            labels_by_edge_id[edge_id] = int(item["label"])

    elabel_list = [
        label for label, valid in zip(labels_by_edge_id, valid_edge_mask) if valid
    ]

    graph_values = targets.get("graph_coverage", {}).get("values", {})
    y_vals = []
    for key in COVERAGE_KEYS:
        v = graph_values.get(key)
        y_vals.append(float("nan") if v is None else float(v) / 100.0)

    return (
        torch.tensor(elabel_list, dtype=torch.long),
        torch.tensor([y_vals], dtype=torch.float),
    )
