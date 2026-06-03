#!/usr/bin/env python3
"""Tests for dynamic graph batch samplers."""

from __future__ import annotations

import torch

from datasets.data_types import DualGraphData
from datasets.samplers import RtlSizeBucketSampler


def _graph(rtl_nodes: int, asm_nodes: int) -> DualGraphData:
    return DualGraphData(
        node_cell_type=torch.zeros(rtl_nodes, dtype=torch.long),
        asm_node_type=torch.zeros(asm_nodes, dtype=torch.long),
    )


def test_bucket_sampler_budget_counts_rtl_and_asm_nodes() -> None:
    dataset = [
        _graph(rtl_nodes=2, asm_nodes=7),
        _graph(rtl_nodes=2, asm_nodes=7),
        _graph(rtl_nodes=2, asm_nodes=1),
    ]

    sampler = RtlSizeBucketSampler(dataset, token_budget=10, shuffle=False)

    assert list(sampler) == [[0], [1], [2]]


def test_bucket_sampler_shards_batches_across_distributed_ranks(monkeypatch) -> None:
    dataset = [_graph(rtl_nodes=1, asm_nodes=0) for _ in range(5)]

    monkeypatch.setattr(RtlSizeBucketSampler, "_distributed_info", lambda self: (2, 0))
    rank0 = list(RtlSizeBucketSampler(dataset, token_budget=1, shuffle=False))

    monkeypatch.setattr(RtlSizeBucketSampler, "_distributed_info", lambda self: (2, 1))
    rank1 = list(RtlSizeBucketSampler(dataset, token_budget=1, shuffle=False))

    assert len(rank0) == len(rank1)
    assert rank0 == [[0], [2], [4]]
    assert rank1 == [[1], [3], [0]]
