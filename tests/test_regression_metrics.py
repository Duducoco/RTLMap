#!/usr/bin/env python3
"""Tests for full-dataset coverage regression metrics."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from metrics.regression import CoverageRegressionMetrics


def _distributed_regression_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        metrics = CoverageRegressionMetrics()
        if rank == 0:
            predictions = torch.tensor([[0.0], [1.0]])
            targets = torch.tensor([[0.0], [0.0]])
        else:
            predictions = torch.tensor([[2.0], [3.0]])
            targets = torch.tensor([[2.0], [3.0]])
        metrics.collect_epoch(predictions, targets)
        result = metrics.compute_epoch(
            device=torch.device("cpu"), coverage_keys=("branch",)
        )
        result_queue.put((rank, {key: float(value) for key, value in result.items()}))
    finally:
        dist.destroy_process_group()


def test_epoch_r2_is_computed_from_all_validation_samples() -> None:
    metrics = CoverageRegressionMetrics()

    # The first batch has zero target variance. Computing R² per batch would
    # produce a huge negative value, while the complete validation set has R² > 0.
    metrics.collect_epoch(
        torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [0.0]])
    )
    metrics.collect_epoch(
        torch.tensor([[2.0], [3.0]]), torch.tensor([[2.0], [3.0]])
    )

    result = metrics.compute_epoch(
        device=torch.device("cpu"), coverage_keys=("branch",)
    )

    assert math.isclose(float(result["graph_r2"]), 0.8518519, rel_tol=1e-6)
    assert math.isclose(float(result["graph_r"]), 0.9467293, rel_tol=1e-6)


def test_epoch_r2_is_finite_for_imperfect_constant_targets() -> None:
    metrics = CoverageRegressionMetrics()
    metrics.collect_epoch(
        torch.tensor([[0.4], [0.6]]), torch.tensor([[0.5], [0.5]])
    )

    result = metrics.compute_epoch(
        device=torch.device("cpu"), coverage_keys=("branch",)
    )

    assert float(result["graph_r2"]) == 0.0


def test_epoch_r2_is_one_for_perfect_constant_targets() -> None:
    metrics = CoverageRegressionMetrics()
    metrics.collect_epoch(
        torch.tensor([[0.5], [0.5]]), torch.tensor([[0.5], [0.5]])
    )

    result = metrics.compute_epoch(
        device=torch.device("cpu"), coverage_keys=("branch",)
    )

    assert float(result["graph_r2"]) == 1.0


def test_epoch_r2_uses_samples_from_every_ddp_rank(tmp_path: Path) -> None:
    world_size = 2
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    mp.spawn(
        _distributed_regression_worker,
        args=(world_size, str(tmp_path / "distributed-init"), result_queue),
        nprocs=world_size,
        join=True,
    )

    results = dict(result_queue.get() for _ in range(world_size))
    assert set(results) == {0, 1}
    for result in results.values():
        assert math.isclose(result["graph_r2"], 0.8518519, rel_tol=1e-6)
        assert math.isclose(result["graph_r"], 0.9467293, rel_tol=1e-6)
