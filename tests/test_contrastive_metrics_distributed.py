from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from metrics.contrastive import ContrastiveMetrics


def _distributed_spearman_worker(
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
        predictions = (
            torch.tensor([0.1, 0.2])
            if rank == 0
            else torch.tensor([0.3, 0.4])
        )
        targets = (
            torch.tensor([0.1, 0.2])
            if rank == 0
            else torch.tensor([0.4, 0.3])
        )
        geometry = SimpleNamespace(
            volume_a=predictions[0].reshape(1, 1),
            volume_b=predictions[1].reshape(1, 1),
            iou=predictions.reshape(1, 2),
        )
        metrics = ContrastiveMetrics()
        metrics.collect(
            geometry=geometry,
            density_a=targets[0].reshape(1, 1),
            density_b=targets[1].reshape(1, 1),
            size_mask=torch.ones(1, 1, dtype=torch.bool),
            jaccard=targets.reshape(1, 2),
            iou_mask=torch.ones(1, 2, dtype=torch.bool),
        )

        result_queue.put(
            (rank, metrics.compute_epoch(device=torch.device("cpu")))
        )
    finally:
        dist.destroy_process_group()


def test_epoch_spearman_uses_global_ddp_samples(tmp_path: Path) -> None:
    world_size = 2
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    mp.spawn(
        _distributed_spearman_worker,
        args=(world_size, str(tmp_path / "distributed-init"), result_queue),
        nprocs=world_size,
        join=True,
    )

    results = dict(result_queue.get() for _ in range(world_size))
    assert set(results) == {0, 1}
    for result in results.values():
        assert result["volume_spearman"] == pytest.approx(0.8)
        assert result["iou_spearman"] == pytest.approx(0.8)
