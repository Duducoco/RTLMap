#!/usr/bin/env python3
"""自定义 PyTorch 数据采样器。"""

import torch


class RtlSizeBucketSampler(torch.utils.data.Sampler):
    """按图节点数分桶的动态 batch sampler。"""

    _BUCKET_EDGES = (128, 512, 2048)

    def __init__(
        self,
        dataset,
        token_budget: int = 8192,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        sizes = self._scan_sizes(dataset)
        buckets: dict[int, list[int]] = {
            k: [] for k in range(len(self._BUCKET_EDGES) + 1)
        }
        for idx, n in enumerate(sizes):
            bkt = sum(n >= e for e in self._BUCKET_EDGES)
            buckets[bkt].append(idx)
        self._buckets = [v for v in buckets.values() if v]
        self._sizes = sizes

    @staticmethod
    def _scan_sizes(dataset) -> list[int]:
        sizes = []
        for i in range(len(dataset)):
            data = dataset[i]
            rtl_nodes = (
                int(data.node_cell_type.size(0))
                if hasattr(data, "node_cell_type")
                else 0
            )
            asm_nodes = (
                int(data.asm_node_type.size(0))
                if hasattr(data, "asm_node_type")
                else 0
            )
            n = rtl_nodes + asm_nodes
            sizes.append(max(n, 1))
        return sizes

    def _make_batches(self) -> list[list[int]]:
        rng = torch.Generator()
        rng.manual_seed(self.seed + self._epoch)
        batches = []
        for bucket in self._buckets:
            indices = list(bucket)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=rng).tolist()
                indices = [indices[p] for p in perm]
            batch, total = [], 0
            for idx in indices:
                n = self._sizes[idx]
                if batch and total + n > self.token_budget:
                    batches.append(batch)
                    batch, total = [], 0
                batch.append(idx)
                total += n
            if batch:
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=rng).tolist()
            batches = [batches[p] for p in perm]
        batches = self._shard_batches(batches)
        return batches

    @staticmethod
    def _distributed_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _shard_batches(self, batches: list[list[int]]) -> list[list[int]]:
        world_size, rank = self._distributed_info()
        if world_size <= 1 or not batches:
            return batches

        remainder = len(batches) % world_size
        if remainder:
            batches = batches + batches[: world_size - remainder]
        return batches[rank::world_size]

    def __iter__(self):
        self._cached_batches = self._make_batches()
        yield from self._cached_batches

    def __len__(self) -> int:
        if not hasattr(self, "_cached_batches"):
            self._cached_batches = self._make_batches()
        return len(self._cached_batches)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if hasattr(self, "_cached_batches"):
            del self._cached_batches
