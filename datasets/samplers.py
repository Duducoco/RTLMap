#!/usr/bin/env python3
"""自定义 PyTorch 数据采样器。"""

import torch


class RtlSizeBucketSampler(torch.utils.data.Sampler):
    """按 RTL 图节点数分桶的动态 batch sampler。"""

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
            n = int(data.node_cell_type.size(0)) if hasattr(data, "node_cell_type") else 1
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
        return batches

    def __iter__(self):
        self._cached_batches = self._make_batches()
        yield from self._cached_batches

    def __len__(self) -> int:
        if not hasattr(self, "_cached_batches"):
            self._cached_batches = self._make_batches()
        return len(self._cached_batches)
