#!/usr/bin/env python3
"""自定义 PyTorch 数据采样器。"""

import math

import torch


class RankingAwarePairBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Spread the full Coverage Similarity ranking across every pair batch."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        world_size, distributed_rank = self._distributed_info()
        self.num_replicas = world_size if num_replicas is None else num_replicas
        self.rank = distributed_rank if rank is None else rank
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        if (
            not self.drop_last
            and len(self.dataset) > 0
            and len(self.dataset) // self.num_replicas < self._batch_count()
        ):
            raise ValueError(
                "dataset is too small for equal-step unpadded DDP pair batches"
            )

    @staticmethod
    def _distributed_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _rank_indices(self) -> list[int]:
        records = self.dataset.pair_records
        if len(records) != len(self.dataset):
            raise ValueError("pair_records must align with the pair dataset")
        ordered = sorted(
            range(len(records)),
            key=lambda index: (records[index].coverage_similarity, index),
        )
        if self.drop_last:
            total_size = self._batch_count() * self.batch_size * self.num_replicas
            ordered = ordered[:total_size]
        return ordered[self.rank :: self.num_replicas]

    def _batch_count(self) -> int:
        global_batch_size = self.batch_size * self.num_replicas
        if self.drop_last:
            return len(self.dataset) // global_batch_size
        return math.ceil(len(self.dataset) / global_batch_size)

    def _make_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        ranked_indices = self._rank_indices()
        if not ranked_indices:
            return []
        batch_count = self._batch_count()
        if len(ranked_indices) < batch_count:
            raise ValueError(
                "each DDP rank must have at least one pair per ranking-aware batch"
            )
        batches: list[list[int]] = [[] for _ in range(batch_count)]
        for position, index in enumerate(ranked_indices):
            sweep, offset = divmod(position, batch_count)
            batch_index = offset if sweep % 2 == 0 else batch_count - 1 - offset
            batches[batch_index].append(index)
        if self.shuffle:
            for batch_index, batch in enumerate(batches):
                permutation = torch.randperm(
                    len(batch), generator=generator
                ).tolist()
                batches[batch_index] = [batch[position] for position in permutation]
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in permutation]
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self) -> int:
        return self._batch_count()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


class DistributedEvalBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Shard evaluation data without the duplicate padding of DistributedSampler."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        world_size, distributed_rank = RankingAwarePairBatchSampler._distributed_info()
        self.num_replicas = world_size if num_replicas is None else num_replicas
        self.rank = distributed_rank if rank is None else rank
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        if (
            self.dataset_size > 0
            and self.dataset_size // self.num_replicas < len(self)
        ):
            raise ValueError(
                "dataset is too small for equal-step unpadded DDP evaluation"
            )

    def __len__(self) -> int:
        return math.ceil(
            self.dataset_size / (self.batch_size * self.num_replicas)
        )

    def __iter__(self):
        batch_count = len(self)
        if batch_count == 0:
            return
        local_indices = list(range(self.rank, self.dataset_size, self.num_replicas))
        if len(local_indices) < batch_count:
            raise ValueError(
                "each DDP rank must have at least one sample per evaluation batch"
            )
        batches: list[list[int]] = [[] for _ in range(batch_count)]
        for position, index in enumerate(local_indices):
            batches[position % batch_count].append(index)
        yield from batches


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
                int(data.asm_node_type.size(0)) if hasattr(data, "asm_node_type") else 0
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
