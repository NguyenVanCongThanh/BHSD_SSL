"""
Semi-Supervised Batch Samplers for Medical Image Segmentation.
Includes:
- TwoStreamBatchSampler: Balanced interleaving of labeled and unlabeled data streams.
"""

from typing import List, Iterator, Tuple, Optional
import itertools
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


def iterate_once(iterable: List[int]) -> np.ndarray:
    """Random permutation of indices once."""
    return np.random.permutation(iterable)


def iterate_eternally(indices: List[int]) -> Iterator[int]:
    """Infinite reshuffling iterator over indices."""
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n: int):
    """Collect data into fixed-length chunks or blocks of size n."""
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """
    Iterate two sets of indices (Primary = Labeled, Secondary = Unlabeled).
    An 'epoch' is defined as one full iteration through the primary indices.
    The secondary indices are iterated eternally as many times as needed.
    """
    def __init__(
        self,
        primary_indices: List[int],
        secondary_indices: List[int],
        batch_size: int,
        secondary_batch_size: int
    ):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0, (
            f"Primary indices count ({len(self.primary_indices)}) must be >= "
            f"primary batch size ({self.primary_batch_size}) > 0"
        )
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0, (
            f"Secondary indices count ({len(self.secondary_indices)}) must be >= "
            f"secondary batch size ({self.secondary_batch_size}) > 0"
        )

    def __iter__(self) -> Iterator[List[int]]:
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            list(primary_batch + secondary_batch)
            for (primary_batch, secondary_batch)
            in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size)
            )
        )

    def __len__(self) -> int:
        return len(self.primary_indices) // self.primary_batch_size


class DistributedTwoStreamBatchSampler(Sampler):
    """
    Distributed version of TwoStreamBatchSampler for DDP training.
    Partitions primary and secondary indices evenly across ranks.
    Batch sizes are LOCAL to each rank; the trainer resolves global sizes.
    Ensures each GPU worker processes an independent subset of the dataset.
    """
    def __init__(
        self,
        primary_indices: List[int],
        secondary_indices: List[int],
        batch_size: int,
        secondary_batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 1337
    ):
        if num_replicas is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        # Subsample indices for this rank
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert self.primary_batch_size > 0, "Primary batch size must be > 0"
        assert self.secondary_batch_size > 0, "Secondary batch size must be > 0"

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic reshuffling across iterations."""
        self.epoch = epoch

    def _get_rank_indices(self, indices: List[int], shuffle: bool = True) -> List[int]:
        if shuffle:
            g = np.random.RandomState(self.seed + self.epoch)
            shuffled = g.permutation(indices).tolist()
        else:
            shuffled = list(indices)
        # Partition among replicas
        # Drop the global remainder so every rank yields the same batch count.
        usable = len(shuffled) // self.num_replicas * self.num_replicas
        return shuffled[self.rank:usable:self.num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        rank_primary = self._get_rank_indices(self.primary_indices, shuffle=True)
        rank_secondary = self._get_rank_indices(self.secondary_indices, shuffle=True)

        # Fallback if partition size is smaller than batch size
        if len(rank_primary) < self.primary_batch_size:
            rank_primary = self.primary_indices[:self.primary_batch_size]
        if len(rank_secondary) < self.secondary_batch_size:
            rank_secondary = self.secondary_indices[:self.secondary_batch_size]

        primary_iter = iterate_once(rank_primary)
        secondary_iter = iterate_eternally(rank_secondary)

        return (
            list(primary_batch + secondary_batch)
            for (primary_batch, secondary_batch)
            in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size)
            )
        )

    def __len__(self) -> int:
        num_primary = len(self.primary_indices) // self.num_replicas
        return max(1, num_primary // self.primary_batch_size)

