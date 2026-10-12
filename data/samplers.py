"""
Semi-Supervised Batch Samplers for Medical Image Segmentation.
Includes:
- TwoStreamBatchSampler: Balanced interleaving of labeled and unlabeled data streams.
"""

from typing import List, Iterator, Tuple
import itertools
import numpy as np
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
