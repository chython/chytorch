# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import torch.distributed as dist
from chython import MoleculeContainer
from collections import defaultdict
from itertools import chain, islice
from math import ceil
from torch import Generator, randperm
from torch.utils.data import Sampler
from typing import Optional, List, Iterator
from .molecule import MoleculeDataset


def _build_index(dataset, sizes):
    if not isinstance(dataset, MoleculeDataset):
        raise TypeError('Unsupported Dataset')
    if sizes is not None:
        return sizes
    elif dataset.unpack:
        return [MoleculeContainer.pack_len(m) for m in dataset.molecules]
    else:
        return [len(m) for m in dataset.molecules]


def _indices(order, sizes, batch_size):
    groups = defaultdict(list)
    for n in order:
        groups[sizes[n]].append(n)

    iterable_groups = {k: iter(v) for k, v in groups.items()}
    sorted_groups = sorted(groups)
    chained_groups = {}
    for k in groups:
        chained_groups[k] = chain(*(iterable_groups[x] for x in sorted_groups[sorted_groups.index(k)::-1]),
                                  *(iterable_groups[x] for x in sorted_groups[sorted_groups.index(k) + 1:]))

    indices = []
    seen = set()
    for n in order:
        if n not in seen:
            for x in islice(chained_groups[sizes[n]], batch_size):
                indices.append(x)
                if x != n:
                    seen.add(x)
            if len(indices) == len(sizes):
                break
        else:
            seen.discard(n)
    return indices


class StructureSampler(Sampler[List[int]]):
    def __init__(self, dataset: MoleculeDataset, batch_size: int, shuffle: bool = True, seed: int = 0, *,
                 sizes: Optional[List[int]] = None):
        """
        Sample molecules locally grouped by size to reduce idle calculations on paddings.

        Example:
         [3, 4, 3, 3, 4, 5, 4] - sizes of molecules in dataset
         [[0, 2, 3], [1, 4, 6], [5]] - output indices for batch_size=3

        :param batch_size: batch size
        :param sizes: precalculated sizes of molecules.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.sizes = _build_index(dataset, sizes)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            generator = Generator()
            generator.manual_seed(self.seed)
            index = _indices(randperm(len(self.sizes), generator=generator).tolist(), self.sizes, self.batch_size)
        else:
            index = _indices(range(len(self.sizes)), self.sizes, self.batch_size)

        index = iter(index)
        while batch := list(islice(index, self.batch_size)):
            yield batch

    def __len__(self):
        return ceil(len(self.sizes) / self.batch_size)


class DistributedStructureSampler(Sampler[List[int]]):
    def __init__(self, dataset: MoleculeDataset, batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, *,
                 sizes: Optional[List[int]] = None):
        """
        Sample molecules locally grouped by size to reduce idle calculations on paddings.

        :param batch_size: expected batch size
        :param sizes: precalculated sizes of molecules.
        :param num_replicas, rank, shuffle, seed: see torch.utils.data.DistributedSampler for details.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.sizes = _build_index(dataset, sizes)

        # adapted from torch/utils/data/distributed.py
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f'Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]')

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = ceil(len(self.sizes) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __len__(self) -> int:
        return ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            generator = Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = _indices(randperm(len(self.sizes), generator=generator).tolist(), self.sizes, self.batch_size)
        else:
            indices = _indices(range(len(self.sizes)), self.sizes, self.batch_size)

        # adapted from torch/utils/data/distributed.py
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        indices = iter(indices)
        while batch := list(islice(indices, self.batch_size)):
            yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


__all__ = ['StructureSampler', 'DistributedStructureSampler']
