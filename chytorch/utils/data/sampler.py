# -*- coding: utf-8 -*-
#
#  Copyright 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of chytorch.
#
#  chytorch is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from chython import MoleculeContainer, ReactionContainer
from collections import defaultdict
from functools import cached_property
from itertools import chain, islice
from torch import Generator, randperm
from torch.utils.data import DistributedSampler, Sampler
from typing import Optional, Union
from .molecule import MoleculeDataset
from .reaction import ReactionDataset


class Mixin:
    def __init__(self, *args, **kwargs):
        if True or isinstance(self.dataset, MoleculeDataset):
            if self.dataset.unpack:
                # 12 bit - atoms count
                self.sizes = [MoleculeContainer.pack_len(m) for m in self.dataset.molecules]
            else:
                self.sizes = [len(m) for m in self.dataset.molecules]
        elif isinstance(self.dataset, ReactionDataset):
            x = int(self.dataset.add_molecule_cls)
            if self.dataset.unpack:
                self.sizes = [sum(m + x for m in ReactionContainer.pack_len(r) for m in m)
                              for r in self.dataset.reactions]
            else:
                self.sizes = [sum(len(m) + x for m in r.molecules()) for r in self.dataset.reactions]
        else:
            raise TypeError
        super().__init__(*args, **kwargs)

    @cached_property
    def index_sequential(self):
        return self.indices(range(len(self.sizes)))

    @property
    def index_random(self):
        return self.indices(randperm(len(self.sizes), generator=self.generator).tolist())

    def indices(self, order):
        sizes = self.sizes
        batch_size = self.batch_size

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


class StructureSampler(Mixin, Sampler):
    def __init__(self, dataset: Union[MoleculeDataset, ReactionDataset], batch_size: int,
                 shuffle: bool = True, seed: int = 0):
        """
        Sample molecules or reactions locally grouped by size to reduce idle calculations on paddings.

        Example:
         [3, 4, 3, 3, 4, 5, 4] - sizes of molecules in dataset
         [0, 2, 3, 1, 4, 6, 5] - output indices for batch_size=3

        :param batch_size: expected batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        super().__init__(dataset)

    def __iter__(self):
        if self.shuffle:
            yield from self.index_random
        else:
            yield from self.index_sequential

    def __len__(self):
        return len(self.dataset)

    @property
    def generator(self):
        generator = Generator()
        generator.manual_seed(self.seed)
        return generator


class DistributedStructureSampler(Mixin, DistributedSampler):
    def __init__(self, dataset: MoleculeDataset, batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0):
        """
        Sample molecules locally grouped by size to reduce idle calculations on paddings.

        :param batch_size: expected batch size
        """
        self.dataset = dataset  # ad-hoc for correct mixin init
        self.batch_size = batch_size
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)

    def __iter__(self):
        if self.shuffle:
            indices = self.index_random
        else:
            indices = self.index_sequential

        rb = self.num_replicas * self.batch_size
        # number of dividable elements
        unique = len(indices) // rb * rb
        for s in range(self.rank * self.batch_size, unique, rb):
            yield from indices[s: s + self.batch_size]

        # check tail batches
        if tail_batches := (len(indices) - unique) // self.batch_size * self.batch_size:
            s = self.rank * self.batch_size
            if s < tail_batches:
                yield from indices[unique + s: unique + s + self.batch_size]
            else:  # reuse unique batch for fill
                # tail batch shift added for preventing duplicate batches
                yield from indices[s - tail_batches: s - tail_batches + self.batch_size]

        # check tail
        if diff := len(indices) % self.batch_size:
            if not self.rank:  # only first replica
                yield from indices[-diff:]
            else:  # reuse unique
                s = -self.rank * self.batch_size
                yield from indices[s - diff: s]

    @property
    def generator(self):
        generator = Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator


__all__ = ['StructureSampler', 'DistributedStructureSampler']
