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
from pathlib import Path
from pickle import load, dump
from torch import Generator, randperm
from torch.utils.data import DistributedSampler, Sampler
from typing import Optional, Union
from .molecule import MoleculeDataset, ContrastiveMethylDataset
from .reaction import ReactionEncoderDataset, ReactionDecoderDataset, PermutedReactionDataset


class Mixin:
    def __init__(self, cache, validate, *args, **kwargs):
        if cache is not None:
            if isinstance(cache, str):
                cache = Path(cache)
            if cache.exists():
                if not isinstance(self.dataset, (MoleculeDataset, ContrastiveMethylDataset, ReactionEncoderDataset,
                                                 ReactionDecoderDataset, PermutedReactionDataset)):
                    raise TypeError('Unsupported Dataset')
                # load existing cache
                with cache.open('rb') as f:
                    sizes = load(f)
                assert isinstance(sizes, list), 'Sampler cache invalid'
                assert not validate or len(sizes) == len(self.dataset), 'Sampler cache size mismatch'
                self.sizes = sizes
                super().__init__(*args, **kwargs)
                return

        if isinstance(self.dataset, (MoleculeDataset, ContrastiveMethylDataset)):
            ds = self.dataset.molecules  # map-like data. not iterable.
            if self.dataset.unpack:
                # 12 bit - atoms count
                self.sizes = [MoleculeContainer.pack_len(ds[m]) for m in range(len(ds))]
            else:
                self.sizes = [len(ds[m]) for m in range(len(ds))]
        elif isinstance(self.dataset, ReactionEncoderDataset):
            ds = self.dataset.reactions
            x = int(self.dataset.add_molecule_cls)
            if self.dataset.unpack:
                self.sizes = sizes = []
                for r in range(len(ds)):
                    rs, _, ps = ReactionContainer.pack_len(ds[r])
                    sizes.append(sum(m + x for m in rs) + sum(m + x for m in ps))
            else:
                self.sizes = [sum(len(m) + x for m in r.reactants) + sum(len(m) + x for m in r.products)
                              for r in (ds[r] for r in range(len(ds)))]
        elif isinstance(self.dataset, (ReactionDecoderDataset, PermutedReactionDataset)):
            ds = self.dataset.reactions
            x = int(self.dataset.add_molecule_cls)
            y = int(self.dataset.add_cls)
            if self.dataset.unpack:
                self.sizes = sizes = []
                for r in range(len(ds)):
                    rs, _, ps = ReactionContainer.pack_len(ds[r])
                    sizes.append(max(sum(m + x for m in rs), sum(m + x for m in ps) + y))
            else:
                self.sizes = [max(sum(len(m) + x for m in r.reactants),
                                  sum(len(m) + x for m in r.products) + y)
                              for r in (ds[r] for r in range(len(ds)))]
        else:
            raise TypeError('Unsupported Dataset')

        if cache is not None:
            # save cache
            with cache.open('wb') as f:
                dump(self.sizes, f)
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
    def __init__(self, dataset: Union[MoleculeDataset, ContrastiveMethylDataset, ReactionEncoderDataset,
                                      ReactionDecoderDataset, PermutedReactionDataset],
                 batch_size: int, shuffle: bool = True, seed: int = 0, *,
                 cache: Union[Path, str, None] = None, validate_cache: bool = True):
        """
        Sample molecules or reactions locally grouped by size to reduce idle calculations on paddings.

        Example:
         [3, 4, 3, 3, 4, 5, 4] - sizes of molecules in dataset
         [0, 2, 3, 1, 4, 6, 5] - output indices for batch_size=3

        :param batch_size: expected batch size
        :param cache: path to cache file for [re]storing size index. caching disabled by default.
        :param validate_cache: check cache-dataset size mismatch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        super().__init__(cache, validate_cache, dataset)

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
    def __init__(self, dataset: Union[MoleculeDataset, ContrastiveMethylDataset, ReactionEncoderDataset,
                                      ReactionDecoderDataset, PermutedReactionDataset],
                 batch_size: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, *,
                 cache: Union[Path, str, None] = None, validate_cache: bool = True):
        """
        Sample molecules locally grouped by size to reduce idle calculations on paddings.

        :param batch_size: expected batch size
        :param cache: path to cache file for [re]storing size index. caching disabled by default.
        :param validate_cache: check cache-dataset size mismatch
        """
        self.dataset = dataset  # ad-hoc for correct mixin init
        self.batch_size = batch_size
        super().__init__(cache, validate_cache, dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                         seed=seed)

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
