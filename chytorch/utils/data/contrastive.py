# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from chython import MoleculeContainer
from itertools import groupby, combinations
from operator import itemgetter
from torch import Size
from torch.utils.data import Dataset
from typing import Union, Sequence, Any, Callable
from ._utils import DataTypeMixin, NamedTuple


class ContrastiveDataPoint(NamedTuple):
    first: Any
    second: Any


class ContrastiveDataBatch(NamedTuple, DataTypeMixin):
    first: Any
    second: Any


def contrastive_collate(collate_fn: Callable) -> Callable:
    """
    Constructor of contrastive collate function. Prepares batches of contrastive molecules.
    """
    def w(batch, *, collate_fn_map=None) -> ContrastiveDataBatch:
        first, second = [], []
        for f, s in batch:
            first.append(f)
            second.append(s)
        return ContrastiveDataBatch(collate_fn(first), collate_fn(second))
    return w


class ContrastiveDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[bytes, MoleculeContainer]], groups: Sequence[int], *,
                 unpack: bool = False):
        """
        Prepare pairs of "similar" molecules from predefined list of groups.
        For multiple similar molecules this dataset enumerate all possible pairs.
        For single element in list molecule returned twice.

        :param molecules: Sequence of molecules.
        :param groups: Sequence of similar group (by any means) labels.
        :param unpack: unpack MoleculeContainer from bytes.
        """
        assert len(molecules) == len(groups), 'Molecules and groups set should be the same size'
        self.molecules = molecules
        self.unpack = unpack

        self.total = total = []
        for _, g in groupby(enumerate(groups), key=itemgetter(1)):
            g = [x for x, _ in g]
            if (n := len(g)) == 1:  # unique mol
                total.append((g[0], g[0]))
            elif n == 2:
                total.append(tuple(g))
            else:
                for x in combinations(g, 2):
                    total.append(x)

    def __getitem__(self, item) -> ContrastiveDataPoint:
        m1, m2 = self.total[item]
        m1, m2 = self.molecules[m1], self.molecules[m2]
        if self.unpack:
            m1, m2 = MoleculeContainer.unpack(m1), MoleculeContainer.unpack(m2)
        return ContrastiveDataPoint(m1, m2)

    def __len__(self):
        """
        Number of possible pairs
        """
        return len(self.total)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['ContrastiveDataset', 'ContrastiveDataPoint', 'ContrastiveDataBatch', 'contrastive_collate']
