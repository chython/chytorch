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
from itertools import groupby, combinations
from operator import itemgetter
from torch import Size
from torch.utils.data import Dataset
from typing import TypeVar, Sequence, Tuple


element = TypeVar('element')


class CombinationsDataset(Dataset):
    def __init__(self, data: Sequence[element], groups: Sequence[int]):
        """
        Prepare pairs of entities from predefined list of groups.
        For multiple elements in group this dataset enumerate all possible pairs.
        Ignores single-element groups.

        :param data: Sequence of entities.
        :param groups: Sequence of group labels.
        """
        assert len(data) == len(groups), 'Molecules and groups set should be the same size'
        self.data = data

        self.total = total = []
        for _, g in groupby(enumerate(groups), key=itemgetter(1)):
            g = [x for x, _ in g]
            if (n := len(g)) == 1:  # skip single element groups
                continue
            elif n == 2:
                total.append(tuple(g))
            else:
                for x in combinations(g, 2):
                    total.append(x)

    def __getitem__(self, item) -> Tuple[element, element]:
        m1, m2 = self.total[item]
        return self.data[m1], self.data[m2]

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


__all__ = ['CombinationsDataset']
