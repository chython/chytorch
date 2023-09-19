# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
