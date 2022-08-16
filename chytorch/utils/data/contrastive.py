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
from chython import MoleculeContainer
from math import sqrt
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Tuple, Union, List
from .molecule import *


def contrastive_collate(batch) -> Tuple[TensorType['2*batch', 'atoms', int], TensorType['2*batch', 'atoms', int],
                                        TensorType['2*batch', 'atoms', 'atoms', int]]:
    """
    Prepares batches of contrastive molecules. First B elements are first elements of pairs, Second > second.

    :return: see collate_molecules
    """
    flat = [x for x, _ in batch]
    flat.extend(x for _, x in batch)
    return collate_molecules(flat)


class ContrastiveDataset(Dataset):
    def __init__(self, data: List[List[Union[bytes, MoleculeContainer]]], distance_cutoff: int = 10,
                 add_cls: bool = True, symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 unpack: bool = False):
        """
        Prepare pairs of similar molecules.
        For multiple similar molecules this dataset enumerate all possible pairs.
        For single element in list molecule returned twice.

        :param data: list of lists of similar (by any means) molecules.

        See MoleculeDataset for other params description.
        """
        self.data = data
        self.distance_cutoff = distance_cutoff
        self.add_cls = add_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.unpack = unpack

        self.total = total = []
        for i, x in enumerate(data):
            if (n := len(x)) <= 2:  # unique mol
                total.append((i, 0))
            else:
                total.extend((i, x) for x in range(n * (n - 1) // 2))

    def __len__(self):
        """
        Number of possible pairs
        """
        return len(self.total)

    def __getitem__(self, item) -> Tuple[Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                               TensorType['atoms', 'atoms', int]],
                                         Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                               TensorType['atoms', 'atoms', int]]]:
        i, p = self.total[item]
        mols = self.data[i]
        if (n := len(mols)) == 1:  # no pairs
            m = MoleculeDataset([MoleculeContainer.unpack(mols[0])], distance_cutoff=self.distance_cutoff,
                                add_cls=self.add_cls, symmetric_cls=self.symmetric_cls, unpack=self.unpack,
                                disable_components_interaction=self.disable_components_interaction)[0]
            return m, m
        elif n == 2:
            m1, m2 = mols
        elif p < n - 1:
            m1 = mols[0]
            m2 = mols[p + 1]
        else:
            # @Andrey suggestion
            # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
            nd = n - 1
            m1 = n - 2 - int(sqrt(4 * n * nd - 8 * p - 7) / 2 - .5)
            m2 = m1 + p + 1 - n * nd // 2 + (n - m1) * (nd - m1) // 2
            m1 = mols[m1]
            m2 = mols[m2]

        ms = MoleculeDataset([m1, m2], distance_cutoff=self.distance_cutoff, add_cls=self.add_cls,
                             symmetric_cls=self.symmetric_cls, unpack=self.unpack,
                             disable_components_interaction=self.disable_components_interaction)
        return ms[0], ms[1]


__all__ = ['ContrastiveDataset', 'contrastive_collate']
