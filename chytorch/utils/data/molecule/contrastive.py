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
from chython.periodictable import C
from math import sqrt
from random import choice, random
from torch import Size
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from torchtyping import TensorType
from typing import Union, List
from .encoder import MoleculeDataset, MoleculeDataPoint, collate_molecules
from .._types import DataTypeMixin, NamedTuple


class ContrastiveDataPoint(NamedTuple):
    first: MoleculeDataPoint
    second: MoleculeDataPoint


class ContrastiveDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['2*batch', 'atoms', int]
    neighbors: TensorType['2*batch', 'atoms', int]
    distances: TensorType['2*batch', 'atoms', 'atoms', int]


def contrastive_collate(batch, *, collate_fn_map=None) -> ContrastiveDataBatch:
    """
    Prepares batches of contrastive molecules. First B elements are first elements of pairs, Second > second.
    """
    flat = [x for x, _ in batch]
    flat.extend(x for _, x in batch)
    return ContrastiveDataBatch(*collate_molecules(flat))


default_collate_fn_map[ContrastiveDataPoint] = contrastive_collate  # add auto_collation to the DataLoader


class ContrastiveDataset(Dataset):
    def __init__(self, molecules: List[List[Union[bytes, MoleculeContainer]]], *, max_distance: int = 10,
                 add_cls: bool = True, symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 unpack: bool = False):
        """
        Prepare pairs of "similar" molecules from predefined list of groups.
        For multiple similar molecules this dataset enumerate all possible pairs.
        For single element in list molecule returned twice.

        :param molecules: list of lists of similar (by any means) molecules.

        See MoleculeDataset for other params description.
        """
        self.molecules = molecules
        self.max_distance = max_distance
        self.add_cls = add_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.unpack = unpack

        self.total = total = []
        for i, x in enumerate(molecules):
            if (n := len(x)) <= 2:  # unique mol
                total.append((i, 0))
            else:
                total.extend((i, x) for x in range(n * (n - 1) // 2))

    def __getitem__(self, item) -> ContrastiveDataPoint:
        i, p = self.total[item]
        mols = self.molecules[i]
        if (n := len(mols)) == 1:  # no pairs
            m = MoleculeDataset(mols, max_distance=self.max_distance,
                                add_cls=self.add_cls, symmetric_cls=self.symmetric_cls, unpack=self.unpack,
                                disable_components_interaction=self.disable_components_interaction)[0]
            return ContrastiveDataPoint(m, m)
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

        ms = MoleculeDataset([m1, m2], max_distance=self.max_distance, add_cls=self.add_cls,
                             symmetric_cls=self.symmetric_cls, unpack=self.unpack,
                             disable_components_interaction=self.disable_components_interaction)
        return ContrastiveDataPoint(ms[0], ms[1])

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


class ContrastiveMethylDataset(Dataset):
    def __init__(self, molecules: List[Union[bytes, MoleculeContainer]], *, rate: float = .15,
                 max_distance: int = 10, add_cls: bool = True, symmetric_cls: bool = True,
                 disable_components_interaction: bool = False, unpack: bool = False):
        """
        Prepare pairs of "similar" molecules.
        First molecule returns as is, second with randomly replaced by methyl aliphatic/aromatic carbon hydrogen.

        This dataset usable for contrastive learning.

        :param molecules: list of molecules.
        :param rate: probability of replacement.

        See MoleculeDataset for other params description.
        """
        self.molecules = molecules
        self.rate = rate
        self.max_distance = max_distance
        self.add_cls = add_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.unpack = unpack

    def __getitem__(self, item) -> ContrastiveDataPoint:
        m1 = MoleculeContainer.unpack(self.molecules[item]) if self.unpack else self.molecules[item]
        m2 = m1.copy()
        hgs = m2._hydrogens  # noqa

        potent = []
        for n, a in m1.atoms():
            if a.atomic_number == 6 and hgs[n] and a.hybridization in (1, 4):
                potent.append(n)
                if random() < self.rate:
                    m2.add_bond(n, m2.add_atom(C()), 1)
                    if hgs[n] is None:  # aryl
                        hgs[n] = 0
        # at least 1 atom should be picked
        if len(m2) == len(m1) and potent:
            n = choice(potent)
            m2.add_bond(n, m2.add_atom(C()), 1)
            if hgs[n] is None:  # aryl
                hgs[n] = 0
        ms = MoleculeDataset([m1, m2], max_distance=self.max_distance, add_cls=self.add_cls,
                             symmetric_cls=self.symmetric_cls,
                             disable_components_interaction=self.disable_components_interaction)
        return ContrastiveDataPoint(ms[0], ms[1])

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['ContrastiveDataset', 'ContrastiveMethylDataset', 'ContrastiveDataPoint', 'ContrastiveDataBatch',
           'contrastive_collate']
