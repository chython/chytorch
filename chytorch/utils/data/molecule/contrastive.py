# -*- coding: utf-8 -*-
#
#  Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from random import choice, random
from torch import Size
from torch.utils.data import Dataset
from typing import Union, Sequence
from .encoder import MoleculeDataset, MoleculeDataPoint, MoleculeDataBatch, collate_molecules
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class ContrastiveMethylDataPoint(NamedTuple):
    first: MoleculeDataPoint
    second: MoleculeDataPoint


class ContrastiveMethylDataBatch(NamedTuple, DataTypeMixin):
    first: MoleculeDataBatch
    second: MoleculeDataBatch


def contrastive_methyl_collate(batch, *, collate_fn_map=None) -> ContrastiveMethylDataBatch:
    """
    Prepares batches of contrastive molecules
    """
    first, second = [], []
    for f, s in batch:
        first.append(f)
        second.append(s)
    return ContrastiveMethylDataBatch(collate_molecules(first), collate_molecules(second))


default_collate_fn_map[ContrastiveMethylDataPoint] = contrastive_methyl_collate  # add auto_collation to the DataLoader


class ContrastiveMethylDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[bytes, MoleculeContainer]], *, rate: float = .15,
                 max_distance: int = 10, add_cls: bool = True, max_neighbors: int = 14, unpack: bool = False):
        """
        Prepare pairs of "similar" molecules.
        First molecule returns as is, second with randomly replaced by methyl carbon/nitrogen's implicit hydrogen atom.

        :param molecules: list of molecules.
        :param rate: probability of replacement.

        See MoleculeDataset for other params description.
        """
        self.molecules = molecules
        self.rate = rate
        self.max_distance = max_distance
        self.add_cls = add_cls
        self.max_neighbors = max_neighbors
        self.unpack = unpack

    def __getitem__(self, item) -> ContrastiveMethylDataPoint:
        m1 = MoleculeContainer.unpack(self.molecules[item]) if self.unpack else self.molecules[item]
        m2 = m1.copy()
        hgs = m2._hydrogens  # noqa

        potent = []
        for n, a in m1.atoms():
            if hgs[n] and a.atomic_number in (6, 7):  # only C,N atoms with hydrogens
                potent.append(n)
                if random() < self.rate:
                    m = m2.add_atom(C(), _skip_hydrogen_calculation=True)
                    m2.add_bond(n, m, 1, _skip_hydrogen_calculation=True)
                    hgs[n] -= 1
                    hgs[m] = 3  # CH3
        # at least 1 atom should be picked
        if len(m2) == len(m1) and potent:
            n = choice(potent)
            m = m2.add_atom(C(), _skip_hydrogen_calculation=True)
            m2.add_bond(n, m, 1, _skip_hydrogen_calculation=True)
            hgs[n] -= 1
            hgs[m] = 3  # CH3
        ms = MoleculeDataset([m1, m2], max_distance=self.max_distance, add_cls=self.add_cls,
                             max_neighbors=self.max_neighbors)
        return ContrastiveMethylDataPoint(ms[0], ms[1])

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['ContrastiveMethylDataset', 'ContrastiveMethylDataPoint', 'ContrastiveMethylDataBatch',
           'contrastive_methyl_collate']
