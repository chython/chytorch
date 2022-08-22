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
from random import random, choice
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Tuple, Union, List
from .molecule import *


class PermutedMoleculeDataset(Dataset):
    def __init__(self, molecules: List[Union[bytes, MoleculeContainer]], rate: float = .15,
                 distance_cutoff: int = 10,
                 add_cls: bool = True, symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 unpack: bool = False):
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
        self.distance_cutoff = distance_cutoff
        self.add_cls = add_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.unpack = unpack

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, item) -> Tuple[Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                               TensorType['atoms', 'atoms', int]],
                                         Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                               TensorType['atoms', 'atoms', int]]]:
        m1 = MoleculeContainer.unpack(self.molecules[item]) if self.unpack else self.molecules[item]
        m2 = m1.copy()

        potent = []
        for n, a in m1.atoms():
            if a.atomic_number == 6 and a.implicit_hydrogens and a.hybridization in (1, 4):
                potent.append(n)
                if random() < self.rate:
                    m2.add_bond(n, m2.add_atom(C()), 1)
        # at least 1 atom should be picked
        if len(m2) == len(m1) and potent:
            m2.add_bond(choice(potent), m2.add_atom(C()), 1)
        print(m1, m2)
        ms = MoleculeDataset([m1, m2], distance_cutoff=self.distance_cutoff, add_cls=self.add_cls,
                             symmetric_cls=self.symmetric_cls, unpack=self.unpack,
                             disable_components_interaction=self.disable_components_interaction)
        return ms[0], ms[1]


__all__ = ['PermutedMoleculeDataset']
