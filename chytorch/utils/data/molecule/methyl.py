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


class AttachedMethylDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[bytes, MoleculeContainer]], *,
                 rate: float = .15, unpack: bool = False, compressed: bool = True):
        """
        Do random replacements of hydrogens to methyl group of carbon/nitrogen atoms.

        :param molecules: list of molecules.
        :param rate: probability of replacement.
        :param unpack: unpacked packed molecule.
        :param compressed: packed molecules are compressed
        """
        self.molecules = molecules
        self.rate = rate
        self.unpack = unpack
        self.compressed = compressed

    def __getitem__(self, item: int) -> MoleculeContainer:
        mol = self.molecules[item]
        if self.unpack:
            mol = MoleculeContainer.unpack(mol, compressed=self.compressed)
        else:
            mol = mol.copy()
        hgs = mol._hydrogens  # noqa

        potent = []
        changed = False
        for n, a in mol.atoms():
            if hgs[n] and a.atomic_number in (6, 7):  # only C,N atoms with hydrogens
                potent.append(n)
                if random() < self.rate:
                    m = mol.add_atom(C(), _skip_hydrogen_calculation=True)
                    mol.add_bond(n, m, 1, _skip_hydrogen_calculation=True)
                    hgs[n] -= 1
                    hgs[m] = 3  # CH3
                    changed = True
        # at least 1 atom should be picked
        if not changed and potent:
            n = choice(potent)
            m = mol.add_atom(C(), _skip_hydrogen_calculation=True)
            mol.add_bond(n, m, 1, _skip_hydrogen_calculation=True)
            hgs[n] -= 1
            hgs[m] = 3  # CH3
        return mol

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['AttachedMethylDataset']
