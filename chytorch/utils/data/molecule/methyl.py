# -*- coding: utf-8 -*-
#
# Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
