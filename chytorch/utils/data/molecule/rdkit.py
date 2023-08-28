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
from numpy import array
from random import choice
from torch import Size
from torch.utils.data import Dataset
from typing import Sequence, Optional, Dict


def generate(smiles, num_conf=10, max_attempts=100, prune=.2):
    from rdkit.Chem import MolFromSmiles, AddHs, RemoveAllHs
    from rdkit.Chem.AllChem import EmbedMultipleConfs

    m = MolFromSmiles(smiles)
    m = AddHs(m)
    EmbedMultipleConfs(m, numConfs=num_conf, maxAttempts=max_attempts, pruneRmsThresh=prune)
    m = RemoveAllHs(m)
    a = array([x.GetAtomicNum() for x in m.GetAtoms()])
    h = array([x.GetNumExplicitHs() for x in m.GetAtoms()])
    return [(a, h, c.GetPositions()) for c in m.GetConformers()]


class RDKitConformerDataset(Dataset):
    """
    Random conformer generator dataset
    """
    def __init__(self, molecules: Sequence[str], num_conf=10, max_attempts=100, prune=.2,
                 cache: Optional[Dict[int, Sequence]] = None):
        """
        :param molecules: list of molecules' SMILES strings
        :param num_conf: numConfs parameter in EmbedMultipleConfs
        :param max_attempts: maxAttempts parameter in EmbedMultipleConfs
        :param prune: pruneRmsThresh parameter in EmbedMultipleConfs
        :param cache: cache for generated conformers
        """
        self.molecules = molecules
        self.num_conf = num_conf
        self.max_attempts = max_attempts
        self.prune = prune
        self.cache = cache

    def __getitem__(self, item: int):
        if self.cache is not None and item in self.cache:
            return choice(self.cache[item])

        confs = generate(self.molecules[item], self.num_conf, self.max_attempts, self.prune)
        if not confs:
            raise ValueError("conformer generation failed")

        if self.cache is not None:
            self.cache[item] = confs
        return choice(confs)

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self.molecules)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['RDKitConformerDataset']
