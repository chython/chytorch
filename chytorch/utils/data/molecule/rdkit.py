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
    if not EmbedMultipleConfs(m, numConfs=num_conf, maxAttempts=max_attempts, pruneRmsThresh=prune):
        # try again ignoring chirality
        EmbedMultipleConfs(m, numConfs=num_conf, maxAttempts=max_attempts, pruneRmsThresh=prune, enforceChirality=False)
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
            raise ValueError('conformer generation failed')

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
