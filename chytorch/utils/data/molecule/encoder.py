# -*- coding: utf-8 -*-
#
#  Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from numpy import minimum, nan_to_num, ones
from scipy.sparse.csgraph import shortest_path
from torch import IntTensor, Size, int32, ones as t_ones, zeros
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class MoleculeDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    neighbors: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]


class MoleculeDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch', 'atoms', int]
    neighbors: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]


def collate_molecules(batch, *, collate_fn_map=None) -> MoleculeDataBatch:
    """
    Prepares batches of molecules.

    Note: Distances padded only on right side. Bottom side filled by 1. Required for MHA softmax nan preventing.

    :return: atoms, neighbors, distances.
    """
    atoms, neighbors, distances = [], [], []

    for a, n, d in batch:
        atoms.append(a)
        neighbors.append(n)
        distances.append(d)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = zeros(b, s, s, dtype=int32)
    tmp[:, :, 0] = 1  # prevent nan in MHA softmax on padding
    for i, d in enumerate(distances):
        s = d.size(0)
        tmp[i, :s, :s] = d
    return MoleculeDataBatch(pa, pad_sequence(neighbors, True), tmp)


default_collate_fn_map[MoleculeDataPoint] = collate_molecules  # add auto_collation to the DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, bytes]], *, max_distance: int = 10,
                 add_cls: bool = True, symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 max_neighbors: int = 14, unpack: bool = False, distance_cutoff=None):
        """
        convert molecules to tuple of:
            atoms vector with atomic numbers + 2,
            neighbors vector with connected neighbored atoms count including implicit hydrogens count shifted by 2,
            distance matrix with the shortest paths between atoms shifted by 2.

        Note: atoms shifted to differentiate from padding equal to zero, special cls token equal to 1, and reserved MLM
              task token equal to 2.
              neighbors shifted to differentiate from padding equal to zero and reserved MLM task token equal to 1.
              distances shifted to differentiate from padding equal to zero and from special distance equal to 1
              that code unreachable atoms (e.g. salts).

        :param molecules: molecules collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param symmetric_cls: do bidirectional attention of cls to atoms and back
        :param disable_components_interaction: treat components as isolated molecules
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        :param unpack: unpack molecules
        """
        assert add_cls or not symmetric_cls, 'add_cls should be True if symmetric_cls is True'
        self.molecules = molecules
        # distance_cutoff is deprecated
        self.max_distance = distance_cutoff if distance_cutoff is not None else max_distance
        self.add_cls = add_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.max_neighbors = max_neighbors
        self.unpack = unpack

    def __getitem__(self, item: int) -> MoleculeDataPoint:
        nc = self.max_neighbors
        mol = self.molecules[item]
        if self.unpack:
            mol = MoleculeContainer.unpack(mol)
        if self.add_cls:
            atoms = t_ones(len(mol) + 1, dtype=int32)  # cls token = 1
            neighbors = zeros(len(mol) + 1, dtype=int32)  # cls centrality-encoder disabled by padding trick
        else:
            atoms = IntTensor(len(mol))
            neighbors = IntTensor(len(mol))

        ngb = mol._bonds  # noqa speedup
        hgs = mol._hydrogens  # noqa
        for i, (n, a) in enumerate(mol.atoms(), self.add_cls):
            atoms[i] = a.atomic_number + 2
            nb = len(ngb[n]) + (hgs[n] or 0)  # treat bad valence as 0-hydrogen
            if nb > nc:
                nb = nc
            neighbors[i] = nb + 2

        sp = shortest_path(mol.adjacency_matrix(), method='FW', directed=False, unweighted=True) + 2
        nan_to_num(sp, copy=False, posinf=(0 if self.disable_components_interaction else 1))
        minimum(sp, self.max_distance + 2, out=sp)
        if self.add_cls:
            tmp = ones((len(atoms), len(atoms)))
            if not self.symmetric_cls:
                tmp[1:, 0] = 0  # disable CLS to atom attention by padding trick
            tmp[1:, 1:] = sp
            sp = tmp
        return MoleculeDataPoint(atoms, neighbors, IntTensor(sp))

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['MoleculeDataset', 'MoleculeDataPoint', 'MoleculeDataBatch', 'collate_molecules']
